"""Differentiable substructure losses for set/point diffusion training.

These attack the documented failure (generated cascades too compact/merged → wrong g(r),
wrong NN spacing, too few sub-cascades). Unlike the radial-density loss — which targets the
statistic the model already matches — these match the *pairwise* structure that carries
substructure.

All operate on padded batches:
    pred, target : (B, N, 3)   coordinates in the model's normalized space
    mask         : (B, N) bool True = real token (collate pads at the end)
Predicted coords should be the model's clean-x0 estimate (gate to low noise t in training,
where x0 is informative). Everything is differentiable w.r.t. ``pred``.
"""
import torch


def _pdist(x):
    """(n,3) -> (n,n) Euclidean distance (clamped for sqrt stability)."""
    d = x[:, None, :] - x[None, :, :]
    return torch.sqrt(torch.clamp((d * d).sum(-1), min=1e-12))


def _upper(d):
    n = d.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=d.device)
    return d[iu[0], iu[1]]


def _soft_hist(values, centers, sigma):
    """RBF-binned soft histogram, normalized to a pdf. Differentiable."""
    if values.numel() == 0:
        return torch.zeros_like(centers)
    k = torch.exp(-((values[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2))
    h = k.sum(0)
    return h / (h.sum() + 1e-8)


def pairwise_hist_loss(pred, target, mask, nbins=24, rmax=6.0, sigma=0.15):
    """Match the pairwise-distance distribution (a differentiable g(r)) per cloud.

    This is the primary substructure lever: it penalizes generated clouds whose
    inter-defect spacing distribution differs from real — i.e. over-merged blobs.
    """
    device = pred.device
    centers = torch.linspace(0.0, rmax, nbins, device=device)
    total = pred.new_zeros(())
    valid = 0
    for b in range(pred.shape[0]):
        n = int(mask[b].sum())
        if n < 2:
            continue
        pd = _upper(_pdist(pred[b, :n]))
        td = _upper(_pdist(target[b, :n].detach()))
        ph = _soft_hist(pd, centers, sigma)
        th = _soft_hist(td, centers, sigma)
        total = total + ((ph - th) ** 2).sum()
        valid += 1
    return total / max(valid, 1)


def rdf_hist_loss(pred, target, mask, nbins=24, rmax_q=0.9, sigma_frac=0.6):
    """Metric-matched differentiable g(r) loss — directly targets the scorecard's ``rdf_l1``
    (its heaviest substructure metric, weight 2.0).

    Mirrors ``cascaide.eval.metrics.rdf``: per-cloud ``rmax`` = the ``rmax_q`` quantile of the
    TARGET pairwise distances (auto-scales with cloud size / energy — unlike the fixed 6 Å of
    ``pairwise_hist_loss``, whose window holds ~1-8% of the real pairs), then the same
    ideal-shell normalization ``g = counts / shell_vol`` and a relative L1 on g(r). pred and
    target share rmax and N, so the density constant cancels and only the metric's defining
    per-bin 1/shell_vol reweighting drives the gradient. Soft (RBF) counts keep it
    differentiable; the target is detached so gradient flows only through ``pred``."""
    total = pred.new_zeros(())
    valid = 0
    for b in range(pred.shape[0]):
        n = int(mask[b].sum())
        if n < 2:
            continue
        pd = _upper(_pdist(pred[b, :n]))
        td = _upper(_pdist(target[b, :n].detach()))
        rmax = torch.clamp(torch.quantile(td, rmax_q), min=1e-3).detach()
        edges = torch.linspace(0.0, float(rmax), nbins + 1, device=pred.device)
        centers = 0.5 * (edges[1:] + edges[:-1])
        sigma = (sigma_frac * (edges[1] - edges[0])).clamp(min=1e-6)
        # soft COUNTS (un-normalized RBF sum) so the ideal-shell normalization applies as in
        # the metric (a pdf-normalized hist would drop the per-bin shell reweighting).
        kp = torch.exp(-((pd[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2)).sum(0)
        kt = torch.exp(-((td[:, None] - centers[None, :]) ** 2) / (2 * sigma ** 2)).sum(0)
        shell = (edges[1:] ** 3 - edges[:-1] ** 3).clamp(min=1e-9)     # ∝ shell volume
        gp = kp / shell
        gt = kt / shell                                                # ideal-shell g(r)
        total = total + (gp - gt).abs().sum() / gt.sum().clamp(min=1e-8)   # relative L1
        valid += 1
    return total / max(valid, 1)


def nn_distance_loss(pred, target, mask, tau=0.05):
    """Match the nearest-neighbor distance distribution per cloud.

    Catches both over-collapse (NN too small) and over-smoothing (NN too uniform). Uses a
    soft-min over neighbors for differentiability, then compares sorted NN-distance vectors
    (a 1-D optimal-transport surrogate; pred and target have equal counts per cloud).
    """
    total = pred.new_zeros(())
    valid = 0
    big = 1e3
    for b in range(pred.shape[0]):
        n = int(mask[b].sum())
        if n < 2:
            continue
        dp = _pdist(pred[b, :n])
        dt = _pdist(target[b, :n].detach())
        eye = torch.eye(n, device=pred.device, dtype=torch.bool)
        dp = dp.masked_fill(eye, big)
        dt = dt.masked_fill(eye, big)
        # soft-min over neighbors for BOTH (consistent → identical clouds give 0);
        # target is detached so only pred carries gradient.
        nn_p = -tau * torch.logsumexp(-dp / tau, dim=1)
        nn_t = -tau * torch.logsumexp(-dt / tau, dim=1)
        total = total + ((torch.sort(nn_p).values - torch.sort(nn_t).values) ** 2).mean()
        valid += 1
    return total / max(valid, 1)


def substructure_loss(pred, target, mask, w_hist=1.0, w_nn=0.5,
                      nbins=24, rmax=6.0, sigma=0.15, tau=0.05, mode="legacy"):
    """Combined substructure loss. Returns (loss, info dict).

    ``mode``: "legacy" = the fixed-6 Å ``pairwise_hist_loss`` (kept for reproducing earlier
    runs; its window holds almost no pairs — see ``rdf_hist_loss``). "rdf" = the
    metric-matched ``rdf_hist_loss`` (adaptive rmax + ideal-shell g(r); directly targets
    ``rdf_l1``). Set ``w_nn=0`` to isolate the RDF lever."""
    lh = (rdf_hist_loss(pred, target, mask, nbins=nbins) if mode == "rdf"
          else pairwise_hist_loss(pred, target, mask, nbins=nbins, rmax=rmax, sigma=sigma))
    ln = nn_distance_loss(pred, target, mask, tau=tau) if w_nn else pred.new_zeros(())
    loss = w_hist * lh + w_nn * ln
    return loss, {"struct_hist": float(lh.detach()), "struct_nn": float(ln.detach())}
