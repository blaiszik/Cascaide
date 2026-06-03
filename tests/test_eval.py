"""Tests for the eval/benchmark toolkit. Numpy-only + synthetic clouds → no ovito, no
real data, CI-safe. The torch test is skipped if torch is unavailable.

NB: run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 on this machine (a pytest plugin aborts on
import here — see .continuity/runbook.md).
"""
import numpy as np
import pytest

from cascaide.eval import metrics as M
from cascaide.eval import scorecard


def _fragmented(rng, n_clusters=4, per=12, spread=2.0, sep=30.0):
    """A cloud that is genuinely fragmented into sub-cascades."""
    pts = []
    for _ in range(n_clusters):
        ctr = rng.normal(0, sep, size=3)
        pts.append(ctr + rng.normal(0, spread, size=(per, 3)))
    return np.concatenate(pts, 0).astype(np.float32)


def _blob(rng, n, radius=5.0):
    """A COMPACT gaussian blob (the 'merged' failure mode)."""
    return rng.normal(0, radius, size=(n, 3)).astype(np.float32)


def test_cluster_spectrum_distinguishes_fragmented_from_blob():
    rng = np.random.default_rng(0)
    frag = _fragmented(rng, n_clusters=5, per=10, spread=2.0, sep=30.0)
    blob = _blob(rng, len(frag), radius=5.0)
    # at a linking length between within-cluster spread (~2 Å) and inter-cluster gap
    # (~30 Å), the fragmented cloud holds its sub-cascades while the compact blob merges.
    link = 13.0
    n_frag = M.cluster_stats(frag, link_length=link)["n_clusters"]
    n_blob = M.cluster_stats(blob, link_length=link)["n_clusters"]
    assert n_frag >= 4, n_frag           # ~5 sub-cascades survive
    assert n_blob <= 2, n_blob           # blob collapses to one
    assert n_frag > n_blob


def test_radial_and_rdf_shapes():
    rng = np.random.default_rng(1)
    c = _blob(rng, 80)
    rp = M.radial_density_profile(c, nbins=20)
    assert rp["pdf"].shape == (20,) and abs(rp["pdf"].sum() - 1.0) < 1e-6
    g = M.rdf(c, nbins=20)
    assert g["g"].shape == (20,) and np.all(np.isfinite(g["g"]))


def test_wasserstein_and_js_basic():
    a = np.zeros(50); b = np.ones(50)
    assert abs(M.wasserstein1d(a, b) - 1.0) < 1e-6
    p = np.array([1.0, 0, 0]);
    assert M.js_divergence(p, p) < 1e-9
    assert M.js_divergence(p, np.array([0, 0, 1.0])) > 0.5


def _sample(coords_v, coords_s, e):
    return {"vac": coords_v, "sia": coords_s, "energy": e}


def test_scorecard_real_vs_self_is_perfect():
    rng = np.random.default_rng(2)
    real = [_sample(_fragmented(rng), _fragmented(rng), 50.0) for _ in range(6)]
    sc = scorecard.compute(real, real, label="self")
    assert sc["score"] < 1e-6
    assert sc["n_requirements_passed"] == sc["n_requirements"]


def test_scorecard_blob_fails_substructure_not_shape():
    rng = np.random.default_rng(3)
    real = [_sample(_fragmented(rng), _fragmented(rng), 50.0) for _ in range(8)]
    # blob: same counts & global extent, destroyed substructure
    def blobify(s):
        out = {"energy": s["energy"]}
        for k in ("vac", "sia"):
            c = s[k]; ctr = c.mean(0); r = np.linalg.norm(c - ctr, axis=1).std() + 1e-3
            out[k] = (ctr + rng.normal(0, r, size=c.shape)).astype(np.float32)
        return out
    blob = [blobify(s) for s in real]
    sc = scorecard.compute(blob, real, label="blob")
    req = {r["key"]: r for r in sc["requirements"]}
    # substructure requirements fail; count passes (right number of defects)
    assert req["cluster_l1"]["pass"] is False
    assert req["rdf_l1"]["pass"] is False
    assert req["count_mape"]["pass"] is True
    assert sc["score"] > 0.5


def test_radial_aux_loss_differentiable():
    torch = pytest.importorskip("torch")
    from cascaide.encoding.tanh_image import TanhImageEncoder
    from cascaide.diffusion.loss import RadialDensityAuxLoss

    rng = np.random.default_rng(4)
    vac = _fragmented(rng); sia = _fragmented(rng)
    allc = np.concatenate([vac, sia], 0)
    enc = TanhImageEncoder(image_size=16, centroid=allc.mean(0),
                           tanh_center=np.zeros(3),
                           tanh_scale=np.percentile(np.abs(allc - allc.mean(0)), 99, 0))
    x0 = enc.encode(torch.from_numpy(vac), torch.from_numpy(sia)).unsqueeze(0)
    x0_pred = x0.clone().requires_grad_(True)
    batch = {"vac_coords": [torch.from_numpy(vac)], "sia_coords": [torch.from_numpy(sia)]}
    loss = RadialDensityAuxLoss(weight=1.0, t_threshold_frac=1.0, diffusion_T=10)
    val = loss.compute(x0=x0, x0_pred=x0_pred, xt=x0, t=torch.zeros(1, dtype=torch.long),
                       batch=batch, encoder=enc, info={})
    assert val is not None and torch.isfinite(val)
    val.backward()
    assert x0_pred.grad is not None and torch.isfinite(x0_pred.grad).all()
