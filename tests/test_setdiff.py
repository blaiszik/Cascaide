"""Tests for the next-iteration set/point modules (substructure losses + pair-relative).

Run with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 KMP_DUPLICATE_LIB_OK=TRUE (see runbook.md).
"""
import numpy as np
import pytest

from cascaide.setdiff import pairing


# -------------------------------------------------------------------- pairing (item 2)
def test_pair_relative_roundtrip_recovers_sia():
    rng = np.random.default_rng(0)
    vac = rng.normal(0, 20, size=(15, 3)).astype(np.float32)
    disp_true = rng.normal(0, 4, size=(15, 3)).astype(np.float32)
    sia = vac + disp_true
    vm, disp = pairing.to_pair_relative(vac, sia)
    _, sia_rec = pairing.from_pair_relative(vm, disp)
    # Hungarian recovers the exact pairing -> SIAs recovered up to ordering
    assert np.allclose(np.sort(sia_rec, axis=0), np.sort(sia, axis=0), atol=1e-3)


def test_pair_relative_shrinks_dynamic_range():
    rng = np.random.default_rng(1)
    vac = rng.normal(0, 30, size=(40, 3)).astype(np.float32)
    sia = vac + rng.normal(0, 3, size=(40, 3)).astype(np.float32)  # small offsets
    _, disp = pairing.to_pair_relative(vac, sia)
    assert np.abs(disp).max() < np.abs(sia).max()        # displacement range << absolute


def test_token_roundtrip():
    vac = np.arange(9, dtype=np.float32).reshape(3, 3)
    disp = np.ones((3, 3), np.float32)
    tok = pairing.pairs_to_tokens(vac, disp)
    assert tok.shape == (3, 6)
    v2, d2 = pairing.tokens_to_pairs(tok)
    assert np.allclose(v2, vac) and np.allclose(d2, disp)


# -------------------------------------------------------------------- losses (item 1)
def test_substructure_loss_zero_when_equal_and_differentiable():
    torch = pytest.importorskip("torch")
    from cascaide.setdiff.losses import substructure_loss
    rng = np.random.default_rng(2)
    x = torch.tensor(rng.normal(0, 1, size=(1, 20, 3)), dtype=torch.float32)
    mask = torch.ones(1, 20, dtype=torch.bool)
    pred = x.clone().requires_grad_(True)
    loss, info = substructure_loss(pred, x, mask)
    assert float(loss) < 1e-4                              # identical clouds -> ~0
    loss2, _ = substructure_loss(pred + 0.5, x, mask)
    assert float(loss2) > float(loss)                     # perturbed -> larger
    loss2.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_rdf_hist_loss_metric_matched():
    torch = pytest.importorskip("torch")
    from cascaide.setdiff.losses import rdf_hist_loss, substructure_loss
    rng = np.random.default_rng(5)
    # a cloud whose pairwise distances span tens of units (nn ~ several units) -> the legacy
    # 6-unit window would be near-empty; the adaptive-rmax rdf loss must still be informative.
    x = torch.tensor(rng.normal(0, 20, size=(1, 24, 3)), dtype=torch.float32)
    mask = torch.ones(1, 24, dtype=torch.bool)
    pred = x.clone().requires_grad_(True)
    loss = rdf_hist_loss(pred, x, mask)
    assert float(loss) < 1e-4                              # identical clouds -> ~0
    worse = rdf_hist_loss(pred * 0.4, x, mask)             # collapse -> different g(r)
    assert float(worse) > 1e-2 and float(worse) > float(loss)
    worse.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    # mode='rdf' routes through rdf_hist_loss; w_nn=0 isolates the RDF term
    lm, info = substructure_loss(pred, x, mask, mode="rdf", w_nn=0.0)
    assert float(lm) < 1e-4 and info["struct_nn"] == 0.0


def test_pair_normalizer_roundtrip_recovers_vac_sia():
    from cascaide.setdiff.paired import PairNormalizer
    from cascaide.setdiff import pairing
    rng = np.random.default_rng(7)
    vac = rng.normal(0, 30, size=(20, 3)).astype(np.float32)
    sia = vac + rng.normal(0, 3, size=(20, 3)).astype(np.float32)
    vm, disp = pairing.to_pair_relative(vac, sia)
    nz = PairNormalizer().fit(vm, disp)
    tok = nz.transform(vm, disp)
    assert tok.shape == (len(vm), 6)
    v2, s2 = nz.inverse(tok)
    assert np.allclose(np.sort(v2, 0), np.sort(vm, 0), atol=1e-2)
    assert np.allclose(np.sort(s2, 0), np.sort(vm + disp, 0), atol=1e-2)


def test_paired_denoiser_forward_and_sample():
    torch = pytest.importorskip("torch")
    from cascaide.setdiff.paired import PairedSetDenoiser
    den = PairedSetDenoiser(d_model=32, depth=2, n_heads=4)
    x = torch.randn(2, 10, 6)
    t = torch.randint(0, 100, (2,))
    e = torch.rand(2)
    out = den(x, t, e)
    assert out.shape == (2, 10, 6) and torch.isfinite(out).all()


def test_build_pair_items():
    torch = pytest.importorskip("torch")
    from cascaide.setdiff.paired import build_pair_items
    rng = np.random.default_rng(8)
    samples = []
    for _ in range(5):
        v = rng.normal(0, 20, (12, 3)).astype(np.float32)
        samples.append({"vac": v, "sia": v + rng.normal(0, 3, (12, 3)).astype(np.float32),
                        "energy": 50.0})
    items, ce, cn, norm = build_pair_items(samples, energy_divisor=300.0)
    assert len(items) == 5 and items[0]["tokens"].shape[1] == 6
    assert len(ce) == 5 and float(cn[0]) == 12


def test_per_cascade_normalizer_preserves_vac_sia_geometry():
    from cascaide.setdiff.data import PerCascadeNormalizer
    rng = np.random.default_rng(11)
    # two cascades at very different absolute positions (mimics mixed supercells)
    a = {"vac": rng.normal(50, 8, (10, 3)).astype(np.float32),
         "sia": rng.normal(70, 8, (10, 3)).astype(np.float32), "energy": 20.0}
    b = {"vac": rng.normal(400, 8, (10, 3)).astype(np.float32),
         "sia": rng.normal(430, 8, (10, 3)).astype(np.float32), "energy": 200.0}
    nz = PerCascadeNormalizer().fit([a, b])
    vn, sn = nz.encode(a["vac"], a["sia"])
    # shared-centroid centering => relative vac-SIA geometry preserved up to the global scale
    rel_raw = a["vac"][0] - a["sia"][0]
    rel_dec = (vn[0] - sn[0]) * nz.s
    assert np.allclose(rel_dec, rel_raw, atol=1e-2)
    # both cascades map to a comparable centered range despite the 8x absolute offset
    assert abs(np.abs(vn).max() - np.abs(nz.encode(b["vac"], b["sia"])[0]).max()) < 3.0
    d = nz.to_dict()
    assert d["G"] == [0.0, 0.0, 0.0] and "s" in d   # CoordNormalizer-compatible (G=0)


def test_set_dataset_builds():
    torch = pytest.importorskip("torch")
    from cascaide.setdiff.data import PerCascadeNormalizer, SetDataset
    rng = np.random.default_rng(12)
    samples = [{"vac": rng.normal(0, 20, (n, 3)).astype(np.float32),
                "sia": rng.normal(30, 20, (n, 3)).astype(np.float32), "energy": e}
               for n, e in [(8, 20.0), (15, 60.0), (0, 5.0)]]  # incl an empty cascade
    nz = PerCascadeNormalizer().fit(samples)
    ds = SetDataset(samples, nz, energy_divisor=300.0)
    assert len(ds) == 2 and len(ds.count_npairs) == 3   # empties excluded from items, kept in counts
    coords, types, energy, npairs = ds[0]
    assert coords.shape[1] == 3 and set(types.tolist()) <= {0, 1}


def test_pairwise_hist_loss_flags_merged_blob():
    torch = pytest.importorskip("torch")
    from cascaide.setdiff.losses import pairwise_hist_loss
    rng = np.random.default_rng(3)
    # target: dispersed (two separated groups); pred: collapsed toward centroid
    g = np.concatenate([rng.normal(-5, 0.5, (10, 3)), rng.normal(5, 0.5, (10, 3))], 0)
    target = torch.tensor(g[None], dtype=torch.float32)
    merged = torch.tensor((g * 0.2)[None], dtype=torch.float32)   # squashed -> over-merged
    mask = torch.ones(1, 20, dtype=torch.bool)
    l_good = pairwise_hist_loss(target, target, mask)
    l_bad = pairwise_hist_loss(merged, target, mask)
    assert float(l_bad) > float(l_good)
