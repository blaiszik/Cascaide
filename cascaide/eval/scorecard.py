"""Turn {generated, reference} cascade sets into a per-energy scorecard.

The scorecard is the single artifact a human (via the dashboard) and an autoresearch loop
(as a fitness function) read to judge "how close is this model to the real distribution,
especially on substructure."

Output is a JSON-serializable dict with a stable ``schema_version`` so results produced on
an HPC cluster months apart stay loadable. Each per-energy block carries both scalar
summaries AND the gen-vs-real curves (radial profile, g(r), cluster fragmentation spectrum)
so the dashboard can overlay them.

A "sample" is a dict {'vac': (Nv,3), 'sia': (Ns,3), 'energy': keV}. Spatial-structure
metrics are computed on the UNION of vac+sia (the whole defect cloud); counts/conservation/
separation are tracked per class.
"""
import json
import numpy as np

from . import metrics as M
from .io import group_by_energy

SCHEMA_VERSION = 1

# Requirement thresholds (tunable). A run "passes" a requirement if value <= target.
# Weights set how much each contributes to the overall fitness score; substructure
# (cluster spectrum + g(r)) is weighted hardest because that's the known failure mode.
REQUIREMENTS = {
    "count_mape":      {"label": "Count error (MAPE)",        "target": 0.15, "unit": "",  "weight": 1.0},
    "conservation":    {"label": "Frenkel violation |nv-ns|", "target": 1.0,  "unit": "",  "weight": 1.0},
    "radial_js":       {"label": "Radial profile (JS div)",   "target": 0.05, "unit": "",  "weight": 1.5},
    "rdf_l1":          {"label": "g(r) shape (rel. L1)",      "target": 0.20, "unit": "",  "weight": 2.0},
    "cluster_l1":      {"label": "Sub-cascade spectrum (L1)", "target": 0.15, "unit": "",  "weight": 3.0},
    "nn_w1":           {"label": "NN-distance W1",            "target": 1.0,  "unit": "Å", "weight": 1.0},
    "vac_sia_sep_err": {"label": "vac-SIA separation error",  "target": 3.0,  "unit": "Å", "weight": 1.0},
}


def _union(s):
    parts = [p for p in (s["vac"], s["sia"]) if len(p)]
    return np.concatenate(parts, 0) if parts else np.zeros((0, 3), np.float32)


def _mean_profile(clouds, kind, rmax, nbins):
    """Average a per-cloud curve onto a fixed grid. kind in {'radial','rdf'}."""
    acc = None
    n = 0
    for c in clouds:
        if len(c) == 0:
            continue
        if kind == "radial":
            y = M.radial_density_profile(c, rmax=rmax, nbins=nbins)["pdf"]
        else:
            y = M.rdf(c, rmax=rmax, nbins=nbins)["g"]
        acc = y if acc is None else acc + y
        n += 1
    if acc is None:
        return np.zeros(nbins)
    return acc / n


def _mean_spectrum(clouds, scales):
    acc = None
    n = 0
    for c in clouds:
        if len(c) == 0:
            continue
        y = M.cluster_spectrum(c, scales)["n_clusters"]
        acc = y if acc is None else acc + y
        n += 1
    return (acc / n) if acc is not None else np.zeros(len(scales))


def _per_energy(real, gen, nbins=24, scales=M.DEFAULT_LINK_SCALES):
    real_u = [_union(s) for s in real]
    gen_u = [_union(s) for s in gen]

    # common radial/rdf grids from the REAL clouds (so gen is judged on real's scale)
    real_rad = [np.linalg.norm(c - c.mean(0), axis=1).max() for c in real_u if len(c)]
    rmax_radial = float(np.percentile(real_rad, 95)) * 1.05 if real_rad else 1.0
    real_pair = []
    for c in real_u:
        if len(c) >= 2:
            d = c[:, None] - c[None]
            real_pair.append(np.percentile(np.sqrt((d * d).sum(-1))[np.triu_indices(len(c), 1)], 90))
    rmax_rdf = float(np.mean(real_pair)) if real_pair else 1.0

    radial_real = _mean_profile(real_u, "radial", rmax_radial, nbins)
    radial_gen = _mean_profile(gen_u, "radial", rmax_radial, nbins)
    rdf_real = _mean_profile(real_u, "rdf", rmax_rdf, nbins)
    rdf_gen = _mean_profile(gen_u, "rdf", rmax_rdf, nbins)
    spec_real = _mean_spectrum(real_u, scales)
    spec_gen = _mean_spectrum(gen_u, scales)

    # counts / conservation / separation
    r_tot = np.array([len(s["vac"]) + len(s["sia"]) for s in real], float)
    g_tot = np.array([len(s["vac"]) + len(s["sia"]) for s in gen], float)
    g_cons = np.mean([abs(len(s["vac"]) - len(s["sia"])) for s in gen]) if gen else float("nan")
    r_sep = np.mean([M.vac_sia_separation(s["vac"], s["sia"])["centroid_dist"] for s in real]) if real else 0.0
    g_sep = np.mean([M.vac_sia_separation(s["vac"], s["sia"])["centroid_dist"] for s in gen]) if gen else 0.0

    nn_real = np.concatenate([M.nn_distances(c) for c in real_u if len(c) >= 2] or [np.zeros(0)])
    nn_gen = np.concatenate([M.nn_distances(c) for c in gen_u if len(c) >= 2] or [np.zeros(0)])

    # ---- additive diagnostics (NOT in the weighted score → keeps scores comparable) ----
    xnn_real = np.concatenate([M.cross_nn_distances(s["vac"], s["sia"]) for s in real]
                              or [np.zeros(0)])
    xnn_gen = np.concatenate([M.cross_nn_distances(s["vac"], s["sia"]) for s in gen]
                             or [np.zeros(0)])
    rg_real = np.array([M.radius_of_gyration(c) for c in real_u if len(c)] or [0.0])
    rg_gen = np.array([M.radius_of_gyration(c) for c in gen_u if len(c)] or [0.0])
    lcf_real = np.mean([M.largest_cluster_frac(c) for c in real_u if len(c)] or [0.0])
    lcf_gen = np.mean([M.largest_cluster_frac(c) for c in gen_u if len(c)] or [0.0])
    diagnostics = {
        # vac->SIA nearest-neighbor (Frenkel separation) distribution — a better vac-SIA
        # metric than the centroid scalar; candidate to PROMOTE into the weighted score.
        "vac_sia_nn_w1": (M.wasserstein1d(xnn_real, xnn_gen)
                          if len(xnn_real) and len(xnn_gen) else float("nan")),
        "vac_sia_nn_real_mean": float(xnn_real.mean()) if len(xnn_real) else 0.0,
        "vac_sia_nn_gen_mean": float(xnn_gen.mean()) if len(xnn_gen) else 0.0,
        # radius of gyration (cascade size) distribution distance
        "rg_w1": M.wasserstein1d(rg_real, rg_gen),
        "rg_real_mean": float(rg_real.mean()), "rg_gen_mean": float(rg_gen.mean()),
        # largest-cluster fraction — blob(≈1) vs fragmented; direct substructure read
        "largest_cluster_frac_real": float(lcf_real),
        "largest_cluster_frac_gen": float(lcf_gen),
    }

    r_grid = np.linspace(0, rmax_radial, nbins + 1)
    r_grid = 0.5 * (r_grid[1:] + r_grid[:-1])
    g_grid = np.linspace(0, rmax_rdf, nbins + 1)
    g_grid = 0.5 * (g_grid[1:] + g_grid[:-1])

    return {
        "n_real": len(real), "n_gen": len(gen),
        "count": {"real_mean": float(r_tot.mean()) if len(r_tot) else 0.0,
                  "gen_mean": float(g_tot.mean()) if len(g_tot) else 0.0,
                  "real_std": float(r_tot.std()) if len(r_tot) else 0.0,
                  "gen_std": float(g_tot.std()) if len(g_tot) else 0.0,
                  "mape": float(abs(g_tot.mean() - r_tot.mean()) / max(r_tot.mean(), 1e-9))
                          if len(r_tot) and len(g_tot) else float("nan"),
                  "w1": M.wasserstein1d(r_tot, g_tot) if len(r_tot) and len(g_tot) else float("nan")},
        "conservation": {"gen_mean_violation": float(g_cons)},
        "radial": {"r": r_grid.tolist(), "real": radial_real.tolist(),
                   "gen": radial_gen.tolist(), "js": M.js_divergence(radial_real, radial_gen)},
        "rdf": {"r": g_grid.tolist(), "real": rdf_real.tolist(), "gen": rdf_gen.tolist(),
                "l1": M.curve_l1(rdf_real, rdf_gen)},
        "cluster_spectrum": {"scales": list(map(float, scales)),
                             "real": spec_real.tolist(), "gen": spec_gen.tolist(),
                             "l1": M.curve_l1(spec_real, spec_gen)},
        "nn": {"w1": M.wasserstein1d(nn_real, nn_gen) if len(nn_real) and len(nn_gen) else float("nan"),
               "real_mean": float(nn_real.mean()) if len(nn_real) else 0.0,
               "gen_mean": float(nn_gen.mean()) if len(nn_gen) else 0.0},
        "vac_sia_sep": {"real": float(r_sep), "gen": float(g_sep),
                        "err": float(abs(r_sep - g_sep))},
        "diagnostics": diagnostics,
    }


def _collect_requirement_values(pe):
    """Pull the scalar each requirement is scored on out of a per-energy block."""
    return {
        "count_mape": pe["count"]["mape"],
        "conservation": pe["conservation"]["gen_mean_violation"],
        "radial_js": pe["radial"]["js"],
        "rdf_l1": pe["rdf"]["l1"],
        "cluster_l1": pe["cluster_spectrum"]["l1"],
        "nn_w1": pe["nn"]["w1"],
        "vac_sia_sep_err": pe["vac_sia_sep"]["err"],
    }


def _bin_energy(samples, width):
    """Snap each sample's energy to the nearest ``width`` keV bin center (for continuous
    energies). Returns new sample dicts; originals untouched."""
    if not width:
        return samples
    return [{**s, "energy": round(s["energy"] / width) * width} for s in samples]


def compute(generated, reference, label="run", nbins=24,
            scales=M.DEFAULT_LINK_SCALES, energy_decimals=0, energy_bin=None):
    """Build a scorecard dict from generated + reference sample lists.

    Returns a JSON-serializable dict (see module docstring / SCHEMA_VERSION).
    The overall ``score`` is a weighted mean of (value/target) across requirements and
    energies — lower is better, <1 on a requirement means it passes. This is the scalar an
    autoresearch loop optimizes.
    """
    if energy_bin:                       # bin continuous energies so gen/ref align
        generated = _bin_energy(generated, energy_bin)
        reference = _bin_energy(reference, energy_bin)
    gen_g = group_by_energy(generated, energy_decimals)
    ref_g = group_by_energy(reference, energy_decimals)
    energies = sorted(set(gen_g) & set(ref_g))

    per_energy, req_accum = {}, {k: [] for k in REQUIREMENTS}
    for e in energies:
        pe = _per_energy(ref_g[e], gen_g[e], nbins=nbins, scales=scales)
        per_energy[str(int(e)) if energy_decimals == 0 else str(e)] = pe
        for k, v in _collect_requirement_values(pe).items():
            if v is not None and np.isfinite(v):
                req_accum[k].append(v)

    # overall requirement values (mean across energies) + pass/fail + weighted score
    requirements, num, den = [], 0.0, 0.0
    for k, spec in REQUIREMENTS.items():
        vals = req_accum[k]
        value = float(np.mean(vals)) if vals else float("nan")
        passed = bool(np.isfinite(value) and value <= spec["target"])
        requirements.append({"key": k, "label": spec["label"], "value": value,
                             "target": spec["target"], "unit": spec["unit"],
                             "weight": spec["weight"], "pass": passed})
        if np.isfinite(value):
            num += spec["weight"] * (value / spec["target"])
            den += spec["weight"]
    score = float(num / den) if den else float("nan")

    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "energies_keV": [float(e) for e in energies],
        "n_reference": len(reference), "n_generated": len(generated),
        "score": score,
        "n_requirements_passed": int(sum(r["pass"] for r in requirements)),
        "n_requirements": len(requirements),
        "requirements": requirements,
        "per_energy": per_energy,
    }


def save_json(scorecard, path):
    with open(path, "w") as f:
        json.dump(scorecard, f, indent=2)
    return path
