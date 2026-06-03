"""Render generated cascades next to real ones in the same image.

The point of the dashboard isn't just numbers — a human needs to *see* whether the
generated defect clouds are well-shaped. These figures put GENERATED samples (top row) and
REAL samples (bottom row) side by side at matched energy, with vacancies (blue) and SIAs
(red), so shape/substructure differences are immediately visible.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io import group_by_energy

VAC_C, SIA_C = "#3b82f6", "#ef4444"


def _scatter(ax, vac, sia, title, lim=None):
    if len(vac):
        ax.scatter(vac[:, 0], vac[:, 1], vac[:, 2], c=VAC_C, s=16, alpha=.8,
                   edgecolors="none", label=f"vac {len(vac)}", depthshade=True)
    if len(sia):
        ax.scatter(sia[:, 0], sia[:, 1], sia[:, 2], c=SIA_C, s=16, alpha=.8,
                   edgecolors="none", label=f"SIA {len(sia)}", depthshade=True)
    ax.set_title(title, fontsize=9, fontweight="bold")
    if len(vac) + len(sia) > 0:
        ax.legend(fontsize=6, loc="upper left")
    if lim is not None:
        ax.set_xlim(lim[0]); ax.set_ylim(lim[1]); ax.set_zlim(lim[2])
    ax.tick_params(labelsize=5)
    ax.view_init(elev=20, azim=45)


def _limits(samples, margin=8.0):
    pts = [np.concatenate([s["vac"], s["sia"]], 0)
           for s in samples if len(s["vac"]) + len(s["sia"]) > 0]
    if not pts:
        return [(-1, 1)] * 3
    allp = np.concatenate(pts, 0)
    return [(allp[:, i].min() - margin, allp[:, i].max() + margin) for i in range(3)]


def render_comparison(generated, reference, out_dir, n_per_energy=4, seed=0):
    """One PNG per energy: GENERATED (top row) vs REAL (bottom row). Returns list of paths.

    Axis limits are shared across both rows per energy so size/shape are comparable.
    Generated samples must be true model output (decoded coords), not re-encoded reals.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    gen_g, ref_g = group_by_energy(generated), group_by_energy(reference)
    paths = []
    for e in sorted(set(gen_g) & set(ref_g)):
        gs, rs = gen_g[e], ref_g[e]
        k = min(n_per_energy, len(gs), len(rs))
        if k == 0:
            continue
        gi = rng.choice(len(gs), k, replace=False)
        ri = rng.choice(len(rs), k, replace=False)
        # shared limits from BOTH rows so the eye compares like-for-like
        lim = _limits([gs[i] for i in gi] + [rs[i] for i in ri])

        fig = plt.figure(figsize=(3.3 * k, 6.8), facecolor="white")
        for j in range(k):
            g = gs[gi[j]]
            ax = fig.add_subplot(2, k, j + 1, projection="3d")
            _scatter(ax, g["vac"], g["sia"], f"GENERATED · {int(e)} keV", lim)
            r = rs[ri[j]]
            ax = fig.add_subplot(2, k, k + j + 1, projection="3d")
            _scatter(ax, r["vac"], r["sia"], f"real · {int(e)} keV", lim)
        fig.suptitle(f"{int(e)} keV  —  generated (top) vs real (bottom)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        p = os.path.join(out_dir, f"compare_{int(e)}keV.png")
        fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(p)
    return paths
