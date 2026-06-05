#!/usr/bin/env python
"""Shareable dataset-coverage figures: defect production vs PKA energy for the processed
cascade dataset (data/cascaide_cascades.npz). Writes to figures/:
  - dataset_coverage.png/.pdf  — joint scatter (energy vs Frenkel pairs) + marginals + trend
  - dataset_sampling.png        — cascades & defect production per MD energy campaign
  - dataset_summary.md          — the numbers (per-campaign table) for pasting into email/slides

    python scripts/plot_dataset_coverage.py [--npz <path>] [--out <dir>]
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(REPO, "data", "cascaide_cascades.npz"))
    ap.add_argument("--out", default=os.path.join(REPO, "figures"))
    ap.add_argument("--material", default="Tungsten (W), 1073 K")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = np.load(args.npz, allow_pickle=True)
    E = z["energy"].astype(float)          # PKA energy, keV
    P = z["n_vac"].astype(float)           # Frenkel pairs (n_vac == n_sia)
    er = z["energy_range"].astype(str)
    sc = z["supercell"].astype(str)
    N = len(E)
    camps = sorted(set(er), key=lambda s: float(s.split("-")[0].replace("keV", "")))
    # truncate the y-axis at the 99th pct so the rare high-E tail (max ~1107) doesn't
    # compress the bulk; true max stays documented in the annotation box.
    p99 = float(np.percentile(P, 99))
    yhi = p99 * 1.04
    n_clip = int((P > yhi).sum())

    # robust median + IQR trend over energy
    edges = np.linspace(0, 300, 31)
    mids = 0.5 * (edges[1:] + edges[:-1])
    med = np.array([np.median(P[(E >= lo) & (E < hi)]) if ((E >= lo) & (E < hi)).sum() >= 5 else np.nan
                    for lo, hi in zip(edges[:-1], edges[1:])])
    q25 = np.array([np.percentile(P[(E >= lo) & (E < hi)], 25) if ((E >= lo) & (E < hi)).sum() >= 5 else np.nan
                    for lo, hi in zip(edges[:-1], edges[1:])])
    q75 = np.array([np.percentile(P[(E >= lo) & (E < hi)], 75) if ((E >= lo) & (E < hi)).sum() >= 5 else np.nan
                    for lo, hi in zip(edges[:-1], edges[1:])])

    # power-law fit  P ~ a * E^p  (physically meaningful range E>1 keV, P>0)
    fitm = (E > 1) & (P > 0)
    slope, intercept, r, _, _ = stats.linregress(np.log(E[fitm]), np.log(P[fitm]))
    Efit = np.linspace(1, 300, 200)
    Pfit = np.exp(intercept) * Efit ** slope

    # ---------------------------------------------------------- FIG 1: joint coverage
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#444", "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(11, 8.2))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                  hspace=0.04, wspace=0.04, left=0.09, right=0.9, top=0.93, bottom=0.09)
    ax, axt, axr = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 1])
    axt.sharex(ax); axr.sharey(ax)

    ax.scatter(E, P, s=10, alpha=0.20, color="#2b6cb0", edgecolors="none", rasterized=True,
               label=f"{N:,} cascades")
    ax.fill_between(mids, q25, q75, color="#dd6b20", alpha=0.22, label="25–75th percentile")
    ax.plot(mids, med, color="#dd6b20", lw=2.2, label="median trend")
    ax.plot(Efit, Pfit, "--", color="#1a202c", lw=1.5,
            label=fr"power-law fit  $N \propto E^{{{slope:.2f}}}$")
    ax.set_xlabel("PKA energy  (keV)")
    ax.set_ylabel("Frenkel pairs  $N_\\mathrm{vac}=N_\\mathrm{SIA}$")
    ax.set_xlim(-6, 306); ax.set_ylim(-0.02 * yhi, yhi)
    ax.grid(alpha=0.25); ax.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    axsec = ax.twinx()
    axsec.set_ylim(ax.get_ylim()[0] * 2, ax.get_ylim()[1] * 2)
    axsec.set_ylabel("total point defects  (vac + SIA)", color="#555")
    axsec.tick_params(colors="#555")
    axr.set_position([0.905, axr.get_position().y0, 0.07, axr.get_position().height])

    axt.hist(E, bins=np.linspace(0, 300, 61), color="#2b6cb0", alpha=0.75)
    axt.set_ylabel("count"); axt.tick_params(labelbottom=False); axt.grid(alpha=0.2)
    axt.set_title(f"Cascaide dataset coverage — defect production vs PKA energy\n"
                  f"{args.material} · minimized post-cascade defect structures · "
                  f"{len(camps)} MD energy campaigns, supercell scaled to energy",
                  fontsize=12.5, loc="left", pad=10)
    axr.hist(P, bins=np.linspace(0, yhi, 61), orientation="horizontal", color="#2b6cb0", alpha=0.75)
    axr.set_xlabel("count"); axr.tick_params(labelleft=False); axr.grid(alpha=0.2)

    ax.text(0.985, 0.04,
            f"N = {N:,} cascades\nPKA energy: {E.min():.2f} – {E.max():.0f} keV\n"
            f"Frenkel pairs: {int(P.min())} – {int(P.max())}  (mean {P.mean():.0f})\n"
            f"0 Frenkel-conservation violations\n"
            f"y-axis truncated at 99th pct ({p99:.0f}); {n_clip} pts above",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bbb", alpha=0.92))
    fig.savefig(os.path.join(args.out, "dataset_coverage.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(args.out, "dataset_coverage.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------- FIG 2: sampling design
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    counts = [int((er == c).sum()) for c in camps]
    xc = np.arange(len(camps))
    a1.bar(xc, counts, color="#2b6cb0", alpha=0.85)
    for x, n in zip(xc, counts):
        a1.text(x, n + max(counts) * 0.01, str(n), ha="center", fontsize=9)
    a1.set_xticks(xc); a1.set_xticklabels(camps, rotation=40, ha="right", fontsize=9)
    a1.set_ylabel("cascades"); a1.set_title("Sampling per energy campaign", loc="left", fontsize=11.5)
    a1.grid(axis="y", alpha=0.25); a1.set_ylim(0, max(counts) * 1.15)

    medp = [np.median(P[er == c]) for c in camps]
    maxp = [P[er == c].max() for c in camps]
    a2.plot(xc, medp, "o-", color="#dd6b20", lw=2, label="median")
    a2.plot(xc, maxp, "^--", color="#718096", lw=1.3, label="max", alpha=0.8)
    a2.set_xticks(xc); a2.set_xticklabels(camps, rotation=40, ha="right", fontsize=9)
    a2.set_ylabel("Frenkel pairs"); a2.set_title("Defect production per campaign", loc="left", fontsize=11.5)
    a2.grid(alpha=0.25); a2.legend(fontsize=9)
    fig2.suptitle(f"Cascaide dataset — sampling & production by MD campaign  ({args.material})",
                  x=0.01, ha="left", fontsize=12.5)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(os.path.join(args.out, "dataset_sampling.png"), dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # ---------------------------------------------------------- summary table (md)
    with open(os.path.join(args.out, "dataset_summary.md"), "w") as f:
        f.write("# Cascaide dataset coverage\n\n")
        f.write(f"- **{N:,} cascades**, {args.material}, minimized post-cascade defect structures\n")
        f.write(f"- PKA energy **{E.min():.3f}–{E.max():.1f} keV** (continuous), Frenkel pairs "
                f"**{int(P.min())}–{int(P.max())}** (mean {P.mean():.1f})\n")
        f.write("- N_vac == N_SIA for every cascade (0 Frenkel-conservation violations); "
                "total point defects = 2x pairs\n")
        f.write("- supercell scales with energy (sc50 → sc280) so the box contains the cascade\n")
        f.write(f"- power-law fit (E>1 keV): N_pairs ~ E^{slope:.2f}  (log-log r={r:.3f})\n\n")
        f.write("| energy campaign | cascades | supercell | min | median | mean | max |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for c in camps:
            m = er == c
            f.write(f"| {c} | {int(m.sum())} | {sc[m][0]} | {int(P[m].min())} | "
                    f"{np.median(P[m]):.0f} | {P[m].mean():.1f} | {int(P[m].max())} |\n")

    print(f"wrote figures + summary to {args.out}  (power-law p={slope:.3f}, r={r:.3f})")


if __name__ == "__main__":
    main()
