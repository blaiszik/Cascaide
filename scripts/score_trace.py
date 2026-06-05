#!/usr/bin/env python
"""Score a CONVERGENCE TRACE: rescore a sequence of ckpt_<epoch>.pt (a --save_every trace)
offline and plot OVERALL + per-energy-REGIME + rdf_l1 vs epoch. Answers the question
"do they converge early, or keep improving to 500ep?".

Every trace point is scored IDENTICALLY (same reference, same seed -> same sampled conditioning
energies per bin), so the curve is apples-to-apples across epochs. Always score with the corrected
eval default --gen_energy sample (see .continuity/scorecard-findings.md).

    python scripts/score_trace.py \
        --ckpt_dir results/ckpts/percascade_500ep_aug \
        --subset data/cascaide_cascades.npz \
        --device mps --n 24 --sampler dpmpp --steps 20 --energy_bin 25 \
        --baseline results/runs/20260603T082541-set-dit-v2-87ec2c1b/model.pt \
        --out figures/convergence_500ep_aug

Results are cached to <out>.json keyed by (ckpt, n, sampler, steps, energy_bin, gen_energy),
so re-running after more checkpoints land only scores the NEW ones. COST (measured on MPS,
full-range, dpmpp-20): ~220 s/ckpt at n=12, and it's HIGHLY SUBLINEAR in n (n=2 was 194 s) because
the per-bin clouds are batched and the high-E tail (>=200 keV, ~2000 tokens, O(N^2) attn) dominates
a roughly fixed cost -> n=24 is only ~5 min/ckpt, not 2x n=12. So a dense every-25-epoch trace
(20 ckpts) is ~75 min. Use --every K or --only "50,100,500" to thin it further if needed.

Run from the repo root (generators._import_sp_cas() looks for sp_cas.py in CWD).
"""
import argparse, glob, json, os, re, time
import numpy as np
import torch
from cascaide.eval import io, scorecard as scm
from cascaide.eval.generators import generate_set_checkpoint

REGIMES = [("low", 0.0, 50.0), ("mid", 50.0, 150.0), ("high", 150.0, 1e9)]
MIN_REF = 8


def _epoch_of(path):
    """Epoch from the filename (ckpt_0050.pt -> 50); fall back to the checkpoint's stored epoch."""
    m = re.search(r"ckpt_(\d+)\.pt$", os.path.basename(path))
    if m:
        return int(m.group(1))
    try:
        ep = torch.load(path, map_location="cpu", weights_only=False).get("epoch")
        return int(ep) if ep is not None else -1
    except Exception:
        return -1


def _reg(samples, lo, hi):
    return [s for s in samples if lo <= s["energy"] < hi]


def _rdf_bulk(sc, keys):
    pe = sc["per_energy"]
    v = [pe[k]["rdf"]["l1"] for k in keys if k in pe and "rdf" in pe[k]]
    return float(np.mean(v)) if v else float("nan")


def _cache_key(ckpt, args):
    return f"{os.path.basename(ckpt)}|n{args.n}|{args.sampler}|s{args.steps}|b{args.energy_bin}|{args.gen_energy}"


def score_one(ckpt, ref, args, bin_keys, hi_keys):
    """Generate + score a single checkpoint -> dict of OVERALL/regime/rdf scores."""
    t0 = time.time()
    use_sampler = None if args.sampler == "ddpm" else args.sampler
    gen, _ = generate_set_checkpoint(ckpt, ref, device=args.device, n_per_energy=args.n,
                                     energy_bin=args.energy_bin, sampler=use_sampler,
                                     steps=args.steps, gen_energy=args.gen_energy, seed=args.seed)
    sc = scm.compute(gen, ref, label=os.path.basename(ckpt), energy_bin=args.energy_bin)
    rec = {"epoch": _epoch_of(ckpt), "overall": sc["score"],
           "passed": sc["n_requirements_passed"], "total": sc["n_requirements"],
           "rdf": _rdf_bulk(sc, bin_keys), "rdf_hi": _rdf_bulk(sc, hi_keys)}
    for nm, lo, hi in REGIMES:
        r, g = _reg(ref, lo, hi), _reg(gen, lo, hi)
        rec[nm] = scm.compute(g, r, label=nm, energy_bin=args.energy_bin)["score"] \
            if g and len(r) >= MIN_REF else float("nan")
    rec["secs"] = round(time.time() - t0, 1)
    return rec


def plot_trace(records, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = sorted((r for r in records if r["epoch"] >= 0), key=lambda r: r["epoch"])
    eps = [r["epoch"] for r in recs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, lab in [("overall", "OVERALL"), ("low", "low <50"),
                     ("mid", "mid 50-150"), ("high", "high >150")]:
        ys = [r.get(key, float("nan")) for r in recs]
        ax1.plot(eps, ys, marker="o", ms=4, label=lab, lw=(2.2 if key == "overall" else 1.3))
    ax1.set_xlabel("epoch"); ax1.set_ylabel("scorecard score (lower = better)")
    ax1.set_title("Score vs epoch (per regime)"); ax1.legend(fontsize=8); ax1.grid(alpha=.3)
    ax2.plot(eps, [r.get("rdf", float("nan")) for r in recs], marker="o", ms=4, label="rdf_l1 (all)")
    ax2.plot(eps, [r.get("rdf_hi", float("nan")) for r in recs], marker="s", ms=4, label="rdf_l1 (high-E)")
    ax2.axhline(0.20, color="k", ls="--", lw=1, label="rdf target 0.20")
    if args.baseline_rec:
        ax1.axhline(args.baseline_rec["overall"], color="gray", ls=":", lw=1.3,
                    label=f"baseline {args.baseline_rec['overall']:.3f}")
        ax1.legend(fontsize=8)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("rdf_l1"); ax2.set_title("Substructure (rdf_l1) vs epoch")
    ax2.legend(fontsize=8); ax2.grid(alpha=.3)
    fig.suptitle(f"Convergence trace | n={args.n}/energy | {args.sampler}-{args.steps} | "
                 f"gen_energy={args.gen_energy}", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=140, bbox_inches="tight")
    print(f"[trace] wrote {args.out}.png / .pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", required=True, help="dir of ckpt_<epoch>.pt files")
    ap.add_argument("--subset", required=True, help="real dataset npz (scoring reference)")
    ap.add_argument("--energy_max", type=float, default=None, help="restrict reference to <= keV")
    ap.add_argument("--energy_bin", type=float, default=25.0)
    ap.add_argument("--n", type=int, default=24, help="generated samples per energy")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--sampler", choices=["ddpm", "ddim", "dpmpp"], default="dpmpp")
    ap.add_argument("--steps", type=int, default=20, help="#steps for ddim/dpmpp")
    ap.add_argument("--gen_energy", choices=["center", "bin_mean", "sample"], default="sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="score every Kth checkpoint (cost control)")
    ap.add_argument("--only", default=None, help="comma-list of epochs to score, e.g. '50,100,500'")
    ap.add_argument("--include_final", action="store_true",
                    help="also score final_model.pt in --ckpt_dir (epoch from its stored field)")
    ap.add_argument("--baseline", default=None, help="reference checkpoint -> horizontal line")
    ap.add_argument("--out", default="figures/convergence_trace")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    ref = io.load_subset(args.subset)
    if args.energy_max is not None:
        ref = [s for s in ref if s["energy"] <= args.energy_max]
    bin_keys = [str(int(k)) for k in np.arange(args.energy_bin, 301, args.energy_bin)]
    hi_keys = [k for k in bin_keys if int(k) >= 175]

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "ckpt_*.pt")), key=_epoch_of)
    if args.include_final:
        fm = os.path.join(args.ckpt_dir, "final_model.pt")
        if os.path.exists(fm):
            ckpts.append(fm)
    if args.only:
        want = {int(x) for x in args.only.split(",")}
        ckpts = [c for c in ckpts if _epoch_of(c) in want]
    elif args.every > 1:
        ckpts = [c for i, c in enumerate(ckpts) if i % args.every == 0]
    if not ckpts:
        raise SystemExit(f"[trace] no checkpoints matched in {args.ckpt_dir}")

    emax = f"<{int(args.energy_max)}keV" if args.energy_max else "full-range"
    print(f"[trace] {len(ckpts)} ckpt(s) | {len(ref)} ref ({emax}) | n={args.n} | "
          f"{args.sampler}-{args.steps} | gen_energy={args.gen_energy}\n", flush=True)

    cache_path = f"{args.out}.json"
    cache = {}
    if os.path.exists(cache_path):
        cache = {r["_key"]: r for r in json.load(open(cache_path)).get("records", [])}

    # baseline (cached under its own key too)
    args.baseline_rec = None
    if args.baseline:
        bkey = _cache_key(args.baseline, args)
        if bkey in cache:
            args.baseline_rec = cache[bkey]
        else:
            print(f"[trace] baseline {os.path.basename(args.baseline)} ...", flush=True)
            args.baseline_rec = score_one(args.baseline, ref, args, bin_keys, hi_keys)
            args.baseline_rec["_key"] = bkey; args.baseline_rec["_baseline"] = True
            cache[bkey] = args.baseline_rec

    hdr = f"{'ckpt':22} {'ep':>4} {'OVERALL':>8} {'p/t':>5} {'rdf':>6} {'rdf_hi':>7} {'low':>6} {'mid':>6} {'high':>6}"
    print(hdr); print("-" * len(hdr))
    records = []
    for ck in ckpts:
        key = _cache_key(ck, args)
        if key in cache:
            rec = cache[key]; tag = "(cached)"
        else:
            rec = score_one(ck, ref, args, bin_keys, hi_keys)
            rec["_key"] = key; cache[key] = rec; tag = f"({rec['secs']:.0f}s)"
            json.dump({"records": list(cache.values())}, open(cache_path, "w"), indent=1)  # incremental save
        records.append(rec)
        print(f"{os.path.basename(ck):22} {rec['epoch']:>4} {rec['overall']:>8.3f} "
              f"{rec['passed']:>2}/{rec['total']:<2} {rec['rdf']:>6.3f} {rec['rdf_hi']:>7.3f} "
              f"{rec['low']:>6.3f} {rec['mid']:>6.3f} {rec['high']:>6.3f}  {tag}", flush=True)

    json.dump({"records": list(cache.values())}, open(cache_path, "w"), indent=1)
    print(f"\n[trace] cache -> {cache_path}")
    if args.baseline_rec:
        b = args.baseline_rec
        print(f"[trace] baseline OVERALL={b['overall']:.3f} ({b['passed']}/{b['total']}) "
              f"rdf={b['rdf']:.3f} high={b['high']:.3f}")
    if not args.no_plot:
        plot_trace(records, args)


if __name__ == "__main__":
    main()
