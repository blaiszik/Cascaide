#!/usr/bin/env python
"""Score ONE trained set-DiT checkpoint: generate n/energy, compute the scorecard (overall +
per-energy-REGIME), write a run to the results store. Generation (T=1000 sampling) is the cost,
so this is meant to run on an A100 (--device cuda) via deploy/polaris/score.pbs — much faster
than local MPS, and the checkpoints already live on Eagle.

    python scripts/score_checkpoint.py --checkpoint <path> --subset <npz> \
        --results results --label "cosine final" --energy_max 100 --energy_bin 25 --n 48

Per-regime: the same generated/real clouds are sliced into energy regimes and scored
independently, so we see WHERE a model fails (the aggregate hides it). Regimes with too few
real cascades are skipped. Stored in the run manifest config under "regime_scores".
"""
import argparse, os
from cascaide.eval import io, scorecard as scm, registry, render, report
from cascaide.eval.generators import generate_set_checkpoint

# energy regimes (keV): edit here to change the lens. half-open [lo, hi).
REGIMES = [("low", 0.0, 50.0), ("mid", 50.0, 150.0), ("high", 150.0, 1e9)]
MIN_REF = 8   # need at least this many real cascades in a regime to score it


def _in(samples, lo, hi):
    return [s for s in samples if lo <= s["energy"] < hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--subset", required=True, help="real dataset npz (the scoring reference)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--label", default=None)
    ap.add_argument("--energy_max", type=float, default=None, help="restrict reference to <= keV")
    ap.add_argument("--energy_bin", type=float, default=25.0)
    ap.add_argument("--n", type=int, default=48, help="generated samples per energy")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sampler", choices=["ddpm", "ddim", "dpmpp"], default="ddpm",
                    help="ddpm = full T=1000 (faithful, default). ddim/dpmpp = fast few-step "
                         "(needs --steps); dpmpp = DPM-Solver++(2M), the higher-order one to prefer.")
    ap.add_argument("--steps", type=int, default=None,
                    help="#steps for ddim/dpmpp (e.g. ~20 dpmpp / ~50 ddim). Ignored for ddpm.")
    ap.add_argument("--gen_energy", choices=["center", "bin_mean", "sample"], default="sample",
                    help="energy each generated cloud is conditioned on. DEFAULT 'sample' = one "
                         "real reference energy per cloud (gold standard: generation is energy-"
                         "matched to the reference within each bin). 'center' = the LEGACY bin-"
                         "center behavior, which biases bin-0 (generates at 0 keV ≈ no cascade "
                         "vs a ref window averaging ~5.5 keV) and inflates the low regime ~2x; "
                         "use it to reproduce pre-2026-06-04 scores. 'bin_mean' = the bin's mean "
                         "ref energy (one energy per bin).")
    ap.add_argument("--tag", action="append", default=[], help="extra tag(s)")
    ap.add_argument("--no_report", action="store_true",
                    help="write the run but skip collect()/build_report() (batch jobs rebuild once)")
    args = ap.parse_args()

    ref = io.load_subset(args.subset)
    if args.energy_max is not None:
        ref = [s for s in ref if s["energy"] <= args.energy_max]
    label = args.label or os.path.basename(os.path.dirname(args.checkpoint)) or "checkpoint"
    emax = f", <{int(args.energy_max)}keV" if args.energy_max else ", full-range"
    print(f"[score] {label}: {len(ref)} ref cascades{emax} | n={args.n}/energy | {args.device}", flush=True)

    import time
    t0 = time.time()
    use_sampler = None if args.sampler == "ddpm" else args.sampler
    defsteps = 20 if args.sampler == "dpmpp" else 50
    samp_label = "DDPM-1000" if args.sampler == "ddpm" else f"{args.sampler}-{args.steps or defsteps}"
    print(f"[score] sampling {samp_label} on {args.device}...", flush=True)
    gen, meta = generate_set_checkpoint(args.checkpoint, ref, device=args.device,
                                        n_per_energy=args.n, energy_bin=args.energy_bin,
                                        steps=args.steps, sampler=use_sampler,
                                        gen_energy=args.gen_energy)
    dt = time.time() - t0
    print(f"[score] generated {len(gen)} clouds in {dt:.0f}s ({dt/max(1,len(gen)):.1f}s/cloud); "
          f"scoring overall + per-regime...", flush=True)

    sc = scm.compute(gen, ref, label=label, energy_bin=args.energy_bin)

    # per-energy-regime sub-scores (same clouds, sliced by energy)
    regime_scores = {}
    for name, lo, hi in REGIMES:
        r, g = _in(ref, lo, hi), _in(gen, lo, hi)
        if len(r) < MIN_REF or len(g) == 0:
            continue
        rsc = scm.compute(g, r, label=f"{label}/{name}", energy_bin=args.energy_bin)
        regime_scores[name] = {"score": round(rsc["score"], 4),
                               "passed": rsc["n_requirements_passed"], "total": rsc["n_requirements"],
                               "n_ref": len(r), "keV": [lo, None if hi > 1e8 else hi]}
        print(f"[score]   regime {name:4s} [{lo:.0f}-{'inf' if hi>1e8 else int(hi)} keV] "
              f"n_ref={len(r):4d}  score={rsc['score']:.3f}  {rsc['n_requirements_passed']}/{rsc['n_requirements']}",
              flush=True)

    manifest = registry.build_manifest(
        label=label, arch="set-dit-v2", normalization=meta.get("normalization", "percascade-p99"),
        checkpoint="model.pt", energies_keV=sc["energies_keV"],
        config={"energy_max": args.energy_max, "energy_bin": args.energy_bin, "score_n": args.n,
                "sampler": samp_label, "steps": args.steps, "gen_energy": args.gen_energy,
                "regime_scores": regime_scores, "source_checkpoint": args.checkpoint},
        tags=["set-dit-v2", "rescore"] + list(args.tag))
    run_dir = registry.write_run(args.results, manifest, scorecard=sc)
    import shutil
    try:
        shutil.copy(args.checkpoint, os.path.join(run_dir, "model.pt"))
    except OSError as e:
        print(f"[score]   (could not co-locate model.pt: {e})", flush=True)
    try:
        render.render_comparison(gen, ref, os.path.join(run_dir, "samples"))
    except Exception as e:
        print(f"[score]   render skipped: {e}", flush=True)

    if not args.no_report:
        registry.collect(args.results); report.build_report(args.results)
    print(f"[score] {label}: OVERALL score={sc['score']:.3f} "
          f"{sc['n_requirements_passed']}/{sc['n_requirements']} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
