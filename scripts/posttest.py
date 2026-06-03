#!/usr/bin/env python
"""Post-test all models on the FULL metric set (incl. the new diagnostics) and refresh the
dashboard / Continuity report.

For each run it ensures the scorecard carries the latest metrics by regenerating:
  - baselines (copy/blob/jitter)         — cheap reference anchors
  - named checkpoints (--ckpt)           — e.g. the local set_local_v1 best/final
  - every run-dir that already has co-located weights (model.pt) — re-scored in place
Runs without model.pt and not regenerated are dropped (they'd be stale). Idempotent:
re-run it whenever a new model (e.g. the EMA / paired run) lands.

    python scripts/posttest.py --subset data/real_subset_full.npz --n 32 --device mps \
        --ckpt "set-dit ep300::runs/set_local_v1/final_model.pt::set" \
        --ckpt "set-dit ep~50 best-val::runs/set_local_v1/best_model.pt::set"
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys


def sh(cmd):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--n", type=int, default=32, help="samples/energy (uniform, for comparability)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt", action="append", default=[],
                    help='"label::path::set|paired" (repeatable)')
    ap.add_argument("--skip_baselines", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    dev = ["--device", args.device] if args.device else []

    # 1. drop stale runs WITHOUT co-located weights (baselines + prior checkpoint benchmarks);
    #    they're regenerated below with the current metric set. model.pt runs are kept + rescored.
    for d in sorted(glob.glob(os.path.join(args.results, "runs", "*"))):
        if os.path.isdir(d) and not os.path.exists(os.path.join(d, "model.pt")):
            print(f"  drop stale (no weights): {os.path.basename(d)}")
            shutil.rmtree(d)

    # 2. baselines (fast; no model)
    if not args.skip_baselines:
        sh([py, "scripts/run_benchmark.py", "--subset", args.subset,
            "--generator", "all-baselines", "--results", args.results])

    # 3. named checkpoints
    for spec in args.ckpt:
        label, path, arch = spec.split("::")
        gen = "paired-checkpoint" if arch == "paired" else "set-checkpoint"
        sh([py, "scripts/run_benchmark.py", "--subset", args.subset, "--generator", gen,
            "--checkpoint", path, "--label", label, "--n_gen_per_energy", str(args.n),
            "--results", args.results] + dev)

    # 4. re-score every run that has co-located weights (sweep / EMA / paired) in place
    sh([py, "scripts/rescore.py", "--results", args.results, "--subset", args.subset,
        "--n_per_energy", str(args.n)] + dev)

    print(f"\nPost-test complete — {args.results}/report.html refreshed "
          f"(Continuity cover updates on reload).")


if __name__ == "__main__":
    main()
