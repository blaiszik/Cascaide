#!/usr/bin/env python
"""Re-score run-dir checkpoints fairly (more samples + optional multi-seed), locally.

The in-training selection (12/energy) and Modal auto-finalize (8/energy) compute the
scorecard on too few samples — small-sample distributional distances (g(r) L1, cluster L1,
NN W1) are biased UPWARD, so those scores read worse than reality. This re-scores each run's
co-located `model.pt` at e.g. 48/energy (pooled over seeds) and overwrites its scorecard so
the dashboard shows apples-to-apples numbers comparable to the baselines.

    python scripts/rescore.py --results results --subset data/real_subset_full.npz \
        --n_per_energy 48 --seeds 2 --device mps
"""
import argparse
import glob
import json
import os

import torch

from cascaide.eval import io, scorecard, registry, render, report
from cascaide.eval.generators import generate_set_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--subset", required=True, help="real reference subset .npz")
    ap.add_argument("--n_per_energy", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=1, help="pool samples over N seeds (lower variance)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on run dir name")
    args = ap.parse_args()

    ref = io.load_subset(args.subset)
    run_dirs = sorted(glob.glob(os.path.join(args.results, "runs", "*")))
    run_dirs = [d for d in run_dirs if os.path.exists(os.path.join(d, "model.pt"))
                and (args.only is None or args.only in d)]
    if not run_dirs:
        raise SystemExit("no run dirs with model.pt found "
                         "(only set-dit-v2 runs co-locate weights)")
    print(f"re-scoring {len(run_dirs)} checkpoint(s) at {args.n_per_energy}/energy "
          f"x {args.seeds} seed(s) vs {len(ref)} real cascades")

    for d in run_dirs:
        ckpt = os.path.join(d, "model.pt")
        label = json.load(open(os.path.join(d, "run.json"))).get("label", os.path.basename(d))
        pooled = []
        for s in range(args.seeds):
            torch.manual_seed(1000 + s)
            gen, _ = generate_set_checkpoint(ckpt, ref, device=args.device,
                                             n_per_energy=args.n_per_energy)
            pooled.extend(gen)
        sc = scorecard.compute(pooled, ref, label=label)
        scorecard.save_json(sc, os.path.join(d, "scorecard.json"))
        if not args.no_render:
            render.render_comparison(pooled, ref, os.path.join(d, "samples"))
        print(f"  {label[:34]:34s} score={sc['score']:.3f} "
              f"({sc['n_requirements_passed']}/{sc['n_requirements']})")

    registry.collect(args.results)
    report.build_report(args.results)
    print(f"\nindex + report rebuilt at {args.results}/ — refresh the dashboard.")


if __name__ == "__main__":
    main()
