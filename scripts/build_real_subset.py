#!/usr/bin/env python
"""Cache a small, stratified REAL cascade subset to a dependency-free .npz.

Run ONCE (needs ovito); everything downstream loads the .npz with numpy alone.

    python scripts/build_real_subset.py \
        --data_root cascaide/dataset/defect_files_20260218/defect_files_20260218 \
        --out data/real_subset.npz --per_energy 64
"""
import argparse
from cascaide.eval import io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="data/real_subset.npz")
    ap.add_argument("--per_energy", type=int, default=64,
                    help="cascades per energy dir (use -1 for all)")
    ap.add_argument("--max_total", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    per_energy = None if args.per_energy < 0 else args.per_energy
    summary = io.cache_real_subset(args.data_root, args.out,
                                   per_energy=per_energy, max_total=args.max_total,
                                   seed=args.seed)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
