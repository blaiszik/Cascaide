#!/usr/bin/env python
"""Train the set/point DiT model (sp_cas.py) on a cached REAL subset, locally.

Uses the dependency-free cached subset (no ovito reload) and the self-contained set-diffusion
trainer. Set models fit cascade point clouds directly (no lossy image encoding), so generated
samples are true coordinates — ideal for judging whether cascades are well-shaped.

    python scripts/train_set_local.py --subset data/real_subset_large.npz \
        --output_dir runs/set_local_v1 --epochs 300 --device mps
"""
import argparse
import importlib.util
import os


def _load_sp():
    path = os.path.join(os.getcwd(), "sp_cas.py")
    spec = importlib.util.spec_from_file_location("sp_cas", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--output_dir", default="runs/set_local_v1")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--energy_divisor", type=float, default=300.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from cascaide.eval import io
    sp = _load_sp()

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    samples = io.load_subset(args.subset)
    raw = [(s["vac"], s["sia"]) for s in samples]
    energies = [s["energy"] for s in samples]  # keV

    class _DS(sp._PairDatasetBase):
        def __init__(self):
            self._finalize(raw, energies, normalizer=None,
                           energy_divisor=args.energy_divisor)

    ds = _DS()
    print(f"[train_set_local] {len(ds)} non-empty cascades | device={device} | "
          f"norm G={ds.normalizer.G.round(1)} s={ds.normalizer.s.round(1)}")

    sp.train(
        ds, output_dir=args.output_dir,
        d_model=args.d_model, depth=args.depth, n_heads=args.n_heads,
        T=1000, schedule="cosine", batch_size=args.batch_size, lr=args.lr,
        weight_decay=1e-2, num_epochs=args.epochs,
        count_lr=1e-3, count_steps_per_epoch=50,
        log_every=5, save_every=50,
        sample_every=0,                       # skip slow in-train sampling; render after
        energy_divisor=args.energy_divisor, count_cap=400, device=device, seed=42,
    )
    print(f"[train_set_local] done -> {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
