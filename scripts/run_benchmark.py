#!/usr/bin/env python
"""Run a benchmark: generate samples, score them against real data, write a run to the
results store (which the dashboard reads).

This is the central INTERFACE between models and the dashboard. A "generator" turns
(energy, n) into cascade samples; we support cheap baselines (calibration reference points)
and real model checkpoints.

    # baselines (no model needed) — populate the dashboard with reference points:
    python scripts/run_benchmark.py --subset data/real_subset.npz --generator all-baselines

    # a real model checkpoint (set / image pipeline):
    python scripts/run_benchmark.py --subset data/real_subset.npz \
        --generator set-checkpoint --checkpoint runs/.../best_model.pt --label "set-dit v3"

Results land in ``--results results/`` as runs/<run_id>/{run.json,scorecard.json}; the
index is rebuilt automatically so the dashboard sees the new run on reload.
"""
import argparse
import os
import numpy as np

from cascaide.eval import io, scorecard, registry, render, report


# --------------------------------------------------------------------- baseline generators
def gen_copy(reference, rng):
    """Resample real clouds (held-out) — the achievable ceiling given finite-sample noise."""
    idx = rng.permutation(len(reference))
    return [reference[i] for i in idx]


def gen_blob(reference, rng):
    """Correct count + global radial extent, destroyed substructure (the model failure)."""
    out = []
    for s in reference:
        o = {"energy": s["energy"]}
        for k in ("vac", "sia"):
            c = s[k]
            if len(c) == 0:
                o[k] = c; continue
            ctr = c.mean(0); rad = np.linalg.norm(c - ctr, axis=1).std() + 1e-3
            o[k] = (ctr + rng.normal(0, rad, size=c.shape)).astype(np.float32)
        out.append(o)
    return out


def gen_jitter(reference, rng, sigma=2.0):
    """Real clouds + small Gaussian jitter — a 'good but blurry' model."""
    out = []
    for s in reference:
        out.append({"energy": s["energy"],
                    "vac": (s["vac"] + rng.normal(0, sigma, s["vac"].shape)).astype(np.float32)
                           if len(s["vac"]) else s["vac"],
                    "sia": (s["sia"] + rng.normal(0, sigma, s["sia"].shape)).astype(np.float32)
                           if len(s["sia"]) else s["sia"]})
    return out


BASELINES = {
    "copy":   ("copy-real",   gen_copy),
    "blob":   ("blob",        gen_blob),
    "jitter": ("jitter-2A",   lambda ref, rng: gen_jitter(ref, rng, 2.0)),
}


def _score_and_write(generated, reference, *, results_dir, label, arch,
                     encoder=None, normalization=None, checkpoint=None, tags=None,
                     render_imgs=True):
    sc = scorecard.compute(generated, reference, label=label)
    manifest = registry.build_manifest(
        label=label, arch=arch, encoder=encoder, normalization=normalization,
        checkpoint=checkpoint, energies_keV=sc["energies_keV"], tags=tags or [arch])
    run_dir = registry.write_run(results_dir, manifest, scorecard=sc)
    if render_imgs:
        # generated (top) vs real (bottom) cascades, in the run's samples/ dir
        render.render_comparison(generated, reference,
                                 os.path.join(run_dir, "samples"))
    registry.collect(results_dir)
    print(f"  [{label}] score={sc['score']:.3f} "
          f"passed {sc['n_requirements_passed']}/{sc['n_requirements']} -> {run_dir}")
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="cached real subset .npz")
    ap.add_argument("--results", default="results", help="results store dir")
    ap.add_argument("--generator", default="all-baselines",
                    help="all-baselines | copy | blob | jitter | "
                         "set-checkpoint | image-checkpoint")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None, help="YAML config (image-checkpoint)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--n_gen_per_energy", type=int, default=48,
                    help="cap generated samples per energy (sampling is slow)")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (checkpoint gen)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reference = io.load_subset(args.subset)
    rng = np.random.default_rng(args.seed)
    print(f"reference: {len(reference)} real cascades from {args.subset}")

    gens = (list(BASELINES) if args.generator in ("all-baselines", "all")
            else [args.generator])

    for g in gens:
        if g in BASELINES:
            arch, fn = BASELINES[g]
            _score_and_write(fn(reference, rng), reference, results_dir=args.results,
                             label=args.label or arch, arch=arch, tags=["baseline"])
        elif g == "set-checkpoint":
            from cascaide.eval.generators import generate_set_checkpoint
            generated, meta = generate_set_checkpoint(
                args.checkpoint, reference, device=args.device,
                n_per_energy=args.n_gen_per_energy)
            _score_and_write(generated, reference, results_dir=args.results,
                             label=args.label or "set-dit", arch="set-dit",
                             normalization=meta.get("normalization"),
                             checkpoint=args.checkpoint, tags=["set-dit"])
        elif g == "paired-checkpoint":
            from cascaide.eval.generators import generate_paired_checkpoint
            generated, meta = generate_paired_checkpoint(
                args.checkpoint, reference, device=args.device,
                n_per_energy=args.n_gen_per_energy)
            _score_and_write(generated, reference, results_dir=args.results,
                             label=args.label or "set-dit-paired", arch="set-dit-paired",
                             normalization=meta.get("normalization"),
                             checkpoint=args.checkpoint, tags=["set-dit-paired", "pair-relative"])
        elif g == "image-checkpoint":
            from cascaide.eval.generators import generate_image_checkpoint
            generated, meta = generate_image_checkpoint(
                args.config, args.checkpoint, reference, n_per_energy=args.n_gen_per_energy)
            _score_and_write(generated, reference, results_dir=args.results,
                             label=args.label or "image-unet", arch="image-unet",
                             encoder=meta.get("encoder"), checkpoint=args.checkpoint,
                             tags=["image-unet"])
        else:
            raise SystemExit(f"unknown generator: {g}")

    report_path = report.build_report(args.results)
    print(f"\nindex rebuilt at {args.results}/index.json")
    print(f"static report at {report_path} (self-contained; browsable via Continuity)")


if __name__ == "__main__":
    main()
