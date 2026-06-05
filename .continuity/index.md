# Cascaide

> Generative diffusion models for **radiation-damage cascade defects in tungsten (W)**.
> Given a primary-knock-on-atom (PKA) energy, generate plausible 3D point clouds of the
> vacancies and self-interstitial atoms (SIAs) a collision cascade leaves behind.

## What this is

Molecular-dynamics (LAMMPS) simulations of collision cascades are expensive. Cascaide
learns the conditional distribution `p(defect cloud | energy)` from a corpus of minimized
cascade dumps so new configurations can be sampled cheaply. Defects come in **Frenkel
pairs** (`n_vac == n_sia`), and counts are heavy-tailed and rise with energy.

There are **two parallel model families** in the repo:

1. **Image pipeline** (`cascaide/` package) — encode the 3D cloud into a 2D image, run a
   conditional 2D-UNet DDPM, decode back to coordinates. Modular, YAML-driven.
2. **Set/point pipeline** (`sp_cas.py`, single file) — a DiT transformer that diffuses
   directly on point coordinates, plus a separate count head (energy → number of pairs).
   No lossy image encoding. Newer; not yet folded into the package.

## Run it

```sh
pip install -e .                                   # editable install (needs ovito for .dump)
pytest                                             # the 2 existing unit tests
# Image pipeline:
python cascaide/training/train.py --config cascaide/configs/config.yaml
python infer.py --config <cfg> --checkpoint runs/<name>/checkpoints/best.pt --output_dir <dir>
# Set pipeline (self-contained; falls back to synthetic data if DATA_ROOT missing):
python sp_cas.py
```

## Start here

- **Agents:** read [`handoff.md`](handoff.md) first, then [`workboard.md`](workboard.md)
  and [`research-directions.md`](research-directions.md).
- **Humans:** skim this page, then the workboard for what's in flight.

## Map

- `handoff.md` — current state, the two known bugs, what to do next.
- `workboard.md` — scope, priorities, now / next / later.
- `decisions.md` — cross-cutting decisions and why.
- `research-directions.md` — benchmarks, normalization, architectures, autoresearch paths.
- `journal.md` — dated build log of the model-progress tooling (start here for 2026-06-02).
- `polaris.md` — **ALCF Polaris handoff** (deploy/polaris kit, resume steps, gotchas). Active 2026-06-03.
- `experiments.md` — **Polaris experiment registry**: exact re-runnable sweep defs (Exp 1 scheduler / Exp 2 / preemptable full-range) + preemption-recovery steps + fast-sampling finding. Active 2026-06-03.
- `sampling-scan.md` — **DPM-Solver++ fast-inference scan** across all models (overall + per-energy-regime; ~20× sampler). Approximate/screening. Active 2026-06-03.
- `scorecard-findings.md` — **what the scorecards reveal about the models**: radial is solved, per_cascade > global ~3×, high-E is convergence; + the bin-0 eval artifact & the `--gen_energy` fix. (Its "RDF is the lever" call was later reversed — see next.) Active 2026-06-04.
- `substructure-loss-negative.md` — **team briefing: the RDF/g(r) aux loss is a dead end** (converged equal-epoch test; hurts every metric incl. `rdf_l1`; `<100 keV` model already passes 7/7). 2nd coordinate aux loss to fail after radial → don't bolt distributional losses on coords. Active 2026-06-04.
- `convergence-500ep-augment.md` — **the new best model: 500 epochs + `--augment_rot`** beats the 250ep baseline by ~0.19 OVERALL / −0.33 high-E (n=24, every regime). Includes the fine-grained n=12 convergence trace (20 ckpts; slow improvement, no early plateau) and the deep n=24 verdict + the n=24 sampling-variance caveat. **`final_model.pt` is the new standing best.** Active 2026-06-05.
- `runbook.md` — env setup (the `cascaide` mamba env w/ ovito), data gotchas, real-data facts.
- `log.md` — running session notes (append-only).

## Track model progress (2026-06-02)

Benchmark + scorecard + human dashboard for "how close are models to the requirements,
especially on **substructure**." See `docs/benchmarks.md`. TL;DR:

```sh
python scripts/build_real_subset.py --data_root <dumps> --out data/real_subset.npz
python scripts/run_benchmark.py --subset data/real_subset.npz --generator all-baselines
python -m http.server 8000   # open http://localhost:8000/viewer/dashboard.html
```
