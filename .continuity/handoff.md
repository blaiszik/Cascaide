# Handoff

> The first thing the next agent reads. Keep it current and forward-looking.

_Last updated: 2026-06-02 by Claude (benchmark/dashboard tooling build — see journal.md)_

## NEW (2026-06-02): model-progress tooling — see `journal.md` + `docs/benchmarks.md`

Built and verified: `cascaide/eval/` (metrics incl. sub-cascade `cluster_spectrum`,
scorecard with requirements pass/fail + fitness score, HPC results `registry`,
checkpoint `generators`), `scripts/build_real_subset.py` + `scripts/run_benchmark.py`,
`RadialDensityAuxLoss` (the previously-missing radial loss, wired + config block),
`viewer/dashboard.html` (self-contained human viewer), `tests/test_eval.py` (6 pass).
The scorecard provably isolates the "captures shape, misses substructure" failure (a
merged blob passes count/radial, fails g(r)/cluster/NN). Results store is shaped for HPC:
each job writes `results/runs/<id>/{run.json,scorecard.json}`; `registry.collect()` rebuilds
`index.json` that the dashboard filters on.

**Key result (set-DiT, 1200 real cascades, MPS):** fully trained (ep300) the sp_cas DiT
scores **0.453** (6/7 pass) — well-shaped *in absolute terms*. The earlier "misses
substructure" was **undertraining**: ep300 vs val-selected ep~50 (0.772) vs ~ep60 (1.08).
NOTE: these are all the SAME architecture at different checkpoints — we have NOT benchmarked
any alternative, so "sp_cas is the BEST architecture" is the researcher's claim, not an
experimental result (it's consistent with our data, not proven by it). Build on it for now;
to actually verify "best", score the image UNet / variants on the same scorecard. Confirmed
experimentally: **select by scorecard, not val coord-MSE** (val froze at ep50 but ep300 is
clearly better). The lone
failing metric is `rdf_l1` (0.229 vs 0.20) — short-range pair correlation — which the new
substructure loss directly targets. Next iteration tooling: `scripts/train_set_v2.py`
(`w_struct=0` == faithful `sp_cas.train`), `cascaide/setdiff/*`, Modal launcher under
`deploy/modal/` (built, NOT submitted — needs go-ahead + hyperparam confirmation).

**Bug fixed this session:** `cascaide/diffusion/core.py` now imports `CosineSchedule`
(the default-schedule path raised `NameError`). **Still open:**
`tests/test_diffusion_core.py` draws `t∈[0,100)` with `T=50` → 3 tests overflow the gather
(test bug, not core). Run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 KMP_DUPLICATE_LIB_OK=TRUE`
(see `runbook.md`).

## State

- **Works now:**
  - Image pipeline trains end-to-end: `Base`, `Hilbert3Ch`, `Hilbert4Ch` encoders →
    2D-UNet DDPM (eps/x0/v; linear/cosine/sigmoid schedules) → DDPM/DDIM sampling, EMA,
    stratified-energy split, best-checkpoint tracking, aux losses (occupancy, count,
    classification, multi-axis projection). Driven entirely by YAML.
  - Set/point pipeline (`sp_cas.py`) is self-contained and runs against either the real
    dumps or a built-in `SyntheticCascadePairDataset` (no ovito needed) — a DiT set
    denoiser + a count head (energy → log-normal over `n_pairs`).
  - `pytest` passes (2 tests: diffusion-core math; Base encoder encode/decode shapes).
- **In progress (current branch `tanhp99-global-centering`):**
  - New `TanhImageEncoder` (`TanhP99`): global centering `G` + per-axis
    `tanh((x-G-c)/s)*coord_range`, `image_size=64`, decodes to **absolute** coords.
    Config: `cascaide/configs/config_tanhp99.yaml`. Built to mirror `sp_cas.py`'s
    normalization/conditioning (energy_keV / 300).
  - Untracked: `cascade_studio_best.html` ("AI Cascade Studio", a browser visualizer) and
    `sp_cas.py` itself — neither is committed yet.
- **Broken / blocked — two confirmed bugs (verified by reading the code):**
  1. **TanhP99 inference is broken.** `infer.py:244` calls
     `build_encoder(cfg, raw_ds.global_centroid)` with **no `dataset=`**, but
     `build_encoder` (`trainer.py:70-78`) raises `ValueError` for `TanhP99` when
     `dataset is None` (it needs the cloud to recompute `tanh_center`/`tanh_scale`).
     So the current branch's headline encoder cannot be used for inference as written.
  2. **`energy_vector` conditioner crashes.** `trainer.py:101` imports
     `EnergyVectorConditioner`, but `conditioner.py:26` defines it as
     `EnergyEmbedConditioner`. `conditioner.name: energy_vector` → `ImportError`.
     (All shipped configs use `energy_channel`, so it's latent.)

## Next step (post-tooling)

The benchmark loop is ready, so the highest-value next move is to **point it at a real
trained checkpoint**: `python scripts/run_benchmark.py --subset data/real_subset.npz
--generator set-checkpoint --checkpoint <best_model.pt>` (the set/sp_cas model is the most
ready). Then turn on `RadialDensityAuxLoss` (config block in `config_tanhp99.yaml`) and
compare scorecards on the dashboard to see if substructure (`cluster_l1`/`rdf_l1`) improves.
If radial alone doesn't close the substructure gap, add a differentiable g(r)/cluster loss.

Original backlog still stands:

Decide whether to **fix TanhP99 inference** or **promote `sp_cas.py` into the package**
(see `research-directions.md` §3 — the set model sidesteps the lossy image encoders).
Quickest unblock for bug #1: persist `tanh_center`/`tanh_scale`/`centroid` in the
checkpoint at train time and reload them in `infer.py` instead of recomputing — `sp_cas.py`
already does exactly this via `CoordNormalizer.to_dict()` (`sp_cas.py:41-50, 475-498`).
Start at `cascaide/training/trainer.py:369` (`_build_checkpoint_dict`) and
`infer.py:244`.

## Watch out for

- **`energy_norm_factor` must match between train and infer**, or energy conditioning is
  silently miscalibrated. It differs across configs: `config.yaml`=170000,
  `base_config.yaml`=100000, `config_tanhp99.yaml`=300000. The tanhp99 config comments
  explain why (matching the monolith's keV/300).
- **Image encoders are lossy and cap at `H*W/2` points per type.** A 64×64 image holds
  ≤2048 px → ≤1024 vac + ≤1024 SIA. High-energy cascades can exceed this and are silently
  truncated. This is the core motivation for the set/point model.
- **Encoder normalization params are not persisted** in image-pipeline checkpoints (unlike
  `sp_cas.py`). Any change to the dataset or split between train and infer shifts the
  computed `centroid`/`tanh` params → decode drifts.
- `train.py` does `from trainer import Trainer` (relies on the script's own dir being on
  `sys.path`); run it as `python cascaide/training/train.py` (not `python -m ...`).
- `n_vac == n_sia` (Frenkel conservation) is a hard physical invariant the data obeys —
  the set pipeline exploits it (one count → both classes); generated samples should be
  checked against it as a sanity metric.

## Read these first

- `sp_cas.py` — the entire set/point pipeline in one file (DiT + count head + diffusion).
- `cascaide/training/trainer.py` — `build_*` factories + `Trainer`; the assembly point.
- `cascaide/encoding/tanh_image.py` + `cascaide/data/dataset.py` — the new normalization.
- `cascaide/diffusion/loss.py` — main weighted-MSE + the 4 composable aux losses.
- `research-directions.md` — benchmarks / normalization / architectures / autoresearch.
