# Runbook / field notes

> Environment + data paper-cuts already paid for. Don't re-pay them.

_Last updated: 2026-06-02_

## Python env (ovito) — `cascaide` mamba env

The base `/usr/bin/python3` (3.9, CommandLineTools) has **no numpy/torch/ovito**. A
dedicated mamba env was created:

```sh
mamba create -y -n cascaide -c https://conda.ovito.org -c conda-forge \
    python=3.11 ovito numpy
mamba run -n cascaide python ...        # run anything in it
# or: mamba activate cascaide
```

- Installed: **ovito 3.15.0, numpy 2.4.6, Python 3.11** (matches `pyproject` ≥3.10).
- `ovito` is the OVITO Basic Python module from the official `conda.ovito.org` channel
  (pulls Qt6/ffmpeg/openvino — large, but isolated to this env).
- **Not yet installed:** `torch`, `tqdm`, `einops`, `hilbertcurve`, `PyYAML`,
  `matplotlib`. Add them before training:
  `mamba run -n cascaide pip install torch tqdm einops hilbertcurve PyYAML matplotlib`
  (or `mamba run -n cascaide pip install -e .`).

## Running tests / torch on this machine

- A pytest **plugin auto-imported here aborts the interpreter** (Fatal Python error during
  numpy import under pytest). Run the suite with plugin autoload off:
  ```sh
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 KMP_DUPLICATE_LIB_OK=TRUE \
      mamba run -n cascaide python -m pytest -q
  ```
- `KMP_DUPLICATE_LIB_OK=TRUE` guards against the duplicate-OpenMP abort when torch + conda
  numpy/ovito are imported together (belt-and-suspenders; plain `python` is usually fine).
- `tests/test_eval.py` is numpy-only (synthetic clouds, no ovito) and CI-safe; its torch
  test auto-skips if torch is missing. `tests/test_base_image.py` needs ovito + real dumps.

## New dataset: raw_data/ (continuous energy, 0-300 keV) — 2026-06-02

`raw_data/` (gitignored, 254 MB) holds **7,100 minimized cascades** across 9 continuous
energy ranges (0-300 keV), W, 1073 K. Same dump format as the old 3-energy set. Process to
a single dependency-free .npz:

```sh
python scripts/process_raw_data.py --raw_dir raw_data --out data/cascaide_cascades.npz
```

- Output loads directly via `cascaide.eval.io.load_subset` (+ extra keys: `n_vac`, `n_sia`,
  `centroid`, `energy_range`, `supercell`, `summary_json`). 17 MB, ~1 min to build (parallel,
  pure-numpy parser — no ovito; finds x/y/z by header name so it's robust to sia's extra
  `c_myKE c_myPE` columns).
- **CRITICAL caveat — supercell scales with energy** (sc50 → sc280): absolute coordinate
  frames differ wildly across ranges, so a single GLOBAL centroid/scale is meaningless here.
  Downstream normalization must **center per-cascade** (the `centroid` array is stored for
  this). The existing `CoordNormalizer` (global mean) would break on this mixed-box data.
- **Big cascades:** max defect count is **1,107** (vs ~232 in the old data) → up to ~2,200
  set tokens at high E → O(N²) attention is heavy (this is what OOM'd MPS on synthetic).
  Train-time handling (chunking / efficient attention / cap) needed for the high-E tail.
- Energy is continuous per-cascade (0.01-299.8 keV) — enables held-out-energy
  interpolation/extrapolation tests. (Old 3-point data was sc80; this is sc50-sc280.)

## Polaris (ALCF) — submit kit (2026-06-03)

`deploy/polaris/` — PBS scripts to train on Polaris (4× A100/node, PBS Pro, `mpiexec`/PALS).
Workflow: **code via git** (commit → `git pull` on Polaris), **data/artifacts via Globus**
(local ↔ Eagle); Polaris reads/writes `/eagle` directly.
- `config.sh` — edit once: `PROJECT` (PBS `-A`), `QUEUE`, Eagle paths, Globus endpoint UUIDs.
  (REPO auto-detected; PROJECT/endpoints are placeholders — fill in.)
- `setup_env.sh` — **lightweight venv on Eagle over ALCF base** (`--system-site-packages`
  reuses CUDA torch in place) + `pip install scipy matplotlib` + `pip install -e . --no-deps`.
  Do NOT `conda --clone` to Eagle — it copies ~158k files and Lustre is pathologically slow
  at small files (hung a setup ~15+ min, 2026-06-03). Activate: `module use /soft/modulefiles;
  module load conda; conda activate base; source $ENV_PREFIX/bin/activate`.
- `submit.sh train …` (1 A100, debug/preemptable) | `submit.sh sweep experiments.txt`
  (one experiment/GPU across nodes; sizes nodes=ceil(N/4); prod for ≥10 nodes).
- `train.pbs`, `sweep.pbs`+`gpu_worker.sh` (PALS rank→GPU), `experiments.txt` (design matrix).
- `globus.sh push|pull|pull-ckpts`.
Queues: debug (≤2 nodes/≤1h), debug-scaling (≤10/≤1h), preemptable (≤20/≤72h, preemptible),
prod (≥10 nodes/≤24h). Our runs ~35min on A100 → fit `debug`. Added `--no_report` to
train_set_v2 so parallel sweep ranks don't race on index.json (sweep.pbs rebuilds once).
**Couldn't verify on Polaris from here** — module path / endpoint UUIDs / PROJECT need
confirming on first run. Docs: `deploy/polaris/README.md`.

## Training on raw_data — per-cascade centering (2026-06-02)

The mixed-supercell data needs **per-cascade centering** (a global centroid is meaningless).
Folded into `cascaide/setdiff/data.py`:
- `PerCascadeNormalizer` — centers each cascade on its **(vac+SIA) union centroid**, then one
  **global per-axis p99 scale**. Preserves vac↔SIA relative geometry (shared-centroid shift
  cancels) and the size-vs-energy signal. Generated clouds are origin-centered; it serializes
  as a `G=[0,0,0]` CoordNormalizer so the existing sp_cas load/generate path decodes correctly
  (coords*s) with **no other plumbing changes**. (Per-cascade scale comes out isotropic ~86 Å,
  vs the old global per-axis ~10% anisotropy.)
- `SetDataset` — per-cascade-centered set dataset, drop-in with `sp_cas.collate_dynamic`.

New `train_set_v2.py` flags: `--center per_cascade|global` (default per_cascade),
`--energy_max <keV>` (subset), `--energy_bin <keV>` (bin continuous energies for scoring —
the scorecard/generator now accept `energy_bin`), `--count_cap`. Example (match the old runs
on the <100 keV subset, local MPS, ~2 h):

```sh
python scripts/train_set_v2.py --subset data/cascaide_cascades.npz \
    --energy_max 100 --energy_bin 10 --center per_cascade \
    --epochs 300 --count_cap 200 --output_dir runs/setv2_lt100 --results results --device mps
```

## Data format gotcha — vac vs SIA dump headers differ

The `.dump` files are plain LAMMPS text, but the columns are **not identical**:

- `*_min_vac.dump` header: `ITEM: ATOMS id type x y z`
- `*_min_sia.dump` header: `ITEM: ATOMS id type x y z c_myKE c_myPE`  ← extra columns

A hand-rolled parser that matches the exact vac header string will read **zero SIA atoms**
and produce a *false* "Frenkel conservation violated" signal. Use ovito's `import_file`
(as `CascadeDataset._load_coordinates` does), or key on the `ITEM: ATOMS` prefix and take
columns 2–4 only.

## Real-data facts (verified via ovito, 2026-06-02)

Sampled ~25 cascades per energy from the local corpus (W, 1073 K, sc80):

| energy | n_defects (min / med / max) | Frenkel `n_vac==n_sia` |
|-------:|:----------------------------|:-----------------------|
| 20 keV | 10 / 19 / 28                | holds (0 mismatches)   |
| 50 keV | 16 / 36 / 55                | holds                  |
| 100 keV| 58 / 77 / 112               | holds                  |

- Counts scale ~linearly with energy (~0.8 defects/keV here); conservation is exact.
- **Real cascades are SMALL** (tens to ~112 points). The synthetic generator in
  `sp_cas.py` makes them up to ~1100 with mean ≈ energy_keV — oversized ~10× and
  mis-scaled. Anything calibrated on synthetic (image_size, `count_cap=1300`, the
  `H*W/2≈1024` cap) is calibrated to fiction; the `H*W/2` cap is **not** binding for this
  20–100 keV corpus (correction to an earlier over-emphasis in the handoff).
- Total corpus: 3000 cascades (1000 each at 20/50/100 keV), 6 dump variants each;
  loaders use only `_min_vac.dump` + `_min_sia.dump`.
