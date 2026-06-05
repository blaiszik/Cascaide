# Handoff

> The first thing the next agent reads. Keep it current and forward-looking.

_Last updated: 2026-06-05 by Claude (500ep + augment_rot landed = new best model; convergence trace + n=24 verdict)._
_Branch: `tanhp99-global-centering` (fork `blaiszik/Cascaide`, the push target — `origin`=vigsam is read-only)._

## ✅ LANDED — 500ep + augmentation is the NEW BEST MODEL (full writeup: `convergence-500ep-augment.md`)

The best-bet Modal run (app `ap-KUIg1l1P6BaEjLTsvipaZR`) **finished clean at ep 500/500** and the
result is decisive: **per_cascade · 500 epochs · `--augment_rot`** beats the prior-best 250ep
baseline by **~0.19 OVERALL and −0.33 on high-E** (n=24, full-range, dpmpp-20, `--gen_energy sample`),
winning **every regime and rdf metric** and clearing one more requirement (6/7 vs 5/7).
→ **`results/ckpts/percascade_500ep_aug/final_model.pt` is the new standing best per_cascade model.**

- **Convergence question answered ("do they converge early?"): no early plateau** — a fine-grained
  n=12 trace over all 20 checkpoints shows slow, noisy improvement throughout, best band ep 375–500;
  `rdf_l1`/`rdf_hi`/high-E hit run-minima at ep 500. Figure: `figures/convergence_500ep_aug.png`.
- **Caveat that matters:** n=24 still wobbles ~±0.15 absolute run-to-run (the `seed` arg doesn't pin
  the reverse-diffusion sampler) — trust *within-run* deltas, not cross-run absolutes. Seed the
  sampler before the next cross-run verdict. See `convergence-500ep-augment.md` §4.
- Tooling added: `scripts/score_trace.py` (cached convergence-trace scorer that can trail a live
  run), `/tmp/verdict.py` (n=24 head-to-head). All 20 ckpts pulled to `results/ckpts/percascade_500ep_aug/`.

**Immediate next options:** capacity 512/10 (budget-permitting) and even-longer epochs are the open
levers (curve hadn't flattened by 500); high-E is improved but still the weakest regime (0.64).

## What we know (this session's results — receipts in the linked docs)

- **per_cascade centering > global** — clean matched A/B (same Modal harness, only `--center`
  differs): global is worse on EVERY metric/regime (OVERALL +0.22, rdf_l1 +0.10/~62%). The
  branch name (`global-centering`) is a dead end; **keep per_cascade**. Mechanism: supercells
  scale with energy (sc50→sc280) so each cascade is in its own frame — a single global centroid +
  dataset-wide scale (s≈270 vs per_cascade ≈145) squashes per-cloud structure. See `log.md` 2026-06-05.
- **High-E is a CONVERGENCE problem**, not architecture (250ep high-regime ≈0.47–0.67 vs 2.3–3.6
  for un-converged baselines). This is the only real headroom → the in-flight 500ep run.
- **`<100 keV` is SOLVED** — converged per_cascade passes the whole scorecard **7/7** (incl. rdf_l1
  0.16 < 0.20 target).
- **Auxiliary distributional losses on coordinates DON'T WORK** (proven twice): the radial-density
  loss and the metric-matched RDF g(r) loss (`--struct_mode rdf`) both HURT every metric incl. their
  own. Don't bolt distributional losses on the coord objective — they corrupt the score field. See
  `substructure-loss-negative.md`.
- **`energy_balance` de-motivated**, lattice-snap ruled out (see `experiments.md` / `log.md`).
- **Training loss ≠ sample quality** (seen 3×). Lower eps loss is often a normalization artifact;
  judge ONLY by the offline scorecard.

**Proven recipe to build on:** per_cascade · EMA · cosine LR · 256/6 · `--max_defects 1000` · batch 16
· **500 epochs · `--augment_rot`** (both now PROVEN positive — see `convergence-500ep-augment.md`).
Untested-but-plausible levers left: capacity 512/10 (budget-permitting), even-longer epochs (the
500ep curve hadn't flattened), `--min_snr` (lower priority).

## ⚠️ Eval correctness — ALWAYS score with `--gen_energy sample`

`score_checkpoint.py --gen_energy` defaults to **`sample`** now (energy-matches generation to the
reference). The old default `center` had a **bin-0 artifact** (generated at 0 keV vs a ref window
averaging ~5.5 keV) that inflated the LOW regime ~2× for *every* model. **Any pre-2026-06-04 score,
and Modal's in-run scorecard, uses `center` — treat its low-regime numbers as wrong.** Re-score with
`sample` before comparing. Full story: `scorecard-findings.md`.

## Environment & tooling gotchas (these bit me this session)

- **Python:** `/Users/blaiszik/mamba/envs/cascaide/bin/python`, always with `KMP_DUPLICATE_LIB_OK=TRUE`
  (libomp double-init otherwise aborts). Tests: add `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- **Modal CLI breaks under `~/Desktop` (macOS TCC):** `rich` calls `os.getcwd()` at import, which
  gets EPERM when the CWD is inside `~/Desktop`. **Run all `modal …` commands from `/tmp`** with the
  absolute path to `modal_app.py` (`cd /tmp && modal run /Users/.../deploy/modal/modal_app.py::main …`).
  Plain `python` is fine from the repo root.
- **Rescores must run from the repo root** — `generators._import_sp_cas()` looks for `sp_cas.py` in
  CWD. (The Bash tool resets CWD to the repo root each call, so just don't `cd` away.)
- **Globus:** binary at `/Users/blaiszik/mamba/envs/cascaide/bin/globus`. Needs **Globus Connect
  Personal running** locally (`open -a "Globus Connect Personal"`; check `globus endpoint show
  $LOCAL_ENDPOINT | grep "GCP Connected"`). Pull Eagle results: `deploy/polaris/globus.sh pull`.
  Eagle endpoint `05d2c76a-…`, local `7b4d7fd6-…` (also in `deploy/polaris/config.sh`).
- **Polaris was DOWN 2026-06-05.** Submit kit is `deploy/polaris/` (PBS; `submit.sh sweep|score`).
  Two infra-failure precedents this session (node RPC timeout killed a sweep mid-run) — `--ckpt_every`
  / `--save_every` leave recoverable partials.
- **Modal credits:** ~$15 before the in-flight run; that run ≈ $12 → ~$3 left after. A100 ≈ $3.5–4/hr.

## New code this session (all committed + pushed to fork)

- `score_checkpoint.py --gen_energy {sample,bin_mean,center}` + `generators._energy_targets` +
  `sp_cas.generate` accepts a per-sample energy vector (eval fix).
- `train_set_v2.py --save_every N` (distinct kept `ckpt_<epoch>.pt`); `--struct_mode rdf` /
  `rdf_hist_loss` exist but are **DEMOTED — proven to hurt, don't use**.
- `modal_app.py`: exposes `--augment_rot / --save_every / --no_results / --run_name`, writes
  checkpoints onto the Volume, periodic commits, 5 h timeout.
- `scripts/plot_dataset_coverage.py` → `figures/` (shareable dataset-coverage plots; 7,100 cascades,
  N∝E^0.90). Suite: 26 tests green.

## Map — read these for detail

- `scorecard-findings.md` — per-metric/regime read + the `--gen_energy` eval fix.
- `substructure-loss-negative.md` — the RDF/aux-loss dead end (team-briefing).
- `experiments.md` — re-runnable sweep defs; Exp 1 (scheduler) / Exp 3 (substructure, DONE-NEGATIVE) /
  preemptable; reproducibility policy.
- `log.md` — dated session log (newest at bottom; start at 2026-06-04/05).
- `polaris.md` — ALCF Polaris kit + resume steps. `sampling-scan.md` — DPM-Solver++ fast sampler.
- `runbook.md` — env setup, data facts. `decisions.md` / `research-directions.md` — cross-cutting.

## Known but DEPRIORITIZED (image pipeline — the set pipeline is the active line)

The older image 2D-UNet pipeline (`cascaide/` encoders → UNet) has two latent bugs: (1) TanhP99
inference (`infer.py:244` calls `build_encoder` without `dataset=`); (2) `energy_vector` conditioner
ImportError (`EnergyVectorConditioner` vs `EnergyEmbedConditioner`). Not on the critical path — all
recent work is the self-contained set/point model (`sp_cas.py` + `scripts/train_set_v2.py`).

## Watch out for

- `n_vac == n_sia` (Frenkel conservation) is a hard invariant the data obeys (0 violations); the set
  model bakes it in (one count → both classes). A good sanity metric on generated samples.
- Image encoders are lossy and cap at `H*W/2` points/type (64×64 → ≤1024/type) — high-E cascades
  silently truncate. Core motivation for the set model. (Set model caps via `--max_defects`, default 1000.)
- `energy_divisor=300` (keV/300) must match between train and score — baked into the checkpoint config.

## Read these first (code)

- `sp_cas.py` — the entire set pipeline (DiT denoiser + count head + CoordDiffusion; `generate`,
  `load_model`, `sample_dpmpp`).
- `scripts/train_set_v2.py` — the trainer used on Polaris/Modal (EMA, schedulers, selection, save_every).
- `cascaide/eval/{scorecard,metrics,generators}.py` — the scorecard + checkpoint generation.
- `deploy/modal/modal_app.py`, `deploy/polaris/` — the two launch paths.
