# Handoff

> The first thing the next agent reads. Keep it current and forward-looking.

_Last updated: 2026-06-05 by Claude (set-DiT training on Polaris/Modal; eval fix; centering + substructure results)._
_Branch: `tanhp99-global-centering` (fork `blaiszik/Cascaide`, the push target — `origin`=vigsam is read-only)._

## ⏳ IN FLIGHT — the immediate next action

A **Modal A100 run is training right now** (app `ap-KUIg1l1P6BaEjLTsvipaZR`, detached, launched
2026-06-05 ~14:15 CDT, ~3.3 h ≈ ~$12 of the remaining ~$15 credit). It is the **best-bet model**:
per_cascade · EMA · cosine · 256/6 · full 0–300 keV · **500 epochs · `--augment_rot`** · scoring
DISABLED (`--select_every 99999 --no_results`) · **`--save_every 25`** (a convergence trace).
Checkpoints write straight to the Modal Volume `cascaide-results` under
`ckpts/percascade_500ep_aug/` (`ckpt_0025.pt … ckpt_0500.pt` + `final_model.pt`), with a
**periodic VOL.commit() every ~2 min** so a timeout/crash keeps the trace.

**When it finishes** (check `cd /tmp && modal app list`; state→stopped, or logs show `ep 500/500`):
1. Pull the trace: `cd /tmp && modal volume get --force cascaide-results runs/... ` — actually
   the checkpoints are under `ckpts/percascade_500ep_aug/`; pull that dir, e.g.
   `modal volume get --force cascaide-results ckpts/percascade_500ep_aug /Users/blaiszik/Desktop/git/Cascaide/results/ckpts/percascade_500ep_aug` (or `modal run deploy/modal/modal_app.py::fetch`).
2. **Convergence curve** (the user's question — "do they converge early?"): rescore each
   `ckpt_00NN.pt` OFFLINE with the corrected eval and plot score vs epoch. Use the same recipe as
   the centering A/B (`/tmp/centering_ab.py` is a template): `generate_set_checkpoint(ck, ref,
   device="mps", n_per_energy=24, energy_bin=25.0, sampler="dpmpp", steps=20, gen_energy="sample")`
   then `scorecard.compute`. (n=24 full-range dpmpp-20 ≈ 4 min/ckpt on MPS; 20 ckpts ≈ ~80 min, so
   maybe subsample epochs, or run a few first.)
3. **Verdict:** rescore `final_model.pt` head-to-head vs the current best per_cascade model
   (`results/runs/20260603T082541-set-dit-v2-87ec2c1b/model.pt`, full-range 250ep, OVERALL ~0.66)
   to see if 500ep + augmentation beat it. Report OVERALL + low/mid/high + rdf_l1.
4. Log results to `log.md`; if it's a new best, note it in `experiments.md`.

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

**Proven recipe to build on:** per_cascade · EMA · cosine LR · 256/6 · `--max_defects 1000` · batch 16.
Untested-but-plausible levers left: longer epochs (in flight), `--augment_rot` (in flight),
capacity 512/10 (budget-permitting), `--min_snr` (lower priority).

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
