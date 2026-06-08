# Experiments registry (Polaris)

> Exact, re-runnable definitions of the planned/active Polaris sweeps — and what to re-submit
> after a preemption. All read the same dataset on Eagle: `$DATA =
> /lus/eagle/projects/Cascaide/models/data/cascaide_cascades.npz` (7,100 cascades, 0–300 keV).
> Each sweep = one experiment per GPU; `submit.sh sweep` auto-sizes nodes = ceil(N/4).
> Code lives on fork `blaiszik/Cascaide@tanhp99-global-centering`; `git pull` on Polaris first.

## Update 2026-06-04 — substructure lever CLOSED; high-E is the real headroom

Two rounds of findings on the 2026-06-04 A100 batch:
- **Eval fix (stands):** the low-E "failure" was a **scorecard artifact** (bin-0 generated at
  0 keV) — fixed via `score_checkpoint.py --gen_energy sample` (now default). **Re-score any
  backlog / Wave-1 outputs with `sample`** before trusting the low regime.
- **High-E is a *convergence* problem** (converged 250ep full-range high-regime 0.473 vs 2.3–3.6
  un-converged). This is the **real remaining headroom** → Wave-1 cells A (800ep) and D (512/10).
- **RDF / substructure lever = CLOSED (negative result).** We thought `rdf_l1` (the largest
  metric) was the weakness and built a metric-matched g(r) loss for it (**Exp 3**). Converged,
  equal-epoch test (job 7185401): the loss **hurts every metric incl. `rdf_l1`**, and the
  control already **passes the whole scorecard 7/7** (`rdf_l1` 0.16 < 0.20). 2nd coordinate aux
  loss to fail (radial was 1st). **Don't retry.** Full write-up: `substructure-loss-negative.md`.
- **Wave-1 cell B (`--energy_balance`)** stays de-motivated (high-E starvation disproven; bin-0
  not rare).

## Are they distinct? Yes.

| | **Exp 1** | **Exp 2** (staged) | **Preemptable sweep** |
|---|---|---|---|
| file | `experiments.txt` | `experiments.txt` (commented block) | `experiments_preempt.txt` |
| queue | debug-scaling (≤1 h) | debug-scaling | preemptable (≤72 h) |
| data range | **<100 keV** | <100 keV | **full 0–300 keV** |
| axis varied | **LR scheduler** | aug / arch / epochs | **centering / arch / epochs** |
| held fixed | 256/6, 600 ep, per_cascade, EMA | the Exp-1 winning scheduler | cosine sched, full range |
| cells | cosine · warmup · onecycle · plateau | control · +aug · 384/8 · 1200 ep | base · global · 512/10 · 1500 ep |

No cell is shared between sweeps. Within each sweep, cell 1 is the control and cells 2–4 each
change exactly one axis from it.

---

## Exp 1 — LR-scheduler sweep  (debug-scaling)
`deploy/polaris/experiments.txt` · 4 cells → 1 node. Everything fixed at a converged baseline
(256/6, <100 keV, per_cascade, EMA, 600 ep, finalize_n 16); **only `--lr_sched` varies**:
1. `cosine` (control = current recipe)
2. `cosine_warmup --warmup_frac 0.05`
3. `onecycle --lr_max 8e-4`
4. `plateau --plateau_patience 3 --plateau_factor 0.5` (steps on the scorecard)

```sh
QUEUE=debug-scaling WALLTIME=01:00:00 ./deploy/polaris/submit.sh sweep deploy/polaris/experiments.txt
```
~25–35 min. Judge by scorecard; for the verdict, rescore the 4 `best_by_scorecard.pt` locally
at uniform n≈48 (not the in-run finalize_n 16). Winner → fills `WIN` in Exp 2.

## Exp 2 — aug vs arch vs epochs  (debug-scaling, STAGED)
Lives as commented lines in `experiments.txt`. After Exp 1: replace `WIN` with the winning
scheduler, uncomment that block, comment out Exp 1, then submit as above. Cells: control ·
`+--augment_rot` · 384/8 (epochs trimmed to 450 to fit 1 h) · 1200 ep.

## Exp 3 — Substructure (RDF) sweep  (debug-scaling)  ✗ DONE — NEGATIVE (2026-06-04)
**Verdict: the g(r) loss hurts; do not use. Full write-up `substructure-loss-negative.md`.**
Clean run = **job 7185401** (4 cells, all ep500, scored with `--gen_energy sample`): control
(`w_struct=0`) OVERALL ~0.4–0.55 and **passes 7/7** incl. `rdf_l1` 0.16<0.20; the loss cells
get monotonically worse with weight on *every* metric incl. `rdf_l1` (r1/r2/r3 OVERALL
1.45/1.99/2.26). (First attempt job 7185324 died to a node RPC timeout — infra, not us.)
`--struct_mode rdf` / `--w_struct` are demoted to experimental (default off). Original plan ↓.

`deploy/polaris/experiments_struct.txt` · 4 cells → 1 node. Attacks `rdf_l1`, the only
scorecard metric with headroom (everything else near-ceiling — see `scorecard-findings.md`).
Fixed at the converged recipe (256/6, <100 keV, per_cascade, EMA, cosine, 500 ep, pure-SGD via
`--select_every 99999 --no_results`, `--ckpt_every 100`); **only the substructure loss varies**:
1. `--w_struct 0` (control = current recipe; the `rdf_l1` baseline to beat)
2. `--w_struct 0.3 --struct_mode rdf --w_nn 0 --struct_start_epoch 50`
3. `--w_struct 1.0 --struct_mode rdf --w_nn 0 --struct_start_epoch 50`
4. `--w_struct 3.0 --struct_mode rdf --w_nn 0 --struct_start_epoch 50`

```sh
QUEUE=debug-scaling WALLTIME=01:00:00 ./deploy/polaris/submit.sh sweep deploy/polaris/experiments_struct.txt
```
`--struct_mode rdf` = the metric-matched `rdf_hist_loss` (adaptive per-cloud rmax + ideal-shell
g(r) → directly minimizes `rdf_l1`); it replaces the legacy fixed-6 Å loss whose window held
only 1–8 % of the real pairs (the reason the earlier "w≥0.3 hurts" sweep failed — range
mismatch, not the idea). **Verdict:** offline per-regime rescore of each `runs/sweep_<JOB>_r{0..3}/
final_model.pt` with the corrected eval (`score_checkpoint.py --gen_energy sample`, now default);
compare the `rdf_l1` column / OVERALL vs r0. Distinct from the running Wave-1 preemptable sweep
(different queue, range, and axis — Wave-1 varies data-rebalance/aug/capacity, not the loss).

## Exp 4 — per_cascade vs global centering A/B  (Modal)  ✓ DONE — per_cascade wins, 2026-06-05
Matched Modal runs (EMA/cosine/250ep/full-range, only `--center` differs), rescored identically
(full-range, n=24, dpmpp-20, `--gen_energy sample`). per_cascade beats global on every metric &
regime: OVERALL 0.661 vs 0.882, rdf_l1 0.168 vs 0.272, low/mid/high all worse. **Keep per_cascade.**
Models: `results/runs/20260603T082541-…` (per_cascade) · `…20260605T164803-…` (global). See `log.md`.

## Exp 5 — best-bet model: per_cascade 500ep + augment_rot  (Modal)  ✓ DONE — NEW BEST, 2026-06-05
The accumulated-evidence best bet — convergence (the only headroom is high-E, which is
convergence-limited) + SO(3) augmentation (targets the data-starved high-E band). NO aux losses
(ruled out), NO global (worse), NO energy_balance (de-motivated).
```sh
cd /tmp && modal run --detach /Users/blaiszik/Desktop/git/Cascaide/deploy/modal/modal_app.py::main \
  --center per_cascade --epochs 500 --augment-rot --save-every 25 \
  --select-every 99999 --no-results --run-name percascade_500ep_aug
```
per_cascade · EMA · cosine · 256/6 · full-range · 500ep · `--augment_rot` · scoring OFF · `--save_every 25`.
App `ap-KUIg1l1P6BaEjLTsvipaZR`. Checkpoints → Volume `cascaide-results:ckpts/percascade_500ep_aug/`
(`ckpt_0025..0500.pt` + `final_model.pt`), periodic-commit safe. ~3.3 h ≈ $12. **Verdict = offline
`--gen_energy sample` rescore of the trace (convergence curve) + `final_model.pt` vs the Exp-4
per_cascade baseline.** See `handoff.md` for the post-run steps.
**RESULT (full writeup `convergence-500ep-augment.md`):** ran clean to ep500. The n=12 trace (20
ckpts) shows slow improvement, NO early plateau, best band ep375–500 (high-E + rdf hit run-minima
at ep500). n=24 verdict: ep375/475/500 all **6–7/7** and beat the 250ep baseline on **every regime
& rdf** by ~0.19 OVERALL / **−0.33 high-E** (0.64 vs 0.97). **`final_model.pt` is the new standing
best per_cascade model.** Caveat: n=24 wobbles ±0.15 across draws (baseline drew 0.661/0.710/0.810) —
the sampler isn't seeded; see Exp 6. Tooling added: `scripts/score_trace.py`, `scripts/score_verdict.py`.

## Wave 2 (2026-06-06) — high-E data drop + long Polaris run  ⏳ THE CURRENT PLAN

**Context.** Exp 5 landed: per_cascade **500ep + `--augment_rot` is the new best model** and its
convergence curve had **not flattened** by ep500. **~1000 new high-E cascades arrive Monday
2026-06-08**, hitting our sparsest + weakest regime head-on (200–300 keV is ~400 cascades/band today;
high-E is the only real headroom and is convergence/data-limited). A long Polaris run is being set up
(thousands of epochs / more augmentation possible). Plan: **measure the data cleanly first, then spend
the long run scaling the levers we've proven** (epochs + capacity + augmentation) — not scale blindly.

**Two hard-learned principles (govern every cell below):** (1) **isolate ONE factor per cell** —
confounded A/Bs misled us all session; (2) **fix the ruler before the long run** — n=24 verdicts wobble
±0.15 (the SAME baseline checkpoint drew 0.661 / 0.710 / 0.810 across three draws) because the sampler
isn't seeded, and we've been scoring partly on TRAINING data. Thousands of GPU-hours judged by a noisy,
train-contaminated ruler is the main risk.

### Exp 6 — Phase 0: harden the eval (this weekend, BEFORE the data) — PREREQUISITE
Code-only, cheap, no big training. Foundational for trusting anything the long run produces.
1. **Seed the sampler:** thread `seed` → `torch.manual_seed` into `sp.generate` / `sample_dpmpp`
   (today `generate_set_checkpoint(seed=)` only pins `_energy_targets`, NOT the reverse diffusion → the
   ±0.15 wobble). Makes cross-run verdicts reproducible.
2. **Held-out test split:** add `--test_frac` to `train_set_v2` (≈0.15–0.20, **stratified by energy band**,
   fixed seed) and reserve a high-E test set that NEVER trains. **All verdicts scored on held-out only**
   (with high-E this sparse, train/test leakage hurts most exactly here).
3. **Multi-seed / higher-n high-E scoring:** n≈48–96 × ~3 seeds, report mean±std (so a 0.03 regime delta
   means something). Add a `--seeds` knob to `score_checkpoint` / `score_trace`.
4. **Re-baseline:** rescore `final_model.pt` (500ep+aug, OLD data) on the held-out high-E test set → the
   number to beat in Exp 7/8.

### Exp 7 — Phase 1: does the +1000 high-E data move high-E?  (clean A/B, short queue)
Same proven recipe; scored on the SAME held-out high-E test set; **only the training data differs**:
1. **OLD data only** (= re-confirm `final_model` on held-out)
2. **OLD + 1000 NEW high-E**

Fixed: `--center per_cascade --augment_rot --epochs 500 --d_model 256 --depth 6 --max_defects 1000
--batch_size 16` (+ EMA, cosine). Answers the headline "**is high-E a DATA problem?**" and calibrates
data-vs-compute weighting for Exp 8. Quick — debug-scaling or a short preemptable slice.

### Exp 8 — Phase 2: the long run — epochs × capacity scan on OLD+NEW data  (preemptable, MAIN SPEND)
2×2, 1 node (4 GPU), all on old+new, all `--augment_rot`, with `--save_every 100` + **offline held-out
scoring** (`score_trace.py`) to build convergence curves → **find where high-E plateaus and CATCH OVERFIT**
(do NOT train blind to 4000; sparse high-E + a bigger model = memorization risk only a held-out curve reveals):

1. `--d_model 256 --depth 6  --epochs 2000 --batch_size 16`
2. `--d_model 256 --depth 6  --epochs 4000 --batch_size 16`
3. `--d_model 512 --depth 10 --epochs 2000 --batch_size 8`
4. `--d_model 512 --depth 10 --epochs 4000 --batch_size 8`

All cells also: `--center per_cascade --augment_rot --max_defects 1000 --save_every 100 --select_every 99999 --no_results`.
Answers: where high-E flattens, and whether **512/10** (more capacity for the ~2000-token high-E clouds)
beats 256/6. 512/10 is the long pole — watch ep-1 `s/ep` in `cascaide-sweep.o<jobid>` to size walltime.
```sh
QUEUE=preemptable WALLTIME=24:00:00 ./deploy/polaris/submit.sh sweep deploy/polaris/experiments_wave2.txt
```

### Exp 9 — Phase 3: targeted high-E levers  (CONDITIONAL — only if Exp 7/8 leave a high-E gap)
- **High-E specialist / curriculum:** train a `>150 keV`-only model (with new data) as a **ceiling probe**
  (add `--energy_min`, mirror of `--energy_max`); if a specialist ≫ the full-range model on high-E,
  oversample / curriculum-weight high-E in the main model (`energy_balance` is newly justified now that we
  have the data — it was de-motivated only for the disproven bin-0 artifact).
- **O(3) reflection augmentation:** rotation is already full SO(3); add reflections (W BCC is
  reflection-symmetric → exact, ~free, doubles effective symmetry). This is the real "more augmentation"
  lever; beyond it, real high-E data ≫ synthetic aug.
- **`--min_snr`** (already wired, default off): untested loss weighting for the high-noise/high-t steps that
  matter most for the big clouds. One extra cell, cheap.

**DO NOT re-litigate** (all ruled out): aux distributional losses (radial/RDF — `substructure-loss-negative.md`),
global centering (Exp 4), lattice-snap (ledger). **Don't** train thousands of epochs without a held-out
curve. **Don't** make augmentation the primary lever — the real high-E data dominates.

**Stretch (only if Exp 8 shows epochs help but is wall-clock-bound):** local/windowed attention or a
latent/patchified diffusion to cut the per-step O(N²) on ~2000-token high-E clouds → enables the
deep-epoch regime efficiently. Architecture change → only if the scan proves it's needed.

---

## Preemptable sweep — full-range study  (preemptable)
`deploy/polaris/experiments_preempt.txt` · 4 cells → 1 node. **Full 0–300 keV**, cosine, with
`--max_defects 1000` + small batch (memory: high-E clouds up to ~1100 defects, O(N²) attn):
1. **base**: per_cascade, 256/6, 800 ep, batch 16  — converged full-range baseline
2. **global**: SAME but `--center global` — clean per_cascade-vs-global test (resolves the
   vignesh confound: vignesh = global, ep569, scored 1.489/2-of-7 on <100 keV vs our 30-ep
   per_cascade 1.242/4-of-7 — but vignesh was full-range-trained, so this isolates centering)
3. **big arch**: per_cascade, 512/10, 800 ep, batch 8 — fully converged (busts the 1 h cap)
4. **epochs ceiling**: per_cascade, 256/6, 1500 ep, batch 16

```sh
QUEUE=preemptable WALLTIME=08:00:00 ./deploy/polaris/submit.sh sweep deploy/polaris/experiments_preempt.txt
```
512/10 is the long pole; watch ep-1 `s/ep` in `cascaide-sweep.o<jobid>` to estimate total time.

---

## Preemption recovery — what to re-run

We do **not** auto-resume. With `--ckpt_every N` (now set in all sweeps), a rolling **converged
`latest.pt`** is written to Eagle every N epochs (atomic tmp+replace), so a kill loses at most ~N
epochs AND always leaves a converged checkpoint to rescore. `best_by_scorecard.pt` also survives
but is chosen by a NOISY n=6 selection (see the ledger lesson) — **rescore `latest.pt`/`final_model.pt`, not it.**

If a preemptable job is killed:
1. **Salvage what finished:** `globus ls 05d2c76a…:/Cascaide/models/runs/` for `sweep_<jobid>_r{0..3}/`
   dirs; any with `best_by_scorecard.pt` are usable. Pull + score locally:
   `./deploy/polaris/globus.sh pull-ckpts` (⚠ lands in `~/Desktop/runs` — verify/move), then
   score with the eval tooling.
2. **Re-run = just re-submit the SAME file** (no resume; produces fresh run dirs with uuid ids,
   so no collision with salvaged ones):
   `QUEUE=preemptable WALLTIME=08:00:00 ./deploy/polaris/submit.sh sweep deploy/polaris/experiments_preempt.txt`
3. To re-run only the cells that didn't finish, copy those lines into a temporary file and
   `submit.sh sweep <that file>`.

Tip: to reduce loss, lower `--select_every` on long cells (more frequent best-checkpoint saves)
or split the sweep so each cell is shorter.

---

## Run ledger (scored, best→worst) — dates absolute

| model | data | n | score | pass | notes |
|---|---|---|---|---|---|
| copy-real (ceiling) | <100keV | — | 0.000 | 7/7 | resampled real = achievable floor |
| set-dit ep300 (old 3-pt data) | 3-pt | 48 | 0.444 | 7/7 | prior best real model |
| set-dit v2 300ep | <100keV | 32 | 0.519 | 7/7 | converged per_cascade baseline |
| sweep **cosine** cell (best@ep59) | <100keV | 48 | 0.625 | 6/7 | job 7183204; fails only nn_w1 (1.10) — **confounded, see lesson** |
| set-dit v2 250ep | full 0–300 | 32 | 0.644 | 6/7 | full-range, less optimized |
| **set-dit v2 500ep + augment_rot** (final) | full 0–300 | 24 | **0.58–0.62** | 6–7/7 | **NEW BEST** (Exp 5); per_cascade; beats 250ep baseline ~0.19 OVERALL / −0.33 high-E; `results/ckpts/percascade_500ep_aug/final_model.pt`. Range = ±0.15 sampler-draw variance (seed fix → Exp 6). |
| 30ep Polaris smoke (7183018) | <100keV | 8 | 1.242 | 4/7 | undertrained |
| **vignesh** global-center ep569 | <100keV | 12 | 1.489 | 2/7 | GLOBAL centering — worse than our 30ep per_cascade despite 19× training |

**Standing lesson (job 7183204 scheduler sweep):** `best_by_scorecard.pt` is selected by a noisy
`select_n=6` scorecard → each cell's "best" locked at a scattered early epoch (cosine 59 / warmup 89 /
onecycle 329 / plateau 119) and never beaten despite training to ~ep560. Those checkpoints are NOT
the converged models and aren't comparable. Fix shipped: `--ckpt_every` + verdict from `final_model.pt`.

**Converged-model diagnosis:** `<100 keV` per_cascade models are near the ceiling (6–7/7). Lone
marginal metric = **`nn_w1`** (NN spacing); `rdf_l1` and `vac_sia_sep` already pass. Real headroom is
the **full 0–300 keV** regime (high-E cascades, far less optimized) + the structural levers below.

**RULED OUT — lattice-snap decode (local prototype, ~2026-06-03):** premise was "snap generated
coords to BCC sites (a=3.165 Å) to sharpen nn_w1/rdf" (research-directions §2.3). Empirically FALSE
for this data: REAL defects sit **~1.0 Å (median 1.1 ≈ a/3) from the nearest BCC site** — not
lattice-registered (1073 K minimized data: thermal + strained cores). Snapping generated clouds
therefore moved them *away* from real → nn_w1 1.223→1.235, rdf 0.171→0.210 (broke it), score 0.709→0.756.
Do NOT retry the naive version. (A proper per-cascade lattice *registration* — fit orientation+constant —
is conceivable but the residual looks ~random, so deprioritized.)

**FAST SAMPLING (inference speed) — DPM-Solver++ wins, DDIM doesn't:**
- DDIM ruled out: needs ~500 steps to match DDPM-1000 (50→2.27, 250→1.07, 500→0.93 vs 0.86) = no speedup.
- **DPM-Solver++(2M)** with UNIFORM-log-SNR (lambda) spacing: at ~30 steps scores within n=32 noise of full
  DDPM-1000 (converged 300ep: dpmpp-30 0.789 / dpmpp-50 1.098 vs full 0.959) at **~15-30x speedup**
  (0.1 vs 2.9 s/cloud) — SAME model, no retraining. Opt-in: `score_checkpoint.py --sampler dpmpp --steps 30`.
  Gotchas that made it look broken at first: uniform-TIMESTEP spacing (must be uniform-lambda) + testing on
  the undertrained ep59 ckpt + n=8 noise. Caveat: n=32 single-seed noise (esp. low-E small-cloud regime);
  confirm at higher n / multi-seed on A100 before trusting dpmpp for VERDICTS (use full DDPM for those).
- Bigger speedups if needed (require work): distillation/consistency or rectified-flow (retrain -> 1-8 steps);
  latent diffusion or local attention (cut the per-step O(N^2) that dominates high-E).

### In flight / pending
- **▶ Wave 2 (Exp 6–9) is THE CURRENT PLAN** — high-E data drops Mon 2026-06-08. Exp 6 (eval hardening:
  seed sampler + held-out `--test_frac` + multi-seed scoring) is the do-first prerequisite; then Exp 7
  data A/B, Exp 8 long epochs×capacity scan, Exp 9 conditional high-E levers. See the Wave 2 section above.
- **Exp 1 rerun** — scheduler sweep at **500 ep + `--ckpt_every 50`** (commit `8e652909`). PENDING submit.
- **Preemptable rerun** — full-range, `--ckpt_every 100` (commit `a2d1c563`); old run killed at 8 min. PENDING.
- **Exp 2** — aug vs arch vs epochs, staged in `experiments.txt`. Needs Exp-1 winner.
- **Lattice-snap decode** — RULED OUT (see above); not promoted.
- **Polaris scoring job** (`submit.sh score`) — proposed: rescore checkpoints on A100 (sampling is the
  cost) + per-energy-regime breakdown. Would make every verdict fast vs slow local MPS.

---

## Reproducibility & code separation (so old models always re-run)

Already in place — keep it this way:
- **Self-contained checkpoints.** Each `model.pt` bakes its `config` + `normalizer`, so any old
  checkpoint regenerates/re-scores regardless of later code. `model.pt` travels inside its run dir.
- **Provenance per run.** `run.json` records `provenance.git.{commit,branch,dirty}` (`capture_git`)
  + config. `MANIFEST_SCHEMA_VERSION` / scorecard `SCHEMA_VERSION` are stamped (currently 1).
- **New behavior is OPT-IN, defaults preserve the past.** `--augment_rot`, `--min_snr`, `--w_struct 0`,
  `--ckpt_every 0`, `--lr_sched cosine` are all no-ops by default. The lattice-snap decode, when
  promoted, becomes `--lattice_snap` (default OFF) on the generator — it must NEVER change the default
  generate path. → the default path always reproduces prior results.

**To re-run / re-score a past model:** open its `run.json` → `git checkout <provenance.git.commit>`
(if `dirty:true`, results came from an uncommitted tree — note it), load its `model.pt`, score with
defaults (all flags off). Same data subset + `energy_bin` as recorded ⇒ same score.

**Rules going forward:** (1) commit before any Polaris run (the cluster `git pull`s the fork; the
`dirty` flag marks non-reproducible local runs). (2) If a metric definition changes, **bump
`SCHEMA_VERSION`** so old vs new scores aren't silently compared. (3) Keep `sp_cas.py` / `setdiff/`
back-compatible loaders so old checkpoints still load; if a breaking change is unavoidable, version the
loader rather than mutating it in place.
