# Experiments registry (Polaris)

> Exact, re-runnable definitions of the planned/active Polaris sweeps — and what to re-submit
> after a preemption. All read the same dataset on Eagle: `$DATA =
> /lus/eagle/projects/Cascaide/models/data/cascaide_cascades.npz` (7,100 cascades, 0–300 keV).
> Each sweep = one experiment per GPU; `submit.sh sweep` auto-sizes nodes = ceil(N/4).
> Code lives on fork `blaiszik/Cascaide@tanhp99-global-centering`; `git pull` on Polaris first.

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

### In flight / pending
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
