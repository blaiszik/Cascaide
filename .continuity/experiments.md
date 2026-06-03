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

We do **not** auto-resume. But `best_by_scorecard.pt` is written to Eagle **every time the
scorecard improves** (per `select_every`), so a kill loses only post-last-improvement training
plus the final finalize/report — the best checkpoint of each cell survives.

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
