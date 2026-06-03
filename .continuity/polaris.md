# Polaris (ALCF) — handoff

Status as of 2026-06-03. Goal: run Cascaide training on Polaris. Kit is built + pushed;
first job not yet successfully run (blocked on getting the data onto Eagle + a clean env).

## Where we are (RESUME HERE)
Everything is in `deploy/polaris/` and pushed to **fork `blaiszik/Cascaide`, branch
`tanhp99-global-centering`** (latest commit **c235fa93**). The user has the repo cloned on
Polaris and pulls from the fork.

**Immediate next steps (on Polaris unless noted):**
1. `cd ~/Cascaide && git pull` (get c235fa93), then `source deploy/polaris/config.sh && echo "$DATA"`
   → must show `/lus/eagle/projects/Cascaide/models/data/cascaide_cascades.npz`.
2. **From the Mac** (data not on Eagle yet — that's why `ls $DATA` failed): `./deploy/polaris/globus.sh push`
   then `globus task list`. First verify the collection root: `globus ls "$EAGLE_ENDPOINT:/Cascaide"`.
3. On Polaris: `ls -lh "$DATA"` (should now exist), then the smoke test:
   ```sh
   QUEUE=debug-scaling WALLTIME=00:20:00 ./deploy/polaris/submit.sh train \
     --center per_cascade --energy_max 100 --energy_bin 25 \
     --epochs 30 --select_every 30 --select_n 4 --finalize_n 4 --max_defects 1000
   ```
4. Watch: `qstat -u $USER`; `tail -f cascaide-train.o<jobid>`. On success, from Mac:
   `./deploy/polaris/globus.sh pull` then `python -c "from cascaide.eval import registry,report; registry.collect('results'); report.build_report('results')"`.

If `setup_env.sh` was never finished cleanly: `rm -rf /lus/eagle/projects/Cascaide/models/envs/cascaide`
then `bash deploy/polaris/setup_env.sh` (now a venv, ~1-2 min — see gotchas).

## Files (deploy/polaris/)
`config.sh` (edit-once knobs), `setup_env.sh` (build env), `submit.sh` (train | sweep),
`train.pbs`, `sweep.pbs`+`gpu_worker.sh`, `experiments.txt` (sweep matrix), `globus.sh`
(push/pull/pull-ckpts), `README.md`. See `runbook.md` "Polaris" section too.

## Hard-won gotchas (do NOT relitigate)
- **Git**: `origin` = `vigsam-coder/Cascaide` is **read-only** for blaiszik (push 403s).
  Push to **`fork` = `blaiszik/Cascaide`**; Polaris pulls from the fork. Going forward:
  commit to `tanhp99-global-centering`, `git push fork tanhp99-global-centering`.
- **Eagle path** is `/lus/eagle/projects/<PROJECT>/...` (NOT `/eagle/...`). `EAGLE_BASE` is the
  single source of truth; `EAGLE_PATH` (Globus) is DERIVED from it via `EAGLE_COLLECTION_ROOT`
  (assumed `/lus/eagle/projects` — verify with `globus ls "$EAGLE_ENDPOINT:/"`). If the
  collection is rooted elsewhere, fix that ONE var.
- **config.sh paths use DIRECT assignment** (not `${VAR:-default}`) — `${:-}` let a stale
  `EAGLE_BASE`/`DATA` from a previous `source` win, so edits didn't take effect. Only
  PROJECT/QUEUE/WALLTIME/endpoints keep env-override semantics (used as `QUEUE=... ./submit.sh`).
- **Env: venv-on-base, NOT `conda --clone`.** Cloning copies ~158k files; Eagle/Lustre is
  pathologically slow at small files (hung a setup 15+ min). `setup_env.sh` does
  `module load conda; conda activate base; python -m venv --system-site-packages $ENV_PREFIX`
  (reuses ALCF CUDA torch in place). Activate: `module use /soft/modulefiles; module load
  conda; conda activate base; source $ENV_PREFIX/bin/activate`. (`eval "$(conda shell.bash
  hook)"` added to silence `__conda_exe: command not found`.)
- **Queue**: default `debug-scaling` (≤10 nodes, ≤1h) — less congested than `debug` (which had
  ~23 queued). Our runs ~35 min on A100 → fit debug-scaling; it can even pack a ≤40-experiment
  sweep (10 nodes × 4 GPU) in one ≤1h job. `prod` needs ≥10 nodes (24h); `preemptable` ≤72h
  but can be preempted (we don't checkpoint-resume).
- **Memory**: high-energy cascades are big (up to ~1100 defects → ~2200 tokens; O(N²) attn).
  `--max_defects 1000` (drops 1 cascade) + batch ≤16 keeps it safe on a 40GB A100.
- **Sweeps**: each rank runs with `--no_report` (writes only its run dir); `sweep.pbs` does a
  single `registry.collect()/build_report()` at the end (else parallel writers race on
  index.json). `submit.sh sweep <file>` auto-sizes nodes = ceil(N/4).
- **Untestable from the dev box**: no Polaris access here, so first-run shakeout = confirm
  `-A $PROJECT` (PROJECT=Cascaide), `module load conda` path, endpoint UUIDs, collection root.

## Polaris facts
4× A100 (40GB)/node; PBS Pro; launcher `mpiexec` (PALS: `PALS_RANKID`, `PALS_LOCAL_RANKID`);
`-l select=N:system=polaris -l filesystems=home:eagle -l place=scatter`. Globus UUIDs are in
`config.sh` (EAGLE_ENDPOINT 05d2c76a…, LOCAL_ENDPOINT 7b4d7fd6…). PROJECT/Eagle subdir guessed
(`Cascaide`, `models`) — confirm on first run.

## What runs on Polaris
`scripts/train_set_v2.py` (the set-DiT trainer: per-cascade centering, EMA, energy-bin scoring,
results store). Data = `data/cascaide_cascades.npz` (full continuous 0–300 keV, 7100 cascades —
must be pushed to Eagle). Prior local result for reference: full-range model ~0.64 (6/7), <100keV
~0.52 (7/7). See `journal.md` for the science context.
