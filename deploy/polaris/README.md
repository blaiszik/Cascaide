# Running Cascaide on Polaris (ALCF)

Repo-resident PBS scripts for training Cascaide on Polaris. **Code travels via git**
(commit → `git pull` on Polaris); **data/artifacts travel via Globus** (local ↔ Eagle).
Polaris reads/writes `/eagle` directly during jobs.

> Polaris node = 4× A100 (40 GB). Scheduler = PBS Pro; launcher = `mpiexec` (PALS).
> Our model is small, so a single experiment fits one A100 in ~30–40 min (well under the
> `debug` 1 h cap). Sweeps pack one experiment per GPU across nodes.

## Files
| file | purpose |
|---|---|
| `config.sh` | **edit once** — `PROJECT` (PBS `-A`), `QUEUE`, paths on Eagle, Globus endpoint UUIDs |
| `setup_env.sh` | build a lightweight **venv on Eagle over ALCF base** (reuses CUDA torch in place via `--system-site-packages` — no slow file copy; run once on a login node). Activate: `module use /soft/modulefiles; module load conda; conda activate base; source $ENV_PREFIX/bin/activate` |
| `globus.sh` | `push` data → Eagle, `pull` results / `pull-ckpts` back to local |
| `submit.sh` | submit a `train` (1 GPU) or `sweep` (1 experiment/GPU across nodes) job |
| `train.pbs` | single-experiment PBS job |
| `sweep.pbs` + `gpu_worker.sh` | multi-node sweep over `experiments.txt` |
| `experiments.txt` | **design experiments here** — one `train_set_v2.py` arg-line per experiment |

## Queues (verify against the ALCF docs — limits change)
| queue | nodes | walltime | submit directly? | use for |
|---|---|---|---|---|
| `debug` | 1–2 | ≤ 1 h | yes | quick single runs / tests |
| `debug-scaling` | 1–10 | ≤ 1 h | yes | small sweeps (≤1 h) |
| `preemptable` | 1–20 | ≤ 72 h | yes | long single runs (may be preempted) |
| `prod` | **≥10**–~476 | ≤ 24 h | yes (routes to small/medium/large) | large sweeps |

Per-project: ≤10 running + ≤100 queued jobs.

## Workflow
```sh
# 0. on Polaris: clone/pull the repo; edit deploy/polaris/config.sh (PROJECT, paths, endpoints)
git pull                                   # whenever scripts change

# 1. env (once, login node)
bash deploy/polaris/setup_env.sh

# 2. data in (from your laptop): processed npz -> Eagle
./deploy/polaris/globus.sh push            # needs data/cascaide_cascades.npz locally
#    (or generate the npz on Polaris from raw dumps on Eagle: scripts/process_raw_data.py)

# 3. submit
./deploy/polaris/submit.sh train --epochs 300 --energy_max 100 --energy_bin 10   # one A100
./deploy/polaris/submit.sh sweep deploy/polaris/experiments.txt                   # one exp / GPU
qstat -u $USER                             # track

# 4. results out (to your laptop) + into the dashboard
./deploy/polaris/globus.sh pull
python -c "from cascaide.eval import registry, report; registry.collect('results'); report.build_report('results')"
#    -> refresh the Continuity cover
```

## Notes
- Each run is **self-contained** (`results/runs/<id>/` has `model.pt` + scorecard + images),
  and the registry auto-tags PBS (`PBS_JOBID`) so dashboard rows trace back to the Polaris job.
- Sweep jobs run with `--no_report` (each rank writes only its run dir); `sweep.pbs` rebuilds
  the index/report **once** after all ranks finish (no race on `index.json`).
- High-energy cascades are large (~1100 defects → ~2200 tokens); `--max_defects` caps the
  O(N²) attention memory. Multi-GPU **DDP for a single big model** is a follow-on (current
  scripts use 1 GPU per experiment, which suits the small model + sweeps).
- Eagle is not fast disk: keep the dataset as one `.npz`, and `pull-ckpts` only what you need.
