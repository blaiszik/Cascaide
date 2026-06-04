#!/bin/bash
# Score checkpoints on the Polaris LOGIN node — CPU only, NO PBS/GPU, no queue wait.
# The login node has many free cores, so we run several score_checkpoint.py in parallel,
# each pinned to a few threads. Same scorer + per-energy-regime output as submit.sh score.
#
#   ./deploy/polaris/score_login.sh deploy/polaris/checkpoints.txt [PARALLEL_JOBS] [N_OVERRIDE]
#   ./deploy/polaris/score_login.sh deploy/polaris/checkpoints.txt 6 16      # 6 at a time, n=16
#
# CPU T=1000 sampling is much slower per-sample than an A100, so for a quick login-node pass
# prefer a smaller n (e.g. 12-24) via N_OVERRIDE; use `submit.sh score` (A100) for full n=48.
# Be a good citizen: login nodes are shared — keep JOBS*THREADS well under the free core count.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
cd "$REPO"

CKF="${1:?usage: score_login.sh <checkpoints.txt> [parallel_jobs] [n_override]}"
JOBS="${2:-6}"               # checkpoints scored concurrently
NOVR="${3:-}"                # optional: override --n on every line (argparse takes the last)
THREADS="${THREADS:-4}"      # torch/OMP threads per process
STEPS="${STEPS:-0}"          # DDIM fast-sampling steps; 0 = full DDPM-1000 (DEFAULT, faithful).
                             # DDIM degrades scores on this model (50 steps gave 2.27 vs 0.86 true;
                             # needs ~500 to match = no real speedup) -> rough preview only.
                             # For trustworthy scores use the A100: submit.sh score.

module use /soft/modulefiles 2>/dev/null || true
module load conda 2>/dev/null || true
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null || true
conda activate base 2>/dev/null || true
source "$ENV_PREFIX/bin/activate"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" KMP_DUPLICATE_LIB_OK=TRUE

CLEAN="$(mktemp)"; grep -vE '^\s*(#|$)' "$CKF" > "$CLEAN"
NCK=$(wc -l < "$CLEAN")
NFULL=$(grep -cvE '\-\-energy_max' "$CLEAN" 2>/dev/null || true); NFULL=${NFULL:-0}
LOGDIR="$(mktemp -d)"
echo "[score-login] $NCK checkpoints | $JOBS parallel x $THREADS threads | n_override=${NOVR:-from-file} | logs $LOGDIR"
[ "${NFULL:-0}" -gt 0 ] && echo "[score-login] WARNING: $NFULL full-range checkpoint(s) (no --energy_max) emit huge high-E clouds — VERY slow on CPU. Score those on A100: submit.sh score deploy/polaris/checkpoints_full.txt"
mkdir -p "$RESULTS"

i=0
while IFS= read -r line; do
  [ -n "$NOVR" ] && line="$line --n $NOVR"
  [ "${STEPS:-0}" -gt 0 ] && line="$line --steps $STEPS"
  i=$((i + 1))
  ( python scripts/score_checkpoint.py --subset "$DATA" --results "$RESULTS" \
        --no_report --device cpu $line ) > "$LOGDIR/$i.log" 2>&1 &
  # keep at most JOBS running; refill as each finishes so one slow ckpt can't gate a batch
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 2; done
done < "$CLEAN"
wait

# one race-free index/report rebuild, then a clean summary from the per-checkpoint logs
python -c "from cascaide.eval import registry, report; registry.collect('$RESULTS'); report.build_report('$RESULTS')"
echo "----- summary (OVERALL + per-regime) -----"
grep -hE "OVERALL|regime " "$LOGDIR"/*.log || true
echo "[score-login] done -> $RESULTS  (pull: deploy/polaris/globus.sh pull)"
