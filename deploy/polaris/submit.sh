#!/bin/bash
# Submit Cascaide jobs to Polaris. Fills the PBS -A/-q/-l directives from config.sh.
#
#   ./deploy/polaris/submit.sh train  --epochs 300 --energy_max 100 --energy_bin 10
#   ./deploy/polaris/submit.sh sweep  deploy/polaris/experiments.txt
#
# Queue guidance (set QUEUE in config.sh): debug (<=2 nodes, <=1h) | debug-scaling (<=10
# nodes, <=1h) | preemptable (<=20 nodes, <=72h, can be preempted) | prod (>=10 nodes, <=24h).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
cd "$REPO"                      # so PBS_O_WORKDIR (= submit cwd) is the repo root
cmd="${1:-}"; shift || true

case "$cmd" in
  train)
    EXP="$*"
    echo "submit train: queue=$QUEUE walltime=$WALLTIME | EXP=$EXP"
    qsub -A "$PROJECT" -q "$QUEUE" -l walltime="$WALLTIME" \
         -l select=1:system=polaris -l filesystems="$FILESYSTEMS" \
         -v EXP="$EXP" "$HERE/train.pbs"
    ;;
  sweep)
    EXPF="${1:?sweep needs an experiments file}"
    EXPF="$(cd "$(dirname "$EXPF")" && pwd)/$(basename "$EXPF")"
    NEXP=$(grep -cvE '^\s*(#|$)' "$EXPF")
    NODES=$(( (NEXP + 3) / 4 ))               # 4 GPUs/node
    echo "submit sweep: $NEXP experiments -> $NODES nodes, queue=$QUEUE walltime=$WALLTIME"
    [ "$NODES" -ge 10 ] && [ "$QUEUE" != prod ] && \
      echo "  NOTE: >=10 nodes usually needs QUEUE=prod (set it in config.sh)"
    [ "$NODES" -le 2 ] || [ "$QUEUE" = prod ] || \
      echo "  NOTE: $NODES nodes — use QUEUE=debug-scaling (<=1h) or prod (>=10 nodes)"
    qsub -A "$PROJECT" -q "$QUEUE" -l walltime="$WALLTIME" \
         -l select="${NODES}:system=polaris" -l filesystems="$FILESYSTEMS" \
         -v EXPERIMENTS="$EXPF" "$HERE/sweep.pbs"
    ;;
  *)
    echo "usage: submit.sh train <train_set_v2 args...> | sweep <experiments.txt>"; exit 1
    ;;
esac
echo "track:  qstat -u \$USER    logs:  tail -f cascaide-*.o<jobid>    (pull results: deploy/polaris/globus.sh pull)"
