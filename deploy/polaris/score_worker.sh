#!/bin/bash
# One checkpoint per MPI rank, pinned to one GPU. Launched by score.pbs via `mpiexec --ppn 4`.
# PALS sets PALS_RANKID (global) and PALS_LOCAL_RANKID (0-3 on node). Mirrors gpu_worker.sh.
set -uo pipefail
cd "${REPO:?}"
export CUDA_VISIBLE_DEVICES="${PALS_LOCAL_RANKID:-0}"   # bind this rank to its node-local GPU
RANK="${PALS_RANKID:-0}"

LINE=$(sed -n "$((RANK + 1))p" "${CHECKPOINTS_CLEAN:?}")
if [ -z "$LINE" ]; then echo "[rank $RANK] no checkpoint line; idle"; exit 0; fi
echo "[rank $RANK | gpu $CUDA_VISIBLE_DEVICES | $(hostname)] $LINE"

# --no_report: each rank writes only its own run dir; score.pbs rebuilds index/report ONCE.
# $LINE = args to score_checkpoint.py (at least --checkpoint <path>; optional --label/--energy_max/--n).
python scripts/score_checkpoint.py \
    --subset "$DATA" --results "$RESULTS" \
    --no_report --device cuda $LINE
