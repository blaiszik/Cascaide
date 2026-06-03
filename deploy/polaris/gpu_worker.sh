#!/bin/bash
# One sweep experiment per MPI rank, pinned to one GPU. Launched by sweep.pbs via
# `mpiexec --ppn 4`. PALS sets PALS_RANKID (global) and PALS_LOCAL_RANKID (0-3 on node).
set -uo pipefail
cd "${REPO:?}"
export CUDA_VISIBLE_DEVICES="${PALS_LOCAL_RANKID:-0}"   # bind this rank to its node-local GPU
RANK="${PALS_RANKID:-0}"

EXP=$(sed -n "$((RANK + 1))p" "${EXPERIMENTS_CLEAN:?}")
if [ -z "$EXP" ]; then echo "[rank $RANK] no experiment line; idle"; exit 0; fi
echo "[rank $RANK | gpu $CUDA_VISIBLE_DEVICES | $(hostname)] $EXP"

# --no_report: each rank writes only its own run dir; sweep.pbs rebuilds the index/report
# ONCE after all ranks finish (avoids concurrent writes to index.json/report.html).
python scripts/train_set_v2.py \
    --subset "$DATA" --results "$RESULTS" \
    --output_dir "$RUNS_DIR/sweep_${PBS_JOBID:-job}_r${RANK}" \
    --no_report --device cuda $EXP
