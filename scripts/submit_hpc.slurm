#!/bin/bash
#SBATCH --job-name=cascaide-setv2
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
# ---------------------------------------------------------------------------
# Train set-DiT v2 on the cluster and write a run into the results store, which
# can then be rsync'd back to a workstation where the dashboard reads it.
#
#   sbatch scripts/submit_hpc.slurm
#
# registry.build_manifest() auto-captures SLURM_JOB_ID / nodename / partition, plus the git
# commit + config hash, so every dashboard row traces back to this job. After the job lands,
# on the workstation:
#   rsync -a <cluster>:<path>/results/ results/
#   python -c "from cascaide.eval import registry, report; registry.collect('results'); report.build_report('results')"
# ---------------------------------------------------------------------------
set -euo pipefail

# --- environment (edit for your cluster) ---
# module load conda            # or: source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cascaide
export PYTORCH_ENABLE_MPS_FALLBACK=1     # harmless off-mac; CUDA used if present

DATA_ROOT=${DATA_ROOT:-/path/to/defect_files}
SUBSET=${SUBSET:-data/real_subset_large.npz}
OUTDIR=${OUTDIR:-runs/set_v2_${SLURM_JOB_ID:-local}}

# --- cache the real subset once (skips if present) ---
if [ ! -f "$SUBSET" ]; then
  python scripts/build_real_subset.py --data_root "$DATA_ROOT" --out "$SUBSET" --per_energy -1
fi

# --- train (writes a run + report into results/ on completion) ---
python scripts/train_set_v2.py \
  --subset "$SUBSET" \
  --output_dir "$OUTDIR" \
  --epochs "${EPOCHS:-600}" \
  --w_struct "${W_STRUCT:-0.5}" \
  --select_every "${SELECT_EVERY:-25}" \
  --results results \
  --device cuda

echo "Done. results/ updated; rsync it back to the dashboard host."
