#!/bin/bash -l
# Build the Cascaide Python env on Polaris. Run ONCE on a LOGIN node:
#     cd <repo> && bash deploy/polaris/setup_env.sh
#
# Strategy: layer a lightweight venv ON TOP of ALCF's `conda` module (which already ships a
# CUDA-matched PyTorch + numpy for the A100s) via --system-site-packages, then pip-install
# only what's missing + the cascaide package (no ovito needed — training uses the cached
# .npz). The env lives on /eagle so compute nodes can read it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

echo "[setup] PROJECT=$PROJECT  ENV_PREFIX=$ENV_PREFIX  REPO=$REPO"
mkdir -p "$(dirname "$ENV_PREFIX")" "$EAGLE_BASE/data" "$RESULTS" "$RUNS_DIR"

# Lightweight venv layered ON TOP of ALCF's base conda (which has CUDA torch for the A100s).
# We do NOT clone base: cloning copies ~158k files and Eagle (Lustre) is very slow at many
# small files. --system-site-packages reuses base's torch/numpy IN PLACE (no copy), so the
# venv is tiny and setup is ~1-2 min.
module use /soft/modulefiles
module load conda
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null || true   # enable `conda activate` non-interactively
conda activate base

if [ ! -f "$ENV_PREFIX/bin/activate" ]; then
  echo "[setup] creating venv (reuses base CUDA torch via --system-site-packages; no file copy)"
  python -m venv --system-site-packages "$ENV_PREFIX"
fi
source "$ENV_PREFIX/bin/activate"

# only the few deps not in base; torch/numpy come from base via --system-site-packages
python -m pip install --upgrade pip
python -m pip install scipy matplotlib
python -m pip install -e "$REPO" --no-deps     # the cascaide package (editable)

echo "[setup] verifying..."
python - <<'PY'
import torch, numpy, scipy, matplotlib, cascaide.eval, cascaide.setdiff
print("  torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| #gpus:", torch.cuda.device_count())
PY
echo "[setup] done. Activate later with:"
echo "    module use /soft/modulefiles; module load conda; conda activate base; source $ENV_PREFIX/bin/activate"
