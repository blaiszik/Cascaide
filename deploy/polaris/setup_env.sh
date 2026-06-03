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

# ALCF base conda (CUDA torch for the A100s). Clone it into our own prefix on /eagle so we
# can add packages while keeping ALCF's known-good torch. Activation matches the standard
# Polaris pattern: `module use /soft/modulefiles; module load conda; conda activate <env>`.
module use /soft/modulefiles
module load conda

if [ ! -d "$ENV_PREFIX" ]; then
  echo "[setup] cloning base conda env (ALCF CUDA torch) -> $ENV_PREFIX (this can take a few min)"
  conda create -y --prefix "$ENV_PREFIX" --clone base
fi
conda activate "$ENV_PREFIX"

# only deps not already in base; torch/numpy come from the cloned base
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
echo "    module use /soft/modulefiles; module load conda; conda activate $ENV_PREFIX"
