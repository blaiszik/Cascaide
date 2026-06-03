#!/bin/bash
# Polaris cluster config — EDIT THESE ONCE, then `source` from the other scripts.
# Code lives in the repo (git pull on Polaris). Data/artifacts live on /eagle and move
# local<->eagle via Globus (see globus.sh). Polaris reads/writes /eagle directly.

# --- ALCF allocation / queue ---
export PROJECT="${PROJECT:-Cascaide}"            # PBS -A (your ALCF project/allocation)
export QUEUE="${QUEUE:-debug}"                   # debug | debug-scaling | preemptable | prod
export WALLTIME="${WALLTIME:-01:00:00}"          # HH:MM:SS (debug<=1h; preemptable<=72h; prod<=24h)
export FILESYSTEMS="${FILESYSTEMS:-home:eagle}"  # -l filesystems

# --- paths ---
# Repo root: auto-detected from this file's location (deploy/polaris/ -> repo root).
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export EAGLE_BASE="${EAGLE_BASE:-/eagle/${PROJECT}/cascaide}"   # project space on Eagle (Lustre)
export ENV_PREFIX="${ENV_PREFIX:-${EAGLE_BASE}/envs/cascaide}"  # conda/venv on Eagle
export DATA="${DATA:-${EAGLE_BASE}/data/cascaide_cascades.npz}" # processed dataset (npz)
export RESULTS="${RESULTS:-${EAGLE_BASE}/results}"             # results store (run dirs, scorecards)
export RUNS_DIR="${RUNS_DIR:-${EAGLE_BASE}/runs}"             # checkpoints / training output

# --- Globus endpoints (UUIDs) for local<->Eagle transfers ---
# Find with: globus endpoint search "ALCF Eagle"  and  globus endpoint local-id
export EAGLE_ENDPOINT="${EAGLE_ENDPOINT:-05d2c76a-e867-4f67-aa57-76edeb0beda0}"   # ALCF Eagle Globus collection
export LOCAL_ENDPOINT="${LOCAL_ENDPOINT:-7b4d7fd6-5f5b-11f1-9808-0e9d40238285}"  # your laptop's Globus Connect Personal
export EAGLE_PATH="${EAGLE_PATH:-/Cascaide/models}"               # path within the Eagle collection
