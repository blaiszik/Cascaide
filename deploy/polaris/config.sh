#!/bin/bash
# Polaris cluster config — EDIT THESE ONCE, then `source` from the other scripts.
# Code lives in the repo (git pull on Polaris). Data/artifacts live on Eagle and move
# local<->eagle via Globus (see globus.sh). Polaris reads/writes Eagle directly at
# /lus/eagle/projects/<project>/... .

# --- ALCF allocation / queue ---
export PROJECT="${PROJECT:-Cascaide}"            # PBS -A (your ALCF project/allocation)
export QUEUE="${QUEUE:-debug-scaling}"           # debug-scaling(<=10 nodes,<=1h) | debug | preemptable | prod
export WALLTIME="${WALLTIME:-01:00:00}"          # HH:MM:SS (debug/debug-scaling<=1h; preemptable<=72h; prod<=24h)
export FILESYSTEMS="${FILESYSTEMS:-home:eagle}"  # -l filesystems

# --- paths (EAGLE_BASE is the single source of truth: the POSIX path as seen on Polaris) ---
# NOTE: paths use DIRECT assignment (not ${VAR:-default}) so re-sourcing always reflects this
# file — a stale value exported by a previous `source` must never win. Change the subdir by
# editing EAGLE_BASE below.
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"     # auto-detected repo root
export EAGLE_BASE="/lus/eagle/projects/${PROJECT}/models"       # Eagle project space (Lustre)
export ENV_PREFIX="${EAGLE_BASE}/envs/cascaide"  # venv on Eagle
export DATA="${EAGLE_BASE}/data/cascaide_cascades.npz"          # processed dataset (npz)
export RESULTS="${EAGLE_BASE}/results"           # results store (run dirs, scorecards)
export RUNS_DIR="${EAGLE_BASE}/runs"             # checkpoints / training output

# --- Globus (only used by globus.sh on your LAPTOP; Polaris reads Eagle directly) ---
# Find UUIDs: globus endpoint search "ALCF Eagle"  ;  globus endpoint local-id
export EAGLE_ENDPOINT="${EAGLE_ENDPOINT:-05d2c76a-e867-4f67-aa57-76edeb0beda0}"   # ALCF Eagle collection
export LOCAL_ENDPOINT="${LOCAL_ENDPOINT:-7b4d7fd6-5f5b-11f1-9808-0e9d40238285}"   # your laptop (Globus Connect Personal)
# Globus collection root (verify: globus ls $EAGLE_ENDPOINT:/ — ALCF Eagle is usually /lus/eagle/projects).
export EAGLE_COLLECTION_ROOT="${EAGLE_COLLECTION_ROOT:-/lus/eagle/projects}"
# Collection-relative path is DERIVED from EAGLE_BASE so Globus writes EXACTLY where Polaris reads
# (e.g. /lus/eagle/projects/Cascaide/models -> /Cascaide/models). Direct assignment (no drift).
export EAGLE_PATH="${EAGLE_BASE#$EAGLE_COLLECTION_ROOT}"
