#!/bin/bash
# Move data/artifacts between your LOCAL machine and ALCF Eagle via Globus.
# (Polaris reads/writes /eagle directly during jobs; Globus is only for local<->Eagle.)
#
# One-time: `pip install globus-cli && globus login`, start Globus Connect Personal locally,
#   then set EAGLE_ENDPOINT / LOCAL_ENDPOINT / EAGLE_PATH in config.sh.
#   Find UUIDs:  globus endpoint search "ALCF Eagle"   |   globus endpoint local-id
#
#   ./deploy/polaris/globus.sh push    # local processed npz  -> Eagle (so jobs can read it)
#   ./deploy/polaris/globus.sh pull    # Eagle results/       -> local (for the dashboard)
#   ./deploy/polaris/globus.sh pull-ckpts   # Eagle runs/ (model checkpoints) -> local
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

for v in EAGLE_ENDPOINT LOCAL_ENDPOINT; do
  case "${!v}" in *"<"*) echo "ERROR: set $v in config.sh (a Globus endpoint UUID)"; exit 1;; esac
done
L="$LOCAL_ENDPOINT"; E="$EAGLE_ENDPOINT"

case "${1:-}" in
  push)        # processed dataset -> Eagle
    globus transfer "$L:$REPO/data/cascaide_cascades.npz" \
                    "$E:$EAGLE_PATH/data/cascaide_cascades.npz" \
                    --label "cascaide-data-push" ;;
  pull)        # results store (scorecards, run.json, gen-vs-real images, report.html) -> local
    globus transfer --recursive "$E:$EAGLE_PATH/results" "$L:$REPO/results" \
                    --label "cascaide-results-pull" ;;
  pull-ckpts)  # model checkpoints -> local (Eagle is slow disk; pull only what you need)
    globus transfer --recursive "$E:$EAGLE_PATH/runs" "$L:$REPO/runs" \
                    --label "cascaide-ckpts-pull" ;;
  *)
    echo "usage: globus.sh push | pull | pull-ckpts"; exit 1 ;;
esac
echo "submitted Globus transfer — track with: globus task list   (then locally: refresh the dashboard)"
