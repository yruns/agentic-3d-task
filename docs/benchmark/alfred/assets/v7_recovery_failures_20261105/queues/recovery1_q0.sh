#!/usr/bin/env bash
set -eo pipefail
source /home/ysh/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
set -u
cd /home/ysh/codebase/agentic-3d-task
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,$no_proxy}"
export DISPLAY=:5
export XAUTHORITY="$PWD/tmp/alfred_setup/Xauthority"
exec python tmp/alfred_setup/run_pilot50_recovery_queue.py \
  --manifest tmp/alfred_setup/pilot50_cohort.json --queue-index 0 "$@"
