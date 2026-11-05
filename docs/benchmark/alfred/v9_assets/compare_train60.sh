#!/usr/bin/env bash
set -eo pipefail
source /home/ysh/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
set -u
python -m codex_agent.cli.run_alfred_depth_calibration \
 --manifest outputs/alfred_depth_calibration/train60_v1/manifest.json \
 --da3-model-dir /home/ysh/Models/agentic3d/models/alfred_vision/da3metric-large \
 --unidepth-model-dir /home/ysh/Models/agentic3d/models/alfred_vision/unidepth-v2-vitl14 \
 --output-dir outputs/alfred_depth_calibration/compare_train60_v2 \
 --gpu-lock-path /tmp/agentic3d-alfred-gpu4.lock \
 --physical-gpu-index 4 --device cuda:0
