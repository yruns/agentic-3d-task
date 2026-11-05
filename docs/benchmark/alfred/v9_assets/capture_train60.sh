#!/usr/bin/env bash
set -euo pipefail
source /home/ysh/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
export DISPLAY=:5
export XAUTHORITY=/home/ysh/codebase/agentic-3d-task/tmp/alfred_setup/Xauthority
python -m codex_agent.cli.capture_alfred_calibration plan \
 --data-root /home/ysh/Datasets/FILM/alfred/data/json_2.1.0 \
 --splits /home/ysh/Datasets/FILM/alfred_data_small/splits/oct21.json \
 --alfred-root /home/ysh/Datasets/FILM/alfred \
 --ai2thor-root /home/ysh/codebase/FILM/.venv/lib/python3.8/site-packages/ai2thor \
 --unity-executable /home/ysh/.ai2thor/releases/thor-201909061227-Linux64/thor-201909061227-Linux64 \
 --output-plan outputs/alfred_depth_calibration/capture_plan_v1.json
python -m codex_agent.cli.capture_alfred_calibration capture \
 --plan outputs/alfred_depth_calibration/capture_plan_v1.json \
 --output-dir outputs/alfred_depth_calibration/train60_v1 \
 --simulator-python /home/ysh/codebase/FILM/.venv/bin/python \
 --x-display 5 --worker-timeout 300
