#!/usr/bin/env bash
# Execute one disjoint scene shard with the original SCOPE episode selector.
set -euo pipefail
[[ $# -eq 2 ]] || { printf 'Usage: run_worker.sh SPLIT WORKER\n' >&2; exit 2; }
scope_split=$1
scope_worker=$2
[[ "$scope_split" =~ ^([1-9]|10)$ ]] || exit 2
case "$scope_worker" in
  0) scope_start=0.0; scope_end=0.5 ;;
  1) scope_start=0.5; scope_end=1.0 ;;
  *) exit 2 ;;
esac
scope_status=/home/ysh/Datasets/SCOPE/logs/full_split${scope_split}_worker${scope_worker}.exit
trap 'scope_exit=$?; printf "%s\n" "$scope_exit" > "$scope_status"' EXIT
read -r scope_gpu scope_cfg < <(
  /home/ysh/miniconda3/envs/conceptgraph/bin/python - \
    /home/ysh/Datasets/SCOPE/manifests/storage_plan_parallel_v2.json "$scope_split" "$scope_worker" <<'PY'
from pathlib import Path
import json
import sys
plan = json.loads(Path(sys.argv[1]).read_text())
stages = [stage for stage in plan['stages'] if stage['split'] == int(sys.argv[2]) and stage['worker'] == int(sys.argv[3])]
if len(stages) != 1:
    raise SystemExit('Expected exactly one frozen worker stage')
stage = stages[0]
print(stage['physical_gpu'], stage['cfg_file'])
PY
)
[[ "$scope_gpu" =~ ^[0-6]$ && "$scope_gpu" != 1 && -f "$scope_cfg" ]] || exit 2
source /home/ysh/miniconda3/etc/profile.d/conda.sh
conda activate scope-repro
export CUDA_VISIBLE_DEVICES=$scope_gpu
export HF_HUB_CACHE=/home/ysh/Datasets/SCOPE/weights/huggingface/hub
export HF_HUB_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY
curl --noproxy '*' --fail --silent http://127.0.0.1:8898/health
cd /home/ysh/codecase/SCOPE-repro/SCOPE
[[ -z "$(git status --porcelain)" ]] || exit 2
python /home/ysh/Datasets/SCOPE/scripts/run_scope.py \
  --upstream-root /home/ysh/codecase/SCOPE-repro/SCOPE \
  --cfg-file "$scope_cfg" \
  --bridge-config /home/ysh/Datasets/SCOPE/private/bridge_connection.json \
  --split "$scope_split" --start-ratio "$scope_start" --end-ratio "$scope_end"
