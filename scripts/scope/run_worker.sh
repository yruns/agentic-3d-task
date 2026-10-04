#!/usr/bin/env bash
# Execute one disjoint scene shard with the original SCOPE episode selector.
set -euo pipefail
[[ $# -eq 2 ]] || { printf 'Usage: run_worker.sh SPLIT WORKER\n' >&2; exit 2; }
scope_split=$1
scope_worker=$2
[[ "$scope_split" =~ ^([1-9]|10)$ ]] || exit 2
case "$scope_worker" in
  0) scope_gpu=0; scope_start=0.0; scope_end=0.5 ;;
  1) scope_gpu=2; scope_start=0.5; scope_end=1.0 ;;
  *) exit 2 ;;
esac
scope_status=/home/ysh/Datasets/SCOPE/logs/full_split${scope_split}_worker${scope_worker}.exit
trap 'scope_exit=$?; printf "%s\n" "$scope_exit" > "$scope_status"' EXIT
source /home/ysh/miniconda3/etc/profile.d/conda.sh
conda activate scope-repro
export CUDA_VISIBLE_DEVICES=$scope_gpu
export HF_HUB_CACHE=/home/ysh/Datasets/SCOPE/weights/huggingface/hub
export HF_HUB_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY
curl --noproxy '*' --fail --silent http://127.0.0.1:8899/health
cd /home/ysh/codecase/SCOPE-repro/SCOPE
[[ -z "$(git status --porcelain)" ]] || exit 2
python /home/ysh/Datasets/SCOPE/scripts/run_scope.py \
  --upstream-root /home/ysh/codecase/SCOPE-repro/SCOPE \
  --cfg-file /home/ysh/Datasets/SCOPE/manifests/eval_full2669.yaml \
  --bridge-config /home/ysh/Datasets/SCOPE/private/bridge_connection.json \
  --split "$scope_split" --start-ratio "$scope_start" --end-ratio "$scope_end"
