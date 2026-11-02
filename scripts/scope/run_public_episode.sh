#!/usr/bin/env bash
set -euo pipefail
trap 'result=$?; printf "%s\n" "$result" > /home/ysh/Datasets/SCOPE/logs/public_episode.exit' EXIT
source /home/ysh/miniconda3/etc/profile.d/conda.sh
conda activate scope-repro
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_CACHE=/home/ysh/Datasets/SCOPE/weights/huggingface/hub
export HF_HUB_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY
curl --noproxy '*' --fail --silent http://127.0.0.1:8899/health
cd /home/ysh/codecase/SCOPE-repro/SCOPE
[[ -z "$(git status --porcelain)" ]] || { printf 'Upstream working tree is dirty\n'; exit 2; }
python /home/ysh/Datasets/SCOPE/scripts/run_scope.py \
  --upstream-root /home/ysh/codecase/SCOPE-repro/SCOPE \
  --cfg-file /home/ysh/Datasets/SCOPE/manifests/eval_public_example.yaml \
  --bridge-config /home/ysh/Datasets/SCOPE/private/bridge_connection.json \
  --split 1
