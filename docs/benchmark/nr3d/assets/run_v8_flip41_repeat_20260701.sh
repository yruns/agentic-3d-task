#!/usr/bin/env bash
set -uo pipefail

cd /Users/bytedance/project/agentic-3d-task
source .venv/bin/activate

export AIDP_MODELHUB_UPSTREAMS_TOML=/Users/bytedance/project/agentic-3d-task/codex_modelhub_adapter/.modelhub_upstreams.toml
export AIDP_CODEX_PROXY_UPSTREAM_API=responses
export AIDP_CODEX_PROXY_UPSTREAM_ENV=office
export CODEX_AGENT_MODEL_CONTEXT_WINDOW=900000

output_dir=tmp/nr3d_tools_flip41_v8_repeat_20260701
log_path=tmp/nr3d_tools_flip41_v8_repeat_20260701.log
exit_path=tmp/nr3d_tools_flip41_v8_repeat_20260701.exit

mkdir -p tmp

{
  echo "START $(date) HEAD $(git rev-parse HEAD)"
  PYTHONPATH=src python -m codex_agent.cli.run_nr3d \
    --sample-ids docs/benchmark/nr3d/assets/v7_v8_flip41_sample_ids_20260701.json \
    --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
    --output-dir "$output_dir" \
    --pack-name pack_nr3d_v9_catalog_first \
    --workers 40 \
    --sample-retries 2 \
    --tools \
    --reasoning-summary auto \
    --turn-timeout 900
  status=$?
  echo "EXIT_STATUS:${status} FINISH $(date)"
  echo "${status}" > "$exit_path"
  exit "$status"
} 2>&1 | tee "$log_path"
exit "${PIPESTATUS[0]}"
