#!/usr/bin/env bash
set -uo pipefail

cd /Users/bytedance/project/agentic-3d-task
source .venv/bin/activate

export AIDP_MODELHUB_UPSTREAMS_TOML=/Users/bytedance/project/agentic-3d-task/codex_modelhub_adapter/.modelhub_upstreams.toml
export AIDP_CODEX_PROXY_UPSTREAM_API=responses
export AIDP_CODEX_PROXY_UPSTREAM_ENV=office
export CODEX_AGENT_MODEL_CONTEXT_WINDOW=900000

output_dir=tmp/nr3d_tools_skill10_minprompt_20260701
log_path=tmp/nr3d_tools_skill10_minprompt_20260701.log
exit_path=tmp/nr3d_tools_skill10_minprompt_20260701.exit

mkdir -p tmp

{
  echo "START $(date) HEAD $(git rev-parse HEAD)"
  echo "WORKTREE_STATUS_START"
  git status --short
  echo "WORKTREE_STATUS_END"
  PYTHONPATH=src python -m codex_agent.cli.run_nr3d \
    --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
    --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
    --output-dir "$output_dir" \
    --pack-name pack_nr3d_v9_catalog_first \
    --limit 10 \
    --workers 10 \
    --sample-retries 2 \
    --tools \
    --skill-path .agents/skills/nr3d-codex-tools/SKILL.md \
    --reasoning-summary auto \
    --turn-timeout 900
  status=$?
  echo "EXIT_STATUS:${status} FINISH $(date)"
  echo "${status}" > "$exit_path"
  exit "$status"
} 2>&1 | tee "$log_path"
exit "${PIPESTATUS[0]}"
