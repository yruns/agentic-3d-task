#!/usr/bin/env bash
set -uo pipefail
cd /Users/bytedance/aispace/codex_modelhub_adapter
export AIDP_MODELHUB_UPSTREAMS_TOML=/Users/bytedance/aispace/codex_modelhub_adapter/.modelhub_upstreams.toml
export AIDP_CODEX_PROXY_UPSTREAM_API=auto
export AIDP_CODEX_PROXY_CHAT_COMPLETIONS_MODELS="gpt-5.4*,gpt-5.5*"
export AIDP_LOG_PROMPT_CACHE=1
exec uv run uvicorn adapter.app:app --host 127.0.0.1 --port 8787 2>&1 | tee /tmp/codex_modelhub_adapter_8787_restart.log
