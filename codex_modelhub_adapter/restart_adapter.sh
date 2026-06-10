#!/usr/bin/env bash
set -uo pipefail
# Resolve to this script's own directory so the adapter runs from wherever the
# repo is checked out (no hardcoded absolute path).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export AIDP_MODELHUB_UPSTREAMS_TOML="$HERE/.modelhub_upstreams.toml"
export AIDP_CODEX_PROXY_UPSTREAM_API=auto
export AIDP_CODEX_PROXY_CHAT_COMPLETIONS_MODELS="gpt-5.4*,gpt-5.5*"
export AIDP_LOG_PROMPT_CACHE=1
exec uv run uvicorn adapter.app:app --host 127.0.0.1 --port 8787 2>&1 | tee /tmp/codex_modelhub_adapter_8787_restart.log
