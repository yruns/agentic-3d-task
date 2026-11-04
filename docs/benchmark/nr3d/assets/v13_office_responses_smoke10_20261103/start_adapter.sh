#!/usr/bin/env bash
set -eo pipefail
ROOT=/home/ysh/codebase/agentic-3d-task
# Launch via zsh -ic so the user's existing ~/.zshrc transport setup is inherited.
source /home/ysh/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
set -a
source "$ROOT/codex_modelhub_adapter/.env"
set +a
# Preserve the user's transparent forwarding; no HTTP proxy/NO_PROXY override.
cd "$ROOT"
exec python -m uvicorn adapter.app:app --app-dir "$ROOT/codex_modelhub_adapter" \
  --host 127.0.0.1 --port 8787 --log-level warning --no-access-log \
  > "$ROOT/tmp/nr3d_office_smoke_20261103/preflight/adapter.log" 2>&1
