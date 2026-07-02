#!/usr/bin/env bash
set -euo pipefail

"${PYTHON:-python3}" -m codex_agent.scenefunc3d.servers.scenefunc_sidecar_server "$@"
