#!/usr/bin/env bash
set -euo pipefail

"${PYTHON:-python3}" -m codex_agent.scenefunc3d.servers.molmo_point_server "$@"
