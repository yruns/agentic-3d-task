#!/usr/bin/env bash
set -euo pipefail

"${PYTHON:-python}" -m codex_agent.scenefunc3d.servers.sam2_mask_server "$@"
