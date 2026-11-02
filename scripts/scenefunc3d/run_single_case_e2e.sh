#!/usr/bin/env bash
set -euo pipefail

"${PYTHON:-python3}" -m codex_agent.cli.run_scenefunc3d "$@"
