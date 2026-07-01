#!/usr/bin/env bash
set -euo pipefail

backend_config="${1:?usage: scripts/scenefunc3d/check_sidecars.sh /path/to/scenefunc3d_backends.toml}"

"${PYTHON:-python3}" - <<'PY' "$backend_config"
from __future__ import annotations

import sys
from pathlib import Path

from codex_agent.scenefunc3d.runner import check_sidecar_health


config_path = Path(sys.argv[1])
check_sidecar_health(config_path)
print(f"sidecars ok: {config_path}")
PY
