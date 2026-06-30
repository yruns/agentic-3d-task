#!/usr/bin/env bash
set -euo pipefail

project_root="$(pwd)"
codex_home="${project_root}/.codex-home"
adapter_base_url="http://127.0.0.1:8787/v1"
model_name="gpt-5.4-2026-03-05"
python_bin="${BOOTSTRAP_PYTHON_BIN:-python3}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --project-root)
      project_root="${2:?missing value for --project-root}"
      shift 2
      ;;
    --codex-home)
      codex_home="${2:?missing value for --codex-home}"
      shift 2
      ;;
    --adapter-base-url)
      adapter_base_url="${2:?missing value for --adapter-base-url}"
      shift 2
      ;;
    --model)
      model_name="${2:?missing value for --model}"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
usage: scripts/codex_agent/bootstrap_codex_home.sh [options]

Options:
  --project-root PATH     Trusted project root for Codex turns.
  --codex-home PATH       Directory where config.toml is written.
  --adapter-base-url URL  Local ModelHub adapter base URL.
  --model NAME            Codex model name.
USAGE
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

toml_string() {
  BOOTSTRAP_TOML_VALUE="$1" "${python_bin}" - <<'PY'
from __future__ import annotations

import json
import os

print(json.dumps(os.environ["BOOTSTRAP_TOML_VALUE"]))
PY
}

model_name_toml="$(toml_string "${model_name}")"
adapter_base_url_toml="$(toml_string "${adapter_base_url}")"
project_root_toml="$(toml_string "${project_root}")"

mkdir -p "${codex_home}"
cat >"${codex_home}/config.toml" <<TOML
model = ${model_name_toml}
model_provider = "modelhub_adapter"

[model_providers.modelhub_adapter]
name = "ModelHub local adapter"
base_url = ${adapter_base_url_toml}
wire_api = "responses"

[projects.${project_root_toml}]
trust_level = "trusted"
TOML

printf 'wrote %s\n' "${codex_home}/config.toml"
