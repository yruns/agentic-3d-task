#!/usr/bin/env bash
set -euo pipefail

project_root="$(pwd)"
codex_home="${project_root}/.codex-home"
adapter_base_url="http://127.0.0.1:8787/v1"
model_name="gpt-5.4-2026-03-05"
python_bin="${BOOTSTRAP_PYTHON_BIN:-python3}"
force=false

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
    --force)
      force=true
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
usage: scripts/codex_agent/bootstrap_codex_home.sh [options]

Options:
  --project-root PATH     Trusted project root for Codex turns.
  --codex-home PATH       Directory where config.toml is written.
  --adapter-base-url URL  Codex adapter Responses base URL.
  --model NAME            Exact registered Codex model name.
  --force                 Replace an existing config.toml (default: refuse).

The optional raw CODEX_ADAPTER_TOKEN is converted by the Python runtime into
CODEX_ADAPTER_AUTHORIZATION="Bearer ...". No upstream credential belongs here.
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
  BOOTSTRAP_TOML_VALUE="$1" "${python_bin}" -I - <<'PY'
from __future__ import annotations

import json
import os

print(json.dumps(os.environ["BOOTSTRAP_TOML_VALUE"]))
PY
}

model_name_toml="$(toml_string "${model_name}")"
adapter_base_url_toml="$(toml_string "${adapter_base_url}")"
project_root_toml="$(toml_string "${project_root}")"

config_path="${codex_home}/config.toml"
if [[ "${force}" != true ]]; then
  if [[ -e "${config_path}" || -L "${config_path}" ]]; then
    printf 'config already exists: %s; use --force to replace it\n' "${config_path}" >&2
    exit 1
  fi
  set -o noclobber
fi
mkdir -p "${codex_home}"
cat >"${config_path}" <<TOML
model = ${model_name_toml}
model_provider = "codex_adapter"

[model_providers.codex_adapter]
name = "Codex adapter"
base_url = ${adapter_base_url_toml}
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0

[model_providers.codex_adapter.env_http_headers]
X-Codex-Session-ID = "CODEX_ADAPTER_SESSION_ID"
Authorization = "CODEX_ADAPTER_AUTHORIZATION"

[projects.${project_root_toml}]
trust_level = "trusted"
TOML

printf 'wrote %s\n' "${codex_home}/config.toml"
