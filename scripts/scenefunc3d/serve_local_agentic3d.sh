#!/usr/bin/env bash
# Start the native MolmoPoint + Transformers SAM2 service using shared assets.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "agentic3d" ]]; then
  printf 'Activate the agentic3d conda environment before starting this service.\n' >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ ",${CUDA_VISIBLE_DEVICES}," == *,1,* ]]; then
  printf 'Physical GPU 1 is unavailable; select GPU 0, 2, 3, 4, 5, or 6.\n' >&2
  exit 1
fi
bash "$project_root/scripts/link_agentic3d_assets.sh"
staging_root="$project_root/tmp/scenefunc3d/artifacts"
mkdir -p "$staging_root"
exec python -m codex_agent.scenefunc3d.servers.scenefunc_sidecar_server \
  --host 127.0.0.1 \
  --port 9001 \
  --device cuda:0 \
  --molmo-model-path "$project_root/external/models/MolmoPoint-8B" \
  --sam-backend transformers \
  --sam-model-path "$project_root/external/models/sam2.1-hiera-large" \
  --staging-root "$staging_root" \
  "$@"
