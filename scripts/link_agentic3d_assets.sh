#!/usr/bin/env bash
# Link the shared, complete model/source store into this checkout without copying it.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_root="${AGENTIC3D_ASSET_ROOT:-/home/ysh/Models/agentic3d}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --project-root)
      project_root="${2:?missing --project-root value}"
      shift 2
      ;;
    --asset-root)
      asset_root="${2:?missing --asset-root value}"
      shift 2
      ;;
    -h|--help)
      printf 'Usage: %s [--project-root PATH] [--asset-root PATH]\n' "$0"
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

[[ -f "$project_root/pyproject.toml" ]] || {
  printf 'Project pyproject.toml is missing: %s\n' "$project_root" >&2
  exit 1
}

for required in \
  models/MolmoPoint-8B/model.safetensors.index.json \
  models/sam2.1-hiera-large/model.safetensors \
  models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_model.safetensors \
  third_party/sam2/sam2/build_sam.py \
  third_party/open_clip/pyproject.toml \
  third_party/transformers/setup.py; do
  [[ -f "$asset_root/$required" ]] || {
    printf 'Required asset is missing: %s/%s\n' "$asset_root" "$required" >&2
    exit 1
  }
done

mkdir -p "$project_root/external"
for name in models third_party; do
  target="$asset_root/$name"
  link="$project_root/external/$name"
  if [[ -L "$link" ]]; then
    [[ "$(readlink -f "$link")" == "$(readlink -f "$target")" ]] || {
      printf 'Refusing to replace an existing link: %s\n' "$link" >&2
      exit 1
    }
  elif [[ -e "$link" ]]; then
    printf 'Refusing to replace an existing path: %s\n' "$link" >&2
    exit 1
  else
    ln -s "$target" "$link"
  fi
  printf '%s -> %s\n' "$link" "$target"
done
