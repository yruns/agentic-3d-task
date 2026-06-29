# SceneFunc Molmo SAM3D archive

Process archive for Molmo + SAM + 3D lifting feasibility checks on the local
SceneFuncVal-CG wrapper dataset.

The target workflow is the lightweight part of TASA-style 2D guidance:
language-conditioned Molmo pointing, SAM mask generation from the point, then
depth/pose lifting into a 3D point cloud. The Point Transformer refinement stage
from TASA is intentionally out of scope for this smoke.

## Version timeline

| Version | Date | Scope | Headline |
|---|---|---|---|
| [sidecar_e2e_20260628](sidecar_e2e_20260628.md) | 2026-06-28 | Repo-integrated SceneFunc3D MolmoPoint/SAM2.1 sidecar status, including runner hardening, tool-loop inspection/fusion, frame geometry exposure, query-aware lightweight keyframes, result scoring, and local model-cache checks. | SceneFunc3D code-path tests pass, fake final-mask artifacts are rejected, and runner outputs are directly scoreable, but true sidecar E2E is not yet proven: the master shell has no CUDA device and GPU worker launch hit readiness/compliance-gateway failures. The available SAM2.1-Hiera-L cache is Hugging Face `model.safetensors/config.json`, so the server path uses the transformers backend rather than the official `.pt + yaml` format assumed by the first plan. |
| [v2_full_molmo_20260627](v2_full_molmo_20260627.md) | 2026-06-27 | Full Molmo-7B-D + SAM + 3D lifting rerun on `421254/000050` drawer-knob target using local Molmo weights under `/nas` and Merlin worker `974456` (`NVIDIA-B200`). Compared highest-score SAM multimask selection against smallest non-empty candidate selection. | Molmo produced a usable point (`x=81.0%`, `y=61.9%`). Highest-score SAM over-segmented the cabinet front (31.3% image coverage), while smallest-candidate selection isolated the knob (1,119 pixels, 0.0405% coverage) and lifted 1,119 3D points. |
| [v1_smoke_20260627](v1_smoke_20260627.md) | 2026-06-27 | SceneFuncVal-CG small-object feasibility on Merlin worker `974456` (`NVIDIA-B200`). Tried to run Molmo-7B-D + local SAM on `421393/000010` radiator dial; external Molmo download failed from Hugging Face, hf-mirror, and ModelScope due TLS EOF. Added a reusable smoke script and ran manual-point diagnostics for SAM + depth/pose lifting. | Full Molmo + SAM + 3D lifting **blocked by missing Molmo weights**. SAM + 3D lifting is viable when the point is accurate: `421254/000050` drawer knob produced a 1,016-pixel mask (0.037% image coverage) and a 1,016-vertex lifted PLY. |

## Assets

- `assets/molmo_sam3d_smoke.py` — reusable smoke script for Molmo point output,
  SAM point-prompt masks, and SceneFuncVal-CG depth/pose lifting.
- `assets/run_full_421254_000050.sh` — reproducible runner for the full Molmo
  drawer-knob smoke, with environment overrides for `SAM_SELECTION`,
  `ALLOW_FAILURE`, `OUTPUT_DIR`, `LOG_PATH`, and `PROMPT`.
- `assets/test_molmo_sam3d_smoke.py` — unit tests for point parsing, SAM
  candidate selection, success assessment, and Molmo remote-code patching.
