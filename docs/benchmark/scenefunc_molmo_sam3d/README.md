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
| [v3_agent_loop_status_20260630](v3_agent_loop_status_20260630.md) | 2026-06-30 | Current SceneFunc3D Molmo/SAM agent loop status after hardening prompt rules, CLI sequence guards, final fuse provenance validation, and Linux ModelHub adapter launch. Documents the latest observed `421393::729dc0d8-571c-44e5-9dc4-05045524dcf5` r22/r23 trajectories. | The visual tool chain is viable: MolmoPoint-8B, SAM2.1-Hiera-L, 2D-to-3D lift, inspection, and fuse can run. The current blocker is the free-form agent controller: r22 reached two valid inspected fragments but did not fuse, while r23 reached SAM but tried to inspect before `lift_mask_to_3d` and did not recover from the guard rejection. The next recommended path is runner-side FSM control plus auto-fuse fallback. |
| [agent_runner_e2e_20260629](agent_runner_e2e_20260629.md) | 2026-06-29 | Current-commit SceneFunc3D agent runner E2E on `421393::729dc0d8-571c-44e5-9dc4-05045524dcf5` with `MolmoPoint-8B + SAM2.1-Hiera-L` on Merlin worker `975499` (`Tesla V100-SXM2-32GB`), plus a later B200 rerun/preflight after adding initial keyframe crop recommendations. Includes a direct CLI regression for `suggest_additional_views` when `accepted_frame_id` is omitted. | The runner produced valid `summary.json`, `result.json`, `fused/mask_artifact.json`, `fused/mask_data.npz`, and `fused/lifted_points.ply` artifacts from one accepted fragment `000010_mask_02`. The scored result was a semantic miss: IoU/precision/recall/F1 were all `0.0` (`predicted_count=99`, `gt_count=199`). The B200 rerun loaded both sidecars but the Codex model stream disconnected on the direct OpenAI path; Linux adapter preflight correctly selected `https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online` and then stopped before GPU sidecars because no gitignored ModelHub AK/TOML was present. |
| [sidecar_e2e_20260628](sidecar_e2e_20260628.md) | 2026-06-28/29 | Repo-integrated SceneFunc3D MolmoPoint/SAM2.1 sidecar status, including runner hardening, tool-loop inspection/fusion, frame geometry exposure, query-aware lightweight keyframes, result scoring, local model-cache checks, and real H100 single-frame sidecar smokes. | On 2026-06-29, `MolmoPoint-8B + SAM2.1-Hiera-L` ran on H100 worker `975402` for `421254/000050` with the latest tool contract: Molmo point output, SAM contact sheet, 2D-to-3D lift, inspect, `suggest_additional_views`, and fuse completed with a 401-point fused artifact and `multi_view_decision.action=stop`. This proves the repo sidecar tool chain for one frame, but not yet the full multi-view agent policy or scored benchmark runner. |
| [v2_full_molmo_20260627](v2_full_molmo_20260627.md) | 2026-06-27 | Full Molmo-7B-D + SAM + 3D lifting rerun on `421254/000050` drawer-knob target using local Molmo weights under `/nas` and Merlin worker `974456` (`NVIDIA-B200`). Compared highest-score SAM multimask selection against smallest non-empty candidate selection. | Molmo produced a usable point (`x=81.0%`, `y=61.9%`). Highest-score SAM over-segmented the cabinet front (31.3% image coverage), while smallest-candidate selection isolated the knob (1,119 pixels, 0.0405% coverage) and lifted 1,119 3D points. |
| [v1_smoke_20260627](v1_smoke_20260627.md) | 2026-06-27 | SceneFuncVal-CG small-object feasibility on Merlin worker `974456` (`NVIDIA-B200`). Tried to run Molmo-7B-D + local SAM on `421393/000010` radiator dial; external Molmo download failed from Hugging Face, hf-mirror, and ModelScope due TLS EOF. Added a reusable smoke script and ran manual-point diagnostics for SAM + depth/pose lifting. | Full Molmo + SAM + 3D lifting **blocked by missing Molmo weights**. SAM + 3D lifting is viable when the point is accurate: `421254/000050` drawer knob produced a 1,016-pixel mask (0.037% image coverage) and a 1,016-vertex lifted PLY. |

## Assets

- `assets/molmo_sam3d_smoke.py` — reusable smoke script for Molmo point output,
  SAM point-prompt masks, and SceneFuncVal-CG depth/pose lifting.
- `assets/run_full_421254_000050.sh` — reproducible runner for the full Molmo
  drawer-knob smoke, with environment overrides for `SAM_SELECTION`,
  `ALLOW_FAILURE`, `OUTPUT_DIR`, `LOG_PATH`, and `PROMPT`.
- `assets/run_sam_transformers_debug_20260629.sh` — focused SAM2.1-Hiera-L
  transformers backend debug runner for one point prompt on `421254/000050`.
- `assets/run_sidecar_tool_smoke_20260629.sh` — H100 worker smoke for the
  repo-integrated sidecars and tools: MolmoPoint, SAM, lift, inspect, and fuse.
- `assets/run_agent_runner_e2e_20260629.sh` — intended real E2E driver for
  scored SceneFunc3D agent runs. By default it runs one `SAMPLE_ID`; set
  `SAMPLE_IDS_PATH=/path/to/sample_ids.json` for a batch file or `ALL_SAMPLES=1`
  to enumerate the prepared dataset. Set only one explicit sample source:
  explicit non-empty `SAMPLE_ID` cannot be combined with `SAMPLE_IDS_PATH` or
  `ALL_SAMPLES=1`, and the default `SAMPLE_ID` is used only when no batch source
  is set. On Linux it starts the project-local ModelHub adapter by default;
  other platforms can still set `START_ADAPTER=1` explicitly. The adapter loads
  private credentials from the gitignored `codex_modelhub_adapter/.env` or
  `codex_modelhub_adapter/.modelhub_upstreams.toml` before launching MolmoPoint,
  SAM2.1-Hiera-L, and `codex_agent.scenefunc3d.runner --score`.
- `assets/test_molmo_sam3d_smoke.py` — unit tests for point parsing, SAM
  candidate selection, success assessment, and Molmo remote-code patching.
