# SceneFunc3D agent runner E2E, 2026-06-29

This record covers a current-commit end-to-end run of the SceneFunc3D agent
runner on one `SceneFuncVal-CG` case with MolmoPoint-8B and SAM2.1-Hiera-L.

The run proves that the repo-integrated agent path can produce a scored 3D mask
artifact, but the produced mask did not overlap the hidden GT for this case.

## Run Identity

- Branch: `feat/scenefunc3d-agent-tools`
- Head commit at launch: `056a4f7318b7e998031169ea4763df8fc20538fb`
- Run-time code commit: same as head commit; no worktree drift.
- Worker: Merlin worker `975499`
- GPU: `Tesla V100-SXM2-32GB`, 32768 MiB
- Sample id: `421393::729dc0d8-571c-44e5-9dc4-05045524dcf5`
- Task description: `Adjust the room's temperature using the radiator dial`
- Run root:
  `/tmp/scenefunc_agent_runner_e2e_421393_056a4f7_20260629`

## Invocation

```bash
NO_COLOR=1 TERM=dumb ssh -o StrictHostKeyChecking=no -p 9277 \
  fdbd:dc61:10:599::155 \
  'cd /mlx_devbox/users/yueshuhao/playground/repos/agentic-3d-task/.worktrees/scenefunc3d-agent-tools && \
   USE_CODEX_AUTH=1 \
   SAMPLE_ID=421393::729dc0d8-571c-44e5-9dc4-05045524dcf5 \
   RUN_ROOT=/tmp/scenefunc_agent_runner_e2e_421393_056a4f7_20260629 \
   docs/benchmark/scenefunc_molmo_sam3d/assets/run_agent_runner_e2e_20260629.sh \
   2>&1 | tee /tmp/scenefunc_agent_runner_e2e_421393_056a4f7_20260629.log'
```

Models loaded by the launcher:

- Molmo sidecar: `MolmoPoint-8B`
- SAM sidecar: `SAM2.1-Hiera-L`

Both sidecars reported `model_loaded=True` before the runner started.

## Result

Runner stdout:

```json
{
  "result_path": "/tmp/scenefunc_agent_runner_e2e_421393_056a4f7_20260629/agent_outputs/421393/729dc0d8-571c-44e5-9dc4-05045524dcf5/result.json",
  "score": {
    "sample_id": "421393::729dc0d8-571c-44e5-9dc4-05045524dcf5",
    "failure_type": "",
    "metrics": {
      "iou": 0.0,
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "predicted_count": 99,
      "gt_count": 199
    }
  }
}
```

Final summary:

- `status`: `success`
- `selected_frame_ids`: `["000010"]`
- `accepted_fragment_ids`: `["000010_mask_02"]`
- `final_point_count`: `3881`
- `confidence`: `0.82`
- `multi_view_decision.action`: `stop`
- `multi_view_decision.suggested_frame_ids`:
  `["000101", "000100", "000099", "000086"]`

The result is a valid artifact-generation success but an evaluation failure for
this case: the final predicted scene-point ids had zero overlap with the hidden
GT ids.

## Artifact Paths

Primary outputs:

```text
/tmp/scenefunc_agent_runner_e2e_421393_056a4f7_20260629/agent_outputs/421393/729dc0d8-571c-44e5-9dc4-05045524dcf5/summary.json
/tmp/scenefunc_agent_runner_e2e_421393_056a4f7_20260629/agent_outputs/421393/729dc0d8-571c-44e5-9dc4-05045524dcf5/result.json
/tmp/scenefunc_agent_runner_e2e_421393_056a4f7_20260629/agent_outputs/421393/729dc0d8-571c-44e5-9dc4-05045524dcf5/events.jsonl
/tmp/scenefunc_agent_runner_e2e_421393_056a4f7_20260629/agent_outputs/421393/729dc0d8-571c-44e5-9dc4-05045524dcf5/fused/mask_artifact.json
/tmp/scenefunc_agent_runner_e2e_421393_056a4f7_20260629/agent_outputs/421393/729dc0d8-571c-44e5-9dc4-05045524dcf5/fused/mask_data.npz
/tmp/scenefunc_agent_runner_e2e_421393_056a4f7_20260629/agent_outputs/421393/729dc0d8-571c-44e5-9dc4-05045524dcf5/fused/lifted_points.ply
```

Artifact sizes on the worker:

| File | Bytes |
|---|---:|
| `summary.json` | 1605 |
| `result.json` | 2551 |
| `events.jsonl` | 23113 |
| `fused/mask_artifact.json` | 2077 |
| `fused/mask_data.npz` | 82360 |
| `fused/lifted_points.ply` | 212305 |
| `molmo/000010_raw.txt` | 105 |
| `molmo/000010_points.jpg` | 11224 |
| `sam/000010/contact_sheet.jpg` | 35341 |

## Tool Trace

The run wrote 22 tool events:

| Tool | Status | Count |
|---|---|---:|
| `scene_summary` | success | 1 |
| `keyframe_selector` | failed | 1 |
| `keyframe_selector` | success | 1 |
| `view_bev` | failed | 2 |
| `view_frame` | success | 8 |
| `view_crop` | success | 2 |
| `molmo_point` | success | 1 |
| `sam_mask` | success | 1 |
| `lift_mask_to_3d` | success | 1 |
| `inspect_mask_artifact` | success | 1 |
| `suggest_additional_views` | success | 1 |
| `fuse_accepted_masks` | success | 1 |
| `run_completed` | success | 1 |

Recoverable failures observed:

- `keyframe_selector` rejected `top_k` as an extra field, then the agent retried
  without it and succeeded.
- `view_bev` rejected `task_description` as an extra field.
- `view_bev` then failed because this scene has no BEV asset at
  `conceptgraph/bev/scene_bev.png`.

These did not block final artifact generation.

## Agent Decisions

Evidence phase:

- `keyframe_selector` returned `000073`, `000010`, `000011`, and `000072`.
- The agent viewed all four frames.
- The agent generated crops on `000011` and `000010`.

Molmo phase:

- Input image:
  `421393/000010_crop_pixel_xyxy_41aa437cdb42.jpg`
- Prompt: point to the operable radiator temperature dial/thermostatic knob,
  selecting the center of the round beige cylindrical dial and avoiding the
  radiator panel, wall, pipe, label box, and shadows.
- Parsed point: `x_px=143.2`, `y_px=185.98148148148147`
- Raw output and overlay were saved before SAM ran.

SAM phase:

SAM returned three candidates on frame `000010`:

| Candidate | Score | Pixels | Image Coverage |
|---|---:|---:|---:|
| `mask_00` | 0.06103515625 | 157 | 0.005678530092592593 |
| `mask_01` | 0.384765625 | 10442 | 0.37767650462962965 |
| `mask_02` | 0.859375 | 4128 | 0.14930555555555555 |

The agent selected `mask_02`, lifted it to 3D, inspected the lifted artifact,
called `suggest_additional_views`, viewed the four suggested follow-up frames,
and then stopped with a single accepted fragment.

Lift/fuse phase:

- Accepted fragment: `000010_mask_02`
- Lifted point count: `3881`
- Suggested follow-up frames: `000101`, `000100`, `000099`, `000086`
- Final fused artifact reused the single accepted fragment.

## Regression Check For Seed-Fragment Frame Inference

The commit also includes a boundary fix for `suggest_additional_views`: when an
agent omits `accepted_frame_id`, the tool can infer it from
`seed_fragment_id="<frame_id>_<candidate_id>"`.

The E2E agent happened to pass `accepted_frame_id` explicitly, so a direct CLI
regression was run on the real E2E seed fragment with `accepted_frame_id`
intentionally omitted:

```bash
PYTHONPATH=src /mlx_devbox/users/yueshuhao/miniforge3/envs/conceptgraph/bin/python \
  -m codex_agent.scenefunc3d.tools suggest_additional_views \
  --scene-root /mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG/421393 \
  --args '{"seed_fragment_id":"000010_mask_02","seed_mask_npz_path":".../fragments/000010_mask_02/mask_data.npz","seed_mask_ply_path":".../fragments/000010_mask_02/lifted_points.ply","seed_lift_overlay_path":".../fragments/000010_mask_02/lift_overlay.txt","k":1}' \
  --out-dir /tmp/scenefunc_agent_runner_e2e_421393_056a4f7_20260629/inference_regression_cli
```

Result: exit code `0`; the tool returned `seed_lift_status="usable"` and
suggested frame `000101`. This verifies the missing-`accepted_frame_id` path on
the real E2E artifact.

## Quality Gate

Before this E2E launch, the following current-commit checks passed:

```bash
/mlx_devbox/users/yueshuhao/miniforge3/bin/conda run -n conceptgraph ruff check src/
/mlx_devbox/users/yueshuhao/miniforge3/bin/conda run -n conceptgraph black src/
/mlx_devbox/users/yueshuhao/miniforge3/bin/conda run -n conceptgraph mypy src/
/mlx_devbox/users/yueshuhao/miniforge3/bin/conda run -n conceptgraph env PYTHONPATH=src pytest src/keyframe/tests -q
/mlx_devbox/users/yueshuhao/miniforge3/bin/conda run -n conceptgraph env PYTHONPATH=src pytest src/codex_agent/tests/test_scenefunc3d_*.py -q
```

Observed results:

- `ruff check src/`: passed.
- `black src/`: 169 files unchanged; Black emitted the existing Python
  3.11/target 3.12 safety warning.
- `mypy src/`: no issues in 169 source files.
- `src/keyframe/tests`: 37 passed, 1 skipped, 1 pytest config warning.
- `src/codex_agent/tests/test_scenefunc3d_*.py`: 390 passed, 1 pytest config
  warning.

## Interpretation

The current agent task is end-to-end executable and produces the requested
artifact surface:

- `mask_npz_path`
- `mask_ply_path`
- accepted frame ids
- accepted fragment ids
- confidence
- uncertainties
- score metrics

However, this is not yet a good SceneFunc3D policy. The failure mode in this
case is semantic/geometric alignment: the agent chose a visually plausible SAM
candidate for the radiator dial, but the final scene-point ids did not overlap
the hidden GT. The next improvement should focus on mask choice and 3D-lift
validation, not on sidecar availability.
