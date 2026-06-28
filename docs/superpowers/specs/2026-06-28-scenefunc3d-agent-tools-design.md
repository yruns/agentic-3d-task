# SceneFunc3D Agent Tools Design

Date: 2026-06-28

## Goal

Add a SceneFunc3D agent workflow for the prepared ConceptGraph dataset at:

```text
/mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG
```

The workflow should let an agent solve language-conditioned functional-part
segmentation tasks by finding visual evidence, asking Molmo for task-constrained
2D points, asking SAM for 2D mask candidates, lifting accepted masks into 3D,
and optionally expanding from the first accepted 3D seed mask to additional
views.

This is a distinct task from NR3D. NR3D selects one existing bbox/proposal id.
SceneFunc3D must generate and return a 3D mask artifact.

## Current Project Context

The repository already has separate Codex Agent task packages for NR3D and
OpenEQA:

- `src/codex_agent/nr3d/`
- `src/codex_agent/openeqa/`

Those packages provide useful patterns for task-specific samples, playbooks,
runners, tool dispatchers, keyframe retrieval, frame rendering, crop rendering,
BEV rendering, and tests.

SceneFuncVal-CG currently contains two complete ConceptGraph-style scenes:

- `421254`
- `421393`

Each scene has:

- `<visit_id>_descriptions.json`
- `<visit_id>_annotations.json`
- `<visit_id>_motions.json`
- `conceptgraph/` assets

Existing benchmark records under `docs/benchmark/scenefunc_molmo_sam3d/`
confirm that Molmo plus SAM plus depth/pose lifting is feasible, but also show
that naive SAM highest-score selection can over-segment large surfaces.

## Decision

Use a separate SceneFunc3D package and runner:

```text
src/codex_agent/scenefunc3d/
```

This package can reuse low-level evidence-gathering patterns from NR3D/OpenEQA,
but it must not reuse the NR3D task entrypoint, final-answer schema, or
proposal-id semantics.

The first implementation should target a practical end-to-end version:

1. Load SceneFuncVal-CG samples.
2. Let the agent gather and view evidence.
3. Require agent approval after Molmo point overlays.
4. Require agent approval after SAM candidate overlays.
5. Lift only agent-approved masks into 3D.
6. Use the first accepted 3D mask as a seed for optional multi-view expansion.
7. Save auditable mask artifacts and structured run metadata.
8. Add scoring hooks against hidden ground-truth annotation indices.

## Scope

In scope:

- A new SceneFunc3D task namespace, runner, playbook, sample loader, tool
  dispatcher, artifact schema, and scoring skeleton.
- Evidence tools equivalent to the current OpenEQA/NR3D style: scene summary,
  keyframe selection, frame viewing, crop viewing, BEV viewing, and visible
  object listing.
- Molmo point tool with raw text capture, parsed points, and overlay output.
- SAM mask tool with all candidate masks shown in one contact sheet.
- 3D lifting tool using frame depth, intrinsics, and camera pose.
- Agent approval gates between Molmo, SAM, first lift, and optional multi-view
  expansion.
- Durable artifacts for debugging and evaluation.
- Tests that do not require GPUs for schema, parsing, state transitions,
  sample loading, and scoring math.

Out of scope for the first implementation:

- Training or fine-tuning Molmo, SAM, or a 3D refinement model.
- Fully automatic blind top-K multi-view fusion without agent approval.
- Replacing the existing NR3D/OpenEQA runners.
- Committing model weights, private credentials, or generated large artifacts.
- Making the benchmark smoke script the production interface as-is.

## Package Layout

Use a task-specific package with small typed modules:

```text
src/codex_agent/scenefunc3d/
  __init__.py
  sample.py
  task.py
  playbook.py
  runner.py
  tools/
    __init__.py
    __main__.py
    dispatch.py
    models.py
    scene_context.py
    frame_views.py
    keyframe_retrieval.py
    molmo_pointing.py
    sam_masking.py
    mask_lifting.py
    mask_artifacts.py
    mask_inspection.py
  evaluation/
    __init__.py
    metrics.py
    scorer.py
```

The CLI entrypoint should be separate:

```bash
python -m codex_agent.scenefunc3d.tools <tool> \
  --scene-root <scene_dir> \
  --json '<validated-json-payload>'
```

A batch or single-case runner should also be task-specific:

```bash
python -m codex_agent.scenefunc3d.runner \
  --dataset-root /mlx_devbox/users/yueshuhao/playground/nas/Datasets/SceneFuncVal-CG \
  --visit-id 421254 \
  --desc-id <description_id> \
  --output-dir <run_dir>
```

## Sample Model

The sample loader should produce a typed sample for each description:

```text
SceneFunc3dSample
  sample_id: "<visit_id>::<desc_id>"
  visit_id: "421254"
  desc_id: "<description_id>"
  task_description: "<natural language task>"
  annotation_ids: [...]
  motion_hints: [...]
```

Ground-truth annotation indices from `<visit_id>_annotations.json` must not be
placed in the agent prompt or exposed through prediction tools. They are for
scoring only.

Motion metadata may be exposed as task context when it helps the agent
understand the functional affordance. For example, rotation or translation
metadata can help distinguish a knob, dial, handle, switch, drawer face, or
button.

## Agent State Machine

The runner should enforce a stage-based state machine:

```text
evidence_selected
-> molmo_point_proposed
-> molmo_point_agent_approved
-> sam_candidates_proposed
-> sam_candidate_agent_approved
-> first_lift_created
-> first_lift_agent_approved
-> optional_multiview_expansion
-> fused_mask_created
-> final_answer
```

The transitions are strict:

- The agent must inspect Molmo raw output and point overlay before SAM runs.
- SAM must not run until the agent accepts a point or requests a retry.
- The agent must inspect all SAM candidates in one contact sheet before any 3D
  lift runs.
- 3D lifting must not run until the agent accepts one SAM candidate.
- The first accepted 3D lift becomes the seed mask.
- Additional views are considered only after the agent inspects and accepts the
  first 3D seed mask.
- No mask fragment can enter the final fused artifact unless the agent approved
  its point, 2D mask candidate, and 3D lift artifact.

The state machine should be validated in code so a prompt mistake cannot skip
an approval gate.

## Tool Contracts

### Evidence Tools

Evidence tools should let the agent find and inspect visual evidence before
calling Molmo:

| Tool | Purpose |
| --- | --- |
| `scene_summary` | Return available frames, object counts, and indexed assets. |
| `keyframe_selector` | Retrieve candidate frames for the task description or target concept. |
| `view_frame` | Render one RGB frame for `view_image`. |
| `view_crop` | Render a zoomed crop around a visible object, bbox, or image region. |
| `view_bev` | Render top-down scene context with optional highlights. |
| `frame_objects` | List ConceptGraph objects visible in one frame. |

The playbook must tell the agent that a frame or crop only counts as evidence
after it has viewed the returned image.

### Molmo Point Tool

`molmo_point` should run a configured Molmo-family model on one viewed frame or
crop and return raw and parsed outputs:

```json
{
  "frame_id": "000050",
  "prompt": "Point to the small drawer knob used to open the bottom drawer.",
  "points": [
    {
      "x_px": 1166.4,
      "y_px": 1188.48,
      "source": "molmo_percent",
      "label": "small drawer knob"
    }
  ],
  "raw_text_path": "/abs/path/raw_outputs/molmo_000050.txt",
  "overlay_path": "/abs/path/overlays/molmo_points_000050.jpg"
}
```

After this tool runs, the agent must view the overlay and choose one of:

- `accept_point`
- `retry_with_crop`
- `retry_with_new_prompt`
- `try_another_frame`

For small targets such as knobs, handles, switches, dials, and buttons, the
playbook should instruct the agent to first reason about the affordance concept,
then use a full task-constrained point prompt.

The model backend must be injected through configuration. A first
implementation can wrap the existing local Molmo smoke path; production config
should also support a stronger point-specialized backend such as MolmoPoint-8B
when it is installed in the runtime environment.

### SAM Mask Tool

`sam_mask` should take one or more accepted Molmo points and return every
meaningful candidate mask:

```json
{
  "frame_id": "000050",
  "points": [{"x_px": 1166.4, "y_px": 1188.48}],
  "candidates": [
    {
      "candidate_id": "mask_00",
      "score": 0.8247,
      "pixel_count": 1119,
      "coverage_percent": 0.0405,
      "overlay_path": "/abs/path/overlays/mask_00.jpg"
    }
  ],
  "contact_sheet_path": "/abs/path/overlays/sam_candidates_000050.jpg"
}
```

The agent must view the contact sheet and choose one of:

- `accept_candidate`
- `retry_with_different_point`
- `retry_with_crop`
- `try_another_frame`
- `reject_case`

SAM must not silently return only the highest-score candidate. Candidate
selection policy must be explicit and recorded. Supported policies should
include:

- `highest_score`
- `smallest_non_empty`
- `area_range`
- `crop_local`
- `agent_selected`

Prefer SAM2.1-Hiera-L when it is installed and validated. Original SAM ViT-H can
remain a reproducible baseline, but it should not be used as a hidden fallback.

### 3D Lifting Tool

`lift_mask_to_3d` should project an accepted 2D mask through depth, intrinsics,
and camera pose into scene coordinates:

```json
{
  "frame_id": "000050",
  "candidate_id": "mask_00",
  "lifted_point_count": 1119,
  "mask_npz_path": "/abs/path/mask_data.npz",
  "mask_ply_path": "/abs/path/lifted_points.ply",
  "overlay_path": "/abs/path/overlays/selected_mask_000050.jpg"
}
```

The agent must inspect the selected-mask overlay and lifted artifact summary
before the mask becomes an accepted seed.

The long-term preferred artifact is a mask aligned to the scene point cloud or
ConceptGraph point index space. The first implementation may save lifted
coordinates and nearest scene-point ids together so visualization and scoring
can evolve independently.

### Multi-View Expansion Tool

After the first accepted 3D seed, a `suggest_additional_views` tool should find
other likely useful frames by projecting the seed into frames and checking:

- camera coverage of the seed region
- expected occlusion
- depth availability
- view angle diversity
- whether the initial lift is sparse or partial
- whether the functional part likely has unseen sides

The agent decides whether expansion is needed:

- Stop if the first lift already covers the target part well enough.
- Continue if occlusion, missing depth, sparse points, or incomplete geometry is
  visible.

Every additional frame must repeat the same approval path:

```text
Molmo point -> agent approval -> SAM candidates -> agent approval -> 3D lift -> agent approval
```

### Mask Fusion Tool

`fuse_accepted_masks` should merge only accepted 3D fragments. It may use:

- nearest scene-point ids
- connected components
- spatial proximity
- ConceptGraph object membership
- per-view confidence and coverage metadata

The fused artifact must keep all per-view fragments so failures can be traced
back to the responsible point, SAM candidate, or lift.

### Inspection Tool

`inspect_mask_artifact` should summarize any intermediate or final artifact and
return image paths suitable for `view_image`:

- Molmo point overlay
- SAM all-candidate contact sheet
- selected-mask overlay
- lifted-mask stats
- optional BEV or point-cloud projection preview

This is the main tool the agent uses to approve, reject, or retry.

## Artifact Layout

Each sample run should write a durable artifact directory:

```text
<output_dir>/<visit_id>/<desc_id>/
  summary.json
  events.jsonl
  raw_outputs/
    molmo_<frame_id>_<attempt_id>.txt
  overlays/
    frame_<frame_id>.jpg
    crop_<frame_id>_<attempt_id>.jpg
    molmo_points_<frame_id>_<attempt_id>.jpg
    sam_candidates_<frame_id>_<attempt_id>.jpg
    selected_mask_<frame_id>_<attempt_id>.jpg
  fragments/
    <frame_id>_<candidate_id>/
      mask_data.npz
      lifted_points.ply
      summary.json
  fused/
    mask_data.npz
    lifted_points.ply
    summary.json
```

`summary.json` should record:

- sample id, visit id, desc id, and task description
- model backends and checkpoints
- selected frames and crops
- Molmo prompts, raw output paths, parsed points, and approval decisions
- SAM candidate metadata and approval decisions
- lift stats and approval decisions
- multi-view expansion decisions
- fusion inputs and output paths
- final status and failure type when unsuccessful

If a case has multiple Molmo points or multiple SAM candidates, the artifact
must display all of them together in a single image for agent inspection.

## Final Answer Contract

The agent final answer should be compact JSON:

```json
{
  "status": "success",
  "sample_id": "421254::<description_id>",
  "mask_artifact_path": "/abs/path/summary.json",
  "mask_npz_path": "/abs/path/fused/mask_data.npz",
  "mask_ply_path": "/abs/path/fused/lifted_points.ply",
  "selected_frame_ids": ["000050"],
  "accepted_fragment_ids": ["000050_mask_00"],
  "confidence": "medium",
  "uncertainties": ["single-view mask may miss hidden geometry"]
}
```

For failures, the answer should include a standardized failure type and the
latest artifact path so the run remains inspectable.

## Error Handling And Retry Policy

Standard failure types:

- `no_relevant_frame`
- `molmo_point_off_target`
- `molmo_point_ambiguous`
- `sam_empty`
- `sam_over_segmented`
- `sam_wrong_part`
- `lift_depth_missing`
- `lift_too_sparse`
- `multiview_fusion_failed`
- `artifact_invalid`

Standard run stop reasons separate from model-stage failure types:

- `single_view_complete`
- `multiview_not_needed`
- `max_retry_budget_reached_with_partial_mask`

Retry rules:

- Do not repeat the same expensive Molmo or SAM call with identical arguments
  unless the previous call failed at the process level.
- Prefer crop or prompt refinement before switching scenes or abandoning a
  case.
- Limit retries per stage through typed runner configuration.
- Record rejected points, masks, and lift fragments instead of overwriting them.
- Keep process failures distinct from model-quality failures.

## Evaluation

Evaluation should stay separate from generation.

Process evaluation:

- Did the agent view evidence before citing it?
- Did the agent approve Molmo points before SAM?
- Did the agent approve SAM candidates before lifting?
- Did the agent approve the first lift before multi-view expansion?
- How many retries happened at each stage?
- Which stage caused failure?

Mask evaluation:

- Load predicted point ids or nearest scene points from the final artifact.
- Load hidden GT annotation indices for the sample's annotation ids.
- Report IoU, precision, recall, F1, predicted point count, GT point count, and
  failure type.

The scorer should support single-fragment and fused multi-fragment artifacts.

## Testing Strategy

GPU-free tests:

- sample loading for `descriptions`, `motions`, and `annotations`
- task id construction
- hidden-GT separation from agent-visible sample context
- Molmo raw text parsing
- SAM candidate metadata parsing
- state transition validation
- artifact schema validation
- retry and failure-type handling
- scorer metric math

Optional integration tests:

- run wrappers against saved smoke artifacts
- run Molmo/SAM/lift on a manually selected small case in a worker tmux
  session
- compare single-view and multi-view artifacts on one or two descriptions

The repository quality gate from `AGENTS.md` applies to any Python code changes:

```bash
ruff check src/
black src/
mypy src/
PYTHONPATH=src pytest src/keyframe/tests -q
```

If heavy vision dependencies or GPUs are required for optional tests, the run
record must say so explicitly.

## Implementation Standards

All production code must comply with
`docs/python_code_agent_quality_guide.md`.

Design requirements:

- Use `dataclass(frozen=True)` or Pydantic v2 models for samples, tool inputs,
  tool outputs, artifact metadata, state transitions, and scoring records.
- Do not use `Any` or untyped dictionaries except at deserialization
  boundaries, and convert to typed models immediately.
- Validate every CLI JSON payload before running a tool.
- Keep CLI dispatch thin; business logic belongs in typed modules.
- Lazy-import heavy optional dependencies inside Molmo, SAM, image, and point
  cloud functions.
- Inject model paths and backend names through typed configuration.
- Do not scatter `os.environ` reads through business logic.
- Do not log secrets, tokens, private keys, or full private data.
- Do not silently fall back from a requested best model to a baseline model.
  Backend fallback must be explicit in configuration or returned as a failure.

## Milestones

P0: loaders and schemas.

- Add typed sample loader and task models.
- Add artifact and state-machine models.
- Add tests for sample loading and hidden-GT separation.

P1: evidence tools.

- Add SceneFunc3D `scene_summary`, `keyframe_selector`, `view_frame`,
  `view_crop`, `view_bev`, and `frame_objects`.
- Reuse OpenEQA/NR3D patterns while keeping the SceneFunc3D entrypoint
  separate.

P2: Molmo, SAM, and lift tools.

- Refactor benchmark smoke logic into typed production modules.
- Add point, mask, lift, inspection, and artifact-writing tools.
- Preserve raw outputs and all candidates.

P3: agent runner and approval gates.

- Add SceneFunc3D playbook.
- Enforce state transitions in code.
- Run a small single-view smoke over prepared scenes.

P4: multi-view expansion and fusion.

- Use first accepted 3D seed to suggest additional frames.
- Require per-view approval gates.
- Fuse accepted fragments into a final mask artifact.

P5: scoring and benchmark record.

- Score predictions against hidden annotation indices.
- Write durable benchmark records under `docs/benchmark/scenefunc_molmo_sam3d/`
  for meaningful runs.

## Open Risks

- Molmo may point to the wrong part when the task needs functional reasoning
  rather than object naming.
- SAM may return multiple plausible masks, including large surface masks that
  score well but are wrong for small handles, knobs, or switches.
- Single-view depth lifting can miss hidden geometry or fail on missing depth.
- Multi-view fusion can add noise if the seed projection selects visually
  similar but wrong parts.
- Best-model backend availability may differ between macOS and Linux workers.
  The implementation should detect and report unavailable configured backends
  rather than silently changing models.
