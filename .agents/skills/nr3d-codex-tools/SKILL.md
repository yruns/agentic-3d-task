---
name: nr3d-codex-tools
description: Select exactly one object proposal for a single NR3D referring expression from a prepared 3D scene. Use when Codex is running an NR3D tool-enabled turn with a query, scene_dir, attached BEV image, proposal catalog, and access to codex_agent.nr3d.tools plus view_image. Requires tools_enabled=True.
---

# NR3D Tool Grounding

Solve one NR3D visual-grounding sample. The prompt gives the query, `scene_dir`,
an attached top-down BEV image, and the proposal catalog. Pick exactly one
`proposal_id`, or `-1` only when the described target is genuinely absent (in
NR3D benchmark samples the target is virtually always present in the pool —
never output `-1` merely because an offline VLM `note` / `enriched` label was
polluted by a foreground occluder).

Read this skill once, then work only from the task prompt, proposal catalog, tool
outputs, and viewed images. Do not re-open this skill. Do not read, cat, sed,
head, grep, rg, or open `AGENTS.md`, `README`, docs, source files, or any other
project file. Do not list or search the repository.

## Final Answer

Return exactly one JSON object, with no prose or markdown:

```json
{
  "proposal_id": 12,
  "confidence": 0.8,
  "summary": "short evidence summary",
  "uncertainties": [],
  "cited_frame_indices": [52]
}
```

- Use an integer id from the proposal pool, or `-1` if absent.
- Include every required key: `proposal_id`, `confidence`, `summary`,
  `uncertainties`, `cited_frame_indices`.
- Put only frame ids you actually viewed in `cited_frame_indices`.
- Never use benchmark ground-truth fields; none are provided.

## Tools

Run tools in the shell:

```bash
python -m codex_agent.nr3d.tools <tool> --scene-dir <scene_dir> --args '<json>'
```

Text tools print one JSON object. `mark_frame_with_bbox` and `view_bev` also
write PNG(s) and print `image_path` (plus `crops_sheet_path` / `contact_sheet`);
open that path with `view_image` before using it as evidence. A frame you have
not viewed is not visual evidence.

Use only these NR3D tools plus `view_image`:

- `inspect_proposal`: full detail for one id, including color, description,
  location, nearby objects, 3D bbox, and visible frames. Args:
  `{"proposal_id": 15}`.
- `list_scene_proposals`: filter the pool. Args: `{"category": "chair"}`,
  `{"region_bev": [xmin, ymin, xmax, ymax]}`, or `{"limit": 20}`.
- `select_by_proposal`: fetch up to 4 spatially-diverse, high-visibility frames
  for given ids, plus `per_proposal_best_frames` and `requested_left_to_right`.
  Set `require_all=true` for co-visible target+anchor frames (auto-falls back to
  maximal partial co-visibility when >2 ids cannot all fit in one view); set
  `require_all=false` for balanced per-id coverage. Args:
  `{"proposal_ids": [15, 23], "require_all": true, "k": 3}`.
- `mark_frame_with_bbox`: draw labeled boxes on one frame (`frame_id`) or up to
  4 frames (`frame_ids`, writing a combined `contact_sheet`), AND automatically
  write `crops_sheet_path` (high-res unoccluded raw RGB crops of each target).
  Then `view_image` the returned `image_path` / `crops_sheet_path`. Args:
  `{"frame_id": 212, "ids": [15, 23]}` or `{"frame_ids": [45, 63], "ids": [12, 13]}`.
- `list_frame_proposals`: proposals in one frame, ordered left-to-right by 2D
  position. Args: `{"frame_id": 212}`.
- `compare_proposals_spatial`: deterministic 3D + co-visible 2D + virtual-viewer
  ranking. Relations: `closest_to`, `near`, `next_to`, `farthest_from`, `above`,
  `below`, `left_of`, `right_of`, `in_front_of`, `behind`, `between`, `opposite`.
  Supports optional `anchor_id` (omit when comparing a group directly) and
  optional `observer_id` (when looking from a specific object). Returns both
  co-visible frame votes and 3D virtual-observer rankings
  (`facing_left_to_right_ids`, `facing_front_to_back_ids`). Args:
  `{"candidate_ids": [15, 16, 17], "anchor_id": 23, "relation": "closest_to"}`.
- `compare_candidates_to_anchors`: rank candidates against several anchors at
  once (including `relation="between"` for exact 3D segment distance between two
  anchors). Args:
  `{"candidate_ids": [15, 16], "anchor_ids": [23, 24], "relation": "between"}`.
- `view_bev`: render a top-down map with highlighted 3D oriented footprints and
  labels. Args: `{"highlight": [15, 23]}`.
- `keyframe_selector`: language to up to 3 first-person frames when the catalog
  shortlist is unclear. Args: `{"query": "the trash can by the tv", "k": 3}`.

## Workflow

1. Parse the query into target category, attributes, anchors, and relation.
2. Shortlist candidates from the catalog. Remember that 3D `category` and
   `size/center` are authoritative whereas offline `enriched/note` text can be
   polluted by foreground occluders.
3. Resolve spatial language with `compare_proposals_spatial` or
   `compare_candidates_to_anchors`, and fetch co-visible/balanced frames with
   `select_by_proposal`.
4. Confirm the winner with `mark_frame_with_bbox` + `view_image` (and inspect
   `crops_sheet_path` for fine appearance/color).
5. Emit the final JSON immediately.

## Stop Conditions

- Aim to decide within 6-10 tool calls.
- Never repeat a tool with identical arguments. Never re-view an image already
  seen. If a result is empty or errors, change approach instead of retrying.
- Once the top candidates are inspected and one co-visible frame or spatial
  comparison supports the choice, do not keep gathering evidence to chase
  certainty.
