---
name: nr3d-codex-tools
description: Select exactly one object proposal for a single NR3D referring expression from a prepared 3D scene. Use when Codex is running an NR3D tool-enabled turn with a query, scene_dir, attached BEV image, proposal catalog, and access to codex_agent.nr3d.tools plus view_image. Requires tools_enabled=True.
---

# NR3D Tool Grounding

Solve one NR3D visual-grounding sample. The prompt gives the query, `scene_dir`,
an attached top-down BEV image, and the proposal catalog. Pick exactly one
`proposal_id`, or `-1` only when the described target is genuinely absent.

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
write a PNG and print `image_path`; open that path with `view_image` before
using it as evidence. A frame you have not viewed is not visual evidence.

Use only these NR3D tools plus `view_image`:

- `inspect_proposal`: full detail for one id, including color, description,
  nearby objects, and visible frames. Args: `{"proposal_id": 15}`.
- `list_scene_proposals`: filter the pool. Args: `{"category": "chair"}`,
  `{"region_bev": [xmin, ymin, xmax, ymax]}`, or `{"limit": 20}`.
- `keyframe_selector`: language to up to 3 first-person frames when the catalog
  shortlist is unclear. Args: `{"query": "the trash can by the tv", "k": 3}`.
- `select_by_proposal`: frames that show given ids; set `require_all` for a
  target and anchor co-visible frame. Args:
  `{"proposal_ids": [15, 23], "require_all": true, "k": 3}`.
- `mark_frame_with_bbox`: draw labeled boxes on a frame, then `view_image` the
  returned path. Args: `{"frame_id": 212, "ids": [15, 23]}`.
- `list_frame_proposals`: proposals in one frame, ordered left-to-right by 2D
  position. Args: `{"frame_id": 212}`.
- `compare_proposals_spatial`: rank candidates against one anchor. Args:
  `{"candidate_ids": [15, 16, 17], "anchor_id": 23, "relation": "closest_to"}`.
  Relations: `closest_to`, `near`, `next_to`, `farthest_from`, `above`, `below`,
  `left_of`, `right_of`. Metric relations use 3D centers; `left_of` and
  `right_of` use co-visible-frame votes.
- `compare_candidates_to_anchors`: rank candidates against several plausible
  anchors and check agreement. Args:
  `{"candidate_ids": [15, 16], "anchor_ids": [23, 24], "relation": "near"}`.
- `view_bev`: render a top-down map with highlighted ids/categories. Args:
  `{"highlight": [15, 23]}`.

## Workflow

1. Parse the query into target category, attributes, anchors, and relation.
2. Shortlist candidates from the catalog using `list_scene_proposals` and
   `inspect_proposal` for category, color, shape, material, and role.
3. If one candidate is obvious from the catalog, still confirm it once with a
   frame or spatial comparison before answering.
4. Resolve spatial language:
   - Metric relations such as near, closest, above, below, and between: use
     `compare_proposals_spatial`, or `compare_candidates_to_anchors` when the
     anchor is ambiguous.
   - View-dependent relations such as left, right, in front of, and behind: get
     a co-visible frame with `select_by_proposal(require_all=true)`, then run
     `mark_frame_with_bbox`, open the returned image with `view_image`, and read
     left/right from the pixels.
5. Confirm the winner with one viewed marked frame or one decisive spatial
   comparison. Check that color, shape, category, and relation match the query.
6. Emit the final JSON immediately.

## Stop Conditions

- Aim to decide within 6-10 tool calls. Unique targets often need only one
  confirming frame.
- View at most one or two images unless the first evidence is genuinely
  ambiguous.
- Never repeat a tool with identical arguments. Never re-view an image already
  seen. If a result is empty or errors, change approach instead of retrying.
- If two candidates remain tied, pick the better-supported one, lower
  `confidence`, record the rival in `uncertainties`, and stop.
- Once the top candidates are inspected and one co-visible frame or spatial
  comparison supports the choice, do not keep gathering evidence to chase
  certainty.
