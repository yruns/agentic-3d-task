---
name: nr3d-codex-tools
description: Select one object proposal for a single NR3D referring expression from a prepared 3D scene, using first-person frames, spatial ranking, and BEV CLI tools plus view_image.
---

# NR3D Codex SDK Visual Grounding (tool-using)

You select exactly one object proposal for one NR3D referring expression over a
prepared 3D scene. The prompt gives you the query, an attached top-down BEV
image, and the full proposal catalog (ids, categories, centers, notes). You also
have a set of CLI tools that fetch real first-person evidence so you can verify
your choice instead of guessing from text alone.

## Output contract

- Choose exactly one `proposal_id` from the catalog. Use `-1` only if the target
  is genuinely absent.
- Your FINAL message must be a single JSON object matching the requested schema
  (`proposal_id`, `confidence`, `summary`, `uncertainties`, `cited_frame_indices`)
  — no prose, no tool commands.
- Put the frame ids you actually looked at in `cited_frame_indices`.
- Never use benchmark ground-truth fields; none are provided.

## How to run a tool

Run in the shell (the scene pack dir is given in the prompt as `scene_dir`):

```
python -m codex_agent.nr3d.tools <tool> --scene-dir <scene_dir> --args '<json>'
```

- Text tools print one JSON object on stdout.
- `mark_frame_with_bbox` and `view_bev` also write a PNG and print its
  `image_path`. **You must open that path with the `view_image` tool before you
  trust what the frame shows.** A frame you have not viewed is not evidence.
- Recoverable mistakes come back as `{"error": "..."}` — read it and adjust
  (e.g. use a valid id, switch `require_all` off, pick a frame that exists).

## Tool catalog

Narrow the candidates (text):
- `inspect_proposal` — full detail for one id (color, description, nearby
  objects, every frame it appears in). `{"proposal_id": 15}`
- `list_scene_proposals` — filter the pool. `{"category": "chair"}` or
  `{"region_bev": [xmin, ymin, xmax, ymax]}` or `{"limit": 20}`

Fetch first-person frames (image):
- `keyframe_selector` — language → up to 3 frames for a free-form description
  when the catalog shortlist is unclear. `{"query": "the trash can by the tv", "k": 3}`
- `select_by_proposal` — frames that show given ids; set `require_all` to get a
  frame where target AND anchor are co-visible.
  `{"proposal_ids": [15, 23], "require_all": true, "k": 3}`

Read a frame (image + text):
- `mark_frame_with_bbox` — draw labeled boxes for ids/categories on a frame,
  then `view_image` the returned `image_path`.
  `{"frame_id": 212, "ids": [15, 23]}`
- `list_frame_proposals` — which proposals are in a frame, ordered left→right by
  2D position. `{"frame_id": 212}`

Decide spatial relations (text):
- `compare_proposals_spatial` — rank candidates against one anchor.
  `{"candidate_ids": [15, 16, 17], "anchor_id": 23, "relation": "closest_to"}`.
  Relations: `closest_to`, `near`, `next_to`, `farthest_from`, `above`, `below`,
  `left_of`, `right_of`. Metric relations use 3D centers; `left_of`/`right_of`
  use votes over co-visible frames (`shared_frame_counts`,
  `supporting_frame_counts`).
- `compare_candidates_to_anchors` — rank candidates against several plausible
  anchors at once and check agreement (`globally_consistent_top1`,
  `anchor_disagreement`). `{"candidate_ids": [15,16], "anchor_ids": [23,24], "relation": "near"}`

Global alignment (image):
- `view_bev` — top-down map; highlight ids/categories to tie first-person frames
  to scene layout. `{"highlight": [15, 23]}`

## Recommended loop

1. Parse the query: target category, anchor(s), relation, attributes.
2. Shortlist candidates from the catalog (and `list_scene_proposals` /
   `inspect_proposal` for color/description).
3. If one candidate is obvious from appearance, jump to step 5 to confirm.
4. Resolve spatial language:
   - metric (`near`/`closest`/`above`/`below`/`between`) →
     `compare_proposals_spatial` (or `compare_candidates_to_anchors` when the
     anchor itself is ambiguous);
   - view-dependent (`left`/`right`/`in front of`/`behind`) → fetch a
     co-visible frame with `select_by_proposal(require_all=true)`, then
     `mark_frame_with_bbox` + `view_image`, and read left↔right from the pixels
     (the marked layout printout and `compare_*` votes back this up).
5. Confirm the winner: `mark_frame_with_bbox` the top candidate (with its
   anchor) on a frame that shows it, `view_image` it, and check color / shape /
   relation actually match the query.
6. Emit the final JSON. Lower `confidence` and list rivals in `uncertainties`
   when more than one candidate survives.

## Guidance

- Prefer the catalog + co-visible frames. `keyframe_selector` is a fallback for
  hard, anchor-free phrasings; if it errors, switch to `select_by_proposal`.
- Always `view_image` a marked frame before claiming what it shows. Do not
  invent colors, counts, or left/right from coordinates alone.

## Tool budget — be decisive

- Aim to decide within about 6–10 tool calls; simple unique targets need only
  one confirming frame.
- Never repeat a tool with the same arguments, and do not re-view an image you
  have already seen. If a result is empty or errors, change approach instead of
  retrying identically.
- Once you have (a) inspected the top candidates and (b) one co-visible frame or
  one spatial comparison, commit to the best-supported candidate and output the
  final JSON. Do not keep gathering evidence to chase certainty on an already
  well-supported answer.
- If two candidates remain genuinely tied, pick the better-supported one,
  lower `confidence`, and record the rival in `uncertainties` — then stop.
