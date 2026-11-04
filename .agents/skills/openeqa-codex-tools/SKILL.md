---
name: openeqa-codex-tools
description: Answer one OpenEQA open-ended question about a prepared 3D indoor scene by combining the initial top-down BEV map and 3D grid/object overview with on-demand first-person frames and high-resolution crops via CLI tools.
---

# OpenEQA Codex SDK Question Answering (Map-First + Evidence-Seeking Tools)

You answer exactly one OpenEQA question about a prepared ScanNet clip. The prompt
provides the question, its category, the scene id, the **Scene Grid Layout**, a
compact **3D Object Catalog preview** (with each object's `(row, col)` grid cell
and `best_frames`), and attaches the top-down **Schematic BEV map** when
ConceptGraph assets are available — **but NO first-person RGB frames are
attached**. You must fetch and view first-person visual evidence yourself using
the CLI tools below.

## Output contract

- Answer concisely and factually as a short phrase, grounded ONLY in first-person
  frames and crops you have actually fetched and viewed with `view_image`.
- If the answer is not fully determinable, give your single best guess; do not
  refuse and do not answer with a question.
- Your FINAL message must be a single JSON object matching the requested schema
  (`answer`, `supporting_claims`, `confidence`) — no prose, no tool commands.
- Never use benchmark ground-truth fields; none are provided.

## How to run a tool

Run in the shell (the clip directory is given in the prompt as `scene_dir`):

```bash
python -m codex_agent.openeqa.tools <tool> --scene-dir <scene_dir> --args '<json>'
```

- `list_objects` prints one JSON object on stdout.
- `view_frame`, `keyframe_selector`, `view_crop`, and `view_bev` write an image
  and print its `image_path` (`keyframe_selector` and multi-frame `view_frame`
  also print a combined `contact_sheet` plus per-frame `visible_objects` with
  normalized 2D boxes `bbox_norm`). **You must open the path with `view_image`
  before relying on what it shows.**
- Recoverable mistakes come back as `{"error": "..."}` — read the message and
  adjust your next call.

## Tool catalog

Fetch first-person frames & zoom in (image):
- `view_frame` — fetch raw frames by `frame_id` / `frame_ids`, by a trajectory
  neighborhood (`around_frame_id` + `step`), or by BEV `grid_cells` `[[r, c]]`.
  Use this immediately when the printed catalog already lists `best_frames` for
  your target or anchor object!
  `{"frame_ids": [34, 60, 120]}` or `{"around_frame_id": 120, "step": 10}` or
  `{"grid_cells": [[2, 3]]}`
- `keyframe_selector` — language + spatial/grid prior → up to 4 spatially-DIVERSE
  frames + `contact_sheet` + `grounded_objects` + per-frame `visible_objects`
  (`bbox_norm`). Pass a **complete relational phrase** (target + anchor +
  relation) and optional `target_grids`. Successive calls return fresh viewpoints.
  `{"query": "the fire extinguisher under the window", "k": 3, "target_grids": [[1, 2]]}`
- `view_crop` — HIGH-RESOLUTION zoom from the full-res raw RGB. Mandatory for
  small objects, colors, brands, labels, and fine states. Aim by `object_id`
  (optionally with `frame_id`) or by `frame_id` + `bbox` (`[0, 1]` normalized or
  raw pixels).
  `{"object_id": 12}` or `{"object_id": 12, "frame_id": 420}` or
  `{"frame_id": 420, "bbox": [0.35, 0.25, 0.65, 0.70]}`

Search object details & highlight map (text / image):
- `list_objects` — search or inspect the 3D object catalog (`obj_id`, `category`,
  `grid_cell`, `best_frame_ids`, `color`, `note`, `location`, `nearby_objects`,
  `usability`). Filter by `query`, `category`, `grid_cells`, or `object_ids`.
  `{"query": "microwave counter"}` or `{"object_ids": [0, 5]}`
- `view_bev` — re-render the schematic BEV highlighting specific object ids,
  categories, or grid cells.
  `{"highlight": [4, 11]}` or `{"categories": ["table"]}` or `{"highlight_grids": [[1, 2]]}`

## Critical rules by question category

1. **Never treat the 3D catalog as complete or infallible**: Small tabletop items
   (erasers, pens, tissues, trophies, toiletries) and wall fixtures (phones,
   thermostats, paintings, hooks) are often omitted or mislabeled in 3D
   detection. Never claim an item is absent without inspecting first-person
   frames of the relevant surfaces/walls.
2. **Functional reasoning & World knowledge**: Look for the **most specific,
   dedicated item(s)** in the room (e.g., an *hourglass* rather than a phone;
   *furry earmuffs* or a *pink jacket* rather than bed covers; *green boxes of
   paper* on the filing cabinet), not a generic room default. In your answer
   phrase, name **both the specific item (plus any paired accessory like a
   remote) and its location/surface** (e.g., *"use the paper cutter on the black
   counter"*).
3. **Object localization & Spatial understanding**: For *"Where is X?"*, include
   **both** the immediate supporting surface/container AND the nearest salient
   landmark (e.g., *"on the work table to the right of the green garbage can"*).
   Always judge left/right from the first-person camera view.
4. **Object & Attribute recognition**: Always call `view_crop` and `view_image`
   to verify small objects, colors, text, or fine distinctions before answering.
