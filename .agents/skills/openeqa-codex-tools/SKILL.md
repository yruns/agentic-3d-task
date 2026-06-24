---
name: openeqa-codex-tools
description: Answer one OpenEQA open-ended question about a prepared 3D indoor scene by fetching your own visual evidence with CLI tools for keyframe retrieval, frame switching, a top-down BEV, and the object list (no frames are attached).
---

# OpenEQA Codex SDK Question Answering (tool-using)

You answer exactly one OpenEQA question about a prepared ScanNet clip. The prompt
gives you the question, its category, and the scene id — but **no images are
attached**. You start with zero visual information and must fetch every piece of
visual evidence yourself with the CLI tools below.

## Output contract

- Answer concisely and factually as a short phrase, grounded ONLY in visual
  evidence you have actually fetched and viewed.
- If the answer is not fully determinable, give your single best guess; do not
  refuse and do not answer with a question.
- Your FINAL message must be a single JSON object matching the requested schema
  (`answer`, `supporting_claims`, `confidence`) — no prose, no tool commands.
- Never use benchmark ground-truth fields; none are provided.

## How to run a tool

Run in the shell (the clip directory is given in the prompt as `scene_dir`):

```
python -m codex_agent.openeqa.tools <tool> --scene-dir <scene_dir> --args '<json>'
```

- `list_objects` prints one JSON object on stdout.
- `view_frame`, `keyframe_selector`, `view_crop`, and `view_bev` also write an
  image and print its `image_path` (`keyframe_selector` / `view_frame` also print
  a `contact_sheet`). **You must open that path with the `view_image` tool before
  you trust what it shows.** An image you have not viewed is not evidence.
- Recoverable mistakes come back as `{"error": "..."}` — read it and adjust
  (e.g. pick a frame id in range, switch `keyframe_selector` → `view_frame`).

## Tool catalog

Understand the scene (text):
- `list_objects` — the detected objects (ids, category, 3D center/size,
  description). Use it to learn what is in the room and to get object ids for
  `view_bev` / `view_crop`. `{}` or `{"category": "chair"}` or `{"limit": 60}`

Fetch first-person frames (image):
- `keyframe_selector` — language → up to 4 frames most likely to show what you
  describe; best when the answer is about a nameable object. Returns
  spatially-DIVERSE frames plus one combined `contact_sheet`, and successive
  calls return NEW viewpoints (it remembers what it already returned), so
  re-asking explores instead of repeating.
  `{"query": "the fire extinguisher under the window", "k": 3}`
- `view_frame` — fetch any raw frame id (or a few) when you need a specific
  viewpoint. The result reports `total_frames`, the valid `frame_id_range`, and a
  `contact_sheet` when you ask for several.
  `{"frame_id": 420}` or `{"frame_ids": [100, 300, 540]}`
- `view_crop` — HIGH-RESOLUTION zoom for a detail too small/blurry to read in a
  frame (a label, a color, an object's identity). Aim it by region after seeing a
  frame, or by object id.
  `{"frame_id": 420, "bbox": [0.4, 0.3, 0.7, 0.8]}` (bbox normalized 0-1 or
  pixels) or `{"object_id": 12}`

Read scene layout (image):
- `view_bev` — top-down schematic map (object footprints + camera path) to read
  layout, counts, and spatial relations; highlight objects by id/category.
  `{}` or `{"highlight": [4, 11]}` or `{"categories": ["table", "chair"]}`

## Recommended loop

1. Decide what evidence the question needs (you have none yet), then fetch it.
2. For "what / where / how many / layout" questions, call `view_bev` and/or
   `list_objects` for scene context.
3. If the answer is about a specific object, call `keyframe_selector` with a
   short description, then `view_image` the returned `contact_sheet`.
4. If the detail you need is too small/blurry to read, use `view_crop` to zoom
   in before answering — do not guess a label/color/identity from a blurry frame.
5. If you need another viewpoint, call `view_frame` for a chosen frame id, or
   call `keyframe_selector` again (it returns fresh viewpoints), then view it.
6. Answer concisely, grounded only in frames/crops/BEV you have viewed.

## Confirm before you commit (fine-grained questions)

For "what is this object" / color / brand / small-detail questions, do not commit
from a single distant frame. Either `view_crop` to read the detail, or confirm
the object in a SECOND independent frame (a different viewpoint) before
answering. If you still cannot resolve it, give your best answer with a LOWER
confidence rather than asserting a guess at high confidence.

## Tool budget — be decisive but don't under-look

- Most questions resolve in a handful of targeted images; use the `contact_sheet`
  and `view_crop` to get more evidence per action instead of opening many frames.
- Never repeat a tool with identical arguments and never re-view an image you
  have already seen. If a result is empty or errors, change approach (rephrase,
  or switch `keyframe_selector` ↔ `view_frame` ↔ `view_crop`) instead of retrying
  identically.
- Once the evidence actually supports an answer, emit the final JSON
  immediately; do not keep gathering to chase certainty you already have.
