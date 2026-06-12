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
- `view_frame`, `keyframe_selector`, and `view_bev` also write an image and print
  its `image_path`. **You must open that path with the `view_image` tool before
  you trust what it shows.** An image you have not viewed is not evidence.
- Recoverable mistakes come back as `{"error": "..."}` — read it and adjust
  (e.g. pick a frame id in range, switch `keyframe_selector` → `view_frame`).

## Tool catalog

Understand the scene (text):
- `list_objects` — the detected objects (ids, category, 3D center/size,
  description). Use it to learn what is in the room and to get object ids for
  `view_bev`. `{}` or `{"category": "chair"}` or `{"limit": 60}`

Fetch first-person frames (image):
- `keyframe_selector` — language → up to 4 frames most likely to show what you
  describe; best when the answer is about a nameable object.
  `{"query": "the fire extinguisher under the window", "k": 3}`
- `view_frame` — fetch any raw frame id (or a few) when you need a specific or
  closer viewpoint. The result reports `total_frames` and the valid
  `frame_id_range`. `{"frame_id": 420}` or `{"frame_ids": [100, 300, 540]}`

Read scene layout (image):
- `view_bev` — top-down schematic map (object footprints + camera path) to read
  layout, counts, and spatial relations; highlight objects by id/category.
  `{}` or `{"highlight": [4, 11]}` or `{"categories": ["table", "chair"]}`

## Recommended loop

1. Decide what evidence the question needs (you have none yet), then fetch it.
2. For "what / where / how many / layout" questions, call `view_bev` and/or
   `list_objects` for scene context.
3. If the answer is about a specific object, call `keyframe_selector` with a
   short description, then `view_image` the returned frame(s).
4. If you just need another viewpoint, call `view_frame` for a chosen frame id
   (use the BEV / `list_objects` to pick one), then `view_image` it.
5. Answer concisely, grounded only in frames/BEV you have viewed.

## Tool budget — be decisive

- Aim to decide within about 4–8 tool calls; most questions need only one or two
  targeted images. View at most a few images, then commit.
- Never repeat a tool with identical arguments and never re-view an image you
  have already seen. If a result is empty or errors, change approach (e.g.
  `keyframe_selector` → `view_frame`) instead of retrying identically.
- Once you have enough evidence, emit the final JSON immediately. Do not keep
  gathering evidence to chase certainty on an already well-supported answer.
