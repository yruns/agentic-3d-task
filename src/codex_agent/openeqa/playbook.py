"""The OpenEQA tool-using playbook, inlined into the QA turn prompt.

Why this lives in the prompt instead of a Codex *skill* file:

The Codex skill mechanism advertises a skill as ``<skill><path>…</path></skill>``
and expects the model to *read that file when needed*. Once Codex's app-server
windows the context (it drops images and older tool outputs after big images
enter the turn), a "lost" model re-runs its default orientation action — reading
the advertised ``SKILL.md`` path — over and over, an unbounded loop (see
``docs/codex_agent/skill_loop_and_reasoning_dropped_20260609.md``). Inlining the
playbook keeps the guidance inside the always-present, cached prompt prefix and
removes the re-readable on-disk bait entirely.

Keep this in sync with ``.agents/skills/openeqa-codex-tools/SKILL.md`` (kept for
documentation / manual use); the inline constant is the source of truth for the
OpenEQA QA run.
"""

from __future__ import annotations

#: Tool names the inline playbook must document (a test keeps them in sync with
#: the CLI dispatcher and catches a tool added without guidance).
OPENEQA_TOOL_NAMES: tuple[str, ...] = (
    "list_objects",
    "keyframe_selector",
    "view_frame",
    "view_crop",
    "view_bev",
)

OPENEQA_TOOLS_PLAYBOOK = """\
Playbook (everything you need is here — do NOT look for a skill or any other \
file):

NO images are attached to this question — you have zero visual evidence until \
you fetch it. Use the tools below to gather ALL the visual evidence you need: \
name an object to keyframe_selector, read the layout with view_bev / \
list_objects, or grab a specific first-person frame with view_frame.

Tool catalog
- list_objects — the scene's detected objects (ids, category, 3D center/size, \
description). Use it to learn what is in the room and to get object ids for \
view_bev / view_crop. args: {} or {"category": "chair"} or {"limit": 60}
- keyframe_selector — language -> up to 4 first-person frames most likely to \
show what you describe; best when the answer is about a specific object you can \
name. It returns spatially-DIVERSE frames and one combined contact_sheet image, \
and successive calls in this turn return NEW viewpoints (it remembers what it \
already gave you), so re-asking explores rather than repeats. \
args: {"query": "the fire extinguisher under the window", "k": 3}
- view_frame — fetch any raw frame id (or a few) when you need a specific view. \
Frame ids run across the whole clip; the result reports total_frames and the \
valid range, and a contact_sheet when you ask for several. \
args: {"frame_id": 420} or {"frame_ids": [100, 300, 540]}
- view_crop — HIGH-RESOLUTION zoom. When a detail is too small/blurry to read \
in a frame (a label, a color, which object it is), crop in instead of guessing. \
Aim it by region after seeing a frame, or by object id. \
args: {"frame_id": 420, "bbox": [0.4, 0.3, 0.7, 0.8]} (bbox normalized 0-1 or \
raw pixels) or {"object_id": 12}
- view_bev — top-down schematic map (object footprints + camera path) to read \
scene layout, counts, and spatial relations; highlight objects by id/category. \
args: {} or {"highlight": [4, 11]} or {"categories": ["table", "chair"]}

Every image tool writes a file and prints its image_path (keyframe_selector / \
view_frame also print a contact_sheet). You MUST open the path with the \
view_image tool before you trust what it shows — a frame, crop, or BEV you have \
not viewed is not evidence. Prefer view_image on the contact_sheet to inspect a \
whole batch with ONE action instead of opening each frame separately.

Recommended loop
1. Decide what evidence the question needs (you have none yet), then fetch it.
2. If you need scene context (layout, counts, "what/where" questions), call \
view_bev and/or list_objects.
3. If the answer is about a specific object, call keyframe_selector with a short \
description, then view_image the returned contact_sheet.
4. If the detail you need is too small or blurry to read, use view_crop to zoom \
in before answering — do NOT guess a label/color/identity from a blurry frame.
5. If you need another viewpoint, call view_frame for a specific frame id, or \
call keyframe_selector again (it returns fresh viewpoints), then view it.
6. Answer concisely and factually, grounded only in frames/crops/BEV you viewed.

Confirm before you commit (fine-grained questions)
- For "what is this object" / color / brand / small-detail questions, do NOT \
commit from a single distant frame. Either view_crop to read the detail, or \
confirm the object in a SECOND independent frame (a different viewpoint) before \
answering. If you still cannot resolve it, give your best answer with a LOWER \
confidence rather than asserting a guess at high confidence.

Tool budget — be decisive but don't under-look
- Most questions resolve in a handful of targeted images; use the contact_sheet \
and view_crop to get more evidence per action instead of opening many frames.
- Never repeat a tool with identical arguments and never re-view an image you \
have already seen. If a result is empty or errors, change approach (rephrase, \
or switch keyframe_selector <-> view_frame <-> view_crop) instead of retrying \
identically.
- Once the evidence actually supports an answer, emit the final JSON \
immediately; do not keep gathering to chase certainty you already have."""


__all__ = ["OPENEQA_TOOLS_PLAYBOOK", "OPENEQA_TOOL_NAMES"]
