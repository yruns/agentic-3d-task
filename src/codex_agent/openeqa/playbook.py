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

NO first-person RGB frames are attached to this message (only the top-down \
schematic BEV map and 3D catalog preview when available). A 3D catalog entry or \
BEV label is a routing hint, NOT visual proof — small desktop/countertop items, \
wall-mounted fixtures, and fine attributes are frequently missing or mislabeled \
in 3D detection. You MUST fetch and view first-person frames/crops with \
view_image before answering.

Tool catalog
- view_frame — fetch raw first-person frames by id, by trajectory neighborhood \
(around_frame_id + step), or by BEV grid_cells. Returns frame images, a combined \
contact_sheet (when ≥2 frames), and per-frame visible_objects with normalized \
2D boxes (bbox_norm) ready for view_crop. Use this immediately when the printed \
catalog already gives you best_frames for a relevant object or grid cell! \
args: {"frame_ids": [34, 60, 120]} or {"around_frame_id": 120, "step": 10} or \
{"grid_cells": [[2, 3]]}
- keyframe_selector — language + spatial/grid prior -> up to 4 spatially-DIVERSE \
first-person frames + one combined contact_sheet + grounded_objects + per-frame \
visible_objects (with bbox_norm). Pass a COMPLETE relational phrase (target + \
anchor + spatial relation, not a bare noun) and optional target_grids [[r, c]]. \
Successive calls automatically return NEW viewpoints. \
args: {"query": "the framed photo next to the wooden double door", "k": 3, \
"target_grids": [[1, 2]]}
- view_crop — HIGH-RESOLUTION zoom cut from the full-res raw RGB. Mandatory \
whenever you need to read a small object's identity, color, brand, text, or \
state. Aim by object_id (optionally on a specific frame_id) or by frame_id + \
bbox (using bbox_norm from view_frame/keyframe_selector or normalized [0,1] \
coords). \
args: {"object_id": 12} or {"object_id": 12, "frame_id": 420} or \
{"frame_id": 420, "bbox": [0.35, 0.25, 0.65, 0.70]}
- list_objects — search or inspect the 3D object catalog (returns obj_id, \
category, grid_cell, best_frame_ids, color, note, location, nearby_objects, \
usability). Filter by query keywords, category, grid_cells, or object_ids (full \
untruncated text). \
args: {"query": "microwave counter"} or {"object_ids": [0, 5]} or \
{"grid_cells": [[2, 3]]}
- view_bev — re-render the top-down schematic map with highlighted object ids, \
categories, or grid cells. \
args: {"highlight": [4, 11]} or {"categories": ["table", "chair"]} or \
{"highlight_grids": [[1, 2]]}

Every image tool writes a file and prints its image_path (keyframe_selector and \
multi-frame view_frame also print a contact_sheet). You MUST open the path with \
view_image before trusting what it shows. Prefer view_image on the contact_sheet \
to inspect a batch in ONE action, then view_crop + view_image on the best frame \
to read fine details.

Recommended loop
1. Read the inlined Scene Grid Layout, Object Catalog preview, and attached BEV \
map. Identify candidate object ids, their best_frames, and target grid cells \
(row, col).
2. Fetch first-person visual evidence:
   - If candidate object(s) and their best_frames are already listed in the \
catalog preview, jump straight to view_frame({"frame_ids": [...]}) or \
view_crop({"object_id": ...})!
   - If you need to locate an uncataloged object or resolve a spatial relation, \
call keyframe_selector with the full relational description and target_grids, \
then view_image the contact_sheet.
   - If a view partially clips the area of interest, pan along the trajectory \
with view_frame({"around_frame_id": <id>, "step": 10}).
3. Zoom in before committing: for any small object, color, label, state, or \
ambiguous identity, call view_crop on the target region and view_image the crop.
4. Emit the final JSON immediately once visual evidence confirms the answer.

Critical rules by question category
- Never claim an object is absent from the catalog alone: 3D detection often \
misses wall-mounted items (phones, thermostats, paintings, hooks) and small \
tabletop items (erasers, pens, tissues, trophies, toiletries). Always inspect \
first-person frames of the relevant walls, desks, counters, or shelves before \
concluding something is present or absent.
- Functional reasoning & World knowledge — seek the SPECIFIC dedicated item and \
include its location: Do NOT settle for a generic room default (e.g. do not \
answer "use a smartphone" when there is an hourglass on the desk; do not answer \
"get under bed covers" or "hoodie" when there is a jacket or earmuffs; do not \
answer "grab bar" when there is a shower curtain or towel rack). In your answer \
phrase, name BOTH the specific dedicated item (and any paired tool like a \
remote) AND where it is located (e.g. "use the paper cutter on the black \
counter", "open the green boxes of paper on the filing cabinet").
- Object localization & Spatial understanding — dual-reference phrasing: \
When asked "Where is X?", name BOTH the immediate support surface/container AND \
the nearest salient landmark with its spatial relation (e.g. "on the work table \
to the right of the green garbage can", "on the lower closet shelf to the \
right"). Always judge left/right/front/behind from the first-person camera view, \
never from world coordinates.
- Object recognition & Attribute recognition — mandatory high-res crop: \
Do NOT guess an object's identity (e.g. eraser vs. sponge, vacuum vs. fire \
extinguisher, water carbonator vs. coffee maker) or color/brand from a distant \
768px frame. Always view_crop the target and verify from a close-up view.

Tool budget — be decisive and avoid redundant calls
- Never repeat a tool with identical arguments and never re-view an image you \
have already seen.
- Once visual evidence from first-person frames/crops clearly supports the \
answer, output the final JSON immediately."""


__all__ = ["OPENEQA_TOOLS_PLAYBOOK", "OPENEQA_TOOL_NAMES"]
