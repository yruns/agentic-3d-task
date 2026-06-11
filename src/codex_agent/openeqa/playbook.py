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
manual/legacy ``--skill-path`` use); the inline constant is the source of truth
for the default tool-mode run.
"""

from __future__ import annotations

#: Tool names the inline playbook must document (a test keeps them in sync with
#: the CLI dispatcher and catches a tool added without guidance).
OPENEQA_TOOL_NAMES: tuple[str, ...] = (
    "list_objects",
    "keyframe_selector",
    "view_frame",
    "view_bev",
)

OPENEQA_TOOLS_PLAYBOOK = """\
Playbook (everything you need is here — do NOT look for a skill or any other \
file):

You already have a uniform sample of first-person frames attached. The tools \
below let you fetch MORE targeted evidence when the attached frames do not \
settle the answer.

Tool catalog
- list_objects — the scene's detected objects (ids, category, 3D center/size, \
description). Use it to learn what is in the room and to get object ids for \
view_bev. args: {} or {"category": "chair"} or {"limit": 60}
- keyframe_selector — language -> up to 4 first-person frames most likely to \
show what you describe; best when the answer is about a specific object you can \
name. args: {"query": "the fire extinguisher under the window", "k": 3}
- view_frame — fetch any raw frame id (or a few) when you need a different or \
closer view than the attached frames. Frame ids run across the whole clip; the \
result reports total_frames and the valid range. args: {"frame_id": 420} or \
{"frame_ids": [100, 300, 540]}
- view_bev — top-down schematic map (object footprints + camera path) to read \
scene layout, counts, and spatial relations; highlight objects by id/category. \
args: {} or {"highlight": [4, 11]} or {"categories": ["table", "chair"]}

Every image tool writes a file and prints its image_path. You MUST open that \
path with the view_image tool before you trust what it shows — a frame or BEV \
you have not viewed is not evidence.

Recommended loop
1. Read the question and the attached frames. If they already answer it, skip to \
the final JSON.
2. If you need scene context (layout, counts, "what/where" questions), call \
view_bev and/or list_objects.
3. If the answer is about a specific object, call keyframe_selector with a short \
description, then view_image the returned frame(s).
4. If you just need another viewpoint, call view_frame for a specific frame id \
(use list_objects / the BEV / the attached frames to pick one), then view_image \
it.
5. Answer concisely and factually, grounded only in frames/BEV you have viewed.

Tool budget — be decisive
- Aim to decide within about 4-8 tool calls; many questions need zero or one \
extra image beyond what is attached. View at most a few images, then commit.
- Never repeat a tool with identical arguments and never re-view an image you \
have already seen. If a result is empty or errors, change approach (e.g. switch \
keyframe_selector -> view_frame) instead of retrying identically.
- Once you have enough evidence, emit the final JSON immediately. Do not keep \
gathering evidence to chase certainty on an already well-supported answer."""


__all__ = ["OPENEQA_TOOLS_PLAYBOOK", "OPENEQA_TOOL_NAMES"]
