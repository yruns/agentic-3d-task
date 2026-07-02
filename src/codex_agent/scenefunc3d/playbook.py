"""Inline SceneFunc3D tool-using playbook."""

from __future__ import annotations

#: Tool names the inline playbook currently documents for SceneFunc3D runs.
SCENEFUNC3D_TOOL_NAMES: tuple[str, ...] = (
    "scene_summary",
    "keyframe_selector",
    "view_frame",
    "view_bev",
    "frame_objects",
    "molmo_point",
    "sam_mask",
)

SCENEFUNC3D_TOOLS_PLAYBOOK = """\
Playbook for SceneFunc3D mask generation.

No image is evidence until you open it with view_image. First use scene_summary,
keyframe_selector, view_frame, view_bev, or frame_objects to find visual evidence
for the task.

Tool catalog
- scene_summary: summarize available scene assets and task context.
- keyframe_selector: retrieve task-relevant first-person frames.
- view_frame: inspect selected raw frames.
- view_bev: inspect the top-down scene layout.
- frame_objects: list detected objects visible in a frame.
- molmo_point: propose a task-conditioned point on the target affordance.
- sam_mask: generate SAM candidates from an approved point.

molmo_point args are exactly image_path, image_width, image_height, prompt. Use
prompt, not point_prompt or task_description. When calling molmo_point, pass the
image_path, image_width, and image_height returned by view_frame. Do not guess
evidence image dimensions.

sam_mask args are exactly frame_id, image_path, point_xy. point_xy must be
[x_px, y_px] copied from the approved Molmo point. Use the same evidence
image_path that produced the approved Molmo point, and do not pass a point
object without frame_id.

Approval gates are mandatory:
1. After molmo_point, inspect the Molmo point overlay. You must approve the
Molmo point before calling sam_mask.
After a successful molmo_point call, do not spend extra reasoning turns if the
point lands on the intended affordance; immediately call sam_mask with the
approved point_xy, the same image_path, and the matching frame_id.
After at most three successful molmo_point attempts before SAM, stop trying new
Molmo prompts or frames and call sam_mask with the best approved point_xy.
2. After sam_mask, inspect the SAM candidates contact sheet. You must approve
one SAM candidate before confirming the seed.
Do not choose a SAM candidate by highest score alone. Use each candidate's
pixel_count and coverage_percent together with the contact sheet. For small
knobs, dials, handles, switches, buttons, and valves, prefer a compact candidate
that tightly covers the operable part; reject a broad panel, radiator body,
cabinet face, wall patch, pipe run, or shadow even if its SAM score is higher.

For small knobs, handles, dials, switches, buttons, and pinch_pull annotations,
first identify the affordance concept, then use a complete task-constrained
Molmo point prompt. After opening a selected frame, call molmo_point as the next
tool; do not spend extra reasoning turns comparing already-opened frames. For
drawer or cabinet pull tasks, inspect nearby frames around the best drawer/cabinet
view (+/- 8 frame ids when available) before the first Molmo call. Prefer a
visible knob, handle, pull tab, or recessed grip; use a seam or lip only after
the nearby frames do not show a distinct small operable component. The target is
not the drawer front panel center or the broad cabinet body.
Never repeat an expensive Molmo or SAM call with identical arguments after a
model-quality failure. Change prompt or frame.

After you approve one compact SAM candidate whose mask (drawn on the original
image) covers only the target operable part, STOP. Do not lift to 3D, do not
inspect 3D geometry, do not search more frames, and do not fuse. The runner
takes your confirmed 2D seed, lifts it to 3D deterministically, and completes
the multi-view 3D mask.

Judge the SAM candidate on the original-image overlay: reject a broad panel,
body, cabinet face, wall patch, pipe run, or shadow even if its SAM score is
higher; prefer a compact mask that tightly covers the knob/handle/dial/switch/
button/valve.

Final answer must be compact JSON matching the SeedConfirmation schema:
seed_frame_id, seed_candidate_id, affordance_concept, approval_actions,
uncertainties.
- seed_frame_id / seed_candidate_id: the frame_id you worked on and the
  approved SAM candidate_id from sam_mask.
- affordance_concept: the short affordance noun phrase you targeted (e.g. "the
  drawer handle"); the runner reuses it verbatim as the Molmo prompt subject on
  every additional view, so make it specific and free of task verbs.
- approval_actions: exactly, in order: select_evidence, propose_molmo_point,
  approve_molmo_point, propose_sam_candidates, approve_sam_candidate.
- uncertainties: short notes on anything ambiguous about the seed.
You must have approved the SAM candidate on seed_frame_id before emitting this
JSON; the runner verifies that sam_mask event exists.
"""


__all__ = ["SCENEFUNC3D_TOOL_NAMES", "SCENEFUNC3D_TOOLS_PLAYBOOK"]
