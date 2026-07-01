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
    "lift_mask_to_3d",
    "inspect_mask_artifact",
    "suggest_additional_views",
    "fuse_accepted_masks",
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
- lift_mask_to_3d: lift an approved 2D mask into a 3D fragment.
- inspect_mask_artifact: inspect mask overlays and artifact metadata.
- suggest_additional_views: propose more views for multi-view completion.
- fuse_accepted_masks: fuse approved 3D fragments into the final artifact.

molmo_point args are exactly image_path, image_width, image_height, prompt. Use
prompt, not point_prompt or task_description. When calling molmo_point, pass the
image_path, image_width, and image_height returned by view_frame. Do not guess
evidence image dimensions.

sam_mask args are exactly frame_id, image_path, point_xy. point_xy must be
[x_px, y_px] copied from the approved Molmo point. Use the same evidence
image_path that produced the approved Molmo point, and do not pass a point
object without frame_id.

lift_mask_to_3d args are exactly frame_id, candidate_id, mask_npz_path. Copy
candidate_id and mask_npz_path from the selected sam_mask candidate. Use
mask_npz_path, not mask_path.

Approval gates are mandatory:
1. After molmo_point, inspect the Molmo point overlay. You must approve the
Molmo point before calling sam_mask.
After a successful molmo_point call, do not spend extra reasoning turns if the
point lands on the intended affordance; immediately call sam_mask with the
approved point_xy, the same image_path, and the matching frame_id.
After at most three successful molmo_point attempts before SAM, stop trying new
Molmo prompts or frames and call sam_mask with the best approved point_xy.
2. After sam_mask, inspect the SAM candidates contact sheet. You must approve
one SAM candidate before calling lift_mask_to_3d.
Do not choose a SAM candidate by highest score alone. Use each candidate's
pixel_count and coverage_percent together with the contact sheet. For small
knobs, dials, handles, switches, buttons, and valves, prefer a compact candidate
that tightly covers the operable part; reject a broad panel, radiator body,
cabinet face, wall patch, pipe run, or shadow even if its SAM score is higher.
After a successful sam_mask call, do not spend extra reasoning turns if a
candidate is plausibly compact for the target part; immediately call
lift_mask_to_3d for the most plausible compact candidate and let the 3D geometry
inspection confirm or reject it; do not write a text-only analysis or plan after
sam_mask; your next assistant action must be lift_mask_to_3d unless every
candidate is visibly impossible.
Do not call inspect_mask_artifact after sam_mask before lift_mask_to_3d has
returned mask_npz_path, mask_ply_path, and overlay_path; never guess lifted
artifact paths.
After you approve a compact SAM candidate, call lift_mask_to_3d before searching
again; do not keep searching new frames before the first 3D lift. Use the 3D
geometry review to reject borderline compact candidates instead of remaining in
2D search.
3. After lift_mask_to_3d, inspect the selected mask overlay and artifact
summary. You must approve the first 3D lift before any multi-view expansion.
Use inspect_mask_artifact geometry fields bbox_extent_xyz and max_extent_meters
before approving a 3D lift. A lift can be too broad for the target affordance.
For small affordances, this usually means SAM captured a panel, body, pipe, wall
patch, or shadow instead of the operable component; do not approve it.
Change the SAM candidate, point prompt, or frame before trying again.
inspect_mask_artifact args must include mask_npz_path, mask_ply_path, and
lift_overlay_path returned by lift_mask_to_3d; do not approve an accepted
fragment from an inspection without those exact lifted artifact paths.
4. Every additional view repeats Molmo point approval, SAM candidates approval,
and 3D lift approval before fusion.
Once two inspected fragments are valid for the same small affordance, call
fuse_accepted_masks immediately; do not try a third view unless the first two
valid fragments conflict or clearly cover different non-target parts.

When calling suggest_additional_views after approving the first 3D lift, pass
the approved seed's seed_mask_npz_path, seed_mask_ply_path, and
seed_lift_overlay_path from the lift_mask_to_3d output, and include the original
task_description so the tool can prioritize follow-up frames whose
matched_objects still show the target object. If you already called
inspect_mask_artifact, reuse the same NPZ/PLY paths you inspected. The tool
validates that the seed lift overlay matches the accepted frame and seed
fragment, then returns seed_lift_point_count, seed_lift_status, suggested
views, and an expand/stop recommendation so you can decide whether the target
part needs more views.
Use lift_mask_to_3d.mask_npz_path as seed_mask_npz_path,
lift_mask_to_3d.mask_ply_path as seed_mask_ply_path, lift_mask_to_3d.overlay_path
as seed_lift_overlay_path, and build seed_fragment_id as
<frame_id>_<candidate_id> from that same lift result.
After suggest_additional_views returns action "expand", choose at most two
follow-up frames before the next Molmo call. Prefer the top-ranked suggested
views with useful matched_objects and geometry. After that, do not call
suggest_additional_views again before Molmo or fusion. Once you have opened the
selected follow-up evidence, call molmo_point on a selected follow-up frame, or call
fuse_accepted_masks with rejected_suggested_frame_ids if every suggested
follow-up view has been rejected.

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

Final answer must be compact JSON with mask_artifact_path, mask_npz_path,
mask_ply_path, selected_frame_ids, accepted_fragment_ids, confidence, and
uncertainties. Before giving the final answer, you must call
fuse_accepted_masks. Use mask_artifact_path returned by fuse_accepted_masks as
the final JSON path, not a reviewed fragment path. Set mask_npz_path to its
mask_npz_path, and mask_ply_path to its mask_ply_path.
You must never use lift_overlay_path. Do not use lifted_points.ply, fragment
mask_data.npz, a SAM candidate NPZ, or any review artifact as a final path.
When passing each fragment into fuse_accepted_masks, include approval_actions in
this exact order: select_evidence, propose_molmo_point, approve_molmo_point,
propose_sam_candidates, approve_sam_candidate, create_first_lift,
approve_first_lift. These actions are written into the final artifact and
validated with the SceneFunc3D approval state machine.
Also include review_artifacts for every accepted fragment: molmo_raw_text_path,
molmo_overlay_path, sam_contact_sheet_path, sam_candidate_overlay_path, and
lift_overlay_path. These paths must point to the actual Molmo raw output,
Molmo point overlay, SAM contact sheet, selected SAM candidate overlay, and
lift inspection artifact you reviewed.
The fuse_accepted_masks tool records lift_geometry for every accepted fragment.
lift_geometry is recorded in the final artifact from the actual fragment NPZ.
This is the durable audit record for bbox_extent_xyz and max_extent_meters, so
approve a fragment only after those geometry fields are appropriate for the
target part.
Also include multi_view_decision when calling fuse_accepted_masks. Use action
"stop" only when the accepted fragments all come from the first accepted frame.
multi_view_decision.reason is required; explain why follow-up views were
accepted or rejected.
When multi_view_decision.action is "stop", suggested_frame_ids must be empty.
If you stop after checking suggested views, use rejected_suggested_frame_ids to
record every suggested follow-up frame you inspected and rejected. If
suggest_additional_views returned action "expand", stopping requires rejecting
every frame returned by an expand recommendation.
Use action "expand" when you accepted fragments from later frames; in that case,
suggested_frame_ids must include at least one accepted follow-up frame. The
seed_fragment_id must be the first accepted fragment.
When multi_view_decision.action is "expand", every frame returned by an
expand recommendation must appear in exactly one of suggested_frame_ids or
rejected_suggested_frame_ids. Do not put a frame in both lists.
Set selected_frame_ids to the accepted_frame_ids returned by fuse_accepted_masks.
"""


__all__ = ["SCENEFUNC3D_TOOL_NAMES", "SCENEFUNC3D_TOOLS_PLAYBOOK"]
