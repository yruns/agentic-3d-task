"""Inline SceneFunc3D tool-using playbook."""

from __future__ import annotations

#: Tool names the inline playbook currently documents for SceneFunc3D runs.
SCENEFUNC3D_TOOL_NAMES: tuple[str, ...] = (
    "scene_summary",
    "keyframe_selector",
    "view_frame",
    "view_crop",
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
keyframe_selector, view_frame, view_crop, view_bev, or frame_objects to find
visual evidence for the task.

Tool catalog
- scene_summary: summarize available scene assets and task context.
- keyframe_selector: retrieve task-relevant first-person frames.
- view_frame: inspect selected raw frames.
- view_crop: zoom into a frame region before judging small affordances.
- view_bev: inspect the top-down scene layout.
- frame_objects: list detected objects visible in a frame.
- molmo_point: propose a task-conditioned point on the target affordance.
- sam_mask: generate SAM candidates from an approved point.
- lift_mask_to_3d: lift an approved 2D mask into a 3D fragment.
- inspect_mask_artifact: inspect mask overlays and artifact metadata.
- suggest_additional_views: propose more views for multi-view completion.
- fuse_accepted_masks: fuse approved 3D fragments into the final artifact.

When calling molmo_point, pass the image_path, image_width, and image_height
returned by view_frame or view_crop. Do not guess evidence image dimensions.

Approval gates are mandatory:
1. After molmo_point, inspect the Molmo point overlay. You must approve the
Molmo point before calling sam_mask.
2. After sam_mask, inspect the SAM candidates contact sheet. You must approve
one SAM candidate before calling lift_mask_to_3d.
3. After lift_mask_to_3d, inspect the selected mask overlay and artifact
summary. You must approve the first 3D lift before any multi-view expansion.
4. Every additional view repeats Molmo point approval, SAM candidates approval,
and 3D lift approval before fusion.

When calling suggest_additional_views after approving the first 3D lift, pass
the approved seed's seed_mask_npz_path, seed_mask_ply_path, and
seed_lift_overlay_path from the lift_mask_to_3d output. If you already called
inspect_mask_artifact, reuse the same NPZ/PLY paths you inspected. The tool
validates that the seed lift overlay matches the accepted frame and seed
fragment, then returns seed_lift_point_count, seed_lift_status, and an
expand/stop recommendation so you can decide whether the target part needs more
views.
Use lift_mask_to_3d.mask_npz_path as seed_mask_npz_path,
lift_mask_to_3d.mask_ply_path as seed_mask_ply_path, lift_mask_to_3d.overlay_path
as seed_lift_overlay_path, and build seed_fragment_id as
<frame_id>_<candidate_id> from that same lift result.

For small knobs, handles, dials, switches, and buttons, first identify the
affordance concept, then use a complete task-constrained Molmo point prompt.
Never repeat an expensive Molmo or SAM call with identical arguments after a
model-quality failure. Change crop, prompt, or frame.

Final answer must be compact JSON with mask_artifact_path, mask_npz_path,
mask_ply_path, selected_frame_ids, accepted_fragment_ids, confidence, and
uncertainties.
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
Also include multi_view_decision when calling fuse_accepted_masks. Use action
"stop" only when the accepted fragments all come from the first accepted frame.
Use action "expand" when you accepted fragments from later frames; in that case,
suggested_frame_ids must include at least one accepted follow-up frame. The
seed_fragment_id must be the first accepted fragment.
Set selected_frame_ids to the accepted_frame_ids returned by fuse_accepted_masks.
"""


__all__ = ["SCENEFUNC3D_TOOL_NAMES", "SCENEFUNC3D_TOOLS_PLAYBOOK"]
