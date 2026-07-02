"""Tests for the SceneFunc3D inline playbook and runner skeleton."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from codex_agent.scenefunc3d.playbook import (
    SCENEFUNC3D_TOOL_NAMES,
    SCENEFUNC3D_TOOLS_PLAYBOOK,
)
from codex_agent.scenefunc3d.runner import (
    SceneFunc3dRunnerConfig,
    build_prompt,
    load_runner_sample,
)
from codex_agent.scenefunc3d.sample import SceneFunc3dSample, SceneFuncMotionHint
from codex_agent.scenefunc3d.tools.dispatch import TOOL_NAMES

EXPECTED_TOOL_NAMES: tuple[str, ...] = (
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


def test_tool_names_are_current_task7_contract() -> None:
    assert SCENEFUNC3D_TOOL_NAMES == EXPECTED_TOOL_NAMES


def test_dispatcher_registers_every_playbook_tool() -> None:
    assert set(SCENEFUNC3D_TOOL_NAMES).issubset(set(TOOL_NAMES))


def test_playbook_mentions_every_tool() -> None:
    for tool_name in SCENEFUNC3D_TOOL_NAMES:
        assert tool_name in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_requires_agent_approval_gates() -> None:
    assert "Molmo point" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "SAM candidates" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "3D lift" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "must approve" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_tells_agent_to_pass_evidence_image_dimensions_to_molmo() -> None:
    assert "image_width" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "image_height" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "molmo_point" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_documents_exact_molmo_and_sam_argument_shapes() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert (
        "molmo_point args are exactly image_path, image_width, image_height, prompt"
        in (normalized_playbook)
    )
    assert "Use prompt, not point_prompt or task_description" in normalized_playbook
    assert "sam_mask args are exactly frame_id, image_path, point_xy" in (
        normalized_playbook
    )
    assert "point_xy must be [x_px, y_px]" in normalized_playbook


def test_playbook_tells_agent_selected_frames_must_match_fused_artifact() -> None:
    assert "selected_frame_ids" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "accepted_frame_ids" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "fuse_accepted_masks" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_requires_final_paths_from_fuse_accepted_masks() -> None:
    assert "mask_artifact_path returned by fuse_accepted_masks" in (
        SCENEFUNC3D_TOOLS_PLAYBOOK
    )
    assert "never use lift_overlay_path" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_tells_agent_to_record_fragment_approval_actions() -> None:
    assert "approval_actions" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "select_evidence" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "approve_first_lift" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_tells_agent_to_record_fragment_review_artifacts() -> None:
    assert "review_artifacts" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "molmo_raw_text_path" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "sam_contact_sheet_path" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "lift_overlay_path" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_tells_agent_to_pass_seed_artifact_to_multiview_suggestion() -> None:
    assert "suggest_additional_views" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "seed_mask_npz_path" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "seed_mask_ply_path" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "seed_lift_overlay_path" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "task_description" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "matched_objects" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_tells_agent_to_target_pinch_pull_affordance_points() -> None:
    assert "pinch_pull" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "handle" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "knob" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "not the drawer front panel center" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_does_not_expose_crop_guidance() -> None:
    assert "matched_objects" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "view_crop" not in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "recommended_crops" not in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "crop" not in SCENEFUNC3D_TOOLS_PLAYBOOK.lower()


def test_playbook_requires_nearby_frame_sweep_before_accepting_drawer_seams() -> None:
    assert "nearby frames" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "+/- 8" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "seam or lip only after" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_warns_against_score_only_sam_candidate_choice() -> None:
    assert "Do not choose a SAM candidate by highest score alone" in (
        SCENEFUNC3D_TOOLS_PLAYBOOK
    )
    assert "pixel_count" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "coverage_percent" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "compact candidate" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "broad panel" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_forces_compact_sam_candidate_to_lift_before_more_search() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert "After you approve a compact SAM candidate, call lift_mask_to_3d" in (
        normalized_playbook
    )
    assert "do not keep searching new frames before the first 3D lift" in (
        normalized_playbook
    )
    assert "Use the 3D geometry review to reject borderline compact candidates" in (
        normalized_playbook
    )


def test_playbook_prevents_reasoning_spin_after_successful_sam() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert "After a successful sam_mask call, do not spend extra reasoning turns" in (
        normalized_playbook
    )
    assert (
        "immediately call lift_mask_to_3d for the most plausible compact candidate"
        in (normalized_playbook)
    )
    assert "do not write a text-only analysis or plan after sam_mask" in (
        normalized_playbook
    )
    assert "your next assistant action must be lift_mask_to_3d" in normalized_playbook


def test_playbook_fuses_after_two_valid_small_affordance_fragments() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert "Once two inspected fragments are valid for the same small affordance" in (
        normalized_playbook
    )
    assert "call fuse_accepted_masks immediately" in normalized_playbook
    assert "do not try a third view unless the first two valid fragments conflict" in (
        normalized_playbook
    )


def test_playbook_forbids_inspection_before_lifted_paths_exist() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert "Do not call inspect_mask_artifact after sam_mask" in normalized_playbook
    assert "before lift_mask_to_3d has returned mask_npz_path" in normalized_playbook
    assert "never guess lifted artifact paths" in normalized_playbook


def test_playbook_prevents_reasoning_spin_after_successful_molmo_point() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert (
        "After a successful molmo_point call, do not spend extra reasoning turns"
        in (normalized_playbook)
    )
    assert "immediately call sam_mask with the approved point_xy" in (
        normalized_playbook
    )


def test_playbook_limits_molmo_retries_before_sam() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert "After at most three successful molmo_point attempts before SAM" in (
        normalized_playbook
    )
    assert "call sam_mask with the best approved point_xy" in normalized_playbook


def test_playbook_documents_exact_lift_argument_shape() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert "lift_mask_to_3d args are exactly frame_id, candidate_id, mask_npz_path" in (
        normalized_playbook
    )
    assert "Use mask_npz_path, not mask_path" in normalized_playbook


def test_playbook_prevents_reasoning_spin_after_opened_frame() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert "After opening a selected frame" in normalized_playbook
    assert "call molmo_point as the next tool" in normalized_playbook
    assert "do not spend extra reasoning turns comparing already-opened frames" in (
        normalized_playbook
    )


def test_playbook_bounds_followup_view_search_after_expand() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert 'After suggest_additional_views returns action "expand"' in (
        normalized_playbook
    )
    assert "choose at most two follow-up frames before the next Molmo call" in (
        normalized_playbook
    )
    assert "do not call suggest_additional_views again before Molmo or fusion" in (
        normalized_playbook
    )
    assert "call molmo_point on a selected follow-up frame" in (normalized_playbook)


def test_playbook_tells_agent_to_review_lift_geometry_summary() -> None:
    assert "bbox_extent_xyz" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "max_extent_meters" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "lift_geometry" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "recorded in the final artifact" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "too broad for the target affordance" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "do not approve it" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "Change the SAM candidate, point prompt, or frame" in (
        SCENEFUNC3D_TOOLS_PLAYBOOK
    )


def test_playbook_requires_lift_overlay_when_inspecting_fragments() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert (
        "inspect_mask_artifact args must include mask_npz_path, mask_ply_path, "
        "and lift_overlay_path returned by lift_mask_to_3d"
    ) in normalized_playbook
    assert "do not approve an accepted fragment from an inspection without" in (
        normalized_playbook
    )


def test_playbook_requires_multi_view_decision_reason_for_fuse() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert "multi_view_decision.reason is required" in normalized_playbook
    assert "explain why follow-up views were accepted or rejected" in (
        normalized_playbook
    )


def test_playbook_tells_agent_stop_decision_has_no_suggested_frame_ids() -> None:
    assert 'When multi_view_decision.action is "stop"' in (SCENEFUNC3D_TOOLS_PLAYBOOK)
    assert "suggested_frame_ids must be empty" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "rejected_suggested_frame_ids" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "record every suggested follow-up frame you inspected and rejected" in (
        SCENEFUNC3D_TOOLS_PLAYBOOK
    )
    assert "every frame returned by an expand recommendation" in (
        SCENEFUNC3D_TOOLS_PLAYBOOK
    )
    assert 'Use action "expand" when you accepted fragments from later frames' in (
        SCENEFUNC3D_TOOLS_PLAYBOOK
    )


def test_playbook_tells_agent_expand_decision_accounts_for_suggested_frames() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert 'When multi_view_decision.action is "expand"' in normalized_playbook
    assert "exactly one of suggested_frame_ids or rejected_suggested_frame_ids" in (
        normalized_playbook
    )
    assert "Do not put a frame in both lists" in normalized_playbook


def test_runner_config_is_frozen(tmp_path: Path) -> None:
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path,
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    with pytest.raises(FrozenInstanceError):
        config.__setattr__("output_dir", tmp_path / "other")


def test_build_prompt_includes_playbook_and_sample_context() -> None:
    sample = SceneFunc3dSample(
        sample_id="421254::desc-a",
        visit_id="421254",
        desc_id="desc-a",
        task_description="Open the lower drawer.",
        annotation_ids=("annot-a",),
        motion_hints=(
            SceneFuncMotionHint(
                motion_id="motion-a",
                annotation_id="annot-a",
                motion_type="trans",
                motion_dir=(1.0, 0.0, 0.0),
            ),
        ),
    )

    prompt = build_prompt(sample)

    assert prompt.startswith(SCENEFUNC3D_TOOLS_PLAYBOOK)
    assert "Open the lower drawer." in prompt
    assert "Generate a SceneFunc3D 3D mask artifact for this task." in prompt
    context_text = prompt.split("Task context:\n", maxsplit=1)[1].split(
        "\n\nGenerate", maxsplit=1
    )[0]
    context_payload = json.loads(context_text)
    assert context_payload["annotation_ids"] == ["annot-a"]


def test_load_runner_sample_uses_config_dataset_root(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    config = SceneFunc3dRunnerConfig(
        dataset_root=tmp_path,
        output_dir=tmp_path / "out",
        backend_config_path=tmp_path / "backends.toml",
    )

    sample = load_runner_sample(config, "421254::desc-a")

    assert sample.sample_id == "421254::desc-a"
    assert sample.task_description == "Open the lower drawer."


def _write_scene(root: Path) -> None:
    scene_dir = root / "421254"
    scene_dir.mkdir(parents=True)
    (scene_dir / "421254_descriptions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "descriptions": [
                    {
                        "desc_id": "desc-a",
                        "annot_id": ["annot-a"],
                        "description": "Open the lower drawer.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_motions.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "motions": [
                    {
                        "motion_id": "motion-a",
                        "annot_id": "annot-a",
                        "motion_type": "trans",
                        "motion_dir": [1.0, 0.0, 0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene_dir / "421254_annotations.json").write_text(
        json.dumps(
            {
                "visit_id": "421254",
                "annotations": [
                    {
                        "annot_id": "annot-a",
                        "label": "pinch_pull",
                        "indices": [3, 5, 8],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
