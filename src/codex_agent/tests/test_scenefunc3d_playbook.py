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
)


def test_tool_names_are_2d_seed_contract() -> None:
    assert SCENEFUNC3D_TOOL_NAMES == EXPECTED_TOOL_NAMES


def test_playbook_tool_names_are_2d_seed_only() -> None:
    from codex_agent.scenefunc3d.playbook import SCENEFUNC3D_TOOL_NAMES

    for dropped in (
        "lift_mask_to_3d",
        "inspect_mask_artifact",
        "suggest_additional_views",
        "fuse_accepted_masks",
    ):
        assert dropped not in SCENEFUNC3D_TOOL_NAMES
    assert SCENEFUNC3D_TOOL_NAMES[-1] == "sam_mask"


def test_playbook_ends_at_seed_confirmation() -> None:
    from codex_agent.scenefunc3d.playbook import SCENEFUNC3D_TOOLS_PLAYBOOK

    assert "seed_frame_id" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "seed_candidate_id" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "affordance_concept" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "lift_mask_to_3d" not in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "fuse_accepted_masks" not in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_dispatcher_registers_every_playbook_tool() -> None:
    assert set(SCENEFUNC3D_TOOL_NAMES).issubset(set(TOOL_NAMES))


def test_playbook_mentions_every_tool() -> None:
    for tool_name in SCENEFUNC3D_TOOL_NAMES:
        assert tool_name in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_requires_agent_approval_gates() -> None:
    assert "Molmo point" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "SAM candidates" in SCENEFUNC3D_TOOLS_PLAYBOOK
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


def test_playbook_tells_agent_to_record_seed_approval_actions() -> None:
    assert "approval_actions" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "select_evidence" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "approve_sam_candidate" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_tells_agent_to_target_pinch_pull_affordance_points() -> None:
    assert "pinch_pull" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "handle" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "knob" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "not the drawer front panel center" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_does_not_expose_crop_guidance() -> None:
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


def test_playbook_prevents_reasoning_spin_after_opened_frame() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert "After opening a selected frame" in normalized_playbook
    assert "call molmo_point as the next tool" in normalized_playbook
    assert "do not spend extra reasoning turns comparing already-opened frames" in (
        normalized_playbook
    )


def test_playbook_tells_agent_seed_confirmation_needs_prior_sam_approval() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert "STOP. Do not lift to 3D" in normalized_playbook
    assert "The runner takes your confirmed 2D seed" in normalized_playbook
    assert (
        "You must have approved the SAM candidate on seed_frame_id before "
        "emitting this JSON" in normalized_playbook
    )
    assert "the runner verifies that sam_mask event exists" in normalized_playbook


def test_playbook_warns_against_repeating_failed_molmo_or_sam_calls() -> None:
    normalized_playbook = " ".join(SCENEFUNC3D_TOOLS_PLAYBOOK.split())
    assert (
        "Never repeat an expensive Molmo or SAM call with identical arguments"
        in normalized_playbook
    )
    assert "Change prompt or frame" in normalized_playbook


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
