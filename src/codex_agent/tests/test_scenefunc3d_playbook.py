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


def test_tool_names_are_current_task7_contract() -> None:
    assert SCENEFUNC3D_TOOL_NAMES == EXPECTED_TOOL_NAMES


def test_dispatcher_registers_every_playbook_tool() -> None:
    assert TOOL_NAMES == SCENEFUNC3D_TOOL_NAMES


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


def test_playbook_tells_agent_selected_frames_must_match_fused_artifact() -> None:
    assert "selected_frame_ids" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "accepted_frame_ids" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "fuse_accepted_masks" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_tells_agent_to_record_fragment_approval_actions() -> None:
    assert "approval_actions" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "select_evidence" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "approve_first_lift" in SCENEFUNC3D_TOOLS_PLAYBOOK


def test_playbook_tells_agent_to_record_fragment_review_artifacts() -> None:
    assert "review_artifacts" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "molmo_raw_text_path" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "sam_contact_sheet_path" in SCENEFUNC3D_TOOLS_PLAYBOOK
    assert "lift_overlay_path" in SCENEFUNC3D_TOOLS_PLAYBOOK


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
