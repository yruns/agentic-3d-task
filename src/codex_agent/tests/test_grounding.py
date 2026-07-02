"""Unit tests for the NR3D grounding task: prompt, validation, parsing."""

from __future__ import annotations

import json

import pytest

from codex_agent.errors import CodexResponseError
from codex_agent.models import CodexSkill
from codex_agent.nr3d.grounding import Nr3dGroundingDecision, Nr3dGroundingTask
from codex_agent.nr3d.playbook import NR3D_TOOL_NAMES
from codex_agent.nr3d.sample import Nr3dScene, load_sample, scene_dir_for
from codex_agent.tests.conftest import Nr3dFixture


def _build_task(fixture: Nr3dFixture) -> Nr3dGroundingTask:
    sample = load_sample(fixture.data_root, fixture.sample_id)
    scene = Nr3dScene.load(scene_dir_for(fixture.data_root, fixture.scene_id))
    return Nr3dGroundingTask(sample=sample, scene=scene)


def test_turn_request_attaches_bev_and_schema(nr3d_fixture: Nr3dFixture) -> None:
    task = _build_task(nr3d_fixture)
    request = task.build_turn_request()
    assert request.image_paths == (
        nr3d_fixture.scene_dir / "bev" / "scene_bev_nr3d.png",
    )
    assert request.output_schema == Nr3dGroundingDecision.model_json_schema()
    assert task.task_name == "nr3d_visual_grounding"


def test_prompt_contains_query_and_proposals_without_gt(
    nr3d_fixture: Nr3dFixture,
) -> None:
    task = _build_task(nr3d_fixture)
    prompt = task.build_turn_request().prompt
    assert "the red chair" in prompt
    assert "#3:" in prompt and "#7:" in prompt
    assert "office chair" in prompt
    # The ground-truth target id / box must not leak into the prompt.
    assert "gt_bbox" not in prompt
    assert "target_id" not in prompt


def test_tool_mode_prompt_includes_cli_and_view_image(
    nr3d_fixture: Nr3dFixture,
) -> None:
    sample = load_sample(nr3d_fixture.data_root, nr3d_fixture.sample_id)
    scene = Nr3dScene.load(scene_dir_for(nr3d_fixture.data_root, nr3d_fixture.scene_id))
    task = Nr3dGroundingTask(
        sample=sample,
        scene=scene,
        tools_enabled=True,
        scene_dir=nr3d_fixture.scene_dir,
    )
    prompt = task.build_turn_request().prompt
    assert "python -m codex_agent.nr3d.tools" in prompt
    assert str(nr3d_fixture.scene_dir) in prompt
    assert "view_image" in prompt
    assert "compare_proposals_spatial" in prompt
    # Prompt-only "do not write files" rule must not be present in tool mode.
    assert "Do not write\nfiles" not in prompt and "Do not write files" not in prompt


def test_tool_mode_inlines_playbook_and_anti_exploration_rules(
    nr3d_fixture: Nr3dFixture,
) -> None:
    sample = load_sample(nr3d_fixture.data_root, nr3d_fixture.sample_id)
    scene = Nr3dScene.load(scene_dir_for(nr3d_fixture.data_root, nr3d_fixture.scene_id))
    task = Nr3dGroundingTask(
        sample=sample,
        scene=scene,
        tools_enabled=True,
        scene_dir=nr3d_fixture.scene_dir,
    )
    prompt = task.build_turn_request().prompt
    # The whole tool catalog is inlined so the model never needs an on-disk file.
    for tool_name in NR3D_TOOL_NAMES:
        assert tool_name in prompt
    # Anti-exploration framing that counters the default coding-agent persona.
    assert "Do NOT read" in prompt
    assert "SKILL.md" in prompt
    assert "NOT exploring or editing a codebase" in prompt


def test_tool_mode_defaults_to_no_on_disk_skill(nr3d_fixture: Nr3dFixture) -> None:
    sample = load_sample(nr3d_fixture.data_root, nr3d_fixture.sample_id)
    scene = Nr3dScene.load(scene_dir_for(nr3d_fixture.data_root, nr3d_fixture.scene_id))
    task = Nr3dGroundingTask(
        sample=sample,
        scene=scene,
        tools_enabled=True,
        scene_dir=nr3d_fixture.scene_dir,
    )
    assert task.build_turn_request().skills == ()


def test_tool_mode_attaches_explicit_skill(nr3d_fixture: Nr3dFixture) -> None:
    sample = load_sample(nr3d_fixture.data_root, nr3d_fixture.sample_id)
    scene = Nr3dScene.load(scene_dir_for(nr3d_fixture.data_root, nr3d_fixture.scene_id))
    explicit_skill = CodexSkill(
        name="nr3d-codex-tools", path=nr3d_fixture.scene_dir / "SKILL.md"
    )
    task = Nr3dGroundingTask(
        sample=sample,
        scene=scene,
        skill=explicit_skill,
        tools_enabled=True,
        scene_dir=nr3d_fixture.scene_dir,
    )
    request = task.build_turn_request()
    assert request.skills == (explicit_skill,)
    assert str(nr3d_fixture.scene_dir) in request.prompt
    assert "Use the attached Codex skill" not in request.prompt
    assert "Tool context" not in request.prompt
    assert "invoke:" not in request.prompt
    assert "python -m codex_agent.nr3d.tools" not in request.prompt
    assert "Do NOT read" not in request.prompt
    assert "Tool catalog" not in request.prompt
    assert "Recommended loop" not in request.prompt
    assert "Tool budget" not in request.prompt


def test_prompt_only_mode_rejects_supplied_skill(nr3d_fixture: Nr3dFixture) -> None:
    sample = load_sample(nr3d_fixture.data_root, nr3d_fixture.sample_id)
    scene = Nr3dScene.load(scene_dir_for(nr3d_fixture.data_root, nr3d_fixture.scene_id))
    skill = CodexSkill(name="nr3d-codex-sdk", path=nr3d_fixture.scene_dir / "SKILL.md")
    with pytest.raises(ValueError, match="tools_enabled=True"):
        Nr3dGroundingTask(sample=sample, scene=scene, skill=skill)


def test_prompt_only_mode_keeps_no_write_rule(nr3d_fixture: Nr3dFixture) -> None:
    task = _build_task(nr3d_fixture)
    prompt = task.build_turn_request().prompt
    assert "Do not write files" in prompt
    assert "python -m codex_agent.nr3d.tools" not in prompt


def test_is_valid_response(nr3d_fixture: Nr3dFixture) -> None:
    task = _build_task(nr3d_fixture)
    good = json.dumps(
        {
            "proposal_id": 3,
            "confidence": 0.9,
            "summary": "the red chair",
            "uncertainties": [],
            "cited_frame_indices": [10],
        }
    )
    assert task.is_valid_response(good) is True
    assert task.is_valid_response("not json") is False
    assert task.is_valid_response("") is False


def test_parse_selects_proposal_bbox(nr3d_fixture: Nr3dFixture) -> None:
    task = _build_task(nr3d_fixture)
    response = json.dumps(
        {
            "proposal_id": 3,
            "confidence": 0.8,
            "summary": "picked the chair",
            "uncertainties": ["maybe 7"],
            "cited_frame_indices": [10],
        }
    )
    outcome = task.parse_response(response)
    assert outcome.proposal_id == 3
    assert outcome.target_present is True
    assert outcome.status == "completed"
    assert outcome.selected_bbox_9dof == nr3d_fixture.gt_bbox
    assert outcome.cited_frame_indices == (10,)


def test_parse_rejects_missing_required_fields(nr3d_fixture: Nr3dFixture) -> None:
    # The strict schema requires all fields; a terse {"proposal_id": 3} is invalid
    # (the upstream's strict response_format prevents the model from sending it).
    task = _build_task(nr3d_fixture)
    with pytest.raises(CodexResponseError):
        task.parse_response(json.dumps({"proposal_id": 3}))


def test_parse_rejects_extra_keys(nr3d_fixture: Nr3dFixture) -> None:
    task = _build_task(nr3d_fixture)
    response = json.dumps(
        {
            "proposal_id": 7,
            "confidence": 0.5,
            "summary": "x",
            "uncertainties": [],
            "cited_frame_indices": [],
            "reasoning": "extra",
        }
    )
    with pytest.raises(CodexResponseError):
        task.parse_response(response)


def test_parse_target_absent(nr3d_fixture: Nr3dFixture) -> None:
    task = _build_task(nr3d_fixture)
    response = json.dumps(
        {
            "proposal_id": -1,
            "confidence": 0.2,
            "summary": "not present",
            "uncertainties": [],
            "cited_frame_indices": [],
        }
    )
    outcome = task.parse_response(response)
    assert outcome.target_present is False
    assert outcome.status == "failed"
    assert outcome.selected_bbox_9dof is None


def test_parse_rejects_out_of_pool_id(nr3d_fixture: Nr3dFixture) -> None:
    task = _build_task(nr3d_fixture)
    response = json.dumps(
        {
            "proposal_id": 999,
            "confidence": 0.5,
            "summary": "bad id",
            "uncertainties": [],
            "cited_frame_indices": [],
        }
    )
    with pytest.raises(CodexResponseError):
        task.parse_response(response)


def test_parse_rejects_schema_violation(nr3d_fixture: Nr3dFixture) -> None:
    task = _build_task(nr3d_fixture)
    # confidence out of range -> schema violation.
    response = json.dumps(
        {
            "proposal_id": 3,
            "confidence": 5.0,
            "summary": "x",
            "uncertainties": [],
            "cited_frame_indices": [],
        }
    )
    with pytest.raises(CodexResponseError):
        task.parse_response(response)
