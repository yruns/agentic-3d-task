"""Unit tests for the NR3D grounding task: prompt, validation, parsing."""

from __future__ import annotations

import json

import pytest

from codex_agent.errors import CodexResponseError
from codex_agent.nr3d.grounding import Nr3dGroundingDecision, Nr3dGroundingTask
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
