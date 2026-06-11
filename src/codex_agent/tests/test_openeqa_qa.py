"""Unit tests for the OpenEQA QA task: prompt, frame attachment, parsing."""

from __future__ import annotations

import json

import pytest

from codex_agent.errors import CodexResponseError
from codex_agent.openeqa.qa import OpenEqaAnswerDecision, OpenEqaQuestionAnsweringTask
from codex_agent.openeqa.question import load_questions
from codex_agent.openeqa.scene import OpenEqaScene, scene_dir_for
from codex_agent.tests.conftest import OpenEqaFixture


def _build_task(
    fixture: OpenEqaFixture,
    tmp_path,
    *,
    num_frames: int = 3,
    tools_enabled: bool = False,
) -> OpenEqaQuestionAnsweringTask:
    question = load_questions(fixture.questions_path)[0]
    scene_dir = scene_dir_for(fixture.data_root, fixture.clip_id)
    scene = OpenEqaScene.load(scene_dir)
    frames = scene.prepare_frames(
        cache_dir=tmp_path / "cache", num_frames=num_frames, max_image_size=32
    )
    return OpenEqaQuestionAnsweringTask(
        question=question,
        frames=frames,
        tools_enabled=tools_enabled,
        scene_dir=scene_dir if tools_enabled else None,
    )


def test_turn_request_attaches_frames_and_schema(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    task = _build_task(openeqa_fixture, tmp_path, num_frames=3)
    request = task.build_turn_request()
    assert len(request.image_paths) == 3
    assert all(path.exists() for path in request.image_paths)
    assert request.output_schema == OpenEqaAnswerDecision.model_json_schema()
    assert task.task_name == "openeqa_question_answering"
    assert request.skills == ()


def test_prompt_contains_question_without_leaking_answer(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    task = _build_task(openeqa_fixture, tmp_path)
    prompt = task.build_turn_request().prompt
    assert "What red object is below the windows?" in prompt
    assert "object recognition" in prompt
    assert "scene0709_00" in prompt
    # The ground-truth answer must never appear in the prompt.
    assert "Fire extinguisher" not in prompt


def test_prompt_lists_attached_frames_in_order(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    task = _build_task(openeqa_fixture, tmp_path, num_frames=2)
    prompt = task.build_turn_request().prompt
    assert "#1 frame 0" in prompt
    assert "#2 frame 2" in prompt


def test_is_valid_response(openeqa_fixture: OpenEqaFixture, tmp_path) -> None:
    task = _build_task(openeqa_fixture, tmp_path)
    good = json.dumps(
        {
            "answer": "fire extinguisher",
            "supporting_claims": ["a red cylinder on the wall in frame 1"],
            "confidence": 0.7,
        }
    )
    assert task.is_valid_response(good) is True
    assert task.is_valid_response("not json") is False
    assert task.is_valid_response("") is False


def test_parse_response_extracts_answer(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    task = _build_task(openeqa_fixture, tmp_path)
    response = json.dumps(
        {
            "answer": "  Fire extinguisher  ",
            "supporting_claims": ["frame 1 shows a red extinguisher"],
            "confidence": 0.9,
        }
    )
    outcome = task.parse_response(response)
    assert outcome.answer == "Fire extinguisher"  # trimmed
    assert outcome.confidence == 0.9
    assert outcome.supporting_claims == ("frame 1 shows a red extinguisher",)
    assert outcome.status == "completed"


def test_parse_response_accepts_fenced_json(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    task = _build_task(openeqa_fixture, tmp_path)
    response = (
        "Here is my answer:\n```json\n"
        + json.dumps({"answer": "chair", "supporting_claims": [], "confidence": 0.5})
        + "\n```"
    )
    outcome = task.parse_response(response)
    assert outcome.answer == "chair"


def test_parse_rejects_missing_fields(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    task = _build_task(openeqa_fixture, tmp_path)
    with pytest.raises(CodexResponseError):
        task.parse_response(json.dumps({"answer": "x"}))


def test_parse_rejects_extra_keys(openeqa_fixture: OpenEqaFixture, tmp_path) -> None:
    task = _build_task(openeqa_fixture, tmp_path)
    response = json.dumps(
        {
            "answer": "x",
            "supporting_claims": [],
            "confidence": 0.5,
            "reasoning": "extra",
        }
    )
    with pytest.raises(CodexResponseError):
        task.parse_response(response)


def test_parse_rejects_empty_answer(openeqa_fixture: OpenEqaFixture, tmp_path) -> None:
    task = _build_task(openeqa_fixture, tmp_path)
    response = json.dumps({"answer": "", "supporting_claims": [], "confidence": 0.5})
    with pytest.raises(CodexResponseError):
        task.parse_response(response)


def test_parse_rejects_confidence_out_of_range(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    task = _build_task(openeqa_fixture, tmp_path)
    response = json.dumps({"answer": "x", "supporting_claims": [], "confidence": 5.0})
    with pytest.raises(CodexResponseError):
        task.parse_response(response)


def test_build_turn_request_with_no_frames(
    openeqa_fixture: OpenEqaFixture,
) -> None:
    from codex_agent.openeqa.qa import OpenEqaQuestionAnsweringTask
    from codex_agent.openeqa.question import load_questions

    question = load_questions(openeqa_fixture.questions_path)[0]
    task = OpenEqaQuestionAnsweringTask(question=question, frames=())
    request = task.build_turn_request()
    assert request.image_paths == ()
    assert "attached_frames (in order): []" in request.prompt


# ----- tool mode --------------------------------------------------------------


def test_prompt_only_mode_forbids_tools(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    task = _build_task(openeqa_fixture, tmp_path)
    prompt = task.build_turn_request().prompt
    assert "Do not write " in prompt
    assert "How to run a tool" not in prompt
    assert "Playbook" not in prompt


def test_tool_mode_inlines_playbook_and_scene_dir(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    task = _build_task(openeqa_fixture, tmp_path, tools_enabled=True)
    request = task.build_turn_request()
    prompt = request.prompt
    # Tool mode attaches no on-disk skill (playbook is inlined).
    assert request.skills == ()
    assert "How to run a tool" in prompt
    assert "python -m codex_agent.openeqa.tools" in prompt
    assert str(openeqa_fixture.clip_id) in prompt
    # Every tool is documented in the inlined playbook.
    for name in ("list_objects", "keyframe_selector", "view_frame", "view_bev"):
        assert name in prompt
    # Frames are still attached in tool mode.
    assert len(request.image_paths) == 3


def test_tool_mode_disabled_without_scene_dir(
    openeqa_fixture: OpenEqaFixture, tmp_path
) -> None:
    question = load_questions(openeqa_fixture.questions_path)[0]
    task = OpenEqaQuestionAnsweringTask(
        question=question, frames=(), tools_enabled=True, scene_dir=None
    )
    assert task.tools_enabled is False
    assert "How to run a tool" not in task.build_turn_request().prompt
