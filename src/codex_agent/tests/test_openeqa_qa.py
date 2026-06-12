"""Unit tests for the OpenEQA QA task: no-frames prompt, tool guidance, parsing."""

from __future__ import annotations

import json

import pytest

from codex_agent.errors import CodexResponseError
from codex_agent.openeqa.qa import OpenEqaAnswerDecision, OpenEqaQuestionAnsweringTask
from codex_agent.openeqa.question import load_questions
from codex_agent.openeqa.scene import scene_dir_for
from codex_agent.tests.conftest import OpenEqaFixture


def _build_task(fixture: OpenEqaFixture) -> OpenEqaQuestionAnsweringTask:
    question = load_questions(fixture.questions_path)[0]
    scene_dir = scene_dir_for(fixture.data_root, fixture.clip_id)
    return OpenEqaQuestionAnsweringTask(question=question, scene_dir=scene_dir)


def test_turn_request_attaches_no_images_and_schema(
    openeqa_fixture: OpenEqaFixture,
) -> None:
    task = _build_task(openeqa_fixture)
    request = task.build_turn_request()
    assert request.image_paths == ()
    assert request.skills == ()
    assert request.output_schema == OpenEqaAnswerDecision.model_json_schema()
    assert task.task_name == "openeqa_question_answering"


def test_prompt_contains_question_without_leaking_answer(
    openeqa_fixture: OpenEqaFixture,
) -> None:
    task = _build_task(openeqa_fixture)
    prompt = task.build_turn_request().prompt
    assert "What red object is below the windows?" in prompt
    assert "object recognition" in prompt
    assert "scene0709_00" in prompt
    # The ground-truth answer must never appear in the prompt.
    assert "Fire extinguisher" not in prompt


def test_prompt_says_no_images_and_guides_tools(
    openeqa_fixture: OpenEqaFixture,
) -> None:
    task = _build_task(openeqa_fixture)
    prompt = task.build_turn_request().prompt
    # The agent is told there are no images and must fetch evidence itself.
    assert "NO images are attached" in prompt
    assert "How to run a tool" in prompt
    assert "python -m codex_agent.openeqa.tools" in prompt
    assert str(openeqa_fixture.clip_id) in prompt
    # Every tool is documented in the inlined playbook.
    for name in ("list_objects", "keyframe_selector", "view_frame", "view_bev"):
        assert name in prompt


def test_is_valid_response(openeqa_fixture: OpenEqaFixture) -> None:
    task = _build_task(openeqa_fixture)
    good = json.dumps(
        {
            "answer": "fire extinguisher",
            "supporting_claims": ["a red cylinder on the wall in a fetched frame"],
            "confidence": 0.7,
        }
    )
    assert task.is_valid_response(good) is True
    assert task.is_valid_response("not json") is False
    assert task.is_valid_response("") is False


def test_parse_response_extracts_answer(openeqa_fixture: OpenEqaFixture) -> None:
    task = _build_task(openeqa_fixture)
    response = json.dumps(
        {
            "answer": "  Fire extinguisher  ",
            "supporting_claims": ["a fetched frame shows a red extinguisher"],
            "confidence": 0.9,
        }
    )
    outcome = task.parse_response(response)
    assert outcome.answer == "Fire extinguisher"  # trimmed
    assert outcome.confidence == 0.9
    assert outcome.supporting_claims == ("a fetched frame shows a red extinguisher",)
    assert outcome.status == "completed"


def test_parse_response_accepts_fenced_json(openeqa_fixture: OpenEqaFixture) -> None:
    task = _build_task(openeqa_fixture)
    response = (
        "Here is my answer:\n```json\n"
        + json.dumps({"answer": "chair", "supporting_claims": [], "confidence": 0.5})
        + "\n```"
    )
    outcome = task.parse_response(response)
    assert outcome.answer == "chair"


def test_parse_rejects_missing_fields(openeqa_fixture: OpenEqaFixture) -> None:
    task = _build_task(openeqa_fixture)
    with pytest.raises(CodexResponseError):
        task.parse_response(json.dumps({"answer": "x"}))


def test_parse_rejects_extra_keys(openeqa_fixture: OpenEqaFixture) -> None:
    task = _build_task(openeqa_fixture)
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


def test_parse_rejects_empty_answer(openeqa_fixture: OpenEqaFixture) -> None:
    task = _build_task(openeqa_fixture)
    response = json.dumps({"answer": "", "supporting_claims": [], "confidence": 0.5})
    with pytest.raises(CodexResponseError):
        task.parse_response(response)


def test_parse_rejects_confidence_out_of_range(
    openeqa_fixture: OpenEqaFixture,
) -> None:
    task = _build_task(openeqa_fixture)
    response = json.dumps({"answer": "x", "supporting_claims": [], "confidence": 5.0})
    with pytest.raises(CodexResponseError):
        task.parse_response(response)


def test_scene_dir_is_required(openeqa_fixture: OpenEqaFixture) -> None:
    # scene_dir is mandatory: the agent needs it to invoke the CLI tools.
    question = load_questions(openeqa_fixture.questions_path)[0]
    with pytest.raises(TypeError):
        OpenEqaQuestionAnsweringTask(question=question)  # type: ignore[call-arg]
