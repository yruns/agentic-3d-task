"""Unit tests for the OpenEQA question loader and domain model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_agent.errors import OpenEqaDataError
from codex_agent.openeqa.question import (
    OpenEqaQuestion,
    index_by_question_id,
    load_questions,
    select_questions,
)
from codex_agent.tests.conftest import OpenEqaFixture


def test_load_questions_filters_to_scannet(openeqa_fixture: OpenEqaFixture) -> None:
    questions = load_questions(openeqa_fixture.questions_path)
    # The HM3D record is filtered out; only the two ScanNet questions remain.
    assert [q.question_id for q in questions] == ["q-scannet-1", "q-scannet-2"]


def test_load_questions_maps_clip_and_scene_id(
    openeqa_fixture: OpenEqaFixture,
) -> None:
    questions = load_questions(openeqa_fixture.questions_path)
    first = questions[0]
    assert first.clip_id == "002-scannet-scene0709_00"
    assert first.scene_id == "scene0709_00"
    assert first.answer == "Fire extinguisher"
    assert first.category == "object recognition"
    assert first.episode_history == "scannet-v0/002-scannet-scene0709_00"


def test_load_questions_synthesizes_missing_question_id(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "q?",
                    "answer": "a",
                    "episode_history": "scannet-v0/000-scannet-sceneXXXX_00",
                }
            ]
        ),
        encoding="utf-8",
    )
    questions = load_questions(path)
    assert questions[0].question_id == "0"


def test_load_questions_reads_extra_answers(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "color?",
                    "answer": "tan",
                    "question_id": "x",
                    "episode_history": "scannet-v0/000-scannet-sceneXXXX_00",
                    "extra_answers": ["beige", "cream"],
                }
            ]
        ),
        encoding="utf-8",
    )
    questions = load_questions(path)
    assert questions[0].extra_answers == ("beige", "cream")


def test_load_questions_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OpenEqaDataError):
        load_questions(tmp_path / "nope.json")


def test_load_questions_rejects_non_array(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(OpenEqaDataError):
        load_questions(path)


def test_load_questions_rejects_record_missing_answer(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "q?",
                    "episode_history": "scannet-v0/000-scannet-sceneXXXX_00",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(OpenEqaDataError):
        load_questions(path)


def test_index_by_question_id_rejects_duplicates() -> None:
    a = _question("dup")
    b = _question("dup")
    with pytest.raises(OpenEqaDataError):
        index_by_question_id([a, b])


def test_select_questions_preserves_requested_order() -> None:
    questions = [_question("a"), _question("b"), _question("c")]
    selected = select_questions(questions, ["c", "a"])
    assert [q.question_id for q in selected] == ["c", "a"]


def test_select_questions_missing_id_raises() -> None:
    with pytest.raises(OpenEqaDataError):
        select_questions([_question("a")], ["missing"])


def test_load_questions_empty_clip_id_raises(tmp_path: Path) -> None:
    # episode_history is exactly the prefix -> passes the filter but has no clip.
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps(
            [{"question": "q?", "answer": "a", "episode_history": "scannet-v0/"}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(OpenEqaDataError):
        load_questions(path)


def test_load_questions_rejects_non_object_item(tmp_path: Path) -> None:
    path = tmp_path / "q.json"
    path.write_text(json.dumps(["not-an-object"]), encoding="utf-8")
    with pytest.raises(OpenEqaDataError):
        load_questions(path)


def test_from_raw_rejects_prefix_mismatch() -> None:
    from codex_agent.openeqa.question import _RawOpenEqaRecord

    record = _RawOpenEqaRecord(
        question="q", answer="a", episode_history="hm3d-v0/000-hm3d-X"
    )
    with pytest.raises(OpenEqaDataError):
        OpenEqaQuestion.from_raw(record, index=0, dataset_prefix="scannet-v0/")


def _question(question_id: str) -> OpenEqaQuestion:
    return OpenEqaQuestion(
        question_id=question_id,
        question="q?",
        answer="a",
        category="object recognition",
        episode_history="scannet-v0/000-scannet-sceneXXXX_00",
        clip_id="000-scannet-sceneXXXX_00",
        scene_id="sceneXXXX_00",
    )
