"""Unit tests for the OpenEQA evaluation runner (checkpoint, judge, MNAS)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from codex_agent.evaluation.openeqa_runner import run_questions
from codex_agent.evaluation.question_ids import load_question_ids
from codex_agent.models import CodexTaskResult, CodexTurnMetadata, CodexTurnResult
from codex_agent.openeqa.qa import OpenEqaAnswerOutcome
from codex_agent.openeqa.question import OpenEqaQuestion, load_questions
from codex_agent.openeqa.scene import filter_questions_with_local_scenes
from codex_agent.tasks.base import CodexTask
from codex_agent.tests.conftest import OpenEqaFixture


class _FakeRuntime:
    """A CodexExecutor returning a fixed answer; can fail the first N calls."""

    def __init__(
        self,
        *,
        answer: str = "Fire extinguisher",
        fail_times: int = 0,
        error: str = "boom",
        input_tokens: int | None = None,
        cached_input_tokens: int | None = None,
    ) -> None:
        self.answer = answer
        self.fail_times = fail_times
        self.error = error
        self.input_tokens = input_tokens
        self.cached_input_tokens = cached_input_tokens
        self.calls = 0

    def execute(self, task: CodexTask[Any]) -> CodexTaskResult[Any]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(self.error)
        outcome = OpenEqaAnswerOutcome(
            answer=self.answer, confidence=0.8, supporting_claims=("frame 1",)
        )
        turn = CodexTurnResult(
            final_response="{}",
            metadata=CodexTurnMetadata(
                duration_ms=1,
                input_tokens=self.input_tokens,
                cached_input_tokens=self.cached_input_tokens,
            ),
        )
        return CodexTaskResult(task_name=task.task_name, outcome=outcome, turn=turn)


class _FakeJudge:
    """A JudgeScorer returning a fixed/keyed score, or raising on demand."""

    def __init__(
        self,
        *,
        default: int = 5,
        by_answer: Mapping[str, int] | None = None,
        raises: bool = False,
    ) -> None:
        self.default = default
        self.by_answer = dict(by_answer or {})
        self.raises = raises
        self.calls = 0

    def score(
        self,
        *,
        question: str,
        gt_answer: str,
        prediction: str,
        extra_answers: Sequence[str] = (),
    ) -> int:
        self.calls += 1
        if self.raises:
            from codex_agent.errors import OpenEqaJudgeError

            raise OpenEqaJudgeError("judge down")
        return self.by_answer.get(gt_answer, self.default)


def _scannet_questions(fixture: OpenEqaFixture) -> tuple[OpenEqaQuestion, ...]:
    return filter_questions_with_local_scenes(
        load_questions(fixture.questions_path), fixture.data_root
    )


def _first(fixture: OpenEqaFixture) -> list[OpenEqaQuestion]:
    return [_scannet_questions(fixture)[0]]


def test_correct_answer_scores_mnas_100(
    openeqa_fixture: OpenEqaFixture, tmp_path: Path
) -> None:
    summary = run_questions(
        questions=_first(openeqa_fixture),
        data_root=openeqa_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=_FakeRuntime(),
        judge=_FakeJudge(default=5),
    )
    assert summary.n == 1
    assert summary.n_judged == 1
    assert summary.mnas == pytest.approx(100.0)
    assert summary.completion_rate == pytest.approx(1.0)
    result = summary.results[0]
    assert result.status == "completed"
    assert result.prediction == "Fire extinguisher"
    assert result.judge_score == 5
    assert result.num_frames == openeqa_fixture.num_raw_frames


def test_tools_enabled_threads_to_task(
    openeqa_fixture: OpenEqaFixture, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    class _CapturingRuntime(_FakeRuntime):
        def execute(self, task: CodexTask[Any]) -> CodexTaskResult[Any]:
            captured["tools_enabled"] = getattr(task, "tools_enabled", None)
            captured["scene_dir"] = getattr(task, "scene_dir", None)
            return super().execute(task)

    run_questions(
        questions=_first(openeqa_fixture),
        data_root=openeqa_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=_CapturingRuntime(),
        judge=None,
        tools_enabled=True,
    )
    assert captured["tools_enabled"] is True
    assert captured["scene_dir"] == (
        openeqa_fixture.data_root / openeqa_fixture.clip_id
    )


def test_no_judge_leaves_scores_unset(
    openeqa_fixture: OpenEqaFixture, tmp_path: Path
) -> None:
    summary = run_questions(
        questions=_first(openeqa_fixture),
        data_root=openeqa_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=_FakeRuntime(),
        judge=None,
    )
    assert summary.n_judged == 0
    assert summary.mnas == 0.0
    assert summary.results[0].judge_score is None
    assert summary.results[0].mnas is None


def test_checkpoint_is_reused(openeqa_fixture: OpenEqaFixture, tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = _FakeRuntime()
    run_questions(
        questions=_first(openeqa_fixture),
        data_root=openeqa_fixture.data_root,
        output_dir=out,
        runtime=first,
        judge=_FakeJudge(),
    )
    assert first.calls == 1

    class _ExplodingRuntime:
        def execute(self, task: CodexTask[Any]) -> CodexTaskResult[Any]:
            raise AssertionError("must not run a cached question")

    summary = run_questions(
        questions=_first(openeqa_fixture),
        data_root=openeqa_fixture.data_root,
        output_dir=out,
        runtime=_ExplodingRuntime(),
        judge=_FakeJudge(),
    )
    assert summary.mnas == pytest.approx(100.0)


def test_retryable_error_then_success(
    openeqa_fixture: OpenEqaFixture, tmp_path: Path
) -> None:
    runtime = _FakeRuntime(fail_times=1, error="503 server busy")
    summary = run_questions(
        questions=_first(openeqa_fixture),
        data_root=openeqa_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=runtime,
        judge=_FakeJudge(),
        sample_retries=1,
    )
    assert runtime.calls == 2
    assert summary.results[0].status == "completed"


def test_permanent_failure_scores_zero(
    openeqa_fixture: OpenEqaFixture, tmp_path: Path
) -> None:
    runtime = _FakeRuntime(fail_times=5, error="non-retryable boom")
    summary = run_questions(
        questions=_first(openeqa_fixture),
        data_root=openeqa_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=runtime,
        judge=_FakeJudge(),
        sample_retries=1,
    )
    assert runtime.calls == 1  # non-retryable -> no retry
    result = summary.results[0]
    assert result.status == "failed"
    assert result.error is not None
    assert result.judge_score == 0  # missing prediction counts as floor
    assert summary.mnas == pytest.approx(0.0)
    assert summary.n_judged == 1


def test_judge_failure_is_recorded_not_fatal(
    openeqa_fixture: OpenEqaFixture, tmp_path: Path
) -> None:
    summary = run_questions(
        questions=_first(openeqa_fixture),
        data_root=openeqa_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=_FakeRuntime(),
        judge=_FakeJudge(raises=True),
    )
    result = summary.results[0]
    assert result.status == "completed"  # the answer was produced
    assert result.judge_score is None  # but could not be judged
    assert result.error is not None and "judge_error" in result.error
    assert summary.n_judged == 0  # excluded from MNAS


def test_per_category_mnas(openeqa_fixture: OpenEqaFixture, tmp_path: Path) -> None:
    summary = run_questions(
        questions=list(_scannet_questions(openeqa_fixture)),
        data_root=openeqa_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=_FakeRuntime(),
        judge=_FakeJudge(by_answer={"Fire extinguisher": 5, "Chair": 1}),
    )
    assert summary.per_category_mnas["object recognition"] == pytest.approx(100.0)
    assert summary.per_category_mnas["spatial understanding"] == pytest.approx(0.0)
    assert summary.mnas == pytest.approx(50.0)


def test_summary_and_predictions_written(
    openeqa_fixture: OpenEqaFixture, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    run_questions(
        questions=list(_scannet_questions(openeqa_fixture)),
        data_root=openeqa_fixture.data_root,
        output_dir=out,
        runtime=_FakeRuntime(),
        judge=_FakeJudge(),
    )
    summary_payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["n"] == 2
    assert "MNAS" in summary_payload

    predictions = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
    assert len(predictions) == 2
    assert {"question_id", "answer"} <= set(predictions[0])


def test_cache_stats_reported(openeqa_fixture: OpenEqaFixture, tmp_path: Path) -> None:
    runtime = _FakeRuntime(input_tokens=20000, cached_input_tokens=7040)
    summary = run_questions(
        questions=_first(openeqa_fixture),
        data_root=openeqa_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=runtime,
        judge=_FakeJudge(),
    )
    assert summary.cache_hit_rate == pytest.approx(1.0)
    assert summary.mean_cache_ratio == pytest.approx(0.352)


def test_duplicate_questions_raise(
    openeqa_fixture: OpenEqaFixture, tmp_path: Path
) -> None:
    question = _first(openeqa_fixture)[0]
    with pytest.raises(ValueError):
        run_questions(
            questions=[question, question],
            data_root=openeqa_fixture.data_root,
            output_dir=tmp_path / "out",
            runtime=_FakeRuntime(),
            judge=_FakeJudge(),
        )


def test_load_question_ids_list_of_strings(tmp_path: Path) -> None:
    path = tmp_path / "fold.json"
    path.write_text(json.dumps(["q1", "q2"]), encoding="utf-8")
    assert load_question_ids(path) == ["q1", "q2"]


def test_load_question_ids_list_of_objects(tmp_path: Path) -> None:
    path = tmp_path / "fold.json"
    path.write_text(
        json.dumps([{"question_id": "q1", "category": "x"}]), encoding="utf-8"
    )
    assert load_question_ids(path) == ["q1"]


def test_load_question_ids_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "fold.json"
    path.write_text(json.dumps({"q": 1}), encoding="utf-8")
    from codex_agent.errors import OpenEqaDataError

    with pytest.raises(OpenEqaDataError):
        load_question_ids(path)


def test_load_question_ids_rejects_mixed_types(tmp_path: Path) -> None:
    path = tmp_path / "fold.json"
    path.write_text(json.dumps(["q1", 2, 3]), encoding="utf-8")
    from codex_agent.errors import OpenEqaDataError

    with pytest.raises(OpenEqaDataError):
        load_question_ids(path)


def test_load_question_ids_object_missing_id(tmp_path: Path) -> None:
    path = tmp_path / "fold.json"
    path.write_text(json.dumps([{"category": "x"}]), encoding="utf-8")
    from codex_agent.errors import OpenEqaDataError

    with pytest.raises(OpenEqaDataError):
        load_question_ids(path)


def test_parallel_workers_run_all(
    openeqa_fixture: OpenEqaFixture, tmp_path: Path
) -> None:
    summary = run_questions(
        questions=list(_scannet_questions(openeqa_fixture)),
        data_root=openeqa_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=_FakeRuntime(),
        judge=_FakeJudge(),
        workers=2,
    )
    assert summary.n == 2
    assert summary.completion_rate == pytest.approx(1.0)


def test_run_questions_rejects_non_positive_workers(
    openeqa_fixture: OpenEqaFixture, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        run_questions(
            questions=_first(openeqa_fixture),
            data_root=openeqa_fixture.data_root,
            output_dir=tmp_path / "out",
            runtime=_FakeRuntime(),
            judge=_FakeJudge(),
            workers=0,
        )


def test_run_questions_rejects_negative_retries(
    openeqa_fixture: OpenEqaFixture, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        run_questions(
            questions=_first(openeqa_fixture),
            data_root=openeqa_fixture.data_root,
            output_dir=tmp_path / "out",
            runtime=_FakeRuntime(),
            judge=_FakeJudge(),
            sample_retries=-1,
        )
