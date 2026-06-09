"""Unit tests for the NR3D evaluation runner (checkpointing, metrics, retries)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codex_agent.evaluation.nr3d_runner import run_samples
from codex_agent.evaluation.sample_ids import load_sample_ids
from codex_agent.models import CodexTaskResult, CodexTurnMetadata, CodexTurnResult
from codex_agent.nr3d.grounding import Nr3dGroundingOutcome
from codex_agent.tasks.base import CodexTask
from codex_agent.tests.conftest import Nr3dFixture


class _FakeRuntime:
    """A CodexExecutor that picks a fixed proposal and can fail N times first."""

    def __init__(
        self,
        *,
        proposal_id: int,
        fail_times: int = 0,
        error: str = "boom",
        input_tokens: int | None = None,
        cached_input_tokens: int | None = None,
    ) -> None:
        self.proposal_id = proposal_id
        self.fail_times = fail_times
        self.error = error
        self.input_tokens = input_tokens
        self.cached_input_tokens = cached_input_tokens
        self.calls = 0

    def execute(self, task: CodexTask[Any]) -> CodexTaskResult[Any]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(self.error)
        pool = task.scene.proposal_pool  # type: ignore[attr-defined]
        bbox = None
        if self.proposal_id != -1:
            bbox = pool.require(self.proposal_id).bbox_3d_9dof
        outcome = Nr3dGroundingOutcome(
            proposal_id=self.proposal_id,
            selected_bbox_9dof=bbox,
            confidence=0.9,
            summary="fake",
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


def test_correct_selection_scores_iou_one(
    nr3d_fixture: Nr3dFixture, tmp_path: Path
) -> None:
    runtime = _FakeRuntime(proposal_id=3)
    summary = run_samples(
        sample_ids=[nr3d_fixture.sample_id],
        data_root=nr3d_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=runtime,
    )
    assert summary.n == 1
    assert summary.mean_iou == pytest.approx(1.0)
    assert summary.acc_025 == pytest.approx(1.0)
    assert summary.results[0].status == "completed"
    assert summary.results[0].selected_object_id == 3


def test_target_absent_scores_zero(nr3d_fixture: Nr3dFixture, tmp_path: Path) -> None:
    runtime = _FakeRuntime(proposal_id=-1)
    summary = run_samples(
        sample_ids=[nr3d_fixture.sample_id],
        data_root=nr3d_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=runtime,
    )
    assert summary.mean_iou == 0.0
    assert summary.results[0].status == "failed"
    assert summary.results[0].selected_object_id is None


def test_checkpoint_is_reused(nr3d_fixture: Nr3dFixture, tmp_path: Path) -> None:
    out = tmp_path / "out"
    first = _FakeRuntime(proposal_id=3)
    run_samples(
        sample_ids=[nr3d_fixture.sample_id],
        data_root=nr3d_fixture.data_root,
        output_dir=out,
        runtime=first,
    )
    assert first.calls == 1

    # A second runtime that would explode if used; the cached checkpoint must
    # prevent any new execution.
    class _ExplodingRuntime:
        def execute(self, task: CodexTask[Any]) -> CodexTaskResult[Any]:
            raise AssertionError("must not run a cached sample")

    summary = run_samples(
        sample_ids=[nr3d_fixture.sample_id],
        data_root=nr3d_fixture.data_root,
        output_dir=out,
        runtime=_ExplodingRuntime(),
    )
    assert summary.n == 1
    assert summary.mean_iou == pytest.approx(1.0)


def test_retryable_error_then_success(
    nr3d_fixture: Nr3dFixture, tmp_path: Path
) -> None:
    runtime = _FakeRuntime(proposal_id=3, fail_times=1, error="503 service busy")
    summary = run_samples(
        sample_ids=[nr3d_fixture.sample_id],
        data_root=nr3d_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=runtime,
        sample_retries=1,
    )
    assert runtime.calls == 2
    assert summary.results[0].status == "completed"


def test_permanent_failure_yields_sentinel(
    nr3d_fixture: Nr3dFixture, tmp_path: Path
) -> None:
    runtime = _FakeRuntime(proposal_id=3, fail_times=5, error="non-retryable boom")
    summary = run_samples(
        sample_ids=[nr3d_fixture.sample_id],
        data_root=nr3d_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=runtime,
        sample_retries=1,
    )
    assert runtime.calls == 1  # non-retryable -> no retry
    assert summary.results[0].status == "failed"
    assert summary.results[0].error is not None
    assert summary.mean_iou == 0.0


def test_summary_written_to_disk(nr3d_fixture: Nr3dFixture, tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_samples(
        sample_ids=[nr3d_fixture.sample_id],
        data_root=nr3d_fixture.data_root,
        output_dir=out,
        runtime=_FakeRuntime(proposal_id=3),
    )
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["n"] == 1
    assert payload["Acc@0.25"] == pytest.approx(1.0)


def test_duplicate_sample_ids_raise(nr3d_fixture: Nr3dFixture, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_samples(
            sample_ids=[nr3d_fixture.sample_id, nr3d_fixture.sample_id],
            data_root=nr3d_fixture.data_root,
            output_dir=tmp_path / "out",
            runtime=_FakeRuntime(proposal_id=3),
        )


def test_summary_reports_cache_stats(nr3d_fixture: Nr3dFixture, tmp_path: Path) -> None:
    runtime = _FakeRuntime(proposal_id=3, input_tokens=20000, cached_input_tokens=7040)
    summary = run_samples(
        sample_ids=[nr3d_fixture.sample_id],
        data_root=nr3d_fixture.data_root,
        output_dir=tmp_path / "out",
        runtime=runtime,
    )
    assert summary.cache_hit_rate == pytest.approx(1.0)
    assert summary.mean_cache_ratio == pytest.approx(0.352)
    assert summary.results[0].cached_input_tokens == 7040


def test_load_sample_ids_list_of_strings(tmp_path: Path) -> None:
    path = tmp_path / "fold.json"
    path.write_text(json.dumps(["a::1::x", "b::2::y"]), encoding="utf-8")
    assert load_sample_ids(path) == ["a::1::x", "b::2::y"]


def test_load_sample_ids_list_of_objects(tmp_path: Path) -> None:
    path = tmp_path / "fold.json"
    path.write_text(
        json.dumps([{"sample_id": "a::1::x", "category": "chair"}]),
        encoding="utf-8",
    )
    assert load_sample_ids(path) == ["a::1::x"]
