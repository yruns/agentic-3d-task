"""A worker pauses only after complete progress and a durable readiness record."""

from pathlib import Path

import pytest

from scripts.ingest_goat_bench_run import ExpectedIds, SampleRecord
from scripts.scope.checkpoint_boundary import (
    BoundaryReady,
    BoundarySettings,
    complete_episode_progress,
    wait_and_pause_at_boundary,
)


class FakeProcess:
    def __init__(self) -> None:
        self.alive = True
        self.pauses = 0
        self.resumes = 0

    def is_alive(self) -> bool:
        return self.alive

    def pause(self) -> None:
        self.pauses += 1

    def resume(self) -> None:
        self.resumes += 1


def sample(qid: str) -> SampleRecord:
    return SampleRecord(
        question_id=qid, scene_id="s", episode_id="0", success=1, spl=0.5
    )


def settings(tmp_path: Path) -> BoundarySettings:
    expected = tmp_path / "expected.json"
    expected.write_text(ExpectedIds(root=("s_0_0", "s_1_0", "s_1_1")).model_dump_json())
    return BoundarySettings(
        pid=1234,
        start_ticks=123,
        source_dir=tmp_path,
        expected_ids_file=expected,
        baseline_ids=("s_0_0",),
        ready_file=tmp_path / "ready.json",
    )


def test_partial_episode_is_not_a_boundary() -> None:
    assert not complete_episode_progress(
        (sample("s_0_0"), sample("s_1_0")),
        expected=frozenset(("s_0_0", "s_1_0", "s_1_1")),
        baseline=frozenset(("s_0_0",)),
    )


def test_complete_progress_pauses_and_records_only_this_stage(tmp_path: Path) -> None:
    process = FakeProcess()
    configuration = settings(tmp_path)
    samples = (sample("s_0_0"), sample("s_1_0"), sample("s_1_1"), sample("other_0_0"))

    def read(_: Path) -> tuple[SampleRecord, ...]:
        return samples

    def sleep(_: float) -> None:
        raise AssertionError("complete checkpoint is immediately ready")

    assert wait_and_pause_at_boundary(
        configuration, process=process, read_samples=read, sleep=sleep
    )
    record = BoundaryReady.model_validate_json(configuration.ready_file.read_bytes())
    assert record.observed_ids == ("s_0_0", "s_1_0", "s_1_1")
    assert (process.pauses, process.resumes) == (1, 0)


def test_transient_pair_write_retries_without_pausing(tmp_path: Path) -> None:
    process = FakeProcess()
    calls = 0

    def read(_: Path) -> tuple[SampleRecord, ...]:
        nonlocal calls
        calls += 1
        raise ValueError("pair currently incomplete")

    def sleep(_: float) -> None:
        process.alive = False

    assert not wait_and_pause_at_boundary(
        settings(tmp_path), process=process, read_samples=read, sleep=sleep
    )
    assert calls == 1 and process.pauses == 0


def test_readiness_write_failure_resumes_worker(tmp_path: Path) -> None:
    process = FakeProcess()
    configuration = settings(tmp_path)
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("block")
    configuration = configuration.model_copy(
        update={"ready_file": blocker / "ready.json"}
    )

    def read(_: Path) -> tuple[SampleRecord, ...]:
        return (sample("s_0_0"), sample("s_1_0"), sample("s_1_1"))

    def sleep(_: float) -> None:
        raise AssertionError("no wait expected")

    with pytest.raises(OSError):
        wait_and_pause_at_boundary(
            configuration, process=process, read_samples=read, sleep=sleep
        )
    assert (process.pauses, process.resumes) == (1, 1)


def test_missing_baseline_rows_are_a_conflict() -> None:
    with pytest.raises(ValueError, match="checkpoint_ids_conflict"):
        complete_episode_progress(
            (sample("s_1_0"),),
            expected=frozenset(("s_0_0", "s_1_0")),
            baseline=frozenset(("s_0_0",)),
        )
