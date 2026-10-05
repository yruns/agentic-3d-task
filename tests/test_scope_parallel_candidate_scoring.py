"""Ordered, bounded candidate scoring without model calls or GPU dependencies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

import pytest
from pydantic import ValidationError

from scripts.scope import parallel_candidate_scoring as scoring


def test_overlaps_scoring_and_retains_input_order() -> None:
    started = Barrier(4)
    release = [Event(), Event(), Event()]
    completed: list[int] = []
    lock = Lock()

    def score(candidate: int) -> int:
        started.wait(timeout=3.0)
        assert release[candidate].wait(timeout=3.0)
        with lock:
            completed.append(candidate)
        return candidate * 10

    try:
        with scoring.submit_candidate_scores_in_order(
            [0, 1, 2], score_candidate=score, max_workers=3
        ) as futures:
            started.wait(timeout=3.0)
            for candidate in (2, 1, 0):
                release[candidate].set()
                assert futures[candidate].result(timeout=3.0) == candidate * 10
            assert tuple(future.result() for future in futures) == (0, 10, 20)
    finally:
        for event in release:
            event.set()
    assert completed == [2, 1, 0]


def test_bounds_number_of_active_scorers() -> None:
    two_started = Event()
    release = Event()
    lock = Lock()
    active = 0
    maximum_active = 0

    def score(candidate: int) -> int:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                two_started.set()
        try:
            assert release.wait(timeout=3.0)
            return candidate
        finally:
            with lock:
                active -= 1

    try:
        with scoring.submit_candidate_scores_in_order(
            range(5), score_candidate=score, max_workers=2
        ) as futures:
            assert two_started.wait(timeout=3.0)
            assert maximum_active == 2
            release.set()
            assert tuple(future.result(timeout=3.0) for future in futures) == tuple(
                range(5)
            )
    finally:
        release.set()
    assert maximum_active == 2


def test_preserves_none_as_an_original_scorer_result() -> None:
    def score(candidate: int) -> str | None:
        return None if candidate == 1 else f"score-{candidate}"

    with scoring.submit_candidate_scores_in_order(
        [0, 1, 2], score_candidate=score, max_workers=3
    ) as futures:
        assert tuple(future.result() for future in futures) == (
            "score-0",
            None,
            "score-2",
        )


def test_each_candidate_exception_remains_available_to_original_caller() -> None:
    failure = ValueError("candidate 1 failed")

    def score(candidate: int) -> int:
        if candidate == 1:
            raise failure
        return candidate

    with scoring.submit_candidate_scores_in_order(
        [0, 1, 2], score_candidate=score, max_workers=3
    ) as futures:
        assert futures[0].result() == 0
        with pytest.raises(ValueError) as caught:
            futures[1].result()
        assert caught.value is failure
        assert futures[2].result() == 2


def test_system_exit_is_not_converted_into_a_score() -> None:
    def score(candidate: int) -> int:
        raise SystemExit("model transport failed")

    with pytest.raises(SystemExit, match="model transport failed"):
        with scoring.submit_candidate_scores_in_order(
            [0], score_candidate=score, max_workers=1
        ) as futures:
            futures[0].result()


def test_empty_batch_creates_no_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    class ForbiddenExecutor(ThreadPoolExecutor):
        def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
            raise AssertionError("empty batches need no executor")

    def score(candidate: int) -> int:
        raise AssertionError("empty batches need no scoring")

    monkeypatch.setattr(scoring, "ThreadPoolExecutor", ForbiddenExecutor)
    with scoring.submit_candidate_scores_in_order(
        [], score_candidate=score, max_workers=16
    ) as futures:
        assert futures == ()


def test_worker_count_is_capped_at_number_of_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[int] = []

    class CountingExecutor(ThreadPoolExecutor):
        def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
            configured.append(max_workers)
            super().__init__(
                max_workers=max_workers, thread_name_prefix=thread_name_prefix
            )

    def score(candidate: int) -> int:
        return candidate

    monkeypatch.setattr(scoring, "ThreadPoolExecutor", CountingExecutor)
    with scoring.submit_candidate_scores_in_order(
        [0, 1], score_candidate=score, max_workers=16
    ) as futures:
        assert tuple(future.result() for future in futures) == (0, 1)
    assert configured == [2]


@pytest.mark.parametrize("max_workers", [0, -1, True])
def test_rejects_invalid_concurrency(max_workers: int) -> None:
    def score(candidate: int) -> int:
        return candidate

    with pytest.raises(ValidationError, match="max_workers"):
        with scoring.submit_candidate_scores_in_order(
            [], score_candidate=score, max_workers=max_workers
        ):
            raise AssertionError("invalid concurrency was accepted")
