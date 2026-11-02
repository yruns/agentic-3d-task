"""Run independent candidate calls concurrently, retaining original merge order."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

_CandidateT = TypeVar("_CandidateT")
_ScoreT = TypeVar("_ScoreT")


class _ScoringConcurrency(BaseModel):
    """Validated concurrency supplied by the caller's explicit configuration."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    max_workers: int = Field(ge=1)


@contextmanager
def submit_candidate_scores_in_order(
    candidates: Sequence[_CandidateT],
    *,
    score_candidate: Callable[[_CandidateT], _ScoreT],
    max_workers: int,
) -> Iterator[tuple[Future[_ScoreT], ...]]:
    """Submit independent calls and yield futures in original candidate order.

    Resolve each future inside the caller's original per-candidate exception
    handling, then merge state on the calling thread. No result or exception is
    replaced. Eligibility, duplicate-key handling and state updates belong to
    the caller; candidates must already represent independent scoring inputs.

    Pending calls are cancelled when leaving the context; already running calls
    are joined before it returns, including when a scorer raises SystemExit.
    """
    concurrency = _ScoringConcurrency(max_workers=max_workers)
    scoring_inputs = tuple(candidates)
    if not scoring_inputs:
        yield ()
        return

    with ThreadPoolExecutor(
        max_workers=min(concurrency.max_workers, len(scoring_inputs)),
        thread_name_prefix="scope-candidate-score",
    ) as executor:
        futures = tuple(
            executor.submit(score_candidate, candidate) for candidate in scoring_inputs
        )
        try:
            yield futures
        finally:
            for future in futures:
                future.cancel()
