"""Validate a complete-episode boundary before pausing an owned Linux worker."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING or __package__:
    from scripts.ingest_goat_bench_run import ExpectedIds, SampleRecord, load_samples
else:
    from ingest_goat_bench_run import ExpectedIds, SampleRecord, load_samples

logger = logging.getLogger(__name__)


class BoundarySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pid: int = Field(gt=1)
    start_ticks: int = Field(gt=0)
    source_dir: Path
    expected_ids_file: Path
    baseline_ids: tuple[str, ...]
    ready_file: Path
    poll_seconds: float = Field(default=2.0, gt=0, le=60)


class BoundaryReady(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pid: int
    start_ticks: int
    paused_at_utc: datetime
    observed_ids: tuple[str, ...]
    observed_count: int


class ProcessPort(Protocol):
    def is_alive(self) -> bool: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...


class LinuxProcess:
    """Verify Linux process creation ticks before every signal."""

    def __init__(self, pid: int, start_ticks: int) -> None:
        self._pid = pid
        self._start_ticks = start_ticks
        if not self.is_alive():
            raise ProcessLookupError("worker_process_identity_changed")

    def is_alive(self) -> bool:
        try:
            fields = (
                Path(f"/proc/{self._pid}/stat").read_text().rsplit(")", 1)[1].split()
            )
        except FileNotFoundError:
            return False
        return int(fields[19]) == self._start_ticks and fields[0] != "Z"

    def _send(self, number: int) -> None:
        if not self.is_alive():
            raise ProcessLookupError("worker_process_identity_changed")
        os.kill(self._pid, number)

    def pause(self) -> None:
        self._send(signal.SIGSTOP)

    def resume(self) -> None:
        self._send(signal.SIGCONT)


def complete_episode_progress(
    samples: Sequence[SampleRecord],
    *,
    expected: frozenset[str],
    baseline: frozenset[str],
) -> bool:
    """Accept strict progress only when every observed episode is complete."""
    observed = frozenset(sample.question_id for sample in samples)
    if not baseline <= observed <= expected:
        raise ValueError("checkpoint_ids_conflict_with_baseline_or_expected")
    if observed == baseline:
        return False
    groups: dict[str, set[str]] = {}
    for qid in expected:
        episode, index = qid.rsplit("_", 1)
        if not index.isdigit():
            raise ValueError("expected_subtask_index_invalid")
        groups.setdefault(episode, set()).add(qid)
    return all(not (ids & observed) or ids <= observed for ids in groups.values())


def wait_and_pause_at_boundary(
    settings: BoundarySettings,
    *,
    process: ProcessPort,
    read_samples: Callable[[Path], Sequence[SampleRecord]],
    sleep: Callable[[float], None],
) -> bool:
    """Leave a worker paused only after an atomic readiness record is durable."""
    expected = frozenset(
        ExpectedIds.model_validate_json(settings.expected_ids_file.read_bytes()).root
    )
    baseline = frozenset(settings.baseline_ids)
    if not baseline <= expected or settings.ready_file.exists():
        raise ValueError("boundary_baseline_or_ready_file_invalid")

    def read_stage_samples() -> tuple[SampleRecord, ...]:
        return tuple(
            sample
            for sample in read_samples(settings.source_dir)
            if sample.question_id in expected
        )

    while process.is_alive():
        try:
            samples = read_stage_samples()
        except (ValueError, FileNotFoundError):
            logger.warning("Checkpoint pair not yet readable; retrying")
            sleep(settings.poll_seconds)
            continue
        if not complete_episode_progress(samples, expected=expected, baseline=baseline):
            sleep(settings.poll_seconds)
            continue
        process.pause()
        recorded = False
        try:
            samples = read_stage_samples()
            if not complete_episode_progress(
                samples, expected=expected, baseline=baseline
            ):
                continue
            ready = BoundaryReady(
                pid=settings.pid,
                start_ticks=settings.start_ticks,
                paused_at_utc=datetime.now(timezone.utc),
                observed_ids=tuple(sorted(sample.question_id for sample in samples)),
                observed_count=len(samples),
            )
            settings.ready_file.parent.mkdir(parents=True, exist_ok=True)
            pending = settings.ready_file.with_suffix(".pending")
            pending.write_text(ready.model_dump_json(indent=2) + "\n")
            pending.replace(settings.ready_file)
            recorded = True
            logger.info("Worker paused at complete episode: n=%d", len(samples))
            return True
        finally:
            if not recorded:
                process.resume()
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = BoundarySettings.model_validate_json(args.settings.read_bytes())
    logging.basicConfig(level=logging.INFO)
    process = LinuxProcess(settings.pid, settings.start_ticks)
    return (
        0
        if wait_and_pause_at_boundary(
            settings, process=process, read_samples=load_samples, sleep=time.sleep
        )
        else 3
    )


if __name__ == "__main__":
    raise SystemExit(main())
