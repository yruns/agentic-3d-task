"""Attribute saved SCOPE metrics to frozen runtime epochs.

The active source applies only to observed IDs outside the preserved baseline.
Update the source map before changing the evaluation runtime. An absent older
project launch head remains unknown rather than being inferred from a split.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

if TYPE_CHECKING or __package__:
    from scripts.ingest_goat_bench_run import ExpectedIds, SampleRecord, load_samples
else:
    from ingest_goat_bench_run import ExpectedIds, SampleRecord, load_samples

CommitHash = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{40}$")]
Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
NonemptyText = Annotated[str, StringConstraints(strict=True, min_length=1)]


class RuntimeSource(BaseModel):
    """A recorded runtime and supporting public evidence references."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    runtime_commit_sha: CommitHash
    project_launch_head: CommitHash | None = None
    evidence_records: tuple[NonemptyText, ...] = Field(min_length=1)


class PreservedEpoch(BaseModel):
    """Saved IDs whose execution predates the active attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: RuntimeSource
    completed_subtask_ids: tuple[NonemptyText, ...] = Field(min_length=1)


class LineagePlan(BaseModel):
    """Frozen requested IDs, historical checkpoints and the current runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    expected_subtask_ids: tuple[NonemptyText, ...] = Field(min_length=1)
    active_source: RuntimeSource
    preserved_epochs: tuple[PreservedEpoch, ...] = ()

    @model_validator(mode="after")
    def validate_id_ownership(self) -> LineagePlan:
        expected = set(self.expected_subtask_ids)
        if len(expected) != len(self.expected_subtask_ids):
            raise ValueError("lineage expected IDs must be unique")
        preserved: set[str] = set()
        for epoch in self.preserved_epochs:
            ids = set(epoch.completed_subtask_ids)
            if len(ids) != len(epoch.completed_subtask_ids) or preserved & ids:
                raise ValueError("preserved IDs must have exactly one source")
            if not ids <= expected:
                raise ValueError("preserved IDs must belong to the frozen request")
            preserved.update(ids)
        return self


class SampleLineage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    question_id: NonemptyText
    disposition: Literal["preserved", "active_attempt"]
    source: RuntimeSource


class LineageReport(BaseModel):
    """Only observed samples contribute to the reported runtime label."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    expected_subtasks: int = Field(gt=0)
    observed_subtasks: int = Field(gt=0)
    source_map_sha256: Digest
    runtime_commit_sha: NonemptyText
    samples: tuple[SampleLineage, ...] = Field(min_length=1)


def build_lineage_report(
    *,
    plan: LineagePlan,
    samples: Sequence[SampleRecord],
    requested_ids: tuple[str, ...],
    source_map_sha256: str,
) -> LineageReport:
    """Reject unknown identities and resolve each saved ID exactly once."""
    requested = set(ExpectedIds(root=requested_ids).root)
    if not requested or not requested <= set(plan.expected_subtask_ids):
        raise ValueError("selected fold IDs are absent from the lineage request")
    observed = {sample.question_id for sample in samples}
    if not observed or len(observed) != len(samples):
        raise ValueError("observed sample IDs must be nonempty and unique")
    if not observed <= requested:
        raise ValueError("observed sample ID is outside the selected fold")
    preserved = {
        question_id: epoch.source
        for epoch in plan.preserved_epochs
        for question_id in epoch.completed_subtask_ids
    }
    rows = tuple(
        SampleLineage(
            question_id=question_id,
            disposition="preserved" if question_id in preserved else "active_attempt",
            source=preserved.get(question_id, plan.active_source),
        )
        for question_id in sorted(observed)
    )
    commits = sorted({row.source.runtime_commit_sha for row in rows})
    label = commits[0] if len(commits) == 1 else "mixed:" + "+".join(commits)
    return LineageReport(
        expected_subtasks=len(requested),
        observed_subtasks=len(rows),
        source_map_sha256=source_map_sha256,
        runtime_commit_sha=label,
        samples=rows,
    )


def _write_report(path: Path, report: LineageReport) -> None:
    """Publish a complete report atomically without touching primary metrics."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(report.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_lineage_report(*, output_dir: Path, source_map: Path) -> LineageReport:
    """Validate external artifacts, attribute their saved IDs and write evidence."""
    if source_map.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("lineage source map exceeds the size limit")
    payload = source_map.read_bytes()
    plan = LineagePlan.model_validate_json(payload)
    requested = ExpectedIds.model_validate_json(
        (output_dir / "expected_subtask_ids.json").read_bytes()
    ).root
    report = build_lineage_report(
        plan=plan,
        samples=load_samples(output_dir),
        requested_ids=requested,
        source_map_sha256=hashlib.sha256(payload).hexdigest(),
    )
    _write_report(output_dir / "runtime_lineage.json", report)
    return report


class _CliOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    output_dir: Path
    source_map: Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    options = _CliOptions.model_validate(vars(parser.parse_args(argv)))
    report = write_lineage_report(
        output_dir=options.output_dir, source_map=options.source_map
    )
    print(report.runtime_commit_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
