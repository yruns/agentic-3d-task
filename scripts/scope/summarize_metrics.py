"""Summarize a complete SCOPE fold from primary metrics and explicit task indices.

Run with ``python -m scripts.scope.summarize_metrics``. Partial folds and derived
aggregate-only inputs fail before a report is written. Token/call attribution
belongs to the separate run-level transport record.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

from scripts.ingest_goat_bench_run import (
    _MAX_ARTIFACT_BYTES,
    ExpectedIds,
    GoalType,
    ModalitySummary,
    NonemptyText,
    SampleRecord,
    TaskMetadata,
    _read_payload,
    _summarize,
    load_samples,
)

MetricName = Literal["success_by_distance", "spl_by_distance"]
ArtifactRole = Literal[
    "success_by_distance", "spl_by_distance", "expected_ids", "task_metadata"
]
PositiveCount = Annotated[int, Field(gt=0, strict=True)]
_EXPECTED_COUNT: TypeAdapter[PositiveCount] = TypeAdapter(PositiveCount)
_METRIC_NAMES: tuple[MetricName, ...] = ("success_by_distance", "spl_by_distance")
_GOAL_TYPES: tuple[GoalType, ...] = ("object", "description", "image")
_HASH_CHUNK_BYTES = 1024 * 1024


class IndexedTaskMetadata(TaskMetadata):
    """Original task metadata with a required, explicitly recorded zero-based index."""

    subtask_index: int = Field(ge=0, strict=True)


class _MetadataPayload(RootModel[dict[NonemptyText, object]]):
    """Deserialization boundary; selected rows immediately become indexed metadata."""


class SubtaskMetrics(ModalitySummary):
    """Percentage metrics for one zero-based index and its human-readable ordinal."""

    subtask_index: int = Field(ge=0, strict=True)
    ordinal1based: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_ordinal(self) -> SubtaskMetrics:
        if self.ordinal1based != self.subtask_index + 1:
            raise ValueError("ordinal1based must be subtask_index + 1")
        return self


class InputFingerprint(BaseModel):
    """Exact selected input file, excluding unused derived aggregates."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    role: ArtifactRole
    path: NonemptyText
    size_bytes: int = Field(ge=0, strict=True)
    sha256: Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]


class FoldReport(BaseModel):
    """Complete-fold metrics with explicit grouping and reproducible input hashes."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    complete: Literal[True] = True
    expected_subtasks: PositiveCount
    output_dir: Path
    overall: ModalitySummary
    per_modality: dict[GoalType, ModalitySummary]
    per_subtask_index: dict[int, SubtaskMetrics]
    inputs: tuple[InputFingerprint, ...]

    @model_validator(mode="after")
    def validate_group_counts(self) -> FoldReport:
        counts = (
            self.overall.n,
            sum(group.n for group in self.per_modality.values()),
            sum(group.n for group in self.per_subtask_index.values()),
        )
        if any(count != self.expected_subtasks for count in counts):
            raise ValueError("all metric groups must cover expected_subtasks exactly")
        if any(
            index != group.subtask_index
            for index, group in self.per_subtask_index.items()
        ):
            raise ValueError("subtask index keys must match their grouped metadata")
        return self


@dataclass(frozen=True)
class _InputFile:
    role: ArtifactRole
    path: Path


def _primary_paths(output_dir: Path, metric_name: MetricName) -> tuple[Path, ...]:
    """Match the ingester's per-stem JSON preference, requiring suffixed primaries."""
    paths_by_stem: dict[str, Path] = {}
    for suffix in ("json", "pkl"):
        for path in sorted(output_dir.glob(f"{metric_name}_*.{suffix}")):
            paths_by_stem.setdefault(path.stem, path)
    if not paths_by_stem:
        raise FileNotFoundError(
            f"no suffixed primary {metric_name} metrics under {output_dir}"
        )
    return tuple(paths_by_stem.values())


def _select_input_files(output_dir: Path) -> tuple[_InputFile, ...]:
    by_metric = {
        metric_name: _primary_paths(output_dir, metric_name)
        for metric_name in _METRIC_NAMES
    }
    success_suffixes = {
        path.stem.removeprefix("success_by_distance")
        for path in by_metric["success_by_distance"]
    }
    spl_suffixes = {
        path.stem.removeprefix("spl_by_distance")
        for path in by_metric["spl_by_distance"]
    }
    if success_suffixes != spl_suffixes:
        raise ValueError("success and SPL primary suffix sets must match exactly")
    inputs = [
        _InputFile(metric_name, path)
        for metric_name in _METRIC_NAMES
        for path in by_metric[metric_name]
    ]
    inputs.extend(
        (
            _InputFile("expected_ids", output_dir / "expected_subtask_ids.json"),
            _InputFile("task_metadata", output_dir / "subtask_metadata.json"),
        )
    )
    return tuple(inputs)


def _fingerprint_inputs(
    output_dir: Path, inputs: Sequence[_InputFile]
) -> tuple[InputFingerprint, ...]:
    fingerprints: list[InputFingerprint] = []
    for artifact in inputs:
        if artifact.path.stat().st_size > _MAX_ARTIFACT_BYTES:
            raise ValueError(
                f"input artifact exceeds {_MAX_ARTIFACT_BYTES} bytes: {artifact.path}"
            )
        digest = hashlib.sha256()
        size_bytes = 0
        with artifact.path.open("rb") as stream:
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                size_bytes += len(chunk)
                digest.update(chunk)
        fingerprints.append(
            InputFingerprint(
                role=artifact.role,
                path=str(artifact.path.relative_to(output_dir)),
                size_bytes=size_bytes,
                sha256=digest.hexdigest(),
            )
        )
    return tuple(fingerprints)


def _load_indexed_metadata(
    output_dir: Path, expected_ids: Sequence[str]
) -> dict[str, IndexedTaskMetadata]:
    payload = _MetadataPayload.model_validate(
        _read_payload(output_dir / "subtask_metadata.json")
    )
    missing = set(expected_ids) - payload.root.keys()
    if missing:
        raise ValueError(f"missing original task metadata for IDs: {sorted(missing)}")
    metadata: dict[str, IndexedTaskMetadata] = {}
    for qid in expected_ids:
        task = IndexedTaskMetadata.model_validate(payload.root[qid])
        declared_qid = f"{task.scene_id}_{task.episode_id}_{task.subtask_index}"
        if declared_qid != qid:
            raise ValueError(
                f"metadata scene/episode/subtask_index does not match ID: {qid}"
            )
        metadata[qid] = task
    return metadata


def _group_metrics(samples: Sequence[SampleRecord], index: int) -> SubtaskMetrics:
    n = len(samples)
    if not n:
        raise ValueError(f"subtask_index={index}: cannot aggregate an empty group")
    return SubtaskMetrics(
        subtask_index=index,
        ordinal1based=index + 1,
        n=n,
        sr=100 * sum(sample.success for sample in samples) / n,
        spl=100 * sum(sample.spl for sample in samples) / n,
    )


def summarize_complete_fold(output_dir: Path, *, expected_subtasks: int) -> FoldReport:
    """Validate a complete frozen fold and summarize only terminal primary metrics."""
    expected_subtasks = _EXPECTED_COUNT.validate_python(expected_subtasks)
    output_dir = output_dir.absolute()
    if not output_dir.is_dir():
        raise ValueError(f"output_dir is not an existing directory: {output_dir}")
    inputs = _select_input_files(output_dir)
    fingerprints = _fingerprint_inputs(output_dir, inputs)
    expected_ids = ExpectedIds.model_validate(
        _read_payload(output_dir / "expected_subtask_ids.json")
    ).root
    if len(expected_ids) != expected_subtasks:
        raise ValueError("expected ID count does not match expected_subtasks")
    metadata = _load_indexed_metadata(output_dir, expected_ids)
    samples = load_samples(output_dir)
    recorded_ids = {sample.question_id for sample in samples}
    if recorded_ids != set(expected_ids):
        raise ValueError(
            "incomplete or unexpected terminal IDs: "
            f"missing={sorted(set(expected_ids) - recorded_ids)}, "
            f"unexpected={sorted(recorded_ids - set(expected_ids))}"
        )
    summary = _summarize(output_dir, samples, expected_subtasks)
    if not (
        summary.complete and summary.expected_ids_verified and summary.modality_complete
    ):
        raise ValueError(
            "complete coverage, frozen IDs and explicit modalities are required"
        )
    if summary.sr is None or summary.spl is None:
        raise ValueError("complete folds must contain measured SR and SPL")
    by_index: dict[int, list[SampleRecord]] = {}
    for sample in samples:
        by_index.setdefault(metadata[sample.question_id].subtask_index, []).append(
            sample
        )
    if inputs != _select_input_files(output_dir) or fingerprints != _fingerprint_inputs(
        output_dir, inputs
    ):
        raise ValueError("input artifacts changed during summary validation")
    return FoldReport(
        expected_subtasks=expected_subtasks,
        output_dir=output_dir,
        overall=ModalitySummary(n=summary.n, sr=summary.sr, spl=summary.spl),
        per_modality={
            goal_type: summary.per_modality[goal_type]
            for goal_type in _GOAL_TYPES
            if goal_type in summary.per_modality
        },
        per_subtask_index={
            index: _group_metrics(rows, index)
            for index, rows in sorted(by_index.items())
        },
        inputs=fingerprints,
    )


def write_summary_report(
    output_dir: Path, *, expected_subtasks: int, report_file: Path
) -> FoldReport:
    """Publish a validated report atomically, leaving previous reports intact on failure."""
    report_file = report_file.absolute()
    logical_output_dir = Path(os.path.abspath(output_dir))
    logical_report_file = Path(os.path.abspath(report_file))
    if report_file.name.startswith(_METRIC_NAMES) and (
        logical_report_file.is_relative_to(logical_output_dir)
        or report_file.parent.resolve().is_relative_to(logical_output_dir.resolve())
    ):
        raise ValueError(f"report_file uses a reserved metric namespace: {report_file}")
    report = summarize_complete_fold(output_dir, expected_subtasks=expected_subtasks)
    for artifact in report.inputs:
        input_path = report.output_dir / artifact.path
        if report_file.resolve() == input_path.resolve() or (
            report_file.exists() and report_file.samefile(input_path)
        ):
            raise ValueError(
                f"report_file must not overwrite an input artifact: {report_file}"
            )
    report_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{report_file.name}.", suffix=".tmp", dir=report_file.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(report.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, report_file)
    finally:
        temporary_path.unlink(missing_ok=True)
    return report


def main() -> None:
    """Parse explicit fold boundaries and delegate to complete-report publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-subtasks", type=int, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    options = parser.parse_args()
    report = write_summary_report(
        options.output_dir,
        expected_subtasks=options.expected_subtasks,
        report_file=options.report_file,
    )
    print(report.model_dump_json())


if __name__ == "__main__":
    main()
