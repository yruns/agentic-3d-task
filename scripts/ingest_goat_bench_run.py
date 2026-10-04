#!/usr/bin/env python3
"""Ingest validated SCOPE distance metrics into the canonical GOAT-Bench database.

SR and SPL are reported as percentages over recorded terminal subtasks. Missing
subtasks affect coverage, never the observed success denominator. JSON metrics
are preferred; upstream numeric pickle dictionaries are decoded with a narrow
global allowlist that never imports or executes arbitrary pickle globals.

``subtask_metadata.json`` maps IDs to task_type/answer/scene_id/episode_id
(question is optional). Grouped success_by_task lists cannot identify individual
subtasks and are deliberately not used. ``expected_subtask_ids.json``, when
present, freezes the requested IDs. Tool and LLM call tables remain empty because
the upstream artifacts do not contain per-subtask call traces.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sqlite3
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
NonemptyText = Annotated[str, StringConstraints(strict=True, min_length=1)]
GoalType = Literal["object", "description", "image"]
_DtypeState = tuple[int, Literal["<", ">", "=", "|"], None, None, None, int, int, int]
_DTYPE_STATE_ADAPTER: TypeAdapter[_DtypeState] = TypeAdapter(_DtypeState)
_NUMERIC_FORMATS = {"f8": "d", "f4": "f", "i8": "q", "i4": "i", "b1": "?"}

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, branch TEXT, commit_sha TEXT, answerer_model TEXT,
    judge_model TEXT, tools_enabled INTEGER, workers INTEGER, num_frames INTEGER,
    n INTEGER, n_judged INTEGER, mnas REAL, completion_rate REAL,
    cache_hit_rate REAL, mean_cache_ratio REAL, per_category_mnas TEXT,
    notes TEXT, ingested_at TEXT
)
"""
_SAMPLES_DDL = """
CREATE TABLE IF NOT EXISTS samples (
    run_id TEXT, question_id TEXT, scene_id TEXT, episode_id TEXT, task_type TEXT,
    status TEXT, question TEXT, gt_answer TEXT, success REAL, spl REAL,
    input_tokens INTEGER, cached_input_tokens INTEGER, tool_count INTEGER,
    PRIMARY KEY (run_id, question_id)
)
"""
_TOOL_CALLS_DDL = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, question_id TEXT,
    seq INTEGER, tool_name TEXT, tool_input TEXT, tool_response TEXT, created_at TEXT
)
"""
_LLM_CALLS_DDL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, question_id TEXT,
    seq INTEGER, prompt_tokens INTEGER, cached_tokens INTEGER,
    completion_tokens INTEGER, created_at TEXT
)
"""
_RUN_EXTRA_COLUMNS = (
    ("runtime_commit_sha", "TEXT"),
    ("expected_subtasks", "INTEGER"),
    ("coverage_complete", "INTEGER"),
    ("expected_ids_verified", "INTEGER"),
    ("modality_complete", "INTEGER"),
    ("sr", "REAL"),
    ("spl", "REAL"),
    ("per_modality_metrics", "TEXT"),
    ("api_ledger_present", "INTEGER"),
)


@dataclass(frozen=True)
class _NumericDtype:
    """Inert substitute for the limited NumPy dtype used by scalar pickles."""

    code: str
    align: bool = False
    copy: bool = False
    byte_order: str = "="

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or self.code not in _NUMERIC_FORMATS:
            raise ValueError(f"forbidden pickle dtype: {self.code!r}")
        if type(self.align) is not bool or type(self.copy) is not bool:
            raise ValueError("forbidden pickle dtype flags")
        if self.byte_order not in {"<", ">", "=", "|"}:
            raise ValueError("forbidden pickle byte order")

    def __setstate__(self, state: object) -> None:
        version, order, _, _, _, size, alignment, flags = (
            _DTYPE_STATE_ADAPTER.validate_python(state, strict=True)
        )
        if (version, size, alignment, flags) != (3, -1, -1, 0):
            raise ValueError("forbidden pickle dtype state")
        object.__setattr__(self, "byte_order", "=" if order == "|" else order)


def _numeric_scalar(dtype: _NumericDtype, buffer: bytes) -> float:
    if not isinstance(dtype, _NumericDtype) or not isinstance(buffer, bytes):
        raise ValueError("forbidden pickle scalar arguments")
    binary_format = dtype.byte_order + _NUMERIC_FORMATS[dtype.code]
    if len(buffer) != struct.calcsize(binary_format):
        raise ValueError("invalid pickle scalar byte length")
    return float(struct.unpack(binary_format, buffer)[0])


class _MetricUnpickler(pickle.Unpickler):
    """Decode plain containers and inert numeric scalars, rejecting other globals."""

    def find_class(self, module: str, name: str) -> object:
        if (module, name) == ("numpy", "dtype"):
            return _NumericDtype
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"}:
            if name == "scalar":
                return _numeric_scalar
        raise ValueError(f"forbidden pickle global: {module}.{name}")


def _validate_numeric_metric(value: object) -> float:
    if not isinstance(value, (bool, int, float)):
        raise ValueError("metric must be a numeric or boolean scalar")
    return float(value)


UnitMetric = Annotated[
    float,
    BeforeValidator(_validate_numeric_metric),
    Field(ge=0, le=1, allow_inf_nan=False),
]


class MetricMapping(RootModel[dict[NonemptyText, UnitMetric]]):
    """Validated ID-to-scalar metric artifact."""


class TaskMetadata(BaseModel):
    """Optional per-subtask facts captured from the original episode tasks."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    task_type: GoalType
    answer: str = ""
    question: str = ""
    scene_id: str = ""
    episode_id: str = ""


class TaskMetadataMapping(RootModel[dict[NonemptyText, TaskMetadata]]):
    """Metadata validated before attaching modality information to metrics."""


class ExpectedIds(RootModel[tuple[NonemptyText, ...]]):
    """Frozen requested IDs, with duplicates rejected."""

    @model_validator(mode="after")
    def reject_duplicates(self) -> ExpectedIds:
        if len(set(self.root)) != len(self.root):
            raise ValueError("expected subtask IDs must be unique")
        return self


class SampleRecord(BaseModel):
    """One terminal subtask, with matching distance-based SR and SPL."""

    model_config = ConfigDict(frozen=True)
    question_id: NonemptyText
    scene_id: str
    episode_id: str
    task_type: GoalType | None = None
    question: str = ""
    answer: str = ""
    success: UnitMetric
    spl: UnitMetric

    @model_validator(mode="after")
    def validate_success_and_spl(self) -> SampleRecord:
        if self.success not in (0.0, 1.0):
            raise ValueError(f"success must be binary: {self.question_id}")
        if self.success == 0.0 and self.spl != 0.0:
            raise ValueError(f"failed subtask must have zero SPL: {self.question_id}")
        return self


class RunMetadata(BaseModel):
    """Run provenance and requested coverage, validated at the CLI boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: NonemptyText
    branch: NonemptyText
    commit_sha: NonemptyText
    runtime_commit_sha: NonemptyText
    model: NonemptyText
    expected_subtasks: int = Field(gt=0, strict=True)
    notes: str = ""


class ModalitySummary(BaseModel):
    """Observed sample count and percentage metrics for one modality."""

    model_config = ConfigDict(frozen=True)
    n: int
    sr: float
    spl: float


class RunSummary(BaseModel):
    """Observed metrics and explicit coverage; absent measurements stay null."""

    model_config = ConfigDict(frozen=True)
    n: int
    expected_subtasks: int
    sr: float | None
    spl: float | None
    completion_rate: float
    complete: bool
    expected_ids_verified: bool
    modality_complete: bool
    per_modality: dict[str, ModalitySummary]
    api_ledger_present: bool


def _read_payload(path: Path) -> object:
    if path.stat().st_size > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"metric artifact exceeds {_MAX_ARTIFACT_BYTES} bytes: {path}")
    if path.suffix == ".json":
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        return payload
    stream = io.BytesIO(path.read_bytes())
    try:
        payload = _MetricUnpickler(stream).load()
    except (pickle.UnpicklingError, EOFError, TypeError, struct.error) as error:
        raise ValueError(f"invalid numeric pickle artifact: {path}") from error
    if stream.read(1):
        raise ValueError(f"trailing bytes after pickle artifact: {path}")
    return payload


def _load_metrics(output_dir: Path, metric_name: str) -> dict[str, float]:
    paths_by_stem: dict[str, Path] = {}
    for suffix in ("json", "pkl"):
        for path in sorted(output_dir.glob(f"{metric_name}*.{suffix}")):
            if path.stem == metric_name or path.stem.startswith(metric_name + "_"):
                paths_by_stem.setdefault(path.stem, path)
    if not paths_by_stem:
        raise FileNotFoundError(
            f"no {metric_name} JSON/pickle metrics under {output_dir}"
        )
    metrics: dict[str, float] = {}
    for path in paths_by_stem.values():
        artifact = MetricMapping.model_validate(_read_payload(path))
        for subtask_id, value in artifact.root.items():
            if subtask_id in metrics and metrics[subtask_id] != value:
                raise ValueError(f"conflicting {metric_name} for {subtask_id}: {path}")
            metrics[subtask_id] = value
    return metrics


def load_samples(output_dir: Path) -> tuple[SampleRecord, ...]:
    """Validate terminal metrics and attach only explicitly keyed task metadata."""
    success = _load_metrics(output_dir, "success_by_distance")
    spl = _load_metrics(output_dir, "spl_by_distance")
    if success.keys() != spl.keys():
        raise ValueError("distance success and SPL must contain the same subtask IDs")
    metadata_path = output_dir / "subtask_metadata.json"
    metadata = (
        TaskMetadataMapping.model_validate(_read_payload(metadata_path)).root
        if metadata_path.is_file()
        else {}
    )
    samples: list[SampleRecord] = []
    for subtask_id in sorted(success):
        id_parts = subtask_id.rsplit("_", 2)
        if len(id_parts) != 3 or not all(id_parts) or not id_parts[-1].isdigit():
            raise ValueError(f"invalid GOAT subtask ID: {subtask_id}")
        task = metadata.get(subtask_id)
        if task and (
            (task.scene_id and task.scene_id != id_parts[0])
            or (task.episode_id and task.episode_id != id_parts[1])
        ):
            raise ValueError(f"metadata scene/episode does not match ID: {subtask_id}")
        samples.append(
            SampleRecord(
                question_id=subtask_id,
                scene_id=task.scene_id if task and task.scene_id else id_parts[0],
                episode_id=task.episode_id if task and task.episode_id else id_parts[1],
                task_type=task.task_type if task else None,
                question=task.question if task else "",
                answer=task.answer if task else "",
                success=success[subtask_id],
                spl=spl[subtask_id],
            )
        )
    return tuple(samples)


def _summarize(
    output_dir: Path, samples: Sequence[SampleRecord], expected_subtasks: int
) -> RunSummary:
    if len(samples) > expected_subtasks:
        raise ValueError("recorded terminal subtasks exceed expected_subtasks")
    expected_path = output_dir / "expected_subtask_ids.json"
    if expected_path.is_file():
        expected_ids = ExpectedIds.model_validate(_read_payload(expected_path)).root
        if len(expected_ids) != expected_subtasks:
            raise ValueError("expected ID count does not match expected_subtasks")
        unexpected = {sample.question_id for sample in samples} - set(expected_ids)
        if unexpected:
            raise ValueError(f"unexpected terminal subtask IDs: {sorted(unexpected)}")
    by_modality: dict[str, list[SampleRecord]] = {}
    for sample in samples:
        by_modality.setdefault(sample.task_type or "unknown", []).append(sample)
    return RunSummary(
        n=len(samples),
        expected_subtasks=expected_subtasks,
        sr=(
            100 * sum(sample.success for sample in samples) / len(samples)
            if samples
            else None
        ),
        spl=(
            100 * sum(sample.spl for sample in samples) / len(samples)
            if samples
            else None
        ),
        completion_rate=len(samples) / expected_subtasks,
        complete=len(samples) == expected_subtasks,
        expected_ids_verified=expected_path.is_file(),
        modality_complete=bool(samples)
        and all(s.task_type is not None for s in samples),
        per_modality={
            modality: ModalitySummary(
                n=len(rows),
                sr=100 * sum(row.success for row in rows) / len(rows),
                spl=100 * sum(row.spl for row in rows) / len(rows),
            )
            for modality, rows in sorted(by_modality.items())
        },
        api_ledger_present=(output_dir / "api_ledger.jsonl").is_file(),
    )


def _ensure_schema(connection: sqlite3.Connection) -> None:
    for ddl in (_RUNS_DDL, _SAMPLES_DDL, _TOOL_CALLS_DDL, _LLM_CALLS_DDL):
        connection.execute(ddl)
    existing_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(runs)")
    }
    for name, column_type in _RUN_EXTRA_COLUMNS:
        if name not in existing_columns:
            connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {column_type}")


def _replace_run(
    connection: sqlite3.Connection,
    metadata: RunMetadata,
    summary: RunSummary,
    samples: Sequence[SampleRecord],
) -> None:
    for table in ("runs", "samples", "tool_calls", "llm_calls"):
        connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (metadata.run_id,))
    connection.execute(
        """INSERT INTO runs (
            run_id, branch, commit_sha, answerer_model, n, completion_rate,
            notes, ingested_at, runtime_commit_sha, expected_subtasks,
            coverage_complete, expected_ids_verified, modality_complete,
            sr, spl, per_modality_metrics, api_ledger_present
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            metadata.run_id,
            metadata.branch,
            metadata.commit_sha,
            metadata.model,
            summary.n,
            summary.completion_rate,
            metadata.notes,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            metadata.runtime_commit_sha,
            metadata.expected_subtasks,
            int(summary.complete),
            int(summary.expected_ids_verified),
            int(summary.modality_complete),
            summary.sr,
            summary.spl,
            json.dumps(
                {key: value.model_dump() for key, value in summary.per_modality.items()}
            ),
            int(summary.api_ledger_present),
        ),
    )
    connection.executemany(
        """INSERT INTO samples (
            run_id, question_id, scene_id, episode_id, task_type,
            status, question, gt_answer, success, spl
        ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                metadata.run_id,
                sample.question_id,
                sample.scene_id,
                sample.episode_id,
                sample.task_type,
                "completed",
                sample.question,
                sample.answer,
                sample.success,
                sample.spl,
            )
            for sample in samples
        ],
    )


def ingest_run(*, output_dir: Path, db_path: Path, metadata: RunMetadata) -> RunSummary:
    """Atomically replace one run after validating all artifacts and coverage."""
    samples = load_samples(output_dir)
    summary = _summarize(output_dir, samples, metadata.expected_subtasks)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        with connection:
            _ensure_schema(connection)
            _replace_run(connection, metadata, summary, samples)
    finally:
        connection.close()
    return summary


class _CliOptions(RunMetadata):
    output_dir: Path
    db: Path


def build_arg_parser() -> argparse.ArgumentParser:
    """Expose paths, provenance, and expected coverage without business logic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True, dest="commit_sha")
    parser.add_argument("--runtime-commit", required=True, dest="runtime_commit_sha")
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-subtasks", type=int, required=True)
    parser.add_argument(
        "--db", type=Path, default=Path("docs/benchmark/goat_bench/runs.sqlite")
    )
    parser.add_argument("--notes", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate CLI options, invoke ingestion, and display the measured summary."""
    options = _CliOptions.model_validate(vars(build_arg_parser().parse_args(argv)))
    summary = ingest_run(
        output_dir=options.output_dir, db_path=options.db, metadata=options
    )
    print(summary.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
