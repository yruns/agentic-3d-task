#!/usr/bin/env python3
"""Ingest one OpenEQA evaluation run into the per-benchmark SQLite database.

This is the canonical OpenEQA ingester referenced by ``CLAUDE.md`` §"Mandatory:
SQLite ingestion of per-run logs". It reads the durable artifacts produced by
:mod:`codex_agent.evaluation.openeqa_runner` (``summary.json`` plus the
``per_sample/*.json`` checkpoints) and writes them into the four canonical
tables ``runs`` / ``samples`` / ``tool_calls`` / ``llm_calls`` so per-question
regression analysis (``Δ`` vs a prior run, per category, with traces) is
possible without re-running the eval.

The schema mirrors the OpenEQA process doc: ``runs`` (one row per run),
``samples`` (one row per question), and ``tool_calls`` / ``llm_calls`` (one row
per tool invocation / chat-completion). Per-call tables are created and ready
for population once per-call traces are captured (see ``CLAUDE.md`` §"Per-LLM-call
durability"); the runner does not yet emit per-question tool/LLM traces, so for
runs without traces those tables stay empty and ``samples.tool_count`` is NULL.

Ingestion is idempotent: re-running for the same ``--run-id`` replaces that
run's rows (so a re-judge or rerun overwrites cleanly).

Example::

    python scripts/ingest_openeqa_run.py \\
        --output-dir tmp/openeqa_eval_v3_tools_full1079_20260612/ \\
        --run-id v3_tools_full1079_20260612 \\
        --branch feat/openeqa-codex-task \\
        --commit 06f73cd \\
        --judge-model gemini-2.5-pro \\
        --answerer-model gpt-5.4-2026-03-05 \\
        --tools-enabled --workers 30 --num-frames 8 \\
        --notes "first full ScanNet split, tools+BEV" \\
        --db docs/benchmark/openeqa/runs.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_MIN_SCORE = 1
_MAX_SCORE = 5

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    branch            TEXT,
    commit_sha        TEXT,
    answerer_model    TEXT,
    judge_model       TEXT,
    tools_enabled     INTEGER,
    workers           INTEGER,
    num_frames        INTEGER,
    n                 INTEGER,
    n_judged          INTEGER,
    mnas              REAL,
    completion_rate   REAL,
    cache_hit_rate    REAL,
    mean_cache_ratio  REAL,
    per_category_mnas TEXT,
    notes             TEXT,
    ingested_at       TEXT
)
"""

_SAMPLES_DDL = """
CREATE TABLE IF NOT EXISTS samples (
    run_id              TEXT,
    question_id         TEXT,
    clip_id             TEXT,
    scene_id            TEXT,
    category            TEXT,
    status              TEXT,
    question            TEXT,
    gt_answer           TEXT,
    prediction          TEXT,
    judge_score         INTEGER,
    mnas                REAL,
    confidence          REAL,
    num_frames          INTEGER,
    reasoning_summary   TEXT,
    error               TEXT,
    turn_duration_ms    INTEGER,
    input_tokens        INTEGER,
    cached_input_tokens INTEGER,
    tool_count          INTEGER,
    PRIMARY KEY (run_id, question_id)
)
"""

_TOOL_CALLS_DDL = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    question_id   TEXT,
    seq           INTEGER,
    tool_name     TEXT,
    tool_input    TEXT,
    tool_response TEXT,
    created_at    TEXT
)
"""

_LLM_CALLS_DDL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    question_id       TEXT,
    seq               INTEGER,
    prompt_tokens     INTEGER,
    cached_tokens     INTEGER,
    completion_tokens INTEGER,
    created_at        TEXT
)
"""


class SampleRecord(BaseModel):
    """One per-question checkpoint, validated at the file boundary."""

    model_config = ConfigDict(extra="ignore")

    question_id: str
    clip_id: str = ""
    scene_id: str = ""
    category: str = "unknown"
    status: str
    question: str = ""
    gt_answer: str = ""
    prediction: str = ""
    judge_score: int | None = None
    mnas: float | None = None
    confidence: float | None = None
    num_frames: int = 0
    reasoning_summary: str | None = None
    error: str | None = None
    turn_duration_ms: int | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None


class RunSummary(BaseModel):
    """The aggregate ``summary.json``, validated at the file boundary."""

    model_config = ConfigDict(extra="ignore")

    n: int = 0
    n_judged: int = 0
    mnas: float = Field(default=0.0, alias="MNAS")
    per_category_mnas: dict[str, float] = Field(
        default_factory=dict, alias="per_category_MNAS"
    )
    completion_rate: float = 0.0
    cache_hit_rate: float = 0.0
    mean_cache_ratio: float = 0.0


class RunMetadata(BaseModel):
    """Run-level provenance supplied on the command line."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    branch: str
    commit_sha: str
    answerer_model: str
    judge_model: str
    tools_enabled: bool
    workers: int
    num_frames: int
    notes: str


def score_to_mnas(score: int) -> float:
    """Map a 1-5 judge score to the 0-100 MNAS scale (clipping out-of-range)."""
    clipped = max(_MIN_SCORE, min(_MAX_SCORE, score))
    return 100.0 * (clipped - _MIN_SCORE) / (_MAX_SCORE - _MIN_SCORE)


def load_samples(output_dir: Path) -> tuple[SampleRecord, ...]:
    """Load and validate every per-question checkpoint under ``output_dir``."""
    per_sample = output_dir / "per_sample"
    if not per_sample.is_dir():
        raise FileNotFoundError(f"no per_sample/ checkpoints under {output_dir}")
    records: list[SampleRecord] = []
    for path in sorted(per_sample.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"checkpoint must be a JSON object: {path}")
        records.append(SampleRecord.model_validate(payload))
    if not records:
        raise FileNotFoundError(f"no *.json checkpoints under {per_sample}")
    return tuple(records)


def load_summary(output_dir: Path, samples: Sequence[SampleRecord]) -> RunSummary:
    """Load ``summary.json`` if present, else derive a summary from samples."""
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"summary.json must be a JSON object: {summary_path}")
        return RunSummary.model_validate(payload)
    return _derive_summary(samples)


def load_from_summary(
    summary_path: Path,
) -> tuple[RunSummary, tuple[SampleRecord, ...]]:
    """Load aggregate + per-sample rows from a single summary JSON.

    This is the durable-archive path: a committed ``*_summary.json`` that embeds
    the ``per_sample`` array (see ``docs/benchmark/openeqa/assets/``) can rebuild
    the SQLite DB after the ephemeral ``tmp/`` run dir is gone.
    """
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {summary_path}")
    rows = payload.get("per_sample")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"{summary_path} has no 'per_sample' array to ingest; point "
            "--output-dir at the raw run dir instead"
        )
    samples = tuple(SampleRecord.model_validate(row) for row in rows)
    return RunSummary.model_validate(payload), samples


def _derive_summary(samples: Sequence[SampleRecord]) -> RunSummary:
    judged = [s for s in samples if s.judge_score is not None]
    mnas_values = [
        score_to_mnas(s.judge_score) for s in judged if s.judge_score is not None
    ]
    per_category: dict[str, list[float]] = {}
    for sample in judged:
        if sample.judge_score is None:
            continue
        per_category.setdefault(sample.category, []).append(
            score_to_mnas(sample.judge_score)
        )
    completed = sum(1 for s in samples if s.status == "completed")
    n = len(samples)
    return RunSummary(
        n=n,
        n_judged=len(judged),
        MNAS=(sum(mnas_values) / len(mnas_values)) if mnas_values else 0.0,
        per_category_MNAS={
            category: sum(values) / len(values)
            for category, values in sorted(per_category.items())
        },
        completion_rate=(completed / n) if n else 0.0,
    )


def ingest_run(
    *,
    db_path: Path,
    metadata: RunMetadata,
    output_dir: Path | None = None,
    from_summary: Path | None = None,
) -> RunSummary:
    """Ingest one run's artifacts into ``db_path`` (idempotent per run_id).

    Provide exactly one source: ``output_dir`` (a raw run dir with
    ``per_sample/*.json``) or ``from_summary`` (a committed summary JSON whose
    ``per_sample`` array embeds the rows).
    """
    if (output_dir is None) == (from_summary is None):
        raise ValueError("provide exactly one of output_dir / from_summary")
    if from_summary is not None:
        summary, samples = load_from_summary(from_summary)
    else:
        assert output_dir is not None  # narrowed by the xor check above
        samples = load_samples(output_dir)
        summary = load_summary(output_dir, samples)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        _ensure_schema(connection)
        _replace_run(connection, metadata=metadata, summary=summary)
        _replace_samples(connection, run_id=metadata.run_id, samples=samples)
        connection.commit()
    finally:
        connection.close()
    return summary


def _ensure_schema(connection: sqlite3.Connection) -> None:
    for ddl in (_RUNS_DDL, _SAMPLES_DDL, _TOOL_CALLS_DDL, _LLM_CALLS_DDL):
        connection.execute(ddl)


def _replace_run(
    connection: sqlite3.Connection,
    *,
    metadata: RunMetadata,
    summary: RunSummary,
) -> None:
    connection.execute("DELETE FROM runs WHERE run_id = ?", (metadata.run_id,))
    connection.execute(
        """
        INSERT INTO runs (
            run_id, branch, commit_sha, answerer_model, judge_model,
            tools_enabled, workers, num_frames, n, n_judged, mnas,
            completion_rate, cache_hit_rate, mean_cache_ratio,
            per_category_mnas, notes, ingested_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            metadata.run_id,
            metadata.branch,
            metadata.commit_sha,
            metadata.answerer_model,
            metadata.judge_model,
            int(metadata.tools_enabled),
            metadata.workers,
            metadata.num_frames,
            summary.n,
            summary.n_judged,
            summary.mnas,
            summary.completion_rate,
            summary.cache_hit_rate,
            summary.mean_cache_ratio,
            json.dumps(summary.per_category_mnas, ensure_ascii=False),
            metadata.notes,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )


def _replace_samples(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    samples: Sequence[SampleRecord],
) -> None:
    connection.execute("DELETE FROM samples WHERE run_id = ?", (run_id,))
    connection.executemany(
        """
        INSERT INTO samples (
            run_id, question_id, clip_id, scene_id, category, status,
            question, gt_answer, prediction, judge_score, mnas, confidence,
            num_frames, reasoning_summary, error, turn_duration_ms,
            input_tokens, cached_input_tokens, tool_count
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                run_id,
                s.question_id,
                s.clip_id,
                s.scene_id,
                s.category,
                s.status,
                s.question,
                s.gt_answer,
                s.prediction,
                s.judge_score,
                s.mnas,
                s.confidence,
                s.num_frames,
                s.reasoning_summary,
                s.error,
                s.turn_duration_ms,
                s.input_tokens,
                s.cached_input_tokens,
                None,
            )
            for s in samples
        ],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Raw run dir with per_sample/*.json (+ summary.json). Mutually "
        "exclusive with --from-summary.",
    )
    parser.add_argument(
        "--from-summary",
        type=Path,
        default=None,
        help="A committed *_summary.json whose 'per_sample' array embeds the "
        "rows (durable-archive rebuild). Mutually exclusive with --output-dir.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True, dest="commit_sha")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--answerer-model",
        default="",
        help="The Codex answerer model (for run provenance).",
    )
    parser.add_argument("--tools-enabled", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=0)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("docs/benchmark/openeqa/runs.sqlite"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if (args.output_dir is None) == (args.from_summary is None):
        parser.error("provide exactly one of --output-dir / --from-summary")
    metadata = RunMetadata(
        run_id=args.run_id,
        branch=args.branch,
        commit_sha=args.commit_sha,
        answerer_model=args.answerer_model,
        judge_model=args.judge_model,
        tools_enabled=args.tools_enabled,
        workers=args.workers,
        num_frames=args.num_frames,
        notes=args.notes,
    )
    summary = ingest_run(
        db_path=args.db,
        metadata=metadata,
        output_dir=args.output_dir,
        from_summary=args.from_summary,
    )
    print(
        json.dumps(
            {
                "run_id": metadata.run_id,
                "db": str(args.db),
                "n": summary.n,
                "n_judged": summary.n_judged,
                "MNAS": round(summary.mnas, 2),
                "completion_rate": round(summary.completion_rate, 4),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
