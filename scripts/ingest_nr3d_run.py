#!/usr/bin/env python3
"""Ingest NR3D summary.json and optional indexed Codex JSONL[.gz] rollouts.

Trace-index paths resolve relative to the index file. An index must map every
sample exactly once to a distinct existing trace. Without an index, per-call
rows stay empty and sample tool counts stay NULL; turn usage is never expanded
into fabricated calls. NR3D headline metrics are queried from samples, not runs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

NonNegative = Annotated[int, Field(ge=0)]
Identifier = Annotated[str, Field(min_length=1)]
SqlRow = tuple[str | int | float | None, ...]
Metrics = tuple[int, float, float, float]
_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY, branch TEXT, commit_sha TEXT, answerer_model TEXT,
        judge_model TEXT, tools_enabled INTEGER, workers INTEGER, num_frames INTEGER,
        n INTEGER, n_judged INTEGER, mnas REAL, completion_rate REAL,
        cache_hit_rate REAL, mean_cache_ratio REAL, per_category_mnas TEXT,
        notes TEXT, ingested_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS samples (
        run_id TEXT, sample_id TEXT, scene_id TEXT, query TEXT, status TEXT,
        iou REAL, selected_object_id INTEGER, confidence REAL, error TEXT,
        reasoning_summary TEXT, turn_duration_ms INTEGER, input_tokens INTEGER,
        cached_input_tokens INTEGER, output_tokens INTEGER, turn_usage TEXT,
        tool_count INTEGER, PRIMARY KEY (run_id, sample_id))""",
    """CREATE TABLE IF NOT EXISTS tool_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, question_id TEXT,
        seq INTEGER, tool_name TEXT, tool_input TEXT, tool_response TEXT,
        created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS llm_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, question_id TEXT,
        seq INTEGER, prompt_tokens INTEGER, cached_tokens INTEGER,
        completion_tokens INTEGER, created_at TEXT)""",
)


class Record(BaseModel):
    """Strict JSON boundary; unrelated runner/rollout fields are ignored."""

    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)


class Usage(Record):
    input_tokens: NonNegative
    cached_input_tokens: NonNegative
    output_tokens: NonNegative
    total_tokens: NonNegative | None = None
    reasoning_output_tokens: NonNegative | None = None
    cache_write_input_tokens: NonNegative | None = None


class TurnUsage(Record):
    last: Usage | None = None
    total: Usage | None = None


class Sample(Record):
    sample_id: Identifier
    scene_id: str = ""
    query: str
    status: Identifier
    # Oriented-box intersection introduces tiny numerical overshoots at IoU=1.
    iou: float = Field(ge=0, le=1 + 1e-9)
    selected_object_id: int | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    error: str | None = None
    reasoning_summary: str | None = None
    turn_duration_ms: NonNegative | None = None
    input_tokens: NonNegative | None = None
    cached_input_tokens: NonNegative | None = None
    output_tokens: NonNegative | None = None
    turn_usage: TurnUsage | None = None


class Summary(Record):
    n: Annotated[int, Field(gt=0)]
    mean_iou: float = Field(ge=0, le=1 + 1e-9)
    acc_025: float = Field(alias="Acc@0.25", ge=0, le=1)
    acc_050: float = Field(alias="Acc@0.50", ge=0, le=1)
    cache_hit_rate: float = Field(ge=0, le=1)
    mean_cache_ratio: float = Field(ge=0, le=1)
    per_sample: Annotated[list[Sample], Field(min_length=1)]


class Options(Record):
    output_dir: Path
    run_id: Identifier
    branch: Identifier
    commit_sha: Identifier
    model: Identifier
    db: Path
    trace_index: Path | None = None
    notes: str = ""


class TraceIndexEntry(Record):
    sample_id: Identifier
    path: Identifier


class TraceEvent(Record):
    type: str
    payload: dict[str, JsonValue]
    timestamp: str | None = None


class FunctionCall(Record):
    call_id: Identifier
    name: Identifier
    arguments: str


class FunctionOutput(Record):
    call_id: Identifier
    output: JsonValue


class TokenInfo(Record):
    last_token_usage: Usage
    total_token_usage: dict[str, NonNegative] = Field(min_length=1)


def parse_trace(path: Path, sample_id: str) -> tuple[list[SqlRow], list[SqlRow]]:
    """Pair tools by call_id and emit usage once per cumulative fingerprint."""
    calls: dict[str, tuple[FunctionCall, str | None]] = {}
    outputs: dict[str, str] = {}
    llms: list[SqlRow] = []
    seen_totals: set[tuple[tuple[str, int], ...]] = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = TraceEvent.model_validate_json(line)
                kind = event.payload.get("type")
                if event.type == "response_item" and kind == "function_call":
                    call = FunctionCall.model_validate(event.payload)
                    if call.call_id in calls:
                        raise ValueError(f"duplicate call_id: {call.call_id}")
                    calls[call.call_id] = (call, event.timestamp)
                elif event.type == "response_item" and kind == "function_call_output":
                    output = FunctionOutput.model_validate(event.payload)
                    outputs[output.call_id] = (
                        output.output
                        if isinstance(output.output, str)
                        else json.dumps(output.output, ensure_ascii=False)
                    )
                elif event.type == "event_msg" and kind == "token_count":
                    raw_info = event.payload.get("info")
                    if raw_info is None:
                        continue
                    info = TokenInfo.model_validate(raw_info)
                    fingerprint = tuple(sorted(info.total_token_usage.items()))
                    if fingerprint not in seen_totals:
                        seen_totals.add(fingerprint)
                        usage = info.last_token_usage
                        llms.append(
                            (
                                sample_id,
                                len(llms) + 1,
                                usage.input_tokens,
                                usage.cached_input_tokens,
                                usage.output_tokens,
                                event.timestamp,
                            )
                        )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if outputs.keys() - calls.keys():
        raise ValueError(f"{path}: function_call_output without matching call_id")
    tools: list[SqlRow] = [
        (
            sample_id,
            seq,
            call.name,
            call.arguments,
            outputs.get(call.call_id),
            timestamp,
        )
        for seq, (call, timestamp) in enumerate(calls.values(), 1)
    ]
    return tools, llms


def load_traces(
    options: Options, sample_ids: set[str]
) -> dict[str, tuple[list[SqlRow], list[SqlRow]]]:
    """Validate the complete index and all traces before touching SQLite."""
    if options.trace_index is None:
        return {}
    entries = TypeAdapter(list[TraceIndexEntry]).validate_json(
        options.trace_index.read_text(encoding="utf-8")
    )
    if len(entries) != len(sample_ids) or {e.sample_id for e in entries} != sample_ids:
        raise ValueError(
            "trace-index sample_ids must match summary samples exactly once"
        )
    paths = [(options.trace_index.parent / entry.path).resolve() for entry in entries]
    if len(set(paths)) != len(paths):
        raise ValueError("trace-index paths must be distinct per sample_id")
    return {
        entry.sample_id: parse_trace(path, entry.sample_id)
        for entry, path in zip(entries, paths)
    }


def ingest_run(options: Options) -> Metrics:
    """Validate all files, then atomically replace only the requested run."""
    summary = Summary.model_validate_json(
        (options.output_dir / "summary.json").read_text(encoding="utf-8")
    )
    sample_ids = {sample.sample_id for sample in summary.per_sample}
    if summary.n != len(summary.per_sample) or len(sample_ids) != summary.n:
        raise ValueError(
            "summary.n must match the number of unique per_sample sample_ids"
        )
    traces = load_traces(options, sample_ids)
    options.db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(options.db)
    try:
        with connection:
            connection.execute("BEGIN")
            for ddl in _SCHEMA:
                connection.execute(ddl)
            for table in ("tool_calls", "llm_calls", "samples", "runs"):
                connection.execute(
                    f"DELETE FROM {table} WHERE run_id = ?", (options.run_id,)
                )
            connection.execute(
                """INSERT INTO runs (run_id, branch, commit_sha, answerer_model, n,
                    completion_rate, cache_hit_rate, mean_cache_ratio, notes, ingested_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    options.run_id,
                    options.branch,
                    options.commit_sha,
                    options.model,
                    summary.n,
                    sum(s.status == "completed" for s in summary.per_sample)
                    / summary.n,
                    summary.cache_hit_rate,
                    summary.mean_cache_ratio,
                    options.notes,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
            for sample in summary.per_sample:
                tools, llms = traces.get(sample.sample_id, ([], []))
                usage = sample.turn_usage.last if sample.turn_usage else None
                # Column names come only from the closed, validated Sample model.
                values: dict[str, str | int | float | None] = sample.model_dump(
                    exclude={"turn_usage"}
                )
                values.update(
                    run_id=options.run_id,
                    scene_id=sample.scene_id or sample.sample_id.split("::", 1)[0],
                    output_tokens=(
                        sample.output_tokens
                        if sample.output_tokens is not None
                        else (usage.output_tokens if usage else None)
                    ),
                    turn_usage=(
                        sample.turn_usage.model_dump_json()
                        if sample.turn_usage
                        else None
                    ),
                    tool_count=len(tools) if sample.sample_id in traces else None,
                )
                columns = ",".join(values)
                placeholders = ",".join("?" for _ in values)
                connection.execute(
                    f"INSERT INTO samples ({columns}) VALUES ({placeholders})",
                    tuple(values.values()),
                )
                connection.executemany(
                    """INSERT INTO tool_calls (run_id, question_id, seq, tool_name,
                        tool_input, tool_response, created_at) VALUES (?,?,?,?,?,?,?)""",
                    [(options.run_id, *row) for row in tools],
                )
                connection.executemany(
                    """INSERT INTO llm_calls (run_id, question_id, seq, prompt_tokens,
                        cached_tokens, completion_tokens, created_at) VALUES (?,?,?,?,?,?,?)""",
                    [(options.run_id, *row) for row in llms],
                )
            metrics = TypeAdapter(Metrics).validate_python(
                connection.execute(
                    """SELECT count(*), avg(iou), avg(iou >= 0.25), avg(iou >= 0.50)
                    FROM samples WHERE run_id = ?""",
                    (options.run_id,),
                ).fetchone()
            )
    finally:
        connection.close()
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    for flag in ("run-id", "branch", "commit", "model"):
        parser.add_argument(
            f"--{flag}",
            dest="commit_sha" if flag == "commit" else flag.replace("-", "_"),
            required=True,
        )
    parser.add_argument(
        "--db", type=Path, default=Path("docs/benchmark/nr3d/runs.sqlite")
    )
    parser.add_argument("--trace-index", type=Path)
    parser.add_argument("--notes", default="")
    try:
        options = Options.model_validate(vars(parser.parse_args(argv)))
        n, mean_iou, acc_025, acc_050 = ingest_run(options)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "run_id": options.run_id,
                "db": str(options.db),
                "n": n,
                "mean_iou": mean_iou,
                "Acc@0.25": acc_025,
                "Acc@0.50": acc_050,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
