"""Offline subprocess coverage for canonical NR3D SQLite ingestion."""

from __future__ import annotations

import gzip
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "ingest_nr3d_run.py"
SAMPLE_ID = "scannet/scene0000_00::1::1"
USAGE = {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 7}


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_summary(
    root: Path, overrides: dict[str, object] | None = None, *, iou: float = 1
) -> None:
    sample = {
        "sample_id": SAMPLE_ID,
        "query": "chair",
        "status": "completed",
        "iou": iou,
        "input_tokens": 100,
        "cached_input_tokens": 40,
        "turn_duration_ms": 12,
        "turn_usage": {"last": USAGE, "total": USAGE},
    }
    # Deliberately wrong aggregates: headline metrics must come from sample rows.
    summary = {
        "n": 1,
        "mean_iou": 0.01,
        "Acc@0.25": 0.01,
        "Acc@0.50": 0.01,
        "cache_hit_rate": 1,
        "mean_cache_ratio": 0.4,
        "per_sample": [sample],
    }
    summary.update(overrides or {})
    write_json(root / "summary.json", summary)


def ingest(
    root: Path, *extra: str, run_id: str = "run"
) -> subprocess.CompletedProcess[str]:
    args = ["--output-dir", str(root), "--run-id", run_id]
    args += ["--branch", "test", "--commit", "abc123", "--model", "offline"]
    args += ["--db", str(root / "db" / "runs.sqlite"), *extra]
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_sample_metrics_and_idempotent_replacement(tmp_path: Path) -> None:
    write_summary(tmp_path, iou=1.0000000000000002)
    for run_id in ("run", "other", "run"):
        result = ingest(tmp_path, run_id=run_id)
        assert result.returncode == 0, result.stderr
    with sqlite3.connect(tmp_path / "db" / "runs.sqlite") as db:
        assert db.execute("SELECT count(*) FROM samples").fetchone() == (2,)
        metrics = db.execute(
            "SELECT count(*), avg(iou), avg(iou>=0.25), avg(iou>=0.50) FROM samples WHERE run_id='run'"
        ).fetchone()
        assert metrics == pytest.approx((1, 1, 1, 1))
        assert db.execute(
            "SELECT answerer_model,n,completion_rate,mnas FROM runs WHERE run_id='run'"
        ).fetchone() == ("offline", 1, 1, None)
        assert db.execute("SELECT count(*) FROM llm_calls").fetchone() == (0,)
        assert db.execute(
            "SELECT tool_count,output_tokens FROM samples WHERE run_id='run'"
        ).fetchone() == (None, 7)
    write_summary(
        tmp_path,
        {
            "per_sample": [
                {
                    "sample_id": SAMPLE_ID,
                    "query": "chair",
                    "status": "failed",
                    "iou": 0,
                    "error": "timeout",
                }
            ]
        },
    )
    assert ingest(tmp_path).returncode == 0
    with sqlite3.connect(tmp_path / "db" / "runs.sqlite") as db:
        assert db.execute(
            "SELECT iou,error FROM samples WHERE run_id='run'"
        ).fetchone() == (0, "timeout")
        assert db.execute(
            "SELECT iou FROM samples WHERE run_id='other'"
        ).fetchone() == pytest.approx((1,))


def event(kind: str, payload_type: str, **fields: object) -> dict[str, object]:
    return {"type": kind, "payload": {"type": payload_type, **fields}}


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz"])
def test_trace_pairing_and_token_deduplication(tmp_path: Path, suffix: str) -> None:
    write_summary(tmp_path)
    token = event(
        "event_msg",
        "token_count",
        info={"last_token_usage": USAGE, "total_token_usage": USAGE},
    )
    events = [
        event("session_meta", "unrelated"),
        event(
            "response_item",
            "function_call",
            call_id="a",
            name="inspect",
            arguments='{"id": 1}',
        ),
        event(
            "response_item", "function_call", call_id="b", name="locate", arguments="{}"
        ),
        event("response_item", "function_call_output", call_id="b", output="second"),
        event("response_item", "function_call_output", call_id="a", output="first"),
        event("event_msg", "token_count", info=None),
        token,
        token,
        event(
            "event_msg",
            "token_count",
            info={
                "last_token_usage": USAGE,
                "total_token_usage": {**USAGE, "input_tokens": 200},
            },
        ),
        token,
    ]
    trace = tmp_path / f"rollout{suffix}"
    content = "\n".join(json.dumps(row) for row in events)
    if suffix.endswith(".gz"):
        with gzip.open(trace, "wt", encoding="utf-8") as stream:
            stream.write(content)
    else:
        trace.write_text(content, encoding="utf-8")
    index = write_json(
        tmp_path / "index.json", [{"sample_id": SAMPLE_ID, "path": trace.name}]
    )
    for _ in range(2):
        result = ingest(tmp_path, "--trace-index", str(index))
        assert result.returncode == 0, result.stderr
    with sqlite3.connect(tmp_path / "db" / "runs.sqlite") as db:
        assert db.execute(
            "SELECT seq,tool_name,tool_input,tool_response FROM tool_calls ORDER BY seq"
        ).fetchall() == [
            (1, "inspect", '{"id": 1}', "first"),
            (2, "locate", "{}", "second"),
        ]
        assert db.execute(
            "SELECT seq,prompt_tokens,cached_tokens,completion_tokens FROM llm_calls ORDER BY seq"
        ).fetchall() == [(1, 100, 40, 7), (2, 100, 40, 7)]
        assert db.execute("SELECT tool_count FROM samples").fetchone() == (2,)
    assert ingest(tmp_path).returncode == 0
    with sqlite3.connect(tmp_path / "db" / "runs.sqlite") as db:
        assert db.execute("SELECT count(*) FROM tool_calls").fetchone() == (0,)
        assert db.execute("SELECT count(*) FROM llm_calls").fetchone() == (0,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n": 2},
        {"n": "1"},
        {"per_sample": []},
        {"cache_hit_rate": -1},
        {"per_sample": [{"sample_id": "x"}]},
    ],
)
def test_invalid_summary_rejected_before_database_creation(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    write_summary(tmp_path, overrides)
    assert ingest(tmp_path).returncode != 0
    assert not (tmp_path / "db").exists()


@pytest.mark.parametrize(
    "problem",
    ["unknown-id", "missing-id", "duplicate", "missing-path", "invalid-trace"],
)
def test_invalid_trace_index_rejected_before_database_creation(
    tmp_path: Path, problem: str
) -> None:
    write_summary(tmp_path)
    trace = tmp_path / "trace.jsonl"
    trace.write_text("not json" if problem == "invalid-trace" else "", encoding="utf-8")
    entry = {
        "sample_id": "unknown" if problem == "unknown-id" else SAMPLE_ID,
        "path": "missing.jsonl" if problem == "missing-path" else trace.name,
    }
    entries = (
        []
        if problem == "missing-id"
        else [entry, entry] if problem == "duplicate" else [entry]
    )
    index = write_json(tmp_path / "index.json", entries)
    assert ingest(tmp_path, "--trace-index", str(index)).returncode != 0
    assert not (tmp_path / "db").exists()
