"""Ingest all v3 direct, SDK, bridge, and accounting-control evidence."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    direct_names = ["cache_visible", "cache_reasoning", "cache_reasoning_repeat"]
    sdk_names = [
        "sdk_smoke",
        "sdk_smoke2",
        "sdk_smoke3",
        "sdk_smoke4",
        "sdk_long",
        "sdk_cache",
        "sdk_cache2",
        "sdk_cache3",
    ]
    bridge = [
        json.loads(line)
        for line in (root / "bridge_traces/bridge_trace.jsonl").read_text().splitlines()
    ]
    records: list[
        tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]
    ] = []
    for name in direct_names:
        summary = json.loads((root / name / "summary.json").read_text())
        requests = [
            {"key": str(r["index"]), "kind": "direct_cache", **r}
            for r in summary["requests"]
        ]
        records.append((name, summary, requests, []))
    for name in sdk_names:
        summary = json.loads((root / name / "summary.json").read_text())
        requests = []
        for row in bridge:
            if row.get("session_id") == summary["run_id"]:
                requests.append(
                    {
                        "key": row["request_id"],
                        "kind": row.get("kind", "sdk_bridge"),
                        "http_status": row.get("upstream_status"),
                        **row,
                    }
                )
        if name == "sdk_cache3":
            control = json.loads(
                (root / "sdk_cache3_exact_control/summary.json").read_text()
            )
            requests.append(
                {
                    "key": "exact_visible_control",
                    "kind": "exact_visible_control",
                    **control,
                }
            )
        records.append((name, summary, requests, summary.get("tool_executions", [])))
    args.db.parent.mkdir(parents=True, exist_ok=True)
    counts = {"runs": 0, "requests": 0, "tool_executions": 0}
    with sqlite3.connect(args.db) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, variant TEXT NOT NULL, result TEXT NOT NULL, summary_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (run_id TEXT NOT NULL, request_key TEXT NOT NULL, kind TEXT NOT NULL, http_status INTEGER, metadata_json TEXT NOT NULL, PRIMARY KEY(run_id, request_key));
            CREATE TABLE IF NOT EXISTS tool_executions (run_id TEXT NOT NULL, ordinal INTEGER NOT NULL, metadata_json TEXT NOT NULL, PRIMARY KEY(run_id, ordinal));
        """)
        for name, summary, requests, tools in records:
            run_id = summary["run_id"]
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?)",
                (run_id, name, summary["result"], json.dumps(summary)),
            )
            conn.execute("DELETE FROM requests WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM tool_executions WHERE run_id=?", (run_id,))
            for request in requests:
                conn.execute(
                    "INSERT INTO requests VALUES (?,?,?,?,?)",
                    (
                        run_id,
                        request["key"],
                        request["kind"],
                        request.get("http_status"),
                        json.dumps(request),
                    ),
                )
            for ordinal, tool in enumerate(tools, 1):
                conn.execute(
                    "INSERT INTO tool_executions VALUES (?,?,?)",
                    (run_id, ordinal, json.dumps(tool)),
                )
            counts["runs"] += 1
            counts["requests"] += len(requests)
            counts["tool_executions"] += len(tools)
    assert counts == {"runs": 11, "requests": 65, "tool_executions": 6}, counts
    print(json.dumps({"v3": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
