"""Rebuild this protocol diagnostic's SQLite rows from redacted artifacts."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()
    assets = Path(__file__).resolve().parent
    manifest = json.loads((assets / "manifest.json").read_text())
    args.db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                variant TEXT NOT NULL,
                result TEXT NOT NULL,
                summary_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS requests (
                run_id TEXT NOT NULL,
                request_key TEXT NOT NULL,
                kind TEXT NOT NULL,
                http_status INTEGER,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (run_id, request_key)
            );
            CREATE TABLE IF NOT EXISTS tool_executions (
                run_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (run_id, ordinal)
            );
            """)
        for variant in manifest["variants"]:
            summary = json.loads((assets / variant / "summary.json").read_text())
            run_id = summary["run_id"]
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?)",
                (run_id, variant, summary["result"], json.dumps(summary)),
            )
            conn.execute("DELETE FROM requests WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM tool_executions WHERE run_id = ?", (run_id,))
            for request in summary["requests"]:
                conn.execute(
                    "INSERT INTO requests VALUES (?, ?, ?, ?, ?)",
                    (
                        run_id,
                        str(request["index"]),
                        "chain",
                        request.get("http_status"),
                        json.dumps(request),
                    ),
                )
            for control in summary.get("controls", []):
                conn.execute(
                    "INSERT INTO requests VALUES (?, ?, ?, ?, ?)",
                    (
                        run_id,
                        "control_" + control["mode"],
                        "control",
                        control.get("http_status"),
                        json.dumps(control),
                    ),
                )
            for ordinal, tool in enumerate(summary["tool_executions"], 1):
                conn.execute(
                    "INSERT INTO tool_executions VALUES (?, ?, ?)",
                    (run_id, ordinal, json.dumps(tool)),
                )
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("runs", "requests", "tool_executions")
        }
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
