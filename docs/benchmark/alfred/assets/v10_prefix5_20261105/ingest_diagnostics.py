"""Ingest non-task diagnostics into their own SQLite table, never task attempts.

Input is an explicit JSON manifest with a diagnostics list. Each entry supplies
run_id, kind, terminal status, code (or null with a code_status explanation),
artifact_path, summary and optional metrics. --check-only performs no writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from codex_agent.config import DEFAULT_PROJECT_ROOT

ROOT = DEFAULT_PROJECT_ROOT
JsonObject = dict[str, Any]
OFFICIAL_SCORE_KEYS = {
    "official_score",
    "success_rate",
    "goal_condition_success_rate",
    "path_length_weighted_success_rate",
    "path_length_weighted_goal_condition_success_rate",
    "completed_goal_conditions",
    "total_goal_conditions",
}


class DiagnosticRecord(BaseModel):
    """A diagnostic is a process result, never an ALFRED episode score."""

    model_config = ConfigDict(extra="forbid", strict=True)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    kind: Literal["capture", "depth_comparison", "transport_probe"]
    status: Literal["completed", "failed", "interrupted"]
    code: JsonObject | None
    code_status: Literal["recorded", "not_executed", "unavailable"]
    artifact_path: str
    summary: JsonObject
    metrics: JsonObject | None = None

    @model_validator(mode="after")
    def consistent_code(self) -> DiagnosticRecord:
        if not Path(self.artifact_path).is_absolute():
            raise ValueError("artifact_path must be absolute")
        if self.code_status == "recorded":
            if self.code is None:
                raise ValueError("recorded code status requires actual code provenance")
            if (
                set(self.code)
                != {
                    "branch",
                    "head_sha",
                    "executed_code_sha256",
                    "tracked_worktree_drift",
                }
                or not isinstance(self.code.get("branch"), str)
                or not self.code["branch"]
            ):
                raise ValueError(
                    "recorded code must contain only the explicit provenance fields"
                )
            for name, size in (("head_sha", 40), ("executed_code_sha256", 64)):
                if (
                    re.fullmatch(rf"[0-9a-f]{{{size}}}", str(self.code.get(name)))
                    is None
                ):
                    raise ValueError("recorded code has an invalid digest")
            if self.code.get("tracked_worktree_drift") != "":
                raise ValueError("recorded code must preserve its actual clean state")
        elif self.code is not None:
            raise ValueError(
                "unavailable/not_executed code must be null, not a planned SHA"
            )
        if self.code_status == "not_executed" and self.status == "completed":
            raise ValueError(
                "a completed diagnostic cannot claim code was not executed"
            )
        reject_official_scores(self.summary)
        reject_official_scores(self.metrics)
        return self


def reject_official_scores(value: object) -> None:
    if isinstance(value, dict):
        if OFFICIAL_SCORE_KEYS.intersection(value):
            raise ValueError(
                "diagnostic metadata cannot contain official task-score fields"
            )
        for child in value.values():
            reject_official_scores(child)
    elif isinstance(value, list):
        for child in value:
            reject_official_scores(child)


def sanitized(value: object) -> object:
    """Keep raw gateway errors and credential-bearing URLs out of derived records."""
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, child in value.items():
            if (
                (key == "error" or key.endswith("_error"))
                and isinstance(child, str)
                and child
            ):
                result[key] = {
                    "classification": (
                        "http_502"
                        if "502" in child
                        else "http_403" if "403" in child else "diagnostic_error"
                    ),
                    "sha256": hashlib.sha256(child.encode()).hexdigest(),
                }
            elif key.lower() in {
                "authorization",
                "api_key",
                "password",
                "access_token",
                "secret",
            }:
                result[key] = "[REDACTED]"
            else:
                result[key] = sanitized(child)
        return result
    if isinstance(value, list):
        return [sanitized(child) for child in value]
    if isinstance(value, str):
        return re.sub(r"(?:https?|socks5h?)://[^\s\"'<>]+", "[URL REDACTED]", value)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprint(path: Path) -> tuple[str, list[JsonObject]]:
    root = path.resolve(strict=True)
    if root.is_file():
        digest = file_sha256(root)
        return digest, [
            {"relative_path": root.name, "sha256": digest, "bytes": root.stat().st_size}
        ]
    entries = []
    for child in sorted(root.rglob("*")):
        if not child.is_file():
            continue
        if not child.resolve().is_relative_to(root):
            raise ValueError("diagnostic artifact symlink escapes its source directory")
        entries.append(
            {
                "relative_path": str(child.relative_to(root)),
                "sha256": file_sha256(child),
                "bytes": child.stat().st_size,
            }
        )
    if not entries:
        raise ValueError("diagnostic artifact directory is empty")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), entries


def require_unused_ids(
    connection: sqlite3.Connection, records: list[DiagnosticRecord]
) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    for record in records:
        if (
            "diagnostic_runs" in tables
            and connection.execute(
                "SELECT 1 FROM diagnostic_runs WHERE run_id=?", (record.run_id,)
            ).fetchone()
        ):
            raise ValueError("diagnostic run ID already exists; history is immutable")
        if (
            "runs" in tables
            and connection.execute(
                "SELECT 1 FROM runs WHERE run_id=?", (record.run_id,)
            ).fetchone()
        ):
            raise ValueError("diagnostic run ID collides with a task-evaluation run")


def ingest_records(
    database: Path, records: list[DiagnosticRecord], *, check_only: bool
) -> JsonObject:
    if len({record.run_id for record in records}) != len(records):
        raise ValueError("duplicate diagnostic run IDs in input")
    prepared = []
    for record in records:
        fingerprint, files = artifact_fingerprint(Path(record.artifact_path))
        summary = {**record.summary, "code_status": record.code_status}
        prepared.append(
            (
                record.run_id,
                record.kind,
                record.status,
                (
                    json.dumps(record.code, sort_keys=True)
                    if record.code is not None
                    else None
                ),
                str(Path(record.artifact_path).resolve()),
                json.dumps(sanitized(summary), sort_keys=True, allow_nan=False),
                (
                    json.dumps(
                        sanitized(record.metrics), sort_keys=True, allow_nan=False
                    )
                    if record.metrics is not None
                    else None
                ),
                fingerprint,
                json.dumps(files, sort_keys=True),
            )
        )
    if check_only:
        if database.exists():
            with sqlite3.connect(
                f"file:{database.resolve()}?mode=ro", uri=True
            ) as connection:
                require_unused_ids(connection, records)
        return {
            "status": "validated_not_ingested",
            "records": len(prepared),
            "kinds": [record.kind for record in records],
        }
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS diagnostic_runs ("
            "run_id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('capture','depth_comparison','transport_probe')), "
            "status TEXT NOT NULL CHECK(status IN ('completed','failed','interrupted')), "
            "code_json TEXT, artifact_path TEXT NOT NULL, summary_json TEXT NOT NULL, metrics_json TEXT, "
            "artifact_sha256 TEXT NOT NULL, artifact_manifest_json TEXT NOT NULL)"
        )
        require_unused_ids(connection, records)
        connection.executemany(
            "INSERT INTO diagnostic_runs VALUES (?,?,?,?,?,?,?,?,?)", prepared
        )
    return {
        "status": "ingested_diagnostics_only",
        "records": len(prepared),
        "database": str(database),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--database", type=Path, default=ROOT / "docs/benchmark/alfred/runs.sqlite"
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_bytes())
    if (
        set(payload) != {"schema_version", "diagnostics"}
        or payload["schema_version"] != 1
    ):
        raise ValueError(
            "diagnostic manifest must use schema_version1 and diagnostics list"
        )
    records = [
        DiagnosticRecord.model_validate(entry) for entry in payload["diagnostics"]
    ]
    if not records:
        raise ValueError("diagnostic manifest has no records")
    print(
        json.dumps(
            ingest_records(args.database, records, check_only=args.check_only), indent=2
        )
    )


if __name__ == "__main__":
    main()
