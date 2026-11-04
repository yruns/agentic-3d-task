"""Archive all ALFRED run attempts, including infrastructure failures, in SQLite."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .metrics import EpisodeScore


class EpisodeAttempt(BaseModel):
    """One attempt's terminal state or an explicit error, with retained artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str = Field(min_length=1)
    status: Literal["completed", "error", "interrupted"]
    artifact_dir: str = Field(min_length=1)
    score: EpisodeScore | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> EpisodeAttempt:
        if self.status == "completed" and (
            self.score is None or self.error is not None
        ):
            raise ValueError("completed attempt requires a score and no error")
        if self.status != "completed" and not self.error:
            raise ValueError("error/interrupted attempt requires a nonempty error")
        return self


def ingest_run(
    database_path: Path,
    *,
    run_id: str,
    metadata: Mapping[str, object],
    attempts: Sequence[EpisodeAttempt],
) -> None:
    """Atomically insert a run and all attempts without overwriting history.

    Metadata must include the caller's code, fold, model, command and artifact
    provenance. Partial runs may contain no attempts if startup failed. A new
    run_id is required for each recorded invocation or subsequent snapshot.
    """
    if not run_id.strip():
        raise ValueError("run_id must contain non-whitespace text")
    identifiers = [attempt.episode_id for attempt in attempts]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("attempts contain duplicate episode_id values")
    metadata_json = json.dumps(dict(metadata), sort_keys=True, allow_nan=False)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS attempts ("
            "run_id TEXT NOT NULL REFERENCES runs(run_id), episode_id TEXT NOT NULL, "
            "status TEXT NOT NULL, artifact_dir TEXT NOT NULL, score_json TEXT, error TEXT, "
            "PRIMARY KEY(run_id, episode_id))"
        )
        if connection.execute(
            "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone():
            raise ValueError(
                f"run_id={run_id!r} already exists; history cannot be overwritten"
            )
        connection.execute("INSERT INTO runs VALUES (?, ?)", (run_id, metadata_json))
        connection.executemany(
            "INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    attempt.episode_id,
                    attempt.status,
                    attempt.artifact_dir,
                    None if attempt.score is None else attempt.score.model_dump_json(),
                    attempt.error,
                )
                for attempt in attempts
            ],
        )
