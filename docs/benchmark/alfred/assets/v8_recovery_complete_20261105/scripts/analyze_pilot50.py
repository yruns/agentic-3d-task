"""Audit the frozen ALFRED cohort and derive reporting data from saved artifacts.

This is a read-only, post-hoc analysis. It never invokes an actor or simulator.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from codex_agent.alfred.dataset import load_manifest
from codex_agent.alfred.metrics import EpisodeScore, aggregate_scores

ROOT = Path("/home/ysh/codebase/agentic-3d-task")
PREFIX = "REPRO_20261105_ALFRED_V3_20261105_pilot50_s"


def read_json(path: Path) -> dict[str, Any]:
    """Read an artifact object; field-specific checks follow at the audit boundary."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return value


def read_lines(path: Path) -> list[dict[str, Any]]:
    """Read persisted JSONL events, subsequently checked against terminal records."""
    rows = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"artifact must contain JSON object records: {path}")
    return rows


def main() -> None:
    if not __debug__:
        raise RuntimeError("This audit requires assertions; do not use python -O")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    expected = load_manifest(
        ROOT / "docs/benchmark/alfred/folds/pilot50.json",
        Path("/home/ysh/Datasets/FILM/alfred/data/json_2.1.0"),
        Path("/home/ysh/Datasets/FILM/alfred_data_small/splits/oct21.json"),
    )
    identifiers = {item.episode_id for item in expected}
    expected_by_id = {item.episode_id: item for item in expected}
    runs = []
    rows = []
    planned: list[str] = []
    signatures = []
    for shard in range(4):
        directory = ROOT / f"outputs/alfred/{PREFIX}{shard}_direct"
        run = read_json(directory / "run.json")
        summary = read_json(directory / "summary.json")
        if not args.allow_partial:
            assert run["status"] == "completed", (shard, run["status"])
            assert summary["completed"] == summary["selected"]
            assert (
                summary["errors"] == summary["interrupted"] == summary["pending"] == 0
            )
        signature_keys = (
            "fold_sha256",
            "splits_sha256",
            "model",
            "model_provider",
            "reasoning_effort",
            "execution",
            "geometry",
            "perception_profile",
            "perception",
            "simulator",
            "external_sources",
            "code",
            "num_shards",
        )
        signatures.append({key: run[key] for key in signature_keys})
        assert run["shard_index"] == shard
        episode_metadata = {item["episode_id"]: item for item in run["episodes"]}
        for identity, metadata in episode_metadata.items():
            reference = expected_by_id[identity]
            for field in ("split", "task_type", "trajectory_sha256", "expert_actions"):
                assert metadata[field] == getattr(reference, field)
        planned.extend(episode_metadata)
        assert list(episode_metadata) == [
            item.episode_id for item in expected[shard::4]
        ]
        run_rows = []
        for attempt in summary["attempts"]:
            assert attempt["status"] == "completed" and attempt["score"] is not None
            episode = Path(attempt["artifact_dir"])
            assert read_json(episode / "attempt.json") == attempt
            score = EpisodeScore.model_validate(attempt["score"])
            reference = expected_by_id[attempt["episode_id"]]
            terminal = read_json(episode / "simulator/terminal_evaluation.json")
            assert terminal["trajectory_sha256"] == reference.trajectory_sha256
            assert score.expert_actions == reference.expert_actions
            for field in (
                "success",
                "completed_goal_conditions",
                "total_goal_conditions",
            ):
                assert terminal[field] == getattr(score, field)
            execution = read_json(episode / "execution.json")
            actions = read_lines(episode / "actions.jsonl")
            observations = read_lines(episode / "observations.jsonl")
            responses = sorted((episode / "decisions").glob("*/response.json"))
            assert len(actions) == score.executed_actions == execution["actions"]
            assert sum(not item["success"] for item in actions) == execution["failures"]
            assert len(terminal["actions"]) == len(actions)
            for physical, recorded in zip(terminal["actions"], actions, strict=True):
                assert (physical["action"], physical["success"]) == (
                    recorded["action"],
                    recorded["success"],
                )
            assert len(observations) == len(actions) + 1
            assert len(responses) == execution["decisions"]
            operations: Counter[str] = Counter()
            tokens: Counter[str] = Counter()
            feedbacks: Counter[str] = Counter()
            images = 0
            historical_images = 0
            sdk_attempts = 0
            usage_complete = True
            for response_path in responses:
                response = read_json(response_path)
                wire = json.loads(response["raw_response"])
                operations[wire["operation"]] += 1
                request = read_json(response_path.with_name("request.json"))
                images += len(request["images"])
                historical_images += max(0, len(request["images"]) - 2)
                metadata = response["metadata"]
                assert metadata["status"] == "completed"
                sdk_attempts += len(metadata["attempts"])
                usage = (metadata.get("usage") or {}).get("total")
                if usage is None:
                    usage_complete = False
                else:
                    tokens.update(
                        {
                            key: value
                            for key, value in usage.items()
                            if type(value) is int
                        }
                    )
                # Feedback tags are exact saved strings, never inferred task labels.
                feedback = json.loads(request["prompt"].splitlines()[-1])[
                    "observations"
                ]["feedback"]
                for phrase in (
                    "No route through observed free cells",
                    "No additional historical views",
                    "Rejected interaction: target has no mask",
                    "Camera pitch limit reached",
                ):
                    if phrase in feedback:
                        feedbacks[phrase] += 1
            for field in ("input_tokens", "output_tokens"):
                assert execution[field] == (tokens[field] if usage_complete else None)
            info = episode_metadata[attempt["episode_id"]]
            row = {
                "episode_id": attempt["episode_id"],
                "split": info["split"],
                "task_type": info["task_type"],
                "run_id": run["run_id"],
                "artifact_dir": str(episode),
                "score": score.model_dump(),
                "execution": execution,
                "reported_usage": dict(tokens) if usage_complete else None,
                "sdk_attempts": sdk_attempts,
                "submitted_images": images,
                "submitted_historical_images": historical_images,
                "observations": len(observations),
                "operations": dict(operations),
                "action_types": dict(Counter(item["action"] for item in actions)),
                "failed_action_types": dict(
                    Counter(item["action"] for item in actions if not item["success"])
                ),
                "failure_messages": dict(
                    Counter(item["error"] for item in actions if not item["success"])
                ),
                "selected_feedback_occurrences": dict(feedbacks),
            }
            rows.append(row)
            run_rows.append(row)
        if run_rows:
            calculated = aggregate_scores(
                [EpisodeScore.model_validate(row["score"]) for row in run_rows]
            ).model_dump()
            assert calculated == summary["metrics"]
        if run["status"] == "completed":
            assert run["perception_statistics"]["frames"] == sum(
                row["observations"] for row in run_rows
            )
        runs.append(run)
    assert all(signature == signatures[0] for signature in signatures)
    assert len(planned) == len(set(planned)) == 50 and set(planned) == identifiers
    assert len({row["episode_id"] for row in rows}) == len(rows)
    if not args.allow_partial:
        assert {row["episode_id"] for row in rows} == identifiers
    rows.sort(
        key=lambda row: next(
            i for i, item in enumerate(expected) if item.episode_id == row["episode_id"]
        )
    )
    metrics = (
        aggregate_scores(
            [EpisodeScore.model_validate(row["score"]) for row in rows]
        ).model_dump()
        if rows
        else None
    )
    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for field in ("split", "task_type"):
        grouped[field] = {}
        for value in sorted({row[field] for row in rows}):
            selected = [
                EpisodeScore.model_validate(row["score"])
                for row in rows
                if row[field] == value
            ]
            grouped[field][value] = aggregate_scores(selected).model_dump()
    counts = {}
    for field in (
        "operations",
        "action_types",
        "failed_action_types",
        "failure_messages",
        "selected_feedback_occurrences",
    ):
        total: Counter[str] = Counter()
        for row in rows:
            total.update(row[field])
        counts[field] = dict(total)
    totals = {}
    for field in (
        "actions",
        "failures",
        "decisions",
        "input_tokens",
        "output_tokens",
        "perception_seconds",
        "policy_seconds",
    ):
        values = [row["execution"][field] for row in rows]
        totals[field] = (
            sum(values) if all(value is not None for value in values) else None
        )
    for field in (
        "sdk_attempts",
        "submitted_images",
        "submitted_historical_images",
        "observations",
    ):
        totals[field] = sum(row[field] for row in rows)
    usage_total: Counter[str] = Counter()
    for row in rows:
        if row["reported_usage"] is not None:
            usage_total.update(row["reported_usage"])
    totals["reported_usage"] = (
        dict(usage_total)
        if all(row["reported_usage"] is not None for row in rows)
        else None
    )
    totals["stop_reasons"] = dict(
        Counter(row["execution"]["stop_reason"] for row in rows)
    )
    database_rows = []
    database = ROOT / "docs/benchmark/alfred/runs.sqlite"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        for run in runs:
            stored = connection.execute(
                "SELECT metadata_json FROM runs WHERE run_id=?", (run["run_id"],)
            ).fetchone()
            attempts = connection.execute(
                "SELECT episode_id,status,score_json,error FROM attempts WHERE run_id=? ORDER BY episode_id",
                (run["run_id"],),
            ).fetchall()
            if run["status"] == "completed":
                assert stored and json.loads(stored[0]) == run
                assert len(attempts) == len(run["episodes"])
                assert all(
                    status == "completed" and score and error is None
                    for _, status, score, error in attempts
                )
                expected_scores = {
                    row["episode_id"]: row["score"]
                    for row in rows
                    if row["run_id"] == run["run_id"]
                }
                assert {
                    identity: json.loads(score) for identity, _, score, _ in attempts
                } == expected_scores
            database_rows.append(
                {
                    "run_id": run["run_id"],
                    "ingested": stored is not None,
                    "attempts": len(attempts),
                }
            )
    payload: dict[str, object] = {
        "status": (
            "completed"
            if all(run["status"] == "completed" for run in runs)
            else "partial"
        ),
        "selected": 50,
        "scored": len(rows),
        "metrics": metrics,
        "by_split": grouped["split"],
        "by_task_type": grouped["task_type"],
        "totals": totals,
        "counts": counts,
        "matching_configuration": signatures[0],
        "database_audit": database_rows,
        "run_ids": [run["run_id"] for run in runs],
        "episodes": rows,
    }
    if payload["status"] == "completed":
        started = min(datetime.fromisoformat(run["started_at_utc"]) for run in runs)
        ended = max(datetime.fromisoformat(run["finished_at_utc"]) for run in runs)
        payload["wall_seconds"] = (ended - started).total_seconds()
    path = ROOT / (
        "tmp/alfred_setup/pilot50_analysis_partial.json"
        if args.allow_partial
        else "tmp/alfred_setup/pilot50_analysis.json"
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in ("episodes", "matching_configuration", "counts")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
