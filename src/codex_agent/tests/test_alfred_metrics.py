"""Official ALFRED micro and expert-length-weighted metric denominators."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from codex_agent.alfred.metrics import EpisodeScore, aggregate_scores
from codex_agent.alfred.results import EpisodeAttempt, ingest_run


def test_official_aggregation_uses_condition_and_expert_length_denominators() -> None:
    scores = [
        EpisodeScore(
            success=True,
            completed_goal_conditions=1,
            total_goal_conditions=1,
            expert_actions=10,
            executed_actions=20,
        ),
        EpisodeScore(
            success=False,
            completed_goal_conditions=1,
            total_goal_conditions=4,
            expert_actions=30,
            executed_actions=30,
        ),
    ]
    metrics = aggregate_scores(scores)
    assert metrics.success_rate == 0.5
    assert metrics.goal_condition_success_rate == 0.4
    assert metrics.path_length_weighted_success_rate == 0.125
    assert metrics.path_length_weighted_goal_condition_success_rate == 0.3125


def test_zero_action_stop_is_finite_and_empty_scores_are_invalid() -> None:
    score = EpisodeScore(
        success=False,
        completed_goal_conditions=0,
        total_goal_conditions=1,
        expert_actions=10,
        executed_actions=0,
    )
    assert aggregate_scores([score]).success_rate == 0
    with pytest.raises(ValueError, match="scores"):
        aggregate_scores([])


@pytest.mark.parametrize(
    "changes",
    [
        {"expert_actions": 0},
        {"executed_actions": -1},
        {"total_goal_conditions": 0},
        {"completed_goal_conditions": 2},
        {"success": True},
    ],
)
def test_rejects_invalid_scores(changes: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "success": False,
        "completed_goal_conditions": 0,
        "total_goal_conditions": 1,
        "expert_actions": 10,
        "executed_actions": 2,
    }
    payload.update(changes)
    with pytest.raises(ValueError):
        EpisodeScore.model_validate(payload)


def test_sqlite_preserves_failed_and_interrupted_attempts(tmp_path: Path) -> None:
    attempts = [
        EpisodeAttempt(
            episode_id="valid_seen/type/trial#0",
            status="completed",
            artifact_dir="/results/a",
            score=EpisodeScore(
                success=False,
                completed_goal_conditions=0,
                total_goal_conditions=1,
                expert_actions=10,
                executed_actions=10,
            ),
        ),
        EpisodeAttempt(
            episode_id="valid_unseen/type/trial#0",
            status="error",
            artifact_dir="/results/b",
            error="Simulator disconnected",
        ),
        EpisodeAttempt(
            episode_id="valid_unseen/type/trial2#0",
            status="interrupted",
            artifact_dir="/results/c",
            error="Stopped by user",
        ),
    ]
    database = tmp_path / "runs.sqlite"
    ingest_run(
        database,
        run_id="pilot_v1",
        metadata={"fold_sha256": "abc", "status": "partial"},
        attempts=attempts,
    )
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone() == (3,)
        assert connection.execute(
            "SELECT COUNT(*) FROM attempts WHERE score_json IS NULL"
        ).fetchone() == (2,)
    with pytest.raises(ValueError, match="already exists"):
        ingest_run(database, run_id="pilot_v1", metadata={}, attempts=attempts)
