"""ALFRED official success, micro goal-condition, and path-weighted metrics."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EpisodeScore(BaseModel):
    """Terminal official goal counts and attempted low-level action lengths."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    success: bool
    completed_goal_conditions: int = Field(ge=0)
    total_goal_conditions: int = Field(gt=0)
    expert_actions: int = Field(gt=0)
    executed_actions: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_goal_counts(self) -> EpisodeScore:
        if self.completed_goal_conditions > self.total_goal_conditions:
            raise ValueError(
                "completed_goal_conditions cannot exceed total_goal_conditions"
            )
        if self.success != (
            self.completed_goal_conditions == self.total_goal_conditions
        ):
            raise ValueError("success must agree with completed_goal_conditions")
        return self


class AggregateMetrics(BaseModel):
    """Aggregate values in [0, 1], using official rather than episode-average GC."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_evals: int
    num_successes: int
    completed_goal_conditions: int
    total_goal_conditions: int
    expert_actions: int
    executed_actions: int
    success_rate: float
    goal_condition_success_rate: float
    path_length_weighted_success_rate: float
    path_length_weighted_goal_condition_success_rate: float


def aggregate_scores(scores: Sequence[EpisodeScore]) -> AggregateMetrics:
    """Aggregate exactly the expert-length-weighted official ALFRED formulas.

    GC is the sum of completed conditions divided by the sum of total conditions.
    For score s, efficiency is gold/max(gold, executed). PLWSR and PLWGC average
    efficiency-adjusted episode scores with gold action counts as weights.
    Zero executed actions have efficiency 1, avoiding the upstream zero division.
    Errors without terminal scores must be reported separately, never made into
    synthetic zero-condition episodes.
    """
    if not scores:
        raise ValueError("scores must contain at least one terminal episode")
    successes = sum(score.success for score in scores)
    completed = sum(score.completed_goal_conditions for score in scores)
    total = sum(score.total_goal_conditions for score in scores)
    gold = sum(score.expert_actions for score in scores)
    weighted_success = 0.0
    weighted_gc = 0.0
    for score in scores:
        efficiency = score.expert_actions / max(
            score.expert_actions, score.executed_actions
        )
        weighted_success += score.expert_actions * efficiency * score.success
        weighted_gc += (
            score.expert_actions
            * efficiency
            * score.completed_goal_conditions
            / score.total_goal_conditions
        )
    return AggregateMetrics(
        num_evals=len(scores),
        num_successes=successes,
        completed_goal_conditions=completed,
        total_goal_conditions=total,
        expert_actions=gold,
        executed_actions=sum(score.executed_actions for score in scores),
        success_rate=successes / len(scores),
        goal_condition_success_rate=completed / total,
        path_length_weighted_success_rate=weighted_success / gold,
        path_length_weighted_goal_condition_success_rate=weighted_gc / gold,
    )
