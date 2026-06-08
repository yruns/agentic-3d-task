"""Result models for query execution and keyframe selection.

These replace the original ``dict[str, Any]`` ``metadata`` payloads with
explicit, strongly-typed structures. Heavyweight per-constraint execution
traces (a debugging-only artifact) are intentionally dropped; what remains is
the information the keyframe-selection mainline actually consumes.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from keyframe.models.hypotheses import (
    HypothesisKind,
    HypothesisOutputV1,
    QueryHypothesis,
)
from keyframe.models.scene import SceneObject


class ExecutionMode(str, Enum):
    """Whether execution is strict filtering or recall-biased."""

    STRICT = "strict"
    RECALL = "recall"


class GroundingStatus(str, Enum):
    """Outcome of executing ranked hypotheses."""

    DIRECT_GROUNDED = "direct_grounded"
    PROXY_GROUNDED = "proxy_grounded"
    CONTEXT_ONLY = "context_only"
    NO_EVIDENCE = "no_evidence"


class ExecutionResult(BaseModel):
    """Result of executing one query node (or a full grounding query)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_id: str
    matched_objects: list[SceneObject] = Field(default_factory=list)
    # obj_id -> multiplicative match score
    scores: dict[int, float] = Field(default_factory=dict)
    # obj_id -> {constraint_key -> soft satisfaction score} for SOFT/RANK_ONLY
    soft_match_scores: dict[int, dict[str, float]] = Field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Whether no objects matched."""
        return len(self.matched_objects) == 0

    @property
    def best_object(self) -> SceneObject | None:
        """Highest-scoring matched object, or None when empty."""
        if not self.matched_objects:
            return None
        if self.scores:
            best_id = max(self.scores, key=lambda obj_id: self.scores[obj_id])
            for obj in self.matched_objects:
                if obj.obj_id == best_id:
                    return obj
        return self.matched_objects[0]


#: Outcome of evaluating one ranked hypothesis in ``execute_hypotheses``.
HypothesisAttemptStatus = Literal["skipped_unknown_anchor", "empty", "grounded"]


class HypothesisAttempt(BaseModel):
    """Record of one hypothesis evaluated by ``execute_hypotheses``."""

    kind: HypothesisKind
    rank: int
    status: HypothesisAttemptStatus


class HypothesisExecution(BaseModel):
    """Outcome of executing ranked hypotheses for a query."""

    status: GroundingStatus
    result: ExecutionResult
    hypothesis: QueryHypothesis | None = None
    attempts: list[HypothesisAttempt] = Field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """Whether a hypothesis produced non-empty evidence."""
        return self.hypothesis is not None and not self.result.is_empty


class FrameMapping(BaseModel):
    """Mapping from a selected view index to a resolved RGB frame on disk."""

    requested_view_id: int
    requested_frame_id: int
    resolved_view_id: int
    resolved_frame_id: int
    path: str | None = None


class KeyframeSelectionMetadata(BaseModel):
    """Structured metadata accompanying a ``KeyframeResult``."""

    status: GroundingStatus
    strategy: str
    execution_mode: ExecutionMode
    selected_hypothesis_kind: HypothesisKind | None = None
    selected_hypothesis_rank: int | None = None
    error: str = ""
    all_object_ids: list[int] = Field(default_factory=list)
    frame_mappings: list[FrameMapping] = Field(default_factory=list)
    attempts: list[HypothesisAttempt] = Field(default_factory=list)
    hypothesis_output: HypothesisOutputV1 | None = None


class KeyframeResult(BaseModel):
    """Result of keyframe selection for a query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    target_term: str
    anchor_term: str | None = None

    keyframe_indices: list[int] = Field(default_factory=list)
    keyframe_paths: list[Path] = Field(default_factory=list)

    target_objects: list[SceneObject] = Field(default_factory=list)
    anchor_objects: list[SceneObject] = Field(default_factory=list)

    selection_scores: dict[int, float] = Field(default_factory=dict)
    metadata: KeyframeSelectionMetadata

    def summary(self) -> str:
        """Human-readable one-paragraph summary."""
        lines = [
            f"Query: {self.query}",
            f"Target: '{self.target_term}' -> {len(self.target_objects)} objects",
        ]
        if self.anchor_term:
            lines.append(
                f"Anchor: '{self.anchor_term}' -> {len(self.anchor_objects)} objects"
            )
        lines.append(
            f"Selected {len(self.keyframe_indices)} keyframes: {self.keyframe_indices}"
        )
        return "\n".join(lines)
