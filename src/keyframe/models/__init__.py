"""Strongly-typed data models for the keyframe package."""

from __future__ import annotations

from keyframe.models.hypotheses import (
    DIRECTIONAL_RELATIONS,
    SUPPORTED_RELATIONS,
    SUPPORTED_RELATIONS_STR,
    ConstraintType,
    ExecutionPolicy,
    GroundingQuery,
    HypothesisKind,
    HypothesisOutputV1,
    ParseMode,
    QueryHypothesis,
    QueryNode,
    ReferenceFrame,
    SelectConstraint,
    SpatialConstraint,
    SpatialRelation,
    ViewpointContext,
    ViewpointKind,
)
from keyframe.models.results import (
    ExecutionMode,
    ExecutionResult,
    FrameMapping,
    GroundingStatus,
    HypothesisAttempt,
    HypothesisExecution,
    KeyframeResult,
    KeyframeSelectionMetadata,
)
from keyframe.models.scene import SceneObject

__all__ = [
    # Hypotheses / query structures
    "ConstraintType",
    "SpatialRelation",
    "ReferenceFrame",
    "ViewpointKind",
    "ExecutionPolicy",
    "HypothesisKind",
    "ParseMode",
    "ViewpointContext",
    "QueryNode",
    "SpatialConstraint",
    "SelectConstraint",
    "GroundingQuery",
    "QueryHypothesis",
    "HypothesisOutputV1",
    "DIRECTIONAL_RELATIONS",
    "SUPPORTED_RELATIONS",
    "SUPPORTED_RELATIONS_STR",
    # Scene
    "SceneObject",
    # Results
    "ExecutionMode",
    "GroundingStatus",
    "ExecutionResult",
    "HypothesisAttempt",
    "HypothesisExecution",
    "FrameMapping",
    "KeyframeSelectionMetadata",
    "KeyframeResult",
]
