"""keyframe: query-driven keyframe selection over prepared 3D scenes.

Production entry point::

    from keyframe import KeyframeSelector

    selector = KeyframeSelector.from_scene_path("/path/to/scene")
    result = selector.select_keyframes_v2("the pillow on the sofa nearest the door", k=3)
    print(result.keyframe_indices, result.keyframe_paths)

The pipeline parses a query into structured hypotheses (LLM), grounds target /
anchor objects geometrically, and selects keyframes that jointly cover them.
"""

from __future__ import annotations

from keyframe.keyframe_selector import KeyframeSelector, select_keyframes
from keyframe.lightweight_conceptgraph import load_scene_objects
from keyframe.llm import LLMClient, LLMConfig
from keyframe.models import (
    ConstraintType,
    ExecutionMode,
    ExecutionPolicy,
    ExecutionResult,
    FrameMapping,
    GroundingQuery,
    GroundingStatus,
    HypothesisAttempt,
    HypothesisExecution,
    HypothesisKind,
    HypothesisOutputV1,
    KeyframeResult,
    KeyframeSelectionMetadata,
    ParseMode,
    QueryHypothesis,
    QueryNode,
    ReferenceFrame,
    SceneObject,
    SelectConstraint,
    SpatialConstraint,
    SpatialRelation,
    ViewpointContext,
    ViewpointKind,
)
from keyframe.parsing import QueryParser, parse_query
from keyframe.query_executor import QueryExecutor, execute_query
from keyframe.spatial import RelationResult, SpatialRelationChecker

__version__ = "0.1.0"

__all__ = [
    # Entry points
    "KeyframeSelector",
    "select_keyframes",
    # Scene + results
    "SceneObject",
    "KeyframeResult",
    "KeyframeSelectionMetadata",
    "FrameMapping",
    "ExecutionResult",
    "ExecutionMode",
    "GroundingStatus",
    "HypothesisExecution",
    "HypothesisAttempt",
    # Query structures
    "GroundingQuery",
    "QueryNode",
    "SpatialConstraint",
    "SelectConstraint",
    "HypothesisOutputV1",
    "QueryHypothesis",
    "HypothesisKind",
    "ParseMode",
    "ConstraintType",
    "SpatialRelation",
    "ReferenceFrame",
    "ViewpointContext",
    "ViewpointKind",
    "ExecutionPolicy",
    # Parsing / execution / spatial
    "QueryParser",
    "parse_query",
    "QueryExecutor",
    "execute_query",
    "SpatialRelationChecker",
    "RelationResult",
    # Scene loading + LLM
    "load_scene_objects",
    "LLMClient",
    "LLMConfig",
]
