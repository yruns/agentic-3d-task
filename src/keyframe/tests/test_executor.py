"""Tests for the recursive query executor."""

from __future__ import annotations

import numpy as np

from keyframe.models import (
    ConstraintType,
    GroundingQuery,
    QueryNode,
    SceneObject,
    SelectConstraint,
    SpatialConstraint,
)
from keyframe.query_executor import QueryExecutor


def _bbox(center: tuple[float, float, float], half: float) -> np.ndarray:
    cx, cy, cz = center
    return np.array(
        [[cx - half, cy - half, cz - half], [cx + half, cy + half, cz + half]]
    )


def _scene() -> list[SceneObject]:
    return [
        SceneObject(
            obj_id=0,
            category="sofa",
            centroid=np.array([0.0, 0.0, 0.25]),
            bbox_np=_bbox((0, 0, 0.25), 0.4),
            class_name=["sofa"],
        ),
        SceneObject(
            obj_id=1,
            category="pillow",
            centroid=np.array([0.0, 0.0, 0.6]),
            bbox_np=_bbox((0, 0, 0.6), 0.15),
            class_name=["pillow"],
        ),
        SceneObject(
            obj_id=2,
            category="door",
            centroid=np.array([4.0, 0.0, 1.0]),
            bbox_np=_bbox((4, 0, 1), 0.3),
            class_name=["door"],
        ),
        SceneObject(
            obj_id=3,
            category="pillow",
            centroid=np.array([6.0, 6.0, 0.6]),
            bbox_np=_bbox((6, 6, 0.6), 0.15),
            class_name=["pillow"],
        ),
    ]


def test_pillow_on_sofa_grounds_correct_object() -> None:
    query = GroundingQuery(
        raw_query="the pillow on the sofa",
        expect_unique=True,
        root=QueryNode(
            categories=["pillow"],
            node_id="root",
            spatial_constraints=[
                SpatialConstraint(
                    relation="on",
                    anchors=[QueryNode(categories=["sofa"], node_id="root_sc0_a0")],
                )
            ],
        ),
    )
    result = QueryExecutor(_scene()).execute(query)
    assert [o.obj_id for o in result.matched_objects] == [1]


def test_nearest_pillow_to_door() -> None:
    query = GroundingQuery(
        raw_query="the pillow nearest the door",
        root=QueryNode(
            categories=["pillow"],
            node_id="root",
            select_constraint=SelectConstraint(
                constraint_type=ConstraintType.SUPERLATIVE,
                metric="distance",
                order="min",
                reference=QueryNode(categories=["door"], node_id="root_ref"),
            ),
        ),
    )
    result = QueryExecutor(_scene()).execute(query)
    assert result.best_object is not None
    assert result.best_object.obj_id == 1  # closer to door than the far pillow


def test_no_match_returns_empty() -> None:
    query = GroundingQuery(
        raw_query="the lamp", root=QueryNode(categories=["lamp"], node_id="root")
    )
    result = QueryExecutor(_scene()).execute(query)
    assert result.is_empty
