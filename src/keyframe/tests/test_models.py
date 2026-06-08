"""Tests for the strongly-typed data models."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from keyframe.models import (
    ExecutionResult,
    GroundingQuery,
    HypothesisKind,
    HypothesisOutputV1,
    ParseMode,
    QueryHypothesis,
    QueryNode,
    SceneObject,
    SpatialConstraint,
)


def test_scene_object_from_conceptgraph_computes_centroid() -> None:
    obj = SceneObject.from_conceptgraph(
        7,
        {
            "class_name": ["chair", "chair", "item"],
            "pcd_np": np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
            "clip_ft": np.ones(8, dtype=np.float32),
            "image_idx": np.array([1, 2]),
            "xyxy": np.array([[0, 0, 10, 10], [1, 1, 5, 5]], dtype=float),
        },
    )
    assert obj.obj_id == 7
    assert obj.category == "chair"  # majority vote ignores "item"
    assert obj.centroid is not None
    np.testing.assert_allclose(obj.centroid, [1.0, 1.0, 1.0])
    assert obj.clip_feature is not None and obj.clip_feature.dtype == np.float32
    assert obj.image_idx == [1, 2]
    assert len(obj.xyxy) == 2


def test_scene_object_centroid_from_explicit_field_when_no_pcd() -> None:
    obj = SceneObject.from_conceptgraph(
        0, {"class_name": ["table"], "centroid": [1.0, 2.0, 3.0]}
    )
    assert obj.centroid is not None
    np.testing.assert_allclose(obj.centroid, [1.0, 2.0, 3.0])


def test_hypothesis_output_rejects_noncontiguous_ranks() -> None:
    query = GroundingQuery(raw_query="x", root=QueryNode(categories=["chair"]))
    with pytest.raises(ValidationError):
        HypothesisOutputV1(
            parse_mode=ParseMode.MULTI,
            hypotheses=[
                QueryHypothesis(
                    kind=HypothesisKind.DIRECT, rank=1, grounding_query=query
                ),
                QueryHypothesis(
                    kind=HypothesisKind.PROXY, rank=3, grounding_query=query
                ),
            ],
        )


def test_grounding_query_collects_all_categories() -> None:
    query = GroundingQuery(
        raw_query="the pillow on the sofa",
        root=QueryNode(
            categories=["pillow"],
            spatial_constraints=[
                SpatialConstraint(
                    relation="on", anchors=[QueryNode(categories=["sofa"])]
                )
            ],
        ),
    )
    assert set(query.get_all_categories()) == {"pillow", "sofa"}


def test_execution_result_best_object_uses_scores() -> None:
    objects = [
        SceneObject(obj_id=1, category="a", centroid=np.zeros(3)),
        SceneObject(obj_id=2, category="b", centroid=np.zeros(3)),
    ]
    result = ExecutionResult(
        node_id="root", matched_objects=objects, scores={1: 0.2, 2: 0.9}
    )
    assert result.best_object is not None
    assert result.best_object.obj_id == 2
    assert not result.is_empty
