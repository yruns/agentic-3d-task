"""Tests for the strongly-typed data models and self-healing normalization."""

from __future__ import annotations

import numpy as np

from keyframe.models import (
    ExecutionPolicy,
    ExecutionResult,
    GroundingQuery,
    HypothesisKind,
    HypothesisOutputV1,
    ParseMode,
    QueryHypothesis,
    QueryNode,
    ReferenceFrame,
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


def test_hypothesis_output_auto_heals_noncontiguous_ranks_and_parse_mode() -> None:
    query = GroundingQuery(raw_query="x", root=QueryNode(categories=["chair"]))
    output = HypothesisOutputV1(
        parse_mode=ParseMode.SINGLE,
        hypotheses=[
            QueryHypothesis(kind=HypothesisKind.DIRECT, rank=1, grounding_query=query),
            QueryHypothesis(kind=HypothesisKind.PROXY, rank=5, grounding_query=query),
        ],
    )
    assert output.parse_mode == ParseMode.MULTI
    assert [h.rank for h in output.hypotheses] == [1, 2]


def test_query_node_extracts_inline_id_and_grid_aliases() -> None:
    node = QueryNode.model_validate(
        {
            "categories": ["#7 throw pillow", "#9"],
            "ids": ["#3", 7],
            "grids": ["(1, 2)", {"row": 2, "col": 3}],
        }
    )
    assert node.categories == ["throw pillow"]
    assert node.candidate_object_ids == [3, 7, 9]
    assert node.grid_cells == [(1, 2), (2, 3)]


def test_grounding_query_syncs_target_and_anchor_grids_and_heals_viewpoint() -> None:
    query = GroundingQuery(
        raw_query="the pillow on the sofa",
        root=QueryNode(
            categories=["pillow"],
            grid_cells=[(1, 2)],
            spatial_constraints=[
                SpatialConstraint(
                    relation="left_of",
                    reference_frame=ReferenceFrame.VIEWER,
                    viewpoint_context_id="missing_vp",
                    anchors=[
                        QueryNode(
                            categories=["sofa"],
                            grid_cells=[(1, 2), (1, 3)],
                        )
                    ],
                )
            ],
        ),
    )
    assert query.target_grids == [(1, 2)]
    assert query.anchor_grids == [(1, 2), (1, 3)]
    # Missing viewpoint_context_id heals to AMBIGUOUS + RANK_ONLY instead of crashing
    sc = query.root.spatial_constraints[0]
    assert sc.reference_frame == ReferenceFrame.AMBIGUOUS
    assert sc.execution_policy == ExecutionPolicy.RANK_ONLY
    assert sc.viewpoint_context_id is None


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
