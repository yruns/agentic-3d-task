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
from keyframe.spatial.grid import SceneGridSpec


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


def test_grid_and_object_id_priors_break_ties() -> None:
    scene = _scene()
    spec = SceneGridSpec.from_scene_geometry(scene)
    far_pillow_cell = spec.world_to_cell(6.0, 6.0, clamp=True)
    assert far_pillow_cell is not None

    # Without spatial constraints, pillow #1 (at 0,0) and pillow #3 (at 6,6) both match "pillow".
    # Specifying target_grids=[far_pillow_cell] prioritizes pillow #3!
    query_grid = GroundingQuery(
        raw_query="the pillow in the corner",
        expect_unique=True,
        root=QueryNode(categories=["pillow"], node_id="root"),
        target_grids=[far_pillow_cell],
    )
    res_grid = QueryExecutor(scene, grid_spec=spec).execute(query_grid)
    assert res_grid.best_object is not None
    assert res_grid.best_object.obj_id == 3

    # Specifying candidate_object_ids=[3] also prioritizes pillow #3
    query_id = GroundingQuery(
        raw_query="pillow #3",
        expect_unique=True,
        root=QueryNode(categories=["pillow"], candidate_object_ids=[3], node_id="root"),
    )
    res_id = QueryExecutor(scene, grid_spec=spec).execute(query_id)
    assert res_id.best_object is not None
    assert res_id.best_object.obj_id == 3


def test_virtual_grid_anchor_evaluates_unknown_anchor_geometry() -> None:
    scene = _scene()
    spec = SceneGridSpec.from_scene_geometry(scene)
    origin_cell = spec.world_to_cell(0.0, 0.0, clamp=True)
    assert origin_cell is not None

    # Anchor category is UNKNOW (e.g. "fireplace" not in scene), but VLM localized it to origin_cell.
    # Pillow #1 is at (0, 0) near origin_cell; Pillow #3 is far away at (6, 6).
    query = GroundingQuery(
        raw_query="the pillow near the fireplace",
        expect_unique=True,
        root=QueryNode(
            categories=["pillow"],
            node_id="root",
            spatial_constraints=[
                SpatialConstraint(
                    relation="near",
                    anchors=[
                        QueryNode(
                            categories=["UNKNOW"],
                            grid_cells=[origin_cell],
                            node_id="root_sc0_a0",
                        )
                    ],
                )
            ],
        ),
    )
    result = QueryExecutor(scene, grid_spec=spec).execute(query)
    assert [o.obj_id for o in result.matched_objects] == [1]
