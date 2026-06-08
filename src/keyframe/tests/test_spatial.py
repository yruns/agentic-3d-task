"""Tests for spatial relation checking, quick filters, viewpoint and frustum."""

from __future__ import annotations

import numpy as np

from keyframe.models import SceneObject
from keyframe.spatial import (
    AttributeFilter,
    QuickFilters,
    SpatialRelationChecker,
    frustum_overlap_l1,
    resolve_viewer_pose,
    viewer_frame_relation_score,
)


def _obj(
    obj_id: int,
    category: str,
    centroid: list[float],
    bbox_center: list[float] | None = None,
) -> SceneObject:
    bbox = None
    if bbox_center is not None:
        cx, cy, cz = bbox_center
        bbox = np.array(
            [[cx - 0.2, cy - 0.2, cz - 0.2], [cx + 0.2, cy + 0.2, cz + 0.2]]
        )
    return SceneObject(
        obj_id=obj_id, category=category, centroid=np.array(centroid), bbox_np=bbox
    )


def test_checker_on_relation_true() -> None:
    sofa = _obj(0, "sofa", [0, 0, 0.25], [0, 0, 0.25])
    pillow = _obj(1, "pillow", [0, 0, 0.6], [0, 0, 0.6])
    result = SpatialRelationChecker().check(pillow, sofa, "on")
    assert result.satisfies
    assert result.score > 0


def test_checker_near_and_far() -> None:
    checker = SpatialRelationChecker()
    anchor = _obj(0, "table", [0, 0, 0])
    near = _obj(1, "cup", [0.5, 0, 0])
    far = _obj(2, "lamp", [9, 9, 0])
    assert checker.check(near, anchor, "near").satisfies
    assert not checker.check(far, anchor, "near").satisfies


def test_checker_between() -> None:
    checker = SpatialRelationChecker()
    left = _obj(0, "table", [-2, 0, 0])
    right = _obj(1, "fridge", [2, 0, 0])
    middle = _obj(2, "bag", [0, 0, 0])
    assert checker.check(middle, [left, right], "between").satisfies


def test_quick_filter_vertical_keeps_objects_above_anchor() -> None:
    qf = QuickFilters()
    anchor = _obj(0, "table", [0, 0, 0.5])
    above = _obj(1, "cup", [0, 0, 1.0])
    below = _obj(2, "rug", [0, 0, 0.0])
    kept = {o.obj_id for o in qf.filter_candidates([above, below], [anchor], "on")}
    assert kept == {1}


def test_attribute_filter_color_and_size() -> None:
    af = AttributeFilter()
    assert af.knows_color("red")
    red = _obj(0, "red chair", [0, 0, 0])
    red.summary = "a bright red chair"
    blue = _obj(1, "chair", [0, 0, 0])
    kept = af.filter_by_color([red, blue], "red")
    assert [o.obj_id for o in kept] == [0]


def test_frustum_self_overlap_is_one() -> None:
    pose = np.eye(4)
    k = np.array([[600, 0, 640], [0, 600, 360], [0, 0, 1]], dtype=float)
    assert frustum_overlap_l1(pose, pose, k, (1280, 720)) == 1.0


def test_viewer_frame_relation_score_right_of() -> None:
    # Viewer at origin facing +Y; anchor ahead, candidate to the right (+X).
    anchor = _obj(0, "door", [0, 2, 0])
    pose = resolve_viewer_pose(
        [anchor],
        camera_poses=[],
        all_object_centroids=np.array([[0, 0, 0], [0, 2, 0]]),
        allow_geometric_fallback=True,
    )
    assert pose is not None
    right_score = viewer_frame_relation_score(
        "right_of", np.array([2.0, 2.0, 0.0]), np.array([0.0, 2.0, 0.0]), pose
    )
    left_score = viewer_frame_relation_score(
        "left_of", np.array([2.0, 2.0, 0.0]), np.array([0.0, 2.0, 0.0]), pose
    )
    assert right_score > left_score
