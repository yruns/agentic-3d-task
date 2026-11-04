"""Tests for spatial relation checking, quick filters, viewpoint and frustum."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from keyframe.models import SceneObject
from keyframe.spatial import (
    AttributeFilter,
    QuickFilters,
    SceneGridSpec,
    SpatialRelationChecker,
    build_grid_visibility_index,
    compute_grid_bounds,
    frustum_overlap_l1,
    load_grid_visibility_index,
    resolve_viewer_pose,
    save_grid_visibility_index,
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


def test_scene_grid_spec_world_cell_roundtrip() -> None:
    spec = SceneGridSpec(min_x=-2.0, max_x=2.0, min_y=-2.0, max_y=2.0, rows=4, cols=4)
    # Top-left cell (0, 0) covers x in [-2, -1], y in [1, 2].
    assert spec.world_to_cell(-1.5, 1.5) == (0, 0)
    # Bottom-right cell (3, 3) covers x in [1, 2], y in [-2, -1].
    assert spec.world_to_cell(1.5, -1.5) == (3, 3)
    cx, cy, _cz = spec.cell_center_world(1, 2)
    assert spec.world_to_cell(cx, cy) == (1, 2)
    assert compute_grid_bounds([(0, 2), (2, 1)]) == (0, 2, 1, 2)


def test_cells_for_footprint_dual_coverage_drops_edge_slivers() -> None:
    # 4x4 grid over [-2, 2] x [-2, 2], each cell is 1.0m x 1.0m.
    spec = SceneGridSpec(min_x=-2.0, max_x=2.0, min_y=-2.0, max_y=2.0, rows=4, cols=4)
    # Box mostly inside cell (1, 1) (x in [-1, 0], y in [0, 1]) with a 3cm sliver into (1, 2) (x > 0).
    cells = spec.cells_for_footprint(-0.8, 0.2, 0.03, 0.9, centroid_xy=(-0.4, 0.55))
    assert cells == [(1, 1)]
    # Box spanning substantially across (1, 1) and (1, 2) keeps both, with centroid cell first.
    spanning = spec.cells_for_footprint(-0.6, 0.2, 0.4, 0.9, centroid_xy=(-0.2, 0.55))
    assert spanning == [(1, 1), (1, 2)]


def test_depth_verified_grid_visibility_rejects_occluded_cells(
    tmp_path: Path,
) -> None:
    spec = SceneGridSpec(
        min_x=-2.0,
        max_x=2.0,
        min_y=0.0,
        max_y=4.0,
        min_z=0.0,
        max_z=2.0,
        rows=2,
        cols=2,
    )
    # Camera at (0, 0, 1) looking along +Y:
    # camera X = +world X, camera Y = -world Z, camera Z = +world Y.
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = [1.0, 0.0, 0.0]
    pose[:3, 1] = [0.0, 0.0, -1.0]
    pose[:3, 2] = [0.0, 1.0, 0.0]
    pose[:3, 3] = [0.0, 0.0, 1.0]
    k = np.array([[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]])

    # A wall at depth 1.2m (y=1.2, inside row=1 near cells) occludes row=0 (y in [2, 4]).
    near_wall = np.full((240, 320), 1.2, dtype=np.float64)
    idx_occluded = build_grid_visibility_index(
        spec,
        objects=[],
        camera_poses=[pose],
        intrinsic=k,
        img_wh=(320, 240),
        depth_loader=lambda _vid: near_wall,
    )
    assert idx_occluded.depth_verified_views == 1
    visible_cells = set(idx_occluded.covered_grids_for_view(0))
    # Near row (row=1, y in [0, 2]) is visible; far row (row=0, y in [2, 4]) is occluded.
    assert (1, 0) in visible_cells or (1, 1) in visible_cells
    assert (0, 0) not in visible_cells
    assert (0, 1) not in visible_cells

    # Cache round-trip via save_grid_visibility_index + load_grid_visibility_index.
    cache_file = tmp_path / "indices" / "grid_visibility_index.pkl"
    save_grid_visibility_index(idx_occluded, cache_file)
    assert cache_file.is_file()
    reloaded = load_grid_visibility_index(cache_file)
    assert reloaded.covered_grids_for_view(0) == idx_occluded.covered_grids_for_view(0)
