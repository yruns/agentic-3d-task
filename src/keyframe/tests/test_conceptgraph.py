"""Tests for ConceptGraph object loading and the lightweight cache."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from keyframe.lightweight_conceptgraph import (
    load_scene_objects,
    write_lightweight_conceptgraph_cache,
)


def test_load_scene_objects(build_scene: Callable[[], Path]) -> None:
    scene = build_scene()
    pcd = scene / "pcd_saves" / "full_scene_post.pkl.gz"
    objects = load_scene_objects(pcd, prefer_lightweight=False)
    assert len(objects) == 3
    assert {o.category for o in objects} == {"sofa", "pillow", "door"}
    assert all(o.centroid is not None for o in objects)


def test_lightweight_cache_round_trip(build_scene: Callable[[], Path]) -> None:
    scene = build_scene()
    pcd = scene / "pcd_saves" / "full_scene_post.pkl.gz"
    cache_path = write_lightweight_conceptgraph_cache(pcd)
    assert cache_path.exists()
    objects = load_scene_objects(pcd, prefer_lightweight=True)
    assert len(objects) == 3
    # point cloud is dropped from the lightweight cache, geometry survives via bbox/centroid
    assert all(o.pcd_np is None for o in objects)
    assert all(o.centroid is not None for o in objects)
