"""End-to-end test for KeyframeSelector with a fake LLM and synthetic scene."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from keyframe import KeyframeSelector
from keyframe.llm import LLMClient
from keyframe.models import GroundingStatus

from .conftest import make_fake_llm


def test_select_keyframes_v2_end_to_end(
    build_scene: Callable[[], Path], fake_llm: LLMClient
) -> None:
    scene = build_scene()
    selector = KeyframeSelector.from_scene_path(scene, stride=1, llm_client=fake_llm)

    assert set(selector.scene_categories) == {"sofa", "pillow", "door"}
    assert len(selector.camera_poses) == 3

    result = selector.select_keyframes_v2("the pillow on the sofa", k=2)

    assert result.target_term == "pillow"
    assert result.anchor_term == "sofa"
    assert 1 in [o.obj_id for o in result.target_objects]
    assert result.metadata.status == GroundingStatus.DIRECT_GROUNDED
    assert result.metadata.selected_hypothesis_rank == 1
    assert len(result.keyframe_indices) >= 1
    # selected views should resolve to real frame files
    assert len(result.keyframe_paths) == len(result.keyframe_indices)
    # grid range and visible object IDs attached per selected frame
    assert result.target_grids
    assert result.metadata.grid_shape is not None
    for fm in result.metadata.frame_mappings:
        assert fm.resolved_view_id in result.frame_object_ids
        assert fm.resolved_view_id in result.frame_covered_grids
        assert fm.covered_grids
        assert fm.grid_bounds is not None
        min_r, max_r, min_c, max_c = fm.grid_bounds
        assert 0 <= min_r <= max_r < result.metadata.grid_shape[0]
        assert 0 <= min_c <= max_c < result.metadata.grid_shape[1]


def test_select_keyframes_v2_grid_grounded_when_target_unknown(
    build_scene: Callable[[], Path],
) -> None:
    # Even when target is UNKNOW, VLM-selected target_grids recall views of that region.
    scene = build_scene()
    base_selector = KeyframeSelector.from_scene_path(scene, stride=1)
    # Find the grid cell where object 2 (door at x=4.0, y=0.0) resides.
    door_cells = base_selector.object_to_grids[2]
    assert door_cells
    door_cell = list(door_cells[0])

    grid_payload = json.dumps(
        {
            "format_version": "hypothesis_output_v1",
            "parse_mode": "single",
            "hypotheses": [
                {
                    "kind": "direct",
                    "rank": 1,
                    "grounding_query": {
                        "raw_query": "the poster near the entrance",
                        "expect_unique": True,
                        "root": {"categories": ["UNKNOW"]},
                        "target_grids": [door_cell],
                    },
                }
            ],
        }
    )
    selector = KeyframeSelector.from_scene_path(
        scene, stride=1, llm_client=make_fake_llm(grid_payload)
    )
    result = selector.select_keyframes_v2("the poster near the entrance", k=1)
    assert result.metadata.status == GroundingStatus.GRID_GROUNDED
    assert 2 in result.keyframe_indices
    assert tuple(door_cell) in result.target_grids
    assert result.metadata.frame_mappings[0].source == "grid"
    assert 2 in result.metadata.frame_mappings[0].visible_object_ids


def test_select_keyframes_v2_unknown_anchor_with_grid_executes_direct(
    build_scene: Callable[[], Path],
) -> None:
    # When an anchor is UNKNOW (e.g. "window" not in scene) but has grid_cells,
    # DIRECT hypothesis is NOT skipped and evaluates using a virtual grid anchor!
    scene = build_scene()
    base_selector = KeyframeSelector.from_scene_path(scene, stride=1)
    sofa_cell = list(base_selector.object_to_grids[0][0])

    payload = json.dumps(
        {
            "format_version": "hypothesis_output_v1",
            "parse_mode": "single",
            "hypotheses": [
                {
                    "kind": "direct",
                    "rank": 1,
                    "grounding_query": {
                        "raw_query": "the pillow near the window",
                        "expect_unique": True,
                        "root": {
                            "categories": ["#1 pillows"],
                            "spatial_constraints": [
                                {
                                    "relation": "near",
                                    "anchors": [
                                        {
                                            "categories": ["window"],
                                            "grid_cells": [sofa_cell],
                                        }
                                    ],
                                }
                            ],
                        },
                    },
                }
            ],
        }
    )
    selector = KeyframeSelector.from_scene_path(
        scene, stride=1, llm_client=make_fake_llm(payload)
    )
    result = selector.select_keyframes_v2("the pillow near the window", k=2)
    assert result.metadata.status == GroundingStatus.DIRECT_GROUNDED
    assert [o.obj_id for o in result.target_objects] == [1]


def test_select_keyframes_v2_no_match(build_scene: Callable[[], Path]) -> None:
    # A target category absent from the scene sanitizes to UNKNOW -> no evidence.
    lamp_payload = json.dumps(
        {
            "format_version": "hypothesis_output_v1",
            "parse_mode": "single",
            "hypotheses": [
                {
                    "kind": "direct",
                    "rank": 1,
                    "grounding_query": {
                        "raw_query": "the lamp",
                        "expect_unique": True,
                        "root": {"categories": ["lamp"]},
                    },
                }
            ],
        }
    )
    scene = build_scene()
    selector = KeyframeSelector.from_scene_path(
        scene, stride=1, llm_client=make_fake_llm(lamp_payload)
    )
    result = selector.select_keyframes_v2("the lamp", k=2)
    assert result.metadata.status == GroundingStatus.NO_EVIDENCE
    assert result.keyframe_indices == []


def test_missing_visibility_index_raises_file_not_found(
    build_scene: Callable[[], Path],
) -> None:
    scene = build_scene()
    (scene / "indices" / "visibility_index.pkl").unlink()
    with pytest.raises(FileNotFoundError, match="visibility_index.pkl"):
        KeyframeSelector.from_scene_path(scene, stride=1)


def test_missing_grid_visibility_index_raises_file_not_found(
    build_scene: Callable[[], Path],
) -> None:
    scene = build_scene()
    (scene / "indices" / "grid_visibility_index.pkl").unlink()
    with pytest.raises(FileNotFoundError, match="grid_visibility_index.pkl"):
        KeyframeSelector.from_scene_path(scene, stride=1)
