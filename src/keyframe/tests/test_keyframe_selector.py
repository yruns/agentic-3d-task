"""End-to-end test for KeyframeSelector with a fake LLM and synthetic scene."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

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
