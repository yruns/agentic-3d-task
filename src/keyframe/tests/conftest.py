"""Shared fixtures: a fake LLM client and a synthetic scene builder."""

from __future__ import annotations

import gzip
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from langchain_core.messages import AIMessage
from numpy.typing import NDArray

from keyframe.llm import LLMClient

# A canned parser response: "the pillow on the sofa".
PILLOW_ON_SOFA = json.dumps(
    {
        "format_version": "hypothesis_output_v1",
        "parse_mode": "single",
        "hypotheses": [
            {
                "kind": "direct",
                "rank": 1,
                "grounding_query": {
                    "raw_query": "the pillow on the sofa",
                    "expect_unique": True,
                    "root": {
                        "categories": ["pillow"],
                        "spatial_constraints": [
                            {"relation": "on", "anchors": [{"categories": ["sofa"]}]}
                        ],
                    },
                },
                "lexical_hints": ["pillow", "sofa"],
            }
        ],
    }
)


class _FakeConfig:
    default_model = "fake-model"


class _FakeChatModel:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def invoke(self, _messages: object) -> AIMessage:
        return AIMessage(content=self._payload)


class FakeLLMClient:
    """Duck-typed stand-in for ``LLMClient`` returning a fixed JSON payload."""

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.config = _FakeConfig()

    def key_count(self, model: str | None = None) -> int:
        return 1

    def get_chat_model(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> _FakeChatModel:
        return _FakeChatModel(self._payload)


def _cube(center: tuple[float, float, float], half: float) -> NDArray[np.float64]:
    cx, cy, cz = center
    corners = [
        [cx + dx, cy + dy, cz + dz]
        for dx in (-half, half)
        for dy in (-half, half)
        for dz in (-half, half)
    ]
    return np.array(corners, dtype=np.float64)


def make_fake_llm(payload: str) -> LLMClient:
    """Return a ``FakeLLMClient`` typed as ``LLMClient`` for use in tests."""
    return cast(LLMClient, FakeLLMClient(payload))


@pytest.fixture
def fake_llm() -> LLMClient:
    return make_fake_llm(PILLOW_ON_SOFA)


@pytest.fixture
def build_scene(tmp_path: Path) -> Callable[[], Path]:
    """Build a minimal ConceptGraph-style scene: sofa, pillow-on-sofa, door."""

    def _build() -> Path:
        scene = tmp_path / "scene"
        (scene / "pcd_saves").mkdir(parents=True)
        (scene / "results").mkdir(parents=True)

        objects = [
            {
                "class_name": ["sofa"] * 3,
                "pcd_np": _cube((0.0, 0.0, 0.3), 0.4),
                "bbox_np": _cube((0.0, 0.0, 0.3), 0.4),
                "image_idx": [0, 1],
                "xyxy": np.array(
                    [[100, 100, 500, 400], [100, 100, 500, 400]], dtype=float
                ),
                "num_detections": 2,
            },
            {
                "class_name": ["pillow"] * 3,
                "pcd_np": _cube((0.0, 0.0, 0.65), 0.15),
                "bbox_np": _cube((0.0, 0.0, 0.65), 0.15),
                "image_idx": [0, 1],
                "xyxy": np.array(
                    [[200, 150, 300, 250], [200, 150, 300, 250]], dtype=float
                ),
                "num_detections": 2,
            },
            {
                "class_name": ["door"] * 2,
                "pcd_np": _cube((4.0, 0.0, 1.0), 0.3),
                "bbox_np": _cube((4.0, 0.0, 1.0), 0.3),
                "image_idx": [2],
                "xyxy": np.array([[10, 10, 100, 400]], dtype=float),
                "num_detections": 1,
            },
        ]
        with gzip.open(scene / "pcd_saves" / "full_scene_post.pkl.gz", "wb") as handle:
            pickle.dump({"objects": objects}, handle)

        enrichment = {
            "objects": [
                {
                    "obj_id": 0,
                    "status": "success",
                    "enrichment": {"category": "sofa", "description": "a sofa"},
                },
                {
                    "obj_id": 1,
                    "status": "success",
                    "enrichment": {"category": "pillow", "description": "a pillow"},
                },
                {
                    "obj_id": 2,
                    "status": "success",
                    "enrichment": {"category": "door", "description": "a door"},
                },
            ]
        }
        (scene / "enriched_objects.json").write_text(json.dumps(enrichment))

        pose_lines = []
        for x in (-1.0, 0.0, 4.0):
            pose = np.eye(4)
            pose[:3, 3] = [x, -2.0, 1.0]
            pose_lines.append(" ".join(str(v) for v in pose.flatten()))
        (scene / "traj.txt").write_text("\n".join(pose_lines))

        for i in range(3):
            (scene / "results" / f"frame{i:06d}.jpg").write_bytes(b"\xff\xd8\xff")
        return scene

    return _build
