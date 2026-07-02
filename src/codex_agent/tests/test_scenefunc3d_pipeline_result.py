"""Tests for the Phase-2c pipeline result payload and writers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.models import CodexTurnMetadata
from codex_agent.scenefunc3d.backends.fusion import FusedMask, FusionParams
from codex_agent.scenefunc3d.pipeline import AnchorMultiViewResult, PerFrameOutcome
from codex_agent.scenefunc3d.pipeline_result import (
    SceneFunc3dPipelineOutcome,
    write_pipeline_fused_mask,
    write_pipeline_result,
)
from codex_agent.scenefunc3d.seed_contract import SceneFunc3dSeedOutcome
from codex_agent.scenefunc3d.task import SEED_APPROVAL_ACTIONS

_SCENE_VERTICES = np.array(
    [[0.0, 0.0, 2.0], [0.01, 0.0, 2.0], [0.0, 0.01, 2.0], [9.0, 9.0, 9.0]],
    dtype=np.float64,
)


def _seed_outcome() -> SceneFunc3dSeedOutcome:
    return SceneFunc3dSeedOutcome(
        seed_frame_id="000010",
        seed_candidate_id="cand-a",
        seed_fragment_id="000010_cand-a",
        affordance_concept="the drawer handle",
        approval_actions=SEED_APPROVAL_ACTIONS,
        uncertainties=("partial occlusion",),
        seed_sam_mask_npz_path=Path("/run/sam/000010/cand-a.npz"),
    )


def _pipeline_result(point_indices: tuple[int, ...]) -> AnchorMultiViewResult:
    return AnchorMultiViewResult(
        fused=FusedMask(
            point_indices=point_indices,
            confidence=0.75,
            instances=(),
            params=FusionParams(),
        ),
        selected_frames=(),
        per_frame=(
            PerFrameOutcome(
                frame_id="000010",
                point_indices=point_indices,
                centroid=(0.0, 0.0, 2.0),
                accepted=True,
                reject_reason="",
                molmo_fallback_used=False,
            ),
            PerFrameOutcome(
                frame_id="000020",
                point_indices=(),
                centroid=None,
                accepted=False,
                reject_reason="anchor_gate",
                molmo_fallback_used=False,
            ),
            PerFrameOutcome(
                frame_id="000030",
                point_indices=point_indices,
                centroid=(0.0, 0.0, 2.0),
                accepted=True,
                reject_reason="",
                molmo_fallback_used=False,
            ),
        ),
    )


def test_write_pipeline_fused_mask_writes_npz_and_ply(tmp_path: Path) -> None:
    npz_path, ply_path = write_pipeline_fused_mask(
        tmp_path / "fused", point_indices=(0, 3), scene_vertices=_SCENE_VERTICES
    )
    assert npz_path.is_file() and ply_path.is_file()
    with np.load(npz_path) as archive:
        assert [int(v) for v in archive["point_indices"]] == [0, 3]
        assert np.array_equal(archive["points_world"], _SCENE_VERTICES[[0, 3]])


def test_write_pipeline_fused_mask_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(SceneFunc3dDataError, match="empty fused mask"):
        write_pipeline_fused_mask(
            tmp_path / "fused", point_indices=(), scene_vertices=_SCENE_VERTICES
        )
    assert not (tmp_path / "fused").exists()


def test_write_pipeline_result_writes_scorable_payload(tmp_path: Path) -> None:
    npz_path, ply_path = write_pipeline_fused_mask(
        tmp_path / "fused", point_indices=(0, 1, 2), scene_vertices=_SCENE_VERTICES
    )
    outcome = SceneFunc3dPipelineOutcome(
        seed=_seed_outcome(),
        seed_point_count=2,
        mask_npz_path=npz_path,
        mask_ply_path=ply_path,
        result=_pipeline_result((0, 1, 2)),
    )
    result_path = write_pipeline_result(
        tmp_path / "result.json",
        outcome,
        sample_id="421254::desc-a",
        turn_metadata=CodexTurnMetadata(turn_id="fake-turn", status="completed"),
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["task_name"] == "scenefunc3d_mask_generation"
    assert payload["result_kind"] == "anchor_multiview_pipeline"
    assert payload["outcome"]["mask_npz_path"] == str(npz_path)
    assert payload["outcome"]["confidence"] == pytest.approx(0.75)
    assert payload["outcome"]["selected_frame_ids"] == ["000010", "000030"]
    assert payload["outcome"]["seed"]["seed_fragment_id"] == "000010_cand-a"
    assert payload["outcome"]["seed"]["seed_point_count"] == 2
    assert payload["outcome"]["per_frame"][0]["frame_id"] == "000010"
    assert payload["outcome"]["per_frame"][1] == {
        "frame_id": "000020",
        "point_count": 0,
        "accepted": False,
        "reject_reason": "anchor_gate",
        "molmo_fallback_used": False,
    }
    assert payload["turn"]["turn_id"] == "fake-turn"
