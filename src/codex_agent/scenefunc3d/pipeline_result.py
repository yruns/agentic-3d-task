"""Durable Phase-2c pipeline result payload + fused-mask writers.

``run_single_sample`` writes a ``result.json`` tagged
``result_kind="anchor_multiview_pipeline"`` plus the fused ``mask_data.npz`` /
``lifted_points.ply``. The fused NPZ carries ``point_indices`` so the existing
``score_mask_npz`` can score it directly (no ``FinalMaskArtifactDocument``).
``seed_point_count`` is the point count of the runner's deterministic seed lift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np

from ..errors import SceneFunc3dDataError
from .backends.lift_3d import FloatArray, write_lift_npz, write_lift_ply

if TYPE_CHECKING:
    from ..models import CodexTurnMetadata
    from .pipeline import AnchorMultiViewResult
    from .seed_contract import SceneFunc3dSeedOutcome

TASK_NAME = "scenefunc3d_mask_generation"
PIPELINE_RESULT_KIND: Literal["anchor_multiview_pipeline"] = "anchor_multiview_pipeline"


class SceneFunc3dSeedProvenancePayload(TypedDict):
    """JSON-ready confirmed-seed provenance for a pipeline result."""

    seed_frame_id: str
    seed_candidate_id: str
    seed_fragment_id: str
    affordance_concept: str
    seed_point_count: int


class SceneFunc3dFusionParamsPayload(TypedDict):
    """JSON-ready fusion parameters used for one pipeline result."""

    agreement_tau: float
    radius_scale: float
    cluster_link_eps_m: float
    min_cluster_points: int


class SceneFunc3dPerFramePayload(TypedDict):
    """JSON-ready per-frame provenance row for a pipeline result."""

    frame_id: str
    point_count: int
    accepted: bool
    reject_reason: str
    molmo_fallback_used: bool


class SceneFunc3dPipelineOutcomePayload(TypedDict):
    """JSON-ready outcome of one deterministic anchor multi-view run."""

    result_kind: Literal["anchor_multiview_pipeline"]
    mask_npz_path: str
    mask_ply_path: str
    confidence: float
    selected_frame_ids: list[str]
    seed: SceneFunc3dSeedProvenancePayload
    per_frame: list[SceneFunc3dPerFramePayload]
    fusion_params: SceneFunc3dFusionParamsPayload
    uncertainties: list[str]


class SceneFunc3dPipelineTurnPayload(TypedDict):
    """JSON-ready turn metadata for a pipeline result."""

    turn_id: str | None
    status: str | None
    duration_ms: int | None
    reasoning_summary: str | None
    run_home: str | None


class SceneFunc3dPipelineResultPayload(TypedDict):
    """JSON-ready durable ``result.json`` for one Phase-2c pipeline run."""

    task_name: str
    sample_id: str
    result_kind: Literal["anchor_multiview_pipeline"]
    outcome: SceneFunc3dPipelineOutcomePayload
    turn: SceneFunc3dPipelineTurnPayload


@dataclass(frozen=True)
class SceneFunc3dPipelineOutcome:
    """Deterministic pipeline outcome for one sample (durable result source)."""

    seed: SceneFunc3dSeedOutcome
    seed_point_count: int
    mask_npz_path: Path
    mask_ply_path: Path
    result: AnchorMultiViewResult

    def to_payload(self) -> SceneFunc3dPipelineOutcomePayload:
        """Return the JSON-serializable outcome payload."""
        fused = self.result.fused
        return {
            "result_kind": PIPELINE_RESULT_KIND,
            "mask_npz_path": str(self.mask_npz_path),
            "mask_ply_path": str(self.mask_ply_path),
            "confidence": fused.confidence,
            "selected_frame_ids": [
                outcome.frame_id
                for outcome in self.result.per_frame
                if outcome.accepted
            ],
            "seed": {
                "seed_frame_id": self.seed.seed_frame_id,
                "seed_candidate_id": self.seed.seed_candidate_id,
                "seed_fragment_id": self.seed.seed_fragment_id,
                "affordance_concept": self.seed.affordance_concept,
                "seed_point_count": self.seed_point_count,
            },
            "per_frame": [
                {
                    "frame_id": outcome.frame_id,
                    "point_count": len(outcome.point_indices),
                    "accepted": outcome.accepted,
                    "reject_reason": outcome.reject_reason,
                    "molmo_fallback_used": outcome.molmo_fallback_used,
                }
                for outcome in self.result.per_frame
            ],
            "fusion_params": {
                "agreement_tau": fused.params.agreement_tau,
                "radius_scale": fused.params.radius_scale,
                "cluster_link_eps_m": fused.params.cluster_link_eps_m,
                "min_cluster_points": fused.params.min_cluster_points,
            },
            "uncertainties": list(self.seed.uncertainties),
        }


def write_pipeline_fused_mask(
    fused_dir: Path, *, point_indices: tuple[int, ...], scene_vertices: FloatArray
) -> tuple[Path, Path]:
    """Write the fused mask NPZ (points + ids) and ASCII PLY; fail on empty."""
    if not point_indices:
        raise SceneFunc3dDataError(
            "anchor multi-view pipeline produced an empty fused mask; failing "
            f"closed instead of writing a zero-point artifact: fused_dir={fused_dir}"
        )
    index_array = np.asarray(point_indices, dtype=np.int64)
    scene_points = np.asarray(scene_vertices, dtype=np.float64)
    vertex_count = int(scene_points.shape[0])
    if int(index_array.min()) < 0 or int(index_array.max()) >= vertex_count:
        raise SceneFunc3dDataError(
            "fused mask point index is outside the scene mesh vertex range: "
            f"vertex_count={vertex_count}; min_index={int(index_array.min())}; "
            f"max_index={int(index_array.max())}; fused_dir={fused_dir}"
        )
    points_world = scene_points[index_array]
    fused_dir.mkdir(parents=True, exist_ok=True)
    mask_npz_path = write_lift_npz(
        fused_dir / "mask_data.npz", points_world, point_indices=index_array
    )
    mask_ply_path = write_lift_ply(fused_dir / "lifted_points.ply", points_world)
    return mask_npz_path, mask_ply_path


def write_pipeline_result(
    result_path: Path,
    outcome: SceneFunc3dPipelineOutcome,
    *,
    sample_id: str,
    turn_metadata: CodexTurnMetadata,
) -> Path:
    """Write the durable pipeline ``result.json`` and return its path."""
    payload: SceneFunc3dPipelineResultPayload = {
        "task_name": TASK_NAME,
        "sample_id": sample_id,
        "result_kind": PIPELINE_RESULT_KIND,
        "outcome": outcome.to_payload(),
        "turn": {
            "turn_id": turn_metadata.turn_id,
            "status": turn_metadata.status,
            "duration_ms": turn_metadata.duration_ms,
            "reasoning_summary": turn_metadata.reasoning_summary,
            "run_home": turn_metadata.run_home,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result_path


__all__ = [
    "PIPELINE_RESULT_KIND",
    "SceneFunc3dFusionParamsPayload",
    "SceneFunc3dPerFramePayload",
    "SceneFunc3dPipelineOutcome",
    "SceneFunc3dPipelineOutcomePayload",
    "SceneFunc3dPipelineResultPayload",
    "SceneFunc3dPipelineTurnPayload",
    "SceneFunc3dSeedProvenancePayload",
    "write_pipeline_fused_mask",
    "write_pipeline_result",
]
