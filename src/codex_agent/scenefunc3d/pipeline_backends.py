"""Sidecar-backed per-frame proposer for the anchor multi-view pipeline.

Implements ``pipeline.FrameProposer`` by calling the real Molmo sidecar (RPC,
no overlay), the SAM tool (smallest candidate), and the lift tool, then reading
the fragment's raw-mesh vertex ids back. Molmo emptiness is a *designed*
fallback to the projected anchor, recorded in provenance (spec Stage C step 2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from codex_agent.errors import SceneFunc3dDataError
from codex_agent.scenefunc3d.backends.config import load_backend_settings
from codex_agent.scenefunc3d.backends.frame_assets import (
    resolve_frame_geometry_assets,
)
from codex_agent.scenefunc3d.backends.frame_loader import frame_rgb_path
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry
from codex_agent.scenefunc3d.backends.molmo_rpc import request_molmo_point
from codex_agent.scenefunc3d.pipeline import FrameProposal
from codex_agent.scenefunc3d.tools.mask_lifting import LiftMaskArgs, lift_mask_to_3d
from codex_agent.scenefunc3d.tools.models import ToolInputError
from codex_agent.scenefunc3d.tools.molmo_pointing import (
    MolmoPoint,
    MolmoPointArgs,
    points_from_molmo_response,
)
from codex_agent.scenefunc3d.tools.sam_masking import (
    SamCandidate,
    SamMaskArgs,
    SamPointInput,
    sam_mask,
)
from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

# Molmo points farther than this fraction of the image diagonal from the
# projected anchor are treated as "too far" and trigger the anchor fallback.
_MAX_MOLMO_ANCHOR_DISTANCE_FRACTION = 0.15
_FRAGMENT_POINT_INDICES_KEY = "point_indices"


@dataclass(frozen=True)
class SidecarFrameProposer:
    """Propose one frame's lifted vertex ids via Molmo -> SAM -> lift."""

    scene: SceneFunc3dToolScene
    out_dir: Path
    backend_config_path: Path
    affordance_concept: str
    task_description: str

    def propose(
        self,
        *,
        frame_id: str,
        geometry: CameraGeometry,
        projected_anchor_xy: tuple[float, float] | None,
    ) -> FrameProposal:
        """Return the frame's lifted raw-mesh vertex ids (empty when unusable)."""
        rgb_path = frame_rgb_path(self.scene, frame_id)
        image_width, image_height = _image_size(rgb_path)
        prompt = (
            f"point to {self.affordance_concept} in order to "
            f"{self.task_description}"
        )
        settings = load_backend_settings(self.backend_config_path)
        response = request_molmo_point(
            settings,
            request_id=f"{frame_id}_molmo_mv",
            args=MolmoPointArgs(
                frame_id=frame_id,
                image_path=rgb_path,
                prompt=prompt,
                image_width=image_width,
                image_height=image_height,
            ),
        )
        molmo_points = points_from_molmo_response(
            response, image_width=image_width, image_height=image_height
        )
        prompt_xy, fallback_used = _select_prompt_point(
            molmo_points,
            projected_anchor_xy=projected_anchor_xy,
            image_width=image_width,
            image_height=image_height,
        )
        if prompt_xy is None:
            return FrameProposal(
                frame_id=frame_id, point_indices=(), molmo_fallback_used=fallback_used
            )
        sam_result = sam_mask(
            SamMaskArgs(
                frame_id=frame_id,
                image_path=rgb_path,
                points=(
                    SamPointInput(
                        x_px=prompt_xy[0],
                        y_px=prompt_xy[1],
                        source="anchor_fallback" if fallback_used else "molmo",
                        label=self.affordance_concept,
                    ),
                ),
            ),
            out_dir=self.out_dir,
            backend_config_path=self.backend_config_path,
        )
        smallest = _smallest_candidate(sam_result.candidates)
        if smallest is None:
            return FrameProposal(
                frame_id=frame_id, point_indices=(), molmo_fallback_used=fallback_used
            )
        assets = resolve_frame_geometry_assets(self.scene, frame_id)
        lift_result = lift_mask_to_3d(
            LiftMaskArgs(
                frame_id=frame_id,
                candidate_id=smallest.candidate_id,
                mask_path=smallest.mask_npz_path,
                depth_path=assets.depth_path,
                intrinsics_path=assets.intrinsics_path,
                pose_path=assets.pose_path,
            ),
            out_dir=self.out_dir,
            raw_mesh_path=self.scene.raw_mesh_path,
        )
        point_indices = _read_fragment_point_indices(lift_result.mask_npz_path)
        return FrameProposal(
            frame_id=frame_id,
            point_indices=point_indices,
            molmo_fallback_used=fallback_used,
        )


def _select_prompt_point(
    molmo_points: tuple[MolmoPoint, ...],
    *,
    projected_anchor_xy: tuple[float, float] | None,
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float] | None, bool]:
    diagonal = float(np.hypot(image_width, image_height))
    max_distance = _MAX_MOLMO_ANCHOR_DISTANCE_FRACTION * diagonal
    if projected_anchor_xy is None:
        if molmo_points:
            return (molmo_points[0].x_px, molmo_points[0].y_px), False
        return None, False
    if molmo_points:
        anchor = np.asarray(projected_anchor_xy, dtype=np.float64)
        distances = [
            float(np.hypot(point.x_px - anchor[0], point.y_px - anchor[1]))
            for point in molmo_points
        ]
        best = int(np.argmin(distances))
        if distances[best] <= max_distance:
            return (molmo_points[best].x_px, molmo_points[best].y_px), False
    return projected_anchor_xy, True


def _smallest_candidate(candidates: Sequence[SamCandidate]) -> SamCandidate | None:
    usable = [candidate for candidate in candidates if candidate.pixel_count > 0]
    if not usable:
        return None
    return min(usable, key=lambda candidate: candidate.pixel_count)


def _image_size(rgb_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ToolInputError(
            "Pillow is required to read frame image size; install the 'vision' extra"
        ) from exc
    with Image.open(rgb_path) as image:
        return int(image.width), int(image.height)


def _read_fragment_point_indices(npz_path: Path) -> tuple[int, ...]:
    try:
        with np.load(npz_path) as archive:
            if _FRAGMENT_POINT_INDICES_KEY not in archive.files:
                raise SceneFunc3dDataError(
                    "lift fragment NPZ is missing required key "
                    f"{_FRAGMENT_POINT_INDICES_KEY!r}: npz_path={npz_path}"
                )
            indices = (
                np.asarray(archive[_FRAGMENT_POINT_INDICES_KEY])
                .astype(np.int64)
                .ravel()
            )
    except SceneFunc3dDataError:
        raise
    except (BadZipFile, OSError, ValueError) as exc:
        raise SceneFunc3dDataError(
            "could not load lift fragment NPZ: "
            f"npz_path={npz_path}; error_type={exc.__class__.__name__}"
        ) from exc
    return tuple(int(value) for value in indices)


__all__ = ["SidecarFrameProposer"]
