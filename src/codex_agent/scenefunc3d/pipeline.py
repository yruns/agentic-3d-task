"""Deterministic anchor-centric multi-view mask pipeline (Stages B/C/D).

The pure driver ``fuse_selected_frames`` depends only on injected callables
(a ``FrameProposer`` and a visibility counter) plus in-memory geometry, so it is
unit-testable without a scene, sidecar, or the ModelHub adapter. The scene
wrapper ``run_anchor_multiview_pipeline`` wires the disk/sidecar-backed
implementations (frame selection, geometry loading, per-vertex visibility).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from codex_agent.scenefunc3d.backends.anchor import TargetAnchor
from codex_agent.scenefunc3d.backends.fusion import (
    FrameLift,
    FusedMask,
    FusionParams,
    MultiViewLiftBundle,
    fuse_multiview_points,
)
from codex_agent.scenefunc3d.backends.lift_3d import CameraGeometry, FloatArray
from codex_agent.scenefunc3d.backends.visibility import (
    FrameVisibility,
    project_world_to_pixels,
)

if TYPE_CHECKING:
    from codex_agent.scenefunc3d.tools.scene_context import SceneFunc3dToolScene

VisibilityCounter = Callable[
    [Sequence[int], FloatArray, Sequence[str]], Mapping[int, int]
]

_DEFAULT_FRAME_CAP = 30
# Inert placeholder frame id for the empty-result bundle: satisfies FrameLift's
# non-empty-id and MultiViewLiftBundle's non-empty-frames validators while
# carrying zero lifted points, so fusion yields an empty mask.
_EMPTY_BUNDLE_FRAME_ID = "__none__"


@dataclass(frozen=True)
class FrameProposal:
    """One frame's proposed lift: raw-mesh vertex ids + Molmo-fallback flag."""

    frame_id: str
    point_indices: tuple[int, ...]
    molmo_fallback_used: bool


class FrameProposer(Protocol):
    """Proposes lifted raw-mesh vertex ids for one frame (Molmo->SAM->lift)."""

    def propose(
        self,
        *,
        frame_id: str,
        geometry: CameraGeometry,
        projected_anchor_xy: tuple[float, float] | None,
    ) -> FrameProposal:
        """Return the frame's lifted vertex ids (empty tuple when it found none)."""
        ...


@dataclass(frozen=True)
class PerFrameOutcome:
    """Provenance for one processed frame."""

    frame_id: str
    point_indices: tuple[int, ...]
    centroid: tuple[float, float, float] | None
    accepted: bool
    reject_reason: str
    molmo_fallback_used: bool


@dataclass(frozen=True)
class AnchorMultiViewResult:
    """Fused mask plus per-frame provenance for one sample."""

    fused: FusedMask
    selected_frames: tuple[FrameVisibility, ...]
    per_frame: tuple[PerFrameOutcome, ...]


def _project_anchor_xy(
    anchor: TargetAnchor, geometry: CameraGeometry
) -> tuple[float, float] | None:
    pixels, camera_z = project_world_to_pixels(
        np.asarray([anchor.centroid], dtype=np.float64), geometry
    )
    u, v = float(pixels[0, 0]), float(pixels[0, 1])
    if camera_z[0] <= 0.0 or not (np.isfinite(u) and np.isfinite(v)):
        return None
    return (u, v)


def fuse_selected_frames(
    *,
    anchor: TargetAnchor,
    scene_vertices: FloatArray,
    selected_frames: Sequence[FrameVisibility],
    geometry_by_frame: Mapping[str, CameraGeometry],
    proposer: FrameProposer,
    params: FusionParams,
    visibility_counts_fn: VisibilityCounter,
    gate_radius_scale: float = 1.0,
) -> AnchorMultiViewResult:
    """Run Stage C (per-frame propose + anchor gate) and Stage D (fusion)."""
    vertices = np.asarray(scene_vertices, dtype=np.float64)
    centroid = np.asarray(anchor.centroid, dtype=np.float64)
    gate_radius = anchor.radius_m * gate_radius_scale

    per_frame: list[PerFrameOutcome] = []
    accepted_lifts: list[FrameLift] = []
    for frame_vis in selected_frames:
        geometry = geometry_by_frame[frame_vis.frame_id]
        projected_xy = _project_anchor_xy(anchor, geometry)
        proposal = proposer.propose(
            frame_id=frame_vis.frame_id,
            geometry=geometry,
            projected_anchor_xy=projected_xy,
        )
        outcome = _evaluate_proposal(
            proposal, vertices=vertices, centroid=centroid, gate_radius=gate_radius
        )
        per_frame.append(outcome)
        if outcome.accepted:
            accepted_lifts.append(
                FrameLift(
                    frame_id=proposal.frame_id,
                    point_indices=proposal.point_indices,
                )
            )

    if not accepted_lifts:
        fused = fuse_multiview_points(
            MultiViewLiftBundle(
                frames=(FrameLift(frame_id=_EMPTY_BUNDLE_FRAME_ID, point_indices=()),),
                anchor=anchor,
            ),
            vertices,
            params,
        )
    else:
        union_ids = sorted(
            {index for lift in accepted_lifts for index in lift.point_indices}
        )
        union_coords = vertices[union_ids]
        accepted_frame_ids = [lift.frame_id for lift in accepted_lifts]
        counts = visibility_counts_fn(union_ids, union_coords, accepted_frame_ids)
        bundle = MultiViewLiftBundle(
            frames=tuple(accepted_lifts),
            anchor=anchor,
            visibility_counts=dict(counts),
        )
        fused = fuse_multiview_points(bundle, vertices, params)

    return AnchorMultiViewResult(
        fused=fused,
        selected_frames=tuple(selected_frames),
        per_frame=tuple(per_frame),
    )


def _evaluate_proposal(
    proposal: FrameProposal,
    *,
    vertices: FloatArray,
    centroid: FloatArray,
    gate_radius: float,
) -> PerFrameOutcome:
    if not proposal.point_indices:
        return PerFrameOutcome(
            frame_id=proposal.frame_id,
            point_indices=(),
            centroid=None,
            accepted=False,
            reject_reason="empty_lift",
            molmo_fallback_used=proposal.molmo_fallback_used,
        )
    frame_centroid = vertices[list(proposal.point_indices)].mean(axis=0)
    distance = float(np.linalg.norm(frame_centroid - centroid))
    accepted = distance <= gate_radius
    return PerFrameOutcome(
        frame_id=proposal.frame_id,
        point_indices=proposal.point_indices,
        centroid=(
            float(frame_centroid[0]),
            float(frame_centroid[1]),
            float(frame_centroid[2]),
        ),
        accepted=accepted,
        reject_reason="" if accepted else "anchor_gate",
        molmo_fallback_used=proposal.molmo_fallback_used,
    )


def run_anchor_multiview_pipeline(
    *,
    anchor: TargetAnchor,
    scene: SceneFunc3dToolScene,
    scene_vertices: FloatArray,
    proposer: FrameProposer,
    params: FusionParams,
    frame_cap: int = _DEFAULT_FRAME_CAP,
    depth_tolerance: float = 0.25,
    gate_radius_scale: float = 1.0,
) -> AnchorMultiViewResult:
    """Scene-backed Stage B->C->D: select frames, propose, gate, fuse."""
    from codex_agent.scenefunc3d.backends.frame_loader import load_frame_geometry
    from codex_agent.scenefunc3d.backends.visibility import (
        count_vertex_visibility,
        select_scene_visible_frames,
    )

    selected = select_scene_visible_frames(
        anchor, scene, frame_cap=frame_cap, depth_tolerance=depth_tolerance
    )
    geometry_by_frame = {
        frame_vis.frame_id: load_frame_geometry(scene, frame_vis.frame_id)
        for frame_vis in selected
    }

    def _counts(
        vertex_ids: Sequence[int],
        vertex_coords: FloatArray,
        frame_ids: Sequence[str],
    ) -> Mapping[int, int]:
        return count_vertex_visibility(
            vertex_ids,
            vertex_coords,
            scene,
            frame_ids,
            depth_tolerance=depth_tolerance,
        )

    return fuse_selected_frames(
        anchor=anchor,
        scene_vertices=scene_vertices,
        selected_frames=selected,
        geometry_by_frame=geometry_by_frame,
        proposer=proposer,
        params=params,
        visibility_counts_fn=_counts,
        gate_radius_scale=gate_radius_scale,
    )


__all__ = [
    "AnchorMultiViewResult",
    "FrameProposal",
    "FrameProposer",
    "PerFrameOutcome",
    "VisibilityCounter",
    "fuse_selected_frames",
    "run_anchor_multiview_pipeline",
]
