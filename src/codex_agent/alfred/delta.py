"""Incremental scene memory watermarking and compact SceneDelta computation.

Instead of resending the entire accumulated 3D object catalog and every
intermediate frame on each step, ``EpisodeExecutor`` records a
:class:`RevisionWatermark` before executing a high-level macro command and
returns a compact :class:`SceneDelta` summarizing:
* how many primitive simulator actions were executed during the macro command,
* the agent's updated pose, region, and held object,
* the objects visible and ready for interaction in the current view,
* newly discovered objects, updated object states/confirmations, merged ID
  aliases, newly observed map regions, and top reachable frontiers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .exploration import suggest_frontiers
from .geometry import GridCell

if TYPE_CHECKING:
    from .memory import IncrementalSceneMemory
    from .object_store import ObjectMemory


@dataclass(frozen=True)
class ObjectDigest:
    """Snapshot of one object's externally relevant state at a watermark."""

    object_id: int
    category: str
    region: GridCell | None
    centroid: tuple[float, float, float] | None
    state: str
    provisional: bool
    geometry_valid: bool
    action_states: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class RevisionWatermark:
    """Memory state watermark captured at the end of a high-level tool call."""

    frame_id: int
    objects: dict[int, ObjectDigest]
    regions: frozenset[GridCell]
    aliases: dict[int, int]
    held_object_id: int | None


def _digest_object(memory: IncrementalSceneMemory, node: ObjectMemory) -> ObjectDigest:
    centroid: tuple[float, float, float] | None = None
    region: GridCell | None = None
    if node.geometry_valid:
        raw_c = node.centroid
        centroid = (
            round(float(raw_c[0]), 3),
            round(float(raw_c[1]), 3),
            round(float(raw_c[2]), 3),
        )
        region = memory.world_to_region(float(raw_c[0]), float(raw_c[1]))
    action_states = tuple(
        sorted((name, state.value) for name, state in node.action_states.items())
    )
    return ObjectDigest(
        object_id=node.object_id,
        category=node.category,
        region=region,
        centroid=centroid,
        state=node.state,
        provisional=node.provisional,
        geometry_valid=node.geometry_valid,
        action_states=action_states,
    )


def capture_watermark(memory: IncrementalSceneMemory) -> RevisionWatermark:
    """Capture an immutable watermark of the current scene memory."""
    return RevisionWatermark(
        frame_id=memory._latest_frame_id,
        objects={
            object_id: _digest_object(memory, node)
            for object_id, node in memory.objects.items()
        },
        regions=frozenset(memory.grid_to_views),
        aliases=dict(memory.aliases),
        held_object_id=memory.held_object_id,
    )


def format_visible_objects_now(
    memory: IncrementalSceneMemory,
    *,
    image_size: tuple[int, int] = (300, 300),
) -> list[dict[str, object]]:
    """Return objects detected in the latest frame with distance, bbox_norm, and mask readiness."""
    width, height = image_size
    camera_pos = (
        np.asarray(memory.frames[memory._latest_frame_id].camera_position)
        if memory._latest_frame_id in memory.frames
        else (
            np.asarray(memory.agent_position)
            if memory.agent_position is not None
            else np.zeros(3)
        )
    )
    by_object_id: dict[int, tuple[float, int, dict[str, object]]] = {}
    for det_idx, obj_id in sorted(memory.current_detection_objects.items()):
        resolved_id = memory.resolve_object_id(obj_id)
        if resolved_id not in memory.objects:
            continue
        node = memory.objects[resolved_id]
        eff_mask = memory.current_detection_masks.get(det_idx)
        raw_mask = memory.current_raw_detection_masks.get(det_idx, eff_mask)
        mask_pixels = int(eff_mask.sum()) if eff_mask is not None else 0
        raw_pixels = int(raw_mask.sum()) if raw_mask is not None else mask_pixels
        box = memory.current_detection_boxes.get(det_idx, (0.0, 0.0, 0.0, 0.0))
        conf = memory.current_detection_confidences.get(det_idx, 0.0)
        bbox_norm = [
            round(max(0.0, min(1.0, box[0] / max(1, width))), 3),
            round(max(0.0, min(1.0, box[1] / max(1, height))), 3),
            round(max(0.0, min(1.0, box[2] / max(1, width))), 3),
            round(max(0.0, min(1.0, box[3] / max(1, height))), 3),
        ]
        dist_m = round(float(np.linalg.norm(node.centroid - camera_pos)), 2)
        existing = by_object_id.get(resolved_id)
        if existing is not None and existing[1] >= mask_pixels:
            continue
        top_evidence = dict(
            sorted(node.category_evidence.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        )
        by_object_id[resolved_id] = (
            dist_m,
            mask_pixels,
            {
                "object_id": resolved_id,
                "category": node.category,
                "category_evidence": top_evidence,
                "distance_m": dist_m,
                "confidence": round(float(conf), 3),
                "mask_pixels": mask_pixels,
                "raw_mask_pixels": raw_pixels,
                "bbox_norm": bbox_norm,
                "state": node.state,
                "action_states": {
                    name: state.value for name, state in node.action_states.items()
                },
            },
        )
    ordered = sorted(
        by_object_id.items(),
        key=lambda pair: (pair[1][0], pair[0]),
    )
    return [pair[1][2] for pair in ordered]


def compute_scene_delta(
    before: RevisionWatermark,
    memory: IncrementalSceneMemory,
    *,
    macro_action: str,
    status: str,
    message: str,
    primitive_steps: int,
    failed_steps: int,
    yaw_degrees: float,
    horizon_degrees: float,
    annotated_rgb_path: Path | None = None,
    bev_path: Path | None = None,
    remaining_actions: int = 1000,
    remaining_failures: int = 10,
    remaining_decisions: int = 40,
) -> dict[str, object]:
    """Compute a compact incremental SceneDelta from ``before`` to ``memory``."""
    after = capture_watermark(memory)
    new_objects: list[dict[str, object]] = []
    updated_objects: list[dict[str, object]] = []

    for obj_id, digest in sorted(after.objects.items()):
        node = memory.objects[obj_id]
        recent_frames = [
            view.frame_id
            for view in node.views
            if view.frame_id > node.stale_through_frame
        ][-3:]
        top_evidence = dict(
            sorted(node.category_evidence.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        )
        if obj_id not in before.objects:
            new_objects.append(
                {
                    "object_id": obj_id,
                    "category": digest.category,
                    "category_evidence": top_evidence,
                    "region": (
                        list(digest.region) if digest.region is not None else None
                    ),
                    "centroid": (
                        list(digest.centroid) if digest.centroid is not None else None
                    ),
                    "provisional": digest.provisional,
                    "state": digest.state,
                    "best_frames": recent_frames,
                }
            )
            continue

        prev = before.objects[obj_id]
        changes: list[str] = []
        if prev.category != digest.category:
            changes.append(f"category:{prev.category}->{digest.category}")
        if prev.state != digest.state:
            changes.append(f"state:{prev.state}->{digest.state}")
        if prev.provisional and not digest.provisional:
            changes.append("confirmed")
        if prev.action_states != digest.action_states:
            changes.append(f"action_states:{dict(digest.action_states)}")
        if prev.region != digest.region:
            changes.append(f"region:{prev.region}->{digest.region}")
        elif (
            prev.centroid is not None
            and digest.centroid is not None
            and math.dist(prev.centroid, digest.centroid) >= 0.20
        ):
            changes.append("centroid_shifted")

        if changes:
            updated_objects.append(
                {
                    "object_id": obj_id,
                    "category": digest.category,
                    "category_evidence": top_evidence,
                    "region": (
                        list(digest.region) if digest.region is not None else None
                    ),
                    "centroid": (
                        list(digest.centroid) if digest.centroid is not None else None
                    ),
                    "state": digest.state,
                    "action_states": dict(digest.action_states),
                    "changes": changes,
                    "best_frames": recent_frames,
                }
            )

    new_aliases = {
        str(old_id): new_id
        for old_id, new_id in sorted(after.aliases.items())
        if before.aliases.get(old_id) != new_id
    }
    newly_observed_regions = [
        list(region) for region in sorted(after.regions - before.regions)
    ]

    agent_pos = memory.agent_position or (0.0, 0.0, 0.0)
    agent_region = memory.world_to_region(agent_pos[0], agent_pos[1])
    held_info: dict[str, object] | None = None
    if memory.held_object_id is not None and memory.held_object_id in memory.objects:
        held_node = memory.objects[memory.held_object_id]
        held_info = {
            "object_id": held_node.object_id,
            "category": held_node.category,
        }

    frontiers = [
        {
            "region": list(candidate.region),
            "world_xy": [
                round(candidate.world_position[0], 2),
                round(candidate.world_position[1], 2),
            ],
            "path_cost_m": round(candidate.path_cost_m, 2),
            "info_gain_m2": round(candidate.information_gain_m2, 2),
        }
        for candidate in suggest_frontiers(memory, limit=5)
    ]

    current_view: dict[str, object] = {
        "frame_id": memory._latest_frame_id,
        "visible_objects_now": format_visible_objects_now(memory),
    }
    if annotated_rgb_path is not None:
        current_view["annotated_rgb_path"] = str(annotated_rgb_path)
    if bev_path is not None:
        current_view["bev_path"] = str(bev_path)

    return {
        "macro_action": macro_action,
        "status": status,
        "message": message,
        "steps_executed": {
            "primitive_steps": primitive_steps,
            "failed_steps": failed_steps,
            "frames_integrated": list(
                range(before.frame_id + 1, memory._latest_frame_id + 1)
            ),
        },
        "agent_state": {
            "position_xy": [round(agent_pos[0], 3), round(agent_pos[1], 3)],
            "region": list(agent_region),
            "yaw_degrees": round(yaw_degrees, 1),
            "horizon_degrees": round(horizon_degrees, 1),
            "held_object": held_info,
        },
        "current_view": current_view,
        "scene_delta": {
            "new_objects": new_objects,
            "updated_objects": updated_objects,
            "new_aliases": new_aliases,
            "newly_observed_regions": newly_observed_regions,
            "top_frontiers": frontiers,
            "total_known_objects": len(memory.objects),
            "total_observed_regions": len(after.regions),
        },
        "budget": {
            "remaining_decisions": remaining_decisions,
            "remaining_actions": remaining_actions,
            "remaining_failures": remaining_failures,
        },
    }
