"""Incremental object/view/grid memory built only from received observations.

This first RGB-D implementation uses geometric proximity plus an RGB histogram
for association. It does not consume simulator object identities or unseen
metadata, and does not claim equivalence to a learned appearance encoder.
Historical observations remain on disk when interactions invalidate geometry.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .geometry import (
    Detection,
    FloatArray,
    GridCell,
    Observation,
    appearance_histogram,
    bound_rows,
    project_pixels,
)

if TYPE_CHECKING:
    from keyframe.models.scene import SceneObject

CellState = Literal["free", "occupied", "unknown"]


@dataclass(frozen=True)
class MemoryConfig:
    """Explicit metric thresholds and storage limits for online fusion."""

    cell_size_m: float = 0.25
    association_distance_m: float = 0.45
    held_max_camera_distance_m: float = 1.0
    appearance_similarity_min: float = 0.65
    max_depth_m: float = 8.0
    depth_stride: int = 8
    floor_height_m: float = 0.0
    floor_tolerance_m: float = 0.10
    obstacle_min_height_m: float = 0.15
    obstacle_max_height_m: float = 1.8
    max_points_per_object: int = 1024
    max_object_views: int = 64

    def __post_init__(self) -> None:
        for name in (
            "cell_size_m",
            "association_distance_m",
            "held_max_camera_distance_m",
            "max_depth_m",
            "floor_tolerance_m",
            "obstacle_min_height_m",
            "obstacle_max_height_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if not math.isfinite(self.floor_height_m):
            raise ValueError("floor_height_m must be finite")
        if not 0 <= self.appearance_similarity_min <= 1:
            raise ValueError("appearance_similarity_min must be in [0, 1]")
        if self.obstacle_max_height_m <= self.obstacle_min_height_m:
            raise ValueError("obstacle_max_height_m must exceed obstacle_min_height_m")
        for name in ("depth_stride", "max_points_per_object", "max_object_views"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")


@dataclass(frozen=True)
class FrameRecord:
    """Immutable references and bidirectional index keys for an observed frame."""

    frame_id: int
    rgb_path: Path
    geometry_path: Path
    object_ids: tuple[int, ...]
    cells: tuple[GridCell, ...]
    camera_position: tuple[float, float, float]
    camera_direction: tuple[float, float, float]


@dataclass(frozen=True)
class ObjectView:
    """An image-space observation associated with a persistent memory instance."""

    frame_id: int
    detection_index: int
    box_xyxy: tuple[float, float, float, float]


@dataclass
class ObjectMemory:
    """Mutable fused instance; mutations are owned by one scene-memory instance."""

    object_id: int
    category: str
    points: FloatArray
    colors: FloatArray
    appearance: FloatArray
    first_seen: int
    last_seen: int
    views: list[ObjectView] = field(default_factory=list)
    geometry_valid: bool = True
    stale_through_frame: int = -1
    state: str = "observed"

    @property
    def centroid(self) -> FloatArray:
        """Mean of the current geometry; inspect geometry_valid before navigation."""
        return np.asarray(self.points.mean(axis=0), dtype=np.float64)

    @property
    def bounds(self) -> FloatArray:
        """Axis-aligned minimum/maximum world coordinates."""
        return np.stack((self.points.min(axis=0), self.points.max(axis=0)))


_DEFAULT_CONFIG = MemoryConfig()


class IncrementalSceneMemory:
    """Own one episode's incrementally fused scene; not safe for concurrent writes.

    Grid keys are fixed world ``(floor(x / size), floor(y / size))`` coordinates.
    They never change as the observed extent grows. Only actually traversed
    cells and floor endpoints are considered free. Coverage along camera rays
    is recorded separately and never grants traversability.
    """

    def __init__(
        self, output_dir: Path, config: MemoryConfig = _DEFAULT_CONFIG
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config
        self.objects: dict[int, ObjectMemory] = {}
        self.frames: dict[int, FrameRecord] = {}
        self.object_to_views: dict[int, set[int]] = {}
        self.view_to_objects: dict[int, set[int]] = {}
        self.grid_to_views: dict[GridCell, set[int]] = {}
        self.view_to_grids: dict[int, set[GridCell]] = {}
        self.free_cells: set[GridCell] = set()
        self.occupied_cells: set[GridCell] = set()
        self.collision_cells: set[GridCell] = set()
        self.traversed_cells: set[GridCell] = set()
        self.current_detection_objects: dict[int, int] = {}
        self.agent_position: tuple[float, float, float] | None = None
        self.held_object_id: int | None = None
        self._grid_stale_through: dict[GridCell, int] = {}
        self._latest_frame_id = -1

    def integrate(self, observation: Observation) -> FrameRecord:
        """Persist and fuse one strictly newer frame; return its source record.

        RGB, depth, intrinsics, pose and masks are preserved for later inspection.
        An instance mask with no in-range valid depth remains in the raw record
        but cannot create geometry; no fictitious object location is substituted.
        """
        from PIL import Image

        if observation.frame_id <= self._latest_frame_id:
            raise ValueError(
                f"frame_id={observation.frame_id} must exceed {self._latest_frame_id}"
            )
        with Image.open(observation.rgb_path) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != observation.depth_m.shape:
            raise ValueError("rgb_path image dimensions must match depth_m")
        geometry_path = self._save_geometry(observation)
        self.current_detection_objects = {}
        matched_objects: set[int] = set()
        for index, detection in enumerate(observation.detections):
            valid = (
                detection.mask
                & (observation.depth_m > 0)
                & (observation.depth_m <= self.config.max_depth_m)
            )
            rows, cols = np.nonzero(valid)
            if not len(rows):
                continue
            points = project_pixels(observation, rows, cols)
            colors = rgb[rows, cols]
            appearance = appearance_histogram(colors)
            node = self._associate(
                detection.category,
                points,
                appearance,
                matched_objects,
                observation,
            )
            self._fuse(node, points, colors.astype(float) / 255, appearance)
            node.last_seen = observation.frame_id
            is_held = node.object_id == self.held_object_id
            node.geometry_valid = not is_held
            node.state = "held" if is_held else "observed"
            node.views.append(
                ObjectView(
                    observation.frame_id,
                    index,
                    (
                        float(cols.min()),
                        float(rows.min()),
                        float(cols.max() + 1),
                        float(rows.max() + 1),
                    ),
                )
            )
            del node.views[: -self.config.max_object_views]
            matched_objects.add(node.object_id)
            self.current_detection_objects[index] = node.object_id
            self.object_to_views.setdefault(node.object_id, set()).add(
                observation.frame_id
            )
        cells = self._update_grid(observation)
        frame = FrameRecord(
            frame_id=observation.frame_id,
            rgb_path=observation.rgb_path.resolve(),
            geometry_path=geometry_path,
            object_ids=tuple(sorted(matched_objects)),
            cells=tuple(sorted(cells)),
            camera_position=(
                float(observation.camera_to_world[0, 3]),
                float(observation.camera_to_world[1, 3]),
                float(observation.camera_to_world[2, 3]),
            ),
            camera_direction=(
                float(observation.camera_to_world[0, 2]),
                float(observation.camera_to_world[1, 2]),
                float(observation.camera_to_world[2, 2]),
            ),
        )
        self.frames[frame.frame_id] = frame
        self.view_to_objects[frame.frame_id] = matched_objects
        self.view_to_grids[frame.frame_id] = cells
        for cell in cells:
            self.grid_to_views.setdefault(cell, set()).add(frame.frame_id)
        self.agent_position = observation.agent_position
        self._latest_frame_id = frame.frame_id
        self._save_frame_index(frame)
        return frame

    def world_to_cell(self, x: float, y: float) -> GridCell:
        """Map a finite Z-up world XY location to a stable signed cell key."""
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"world coordinates must be finite, got {(x, y)!r}")
        return (
            math.floor(x / self.config.cell_size_m),
            math.floor(y / self.config.cell_size_m),
        )

    def cell_state(self, cell: GridCell) -> CellState:
        """Return occupancy evidence; an unobserved or merely seen cell is unknown."""
        if cell in self.occupied_cells:
            return "occupied"
        if cell in self.free_cells:
            return "free"
        return "unknown"

    def record_collision(self, cell: GridCell) -> None:
        """Retain physical blockage separately from predicted depth occupancy.

        Later floor predictions cannot clear a failed-motion observation. The
        blockage expires only when the agent actually traverses that cell or
        an interaction invalidates the affected object's previous region.
        """
        if (
            not isinstance(cell, tuple)
            or len(cell) != 2
            or any(type(coordinate) is not int for coordinate in cell)
        ):
            raise ValueError(f"collision cell must be a pair of integers, got {cell!r}")
        self.collision_cells.add(cell)
        self.occupied_cells.add(cell)
        self.free_cells.discard(cell)

    def record_interaction(
        self, action: str, object_id: int | None, *, success: bool
    ) -> None:
        """Invalidate affected historical evidence before integrating the result.

        ``object_id`` is the current image's target memory ID (the receptacle for
        PutObject). Pickup marks the held instance unavailable for spatial use;
        PutObject invalidates both the receptacle and the previously held object.
        Open/close conservatively invalidate neighboring objects whose visibility
        or placement can change. Failed actions leave evidence unchanged.
        """
        if not isinstance(success, bool):
            raise ValueError(f"success must be bool, got {success!r}")
        if object_id is not None and object_id not in self.objects:
            raise KeyError(f"unknown object_id={object_id}")
        if not success:
            return
        affected = {object_id} if object_id is not None else set()
        if action == "PickupObject":
            if object_id is None:
                raise ValueError("PickupObject requires an observed object_id")
            self.held_object_id = object_id
        elif action == "PutObject":
            if self.held_object_id is not None:
                affected.add(self.held_object_id)
            self.held_object_id = None
        elif action in {"OpenObject", "CloseObject"} and object_id is not None:
            center = self.objects[object_id].centroid
            affected.update(
                node.object_id
                for node in self.objects.values()
                if np.linalg.norm(node.centroid - center) <= 0.8
            )
        elif action not in {"SliceObject", "ToggleObjectOn", "ToggleObjectOff"}:
            return
        for affected_id in affected:
            node = self.objects[affected_id]
            node.geometry_valid = False
            node.stale_through_frame = self._latest_frame_id
            node.state = "held" if affected_id == self.held_object_id else "changed"
            for point in node.points:
                cell = self.world_to_cell(float(point[0]), float(point[1]))
                self.free_cells.discard(cell)
                self.occupied_cells.discard(cell)
                self.collision_cells.discard(cell)
                self._grid_stale_through[cell] = self._latest_frame_id

    def retrieve_frames(
        self,
        *,
        object_ids: Sequence[int] = (),
        cells: Sequence[GridCell] = (),
        limit: int = 3,
        include_stale: bool = False,
        exclude_frame_ids: Sequence[int] = (),
    ) -> tuple[FrameRecord, ...]:
        """Greedily cover requested instances/regions with complementary views.

        Unknown regions legitimately return no frames. Unknown object IDs are
        rejected. By default a changed object's pre-action observations do not
        count as current evidence; include_stale explicitly retrieves history.
        Exclusions let an episode avoid returning previously delivered images
        while leaving later observations eligible for progressive retrieval.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit must be a positive int, got {limit!r}")
        for frame_id in exclude_frame_ids:
            if type(frame_id) is not int or frame_id < 0:
                raise ValueError(
                    "exclude_frame_ids must contain nonnegative integers, "
                    f"got {frame_id!r}"
                )
        requested_objects = set(object_ids)
        requested_cells = set(cells)
        candidates: set[int] = set()
        for object_id in requested_objects:
            if object_id not in self.objects:
                raise KeyError(f"unknown object_id={object_id}")
            candidates.update(self.object_to_views[object_id])
        for cell in requested_cells:
            candidates.update(self.grid_to_views.get(cell, set()))
        if not requested_objects and not requested_cells:
            candidates.update(self.frames)
        candidates.difference_update(exclude_frame_ids)
        coverage: dict[int, set[tuple[str, object]]] = {}
        for frame_id in candidates:
            frame = self.frames[frame_id]
            covered: set[tuple[str, object]] = {
                ("object", object_id)
                for object_id in requested_objects.intersection(frame.object_ids)
                if include_stale
                or frame_id > self.objects[object_id].stale_through_frame
            }
            covered.update(
                ("cell", cell)
                for cell in requested_cells.intersection(frame.cells)
                if include_stale or frame_id > self._grid_stale_through.get(cell, -1)
            )
            if covered or (not requested_objects and not requested_cells):
                coverage[frame_id] = covered
        selected: list[FrameRecord] = []
        covered_targets: set[tuple[str, object]] = set()
        while coverage and len(selected) < limit:
            best_id = max(
                coverage,
                key=lambda frame_id: (
                    len(coverage[frame_id] - covered_targets),
                    self._view_novelty(self.frames[frame_id], selected),
                    frame_id,
                ),
            )
            selected.append(self.frames[best_id])
            covered_targets.update(coverage.pop(best_id))
        return tuple(selected)

    def snapshot(self) -> dict[str, object]:
        """Build a compact JSON-safe current catalog without unseen scene data."""
        current_ids = set(self.current_detection_objects.values())
        return {
            "coordinate_system": "world_z_up",
            "grid_convention": "cell=(floor(world_x/size),floor(world_y/size))",
            "cell_size_m": self.config.cell_size_m,
            "frame_count": len(self.frames),
            "latest_frame_id": self._latest_frame_id,
            "agent_position": self.agent_position,
            "held_object_id": self.held_object_id,
            "objects": [
                {
                    "object_id": node.object_id,
                    "category": node.category,
                    "centroid": (
                        node.centroid.round(3).tolist() if node.geometry_valid else None
                    ),
                    "last_known_centroid": node.centroid.round(3).tolist(),
                    "geometry_valid": node.geometry_valid,
                    "state": node.state,
                    "last_seen": node.last_seen,
                    "stale_through_frame": node.stale_through_frame,
                    "visible_now": node.object_id in current_ids,
                    "cell": (
                        self.world_to_cell(*node.centroid[:2])
                        if node.geometry_valid
                        else None
                    ),
                    "recent_frames": [
                        view.frame_id
                        for view in node.views
                        if view.frame_id > node.stale_through_frame
                    ][-3:],
                }
                for node in self.objects.values()
            ],
            "free_cells": sorted(self.free_cells),
            "occupied_cells": sorted(self.occupied_cells),
            "collision_cells": sorted(self.collision_cells),
            "observed_cell_count": len(self.grid_to_views),
        }

    def to_scene_objects(self) -> list[SceneObject]:
        """Export currently valid nodes into the existing keyframe object model."""
        from keyframe.models.scene import SceneObject

        scene_objects: list[SceneObject] = []
        for node in self.objects.values():
            if not node.geometry_valid:
                continue
            current_views = [
                view for view in node.views if view.frame_id > node.stale_through_frame
            ]
            scene_objects.append(
                SceneObject(
                    obj_id=node.object_id,
                    category=node.category,
                    centroid=node.centroid,
                    pcd_np=node.points.copy(),
                    pcd_color_np=node.colors.copy(),
                    bbox_np=node.bounds,
                    image_idx=[view.frame_id for view in current_views],
                    class_name=[node.category for _ in current_views],
                    xyxy=[np.array(view.box_xyxy) for view in current_views],
                    num_detections=len(current_views),
                )
            )
        return scene_objects

    def render_bev(self, path: Path) -> Path:
        """Render occupancy, coverage, stable cell coordinates and observed objects."""
        from .rendering import render_memory_bev

        return render_memory_bev(self, path)

    def _save_geometry(self, observation: Observation) -> Path:
        frame_dir = self.output_dir / "observations"
        frame_dir.mkdir(parents=True, exist_ok=True)
        path = frame_dir / f"frame_{observation.frame_id:06d}.npz"
        if path.exists():
            raise FileExistsError(f"observation artifact already exists: {path}")
        masks = (
            np.stack([detection.mask for detection in observation.detections])
            if observation.detections
            else np.empty((0, *observation.depth_m.shape), dtype=bool)
        )
        np.savez_compressed(
            path,
            depth_m=observation.depth_m.astype(np.float32),
            intrinsics=observation.intrinsics,
            camera_to_world=observation.camera_to_world,
            masks=masks,
            categories=np.array(
                [detection.category for detection in observation.detections]
            ),
        )
        return path.resolve()

    def _save_frame_index(self, frame: FrameRecord) -> None:
        """Persist complete frame/object/grid associations next to raw geometry."""
        record = {
            "frame_id": frame.frame_id,
            "rgb_path": str(frame.rgb_path),
            "geometry_path": str(frame.geometry_path),
            "object_ids": frame.object_ids,
            "detection_object_ids": self.current_detection_objects,
            "cells": frame.cells,
            "camera_position": frame.camera_position,
            "camera_direction": frame.camera_direction,
        }
        with frame.geometry_path.with_suffix(".json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(record, stream, allow_nan=False)

    def _associate(
        self,
        category: str,
        points: FloatArray,
        appearance: FloatArray,
        matched_objects: set[int],
        observation: Observation,
    ) -> ObjectMemory:
        candidates: list[tuple[float, ObjectMemory]] = []
        center = points.mean(axis=0)
        bounds = np.stack((points.min(axis=0), points.max(axis=0)))
        for node in self.objects.values():
            if (
                node.category.casefold() != category.casefold()
                or node.object_id in matched_objects
            ):
                continue
            similarity = float(node.appearance @ appearance)
            if similarity < self.config.appearance_similarity_min:
                continue
            if node.object_id == self.held_object_id:
                camera_distance = float(
                    np.linalg.norm(center - observation.camera_to_world[:3, 3])
                )
                if camera_distance <= self.config.held_max_camera_distance_m:
                    # Successful pickup supplies an identity prior; proximity to
                    # the current camera distinguishes a carried detection from
                    # stationary same-class instances elsewhere in the room.
                    candidates.append((-1.0 + 0.2 * (1 - similarity), node))
                continue
            distance = float(np.linalg.norm(center - node.centroid))
            gap = np.maximum(
                0, np.maximum(node.bounds[0] - bounds[1], bounds[0] - node.bounds[1])
            )
            if (
                distance <= self.config.association_distance_m
                or np.linalg.norm(gap) < 0.08
            ):
                candidates.append((distance + 0.2 * (1 - similarity), node))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
        object_id = len(self.objects)
        node = ObjectMemory(
            object_id=object_id,
            category=category,
            points=np.empty((0, 3)),
            colors=np.empty((0, 3)),
            appearance=appearance,
            first_seen=observation.frame_id,
            last_seen=observation.frame_id,
        )
        self.objects[object_id] = node
        return node

    def _fuse(
        self,
        node: ObjectMemory,
        points: FloatArray,
        colors: FloatArray,
        appearance: FloatArray,
    ) -> None:
        # Invalidated geometry must be replaced, never blended with the old pose.
        if node.geometry_valid:
            points = np.concatenate((node.points, points))
            colors = np.concatenate((node.colors, colors))
        node.points = bound_rows(points, self.config.max_points_per_object)
        node.colors = bound_rows(colors, self.config.max_points_per_object)
        combined = node.appearance + appearance
        node.appearance = combined / np.linalg.norm(combined)

    def _update_grid(self, observation: Observation) -> set[GridCell]:
        valid = (observation.depth_m > 0) & (
            observation.depth_m <= self.config.max_depth_m
        )
        for index, object_id in self.current_detection_objects.items():
            if object_id == self.held_object_id:
                # A carried object's depth must not leave a trail of permanent
                # obstacles as the agent moves through the scene.
                valid &= ~observation.detections[index].mask
        sampled = np.zeros(valid.shape, dtype=bool)
        sampled[:: self.config.depth_stride, :: self.config.depth_stride] = True
        rows, cols = np.nonzero(valid & sampled)
        points = project_pixels(observation, rows, cols)
        seen: set[GridCell] = set()
        new_floor: set[GridCell] = set()
        new_occupied: set[GridCell] = set()
        camera_xy = observation.camera_to_world[:2, 3]
        for point in points:
            cell = self.world_to_cell(float(point[0]), float(point[1]))
            height = point[2] - self.config.floor_height_m
            if abs(height) <= self.config.floor_tolerance_m:
                new_floor.add(cell)
            elif (
                self.config.obstacle_min_height_m
                <= height
                <= self.config.obstacle_max_height_m
            ):
                new_occupied.add(cell)
            sample_count = max(
                2,
                math.ceil(
                    np.linalg.norm(point[:2] - camera_xy) / self.config.cell_size_m
                )
                * 2,
            )
            for position in np.linspace(camera_xy, point[:2], sample_count):
                seen.add(self.world_to_cell(float(position[0]), float(position[1])))
        # Direct current floor evidence supersedes older predicted obstacles,
        # but does not contradict physical collision feedback. Cells outside
        # this observation retain their previous evidence.
        cleared_cells = new_floor - new_occupied - self.collision_cells
        self.occupied_cells.difference_update(cleared_cells)
        self.free_cells.update(cleared_cells)
        self.occupied_cells.update(new_occupied)
        self.occupied_cells.update(self.collision_cells)
        self.free_cells.difference_update(self.occupied_cells)
        agent_cell = self.world_to_cell(*observation.agent_position[:2])
        self.traversed_cells.add(agent_cell)
        self.free_cells.add(agent_cell)
        self.occupied_cells.discard(agent_cell)
        self.collision_cells.discard(agent_cell)
        seen.add(agent_cell)
        return seen

    @staticmethod
    def _view_novelty(frame: FrameRecord, selected: Sequence[FrameRecord]) -> float:
        if not selected:
            return 0.0
        position = np.array(frame.camera_position)
        direction = np.array(frame.camera_direction)
        return min(
            min(
                2.0,
                float(np.linalg.norm(position - np.array(previous.camera_position))),
            )
            + 1.0
            - float(direction @ np.array(previous.camera_direction))
            for previous in selected
        )


__all__ = [
    "Detection",
    "FrameRecord",
    "IncrementalSceneMemory",
    "MemoryConfig",
    "ObjectMemory",
    "Observation",
]
