"""Online object/view/region memory using geometric overlap and CLIP fusion.

Only received observations and action feedback enter the memory. Stable identity
is independent of geometry versions, and navigation evidence is owned separately
by SpatialMap. The implementation does not import the offline dataset pipeline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .geometry import (
    Detection,
    FloatArray,
    GridCell,
    Observation,
    depth_visible_points,
    project_pixels,
)
from .object_mapping import (
    MappingConfig,
    association_similarity,
    feature_cosine,
    fuse_features,
    mapping_masks,
    oriented_envelopes_overlap,
    point_overlap,
    process_cloud,
)
from .object_store import (
    ActionState,
    MappedDetection,
    ObjectMemory,
    ObjectStore,
    ObjectView,
)
from .spatial_map import SpatialMap

if TYPE_CHECKING:
    from keyframe.models.scene import SceneObject

_STATIONARY_RECEPTACLE_CATEGORIES: frozenset[str] = frozenset(
    {
        "ArmChair",
        "BathtubBasin",
        "Bed",
        "Cabinet",
        "CoffeeMachine",
        "CoffeeTable",
        "CounterTop",
        "Desk",
        "DiningTable",
        "Drawer",
        "Dresser",
        "FloorLamp",
        "Fridge",
        "GarbageCan",
        "Microwave",
        "Ottoman",
        "Safe",
        "Shelf",
        "SideTable",
        "SinkBasin",
        "Sofa",
        "StoveBurner",
        "Toilet",
        "ToiletPaperHanger",
    }
)

_COMPATIBLE_FALLBACK_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"SinkBasin", "BathtubBasin"}),
    frozenset(
        {
            "CounterTop",
            "DiningTable",
            "Desk",
            "SideTable",
            "CoffeeTable",
            "Dresser",
            "Shelf",
        }
    ),
    frozenset({"Cabinet", "Drawer", "Shelf"}),
    frozenset({"Fridge", "Cabinet", "Shelf"}),
    frozenset({"DeskLamp", "FloorLamp"}),
    frozenset({"Ladle", "Spoon", "Fork", "Knife", "ButterKnife", "Spatula"}),
    frozenset(
        {
            "Cup",
            "Mug",
            "Bowl",
            "Glassbottle",
            "WineBottle",
            "SprayBottle",
            "SoapBottle",
            "SaltShaker",
            "PepperShaker",
        }
    ),
    frozenset({"Pan", "Pot", "Plate", "Bowl"}),
    frozenset({"Pillow", "Cloth", "HandTowel", "DishSponge", "SoapBar"}),
    frozenset(
        {
            "Apple",
            "Tomato",
            "Potato",
            "Egg",
            "Lettuce",
            "Bread",
            "AppleSliced",
            "TomatoSliced",
            "PotatoSliced",
            "LettuceSliced",
            "BreadSliced",
        }
    ),
    frozenset(
        {
            "KeyChain",
            "Watch",
            "CreditCard",
            "CD",
            "RemoteControl",
            "CellPhone",
            "Pen",
            "Pencil",
        }
    ),
)


def _is_compatible_spatial_fallback(
    target_node: ObjectMemory, cand_node: ObjectMemory, action: str
) -> bool:
    """Return whether ``cand_node`` can safely supply a fallback mask for ``target_node``."""
    target_cat = target_node.category
    cand_cat = cand_node.category
    if target_cat.casefold() == cand_cat.casefold():
        return True
    if action in {"PickupObject", "SliceObject"}:
        if cand_cat in _STATIONARY_RECEPTACLE_CATEGORIES or cand_cat == "Faucet":
            return False
        if target_cat in _STATIONARY_RECEPTACLE_CATEGORIES:
            return False
        if (
            target_cat in cand_node.category_evidence
            or cand_cat in target_node.category_evidence
        ):
            return True
        for family in _COMPATIBLE_FALLBACK_FAMILIES:
            if target_cat in family and cand_cat in family:
                return True
        if not target_node.geometry_valid and target_node.reacquire_bounds is not None:
            return True
        return (
            feature_cosine(target_node.image_feature, cand_node.image_feature) >= 0.78
        )
    if action in {"ToggleObjectOn", "ToggleObjectOff"}:
        if target_cat == "Faucet":
            return cand_cat in {"Faucet", "SinkBasin", "BathtubBasin"}
        return cand_cat in {"DeskLamp", "FloorLamp"}
    if action in {"OpenObject", "CloseObject"}:
        if target_cat in {"Cabinet", "Drawer"}:
            return cand_cat in {"Cabinet", "Drawer"}
        if target_cat == "Fridge":
            if (
                action == "CloseObject"
                and target_node.action_states.get("open") is not None
                and target_node.action_states["open"].value
            ):
                return cand_cat in {"Fridge", "Cabinet", "Shelf", "CounterTop"}
            return cand_cat in {"Fridge", "Cabinet", "Shelf"}
        return False
    if action == "PutObject":
        if target_cat in {"SinkBasin", "BathtubBasin"}:
            return cand_cat in {"SinkBasin", "BathtubBasin"}
        if target_cat == "Fridge":
            return cand_cat in {"Fridge", "Cabinet", "Shelf", "CounterTop"}
        for family in _COMPATIBLE_FALLBACK_FAMILIES:
            if target_cat in family and cand_cat in family:
                return True
        return False
    return False


@dataclass(frozen=True)
class MemoryConfig:
    """Object mapping configuration and bounded source-view storage."""

    mapping: MappingConfig = field(default_factory=MappingConfig)
    max_depth_m: float = 8.0
    max_object_views: int = 64
    visibility_depth_tolerance_m: float = 0.15

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, MappingConfig):
            raise ValueError("mapping must be a MappingConfig")
        for name in ("max_depth_m", "visibility_depth_tolerance_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("max_object_views",):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class FrameRecord:
    """References to one real observation and its current canonical index keys."""

    frame_id: int
    rgb_path: Path
    geometry_path: Path
    object_ids: tuple[int, ...]
    cells: tuple[GridCell, ...]
    camera_position: tuple[float, float, float]
    camera_direction: tuple[float, float, float]
    frame_feature: FloatArray


class IncrementalSceneMemory:
    """Own one episode's scene memory; integration is sequential, not thread-safe.

    Objects use monotonic IDs with merge aliases. Region/view links use the
    spatial map's fixed coarse region coordinates; fine navigation cells never
    substitute for those coordinates. Raw historical observations are immutable.
    """

    def __init__(
        self,
        output_dir: Path,
        config: MemoryConfig | None = None,
        *,
        spatial_map: SpatialMap | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or MemoryConfig()
        self.spatial_map = spatial_map or SpatialMap()
        self._store = ObjectStore()
        self.objects = self._store.objects
        self.aliases = self._store.aliases
        self.frames: dict[int, FrameRecord] = {}
        self.object_to_views: dict[int, set[int]] = {}
        self.view_to_objects: dict[int, set[int]] = {}
        self.grid_to_views: dict[GridCell, set[int]] = {}
        self.view_to_grids: dict[int, set[GridCell]] = {}
        self.current_detection_objects: dict[int, int] = {}
        self.current_detection_masks: dict[int, np.ndarray] = {}
        self.current_raw_detection_masks: dict[int, np.ndarray] = {}
        self.current_detection_boxes: dict[int, tuple[float, float, float, float]] = {}
        self.current_detection_confidences: dict[int, float] = {}
        self.rejected_detection_reasons: dict[int, str] = {}
        self.suppressed_detection_indices: dict[int, int] = {}
        self.agent_position: tuple[float, float, float] | None = None
        self.held_object_id: int | None = None
        self._grid_stale_through: dict[GridCell, int] = {}
        self._latest_frame_id = -1
        self._feature_dimension: int | None = None

    def resolve_object_id(self, object_id: int) -> int:
        """Resolve a previously issued ID after any periodic object merges."""
        return self._store.resolve(object_id)

    def world_to_region(self, x: float, y: float) -> GridCell:
        """Map world XY to the coarse retrieval/BEV region coordinates."""
        return self.spatial_map.world_to_region(x, y)

    def record_collision(self, cell: GridCell) -> None:
        """Forward physical collision evidence in fine navigation cell coordinates."""
        self.spatial_map.record_collision(cell)

    def integrate(self, observation: Observation) -> FrameRecord:
        """Persist and integrate one strictly newer RGB-D/CLIP observation.

        A detection lacking usable depth stays in the raw record but does not
        invent a 3D location. Duplicate masks retain their source identity link
        while contributing only once to object confidence and feature fusion.
        """
        from PIL import Image

        if observation.frame_id <= self._latest_frame_id:
            raise ValueError(
                f"frame_id={observation.frame_id} must exceed {self._latest_frame_id}"
            )
        dimension = len(observation.frame_feature)
        if self._feature_dimension is not None and dimension != self._feature_dimension:
            raise ValueError(
                "feature dimension must remain constant throughout an episode"
            )
        with Image.open(observation.rgb_path) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != observation.depth_m.shape:
            raise ValueError("rgb_path image dimensions must match depth_m")
        geometry_path = self._save_geometry(observation)
        self._feature_dimension = dimension
        self.current_detection_objects = {}
        self.current_detection_masks = {}
        self.current_raw_detection_masks = {}
        self.current_detection_boxes = {}
        self.current_detection_confidences = {}
        self.suppressed_detection_indices = {}
        effective_masks, self.rejected_detection_reasons = mapping_masks(
            observation.detections, self.config.mapping
        )
        regions = self._project_detections(observation, rgb, effective_masks)
        assignments = self._assign_regions(regions, observation)
        for region in regions:
            existing_id = assignments.get(region.index)
            if (
                existing_id is None
                and self.held_object_id is not None
                and self._is_held_self_occlusion(region, observation)
            ):
                existing_id = self.held_object_id
            if existing_id is None:
                node = self._store.create(region, observation.frame_id)
                self._set_slice_parent(node)
            else:
                node = self.objects[existing_id]
                self._fuse_observation(node, region, observation.frame_id)
            node.provisional = (
                node.detection_count < self.config.mapping.confirmation_observations
            )
            is_held = node.object_id == self.held_object_id
            node.geometry_valid = not is_held
            node.state = "held" if is_held else "observed"
            node.reacquire_bounds = None if not is_held else node.reacquire_bounds
            node.views.append(
                ObjectView(
                    observation.frame_id,
                    region.index,
                    region.box_xyxy,
                    node.geometry_epoch,
                    region.detection.confidence,
                    int(effective_masks[region.index].sum()),
                )
            )
            del node.views[: -self.config.max_object_views]
            self.current_detection_objects[region.index] = node.object_id
            self.current_detection_masks[region.index] = effective_masks[region.index]
            self.current_raw_detection_masks[region.index] = observation.detections[
                region.index
            ].mask
            self.current_detection_boxes[region.index] = region.box_xyxy
            self.current_detection_confidences[region.index] = (
                region.detection.confidence
            )
            self.object_to_views.setdefault(node.object_id, set()).add(
                observation.frame_id
            )
        for duplicate, representative in self.suppressed_detection_indices.items():
            if representative in self.current_detection_objects:
                self.current_detection_objects[duplicate] = (
                    self.current_detection_objects[representative]
                )
                self.current_detection_masks[duplicate] = effective_masks[duplicate]
                self.current_raw_detection_masks[duplicate] = observation.detections[
                    duplicate
                ].mask
                self.current_detection_boxes[duplicate] = observation.detections[
                    duplicate
                ].bbox_xyxy
                self.current_detection_confidences[duplicate] = observation.detections[
                    duplicate
                ].confidence
        held_mask = np.zeros(observation.depth_m.shape, dtype=bool)
        for index, object_id in self.current_detection_objects.items():
            if object_id == self.held_object_id:
                held_mask |= self.current_detection_masks[index]
                held_mask |= self.current_raw_detection_masks.get(
                    index, self.current_detection_masks[index]
                )
        if self.held_object_id is not None:
            h_rows = observation.depth_m.shape[0]
            near_fg = (observation.depth_m > 0) & (observation.depth_m <= 0.56)
            near_fg[: int(0.30 * h_rows), :] = False
            held_mask |= near_fg
        cells = set(self.spatial_map.update(observation, ignored_mask=held_mask))
        object_ids = set(self.current_detection_objects.values())
        frame = FrameRecord(
            frame_id=observation.frame_id,
            rgb_path=observation.rgb_path.resolve(),
            geometry_path=geometry_path,
            object_ids=tuple(sorted(object_ids)),
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
            frame_feature=observation.frame_feature,
        )
        self.frames[frame.frame_id] = frame
        self.view_to_objects[frame.frame_id] = object_ids
        self.view_to_grids[frame.frame_id] = cells
        for cell in cells:
            self.grid_to_views.setdefault(cell, set()).add(frame.frame_id)
        self.agent_position = observation.agent_position
        self._latest_frame_id = frame.frame_id
        if len(self.frames) % self.config.mapping.maintenance_interval == 0:
            self.maintain_objects()
            frame = self.frames[frame.frame_id]
        self._save_frame_index(frame, observation.depth_m.shape)
        return frame

    def _project_detections(
        self,
        observation: Observation,
        rgb: np.ndarray,
        effective_masks: dict[int, np.ndarray],
    ) -> list[MappedDetection]:
        from scipy.ndimage import binary_erosion

        projected: list[MappedDetection] = []
        ordered = sorted(
            enumerate(observation.detections),
            key=lambda item: (-item[1].confidence, item[0]),
        )
        erode_iters = self.config.mapping.erode_mask_pixels
        trim_q = self.config.mapping.depth_trim_quantile
        for index, detection in ordered:
            if index not in effective_masks:
                continue
            geom_mask = effective_masks[index]
            if erode_iters > 0 and int(geom_mask.sum()) >= 64:
                eroded = np.asarray(
                    binary_erosion(geom_mask, iterations=erode_iters), dtype=bool
                )
                if int(eroded.sum()) >= max(
                    20, self.config.mapping.min_detection_points
                ):
                    geom_mask = eroded
            valid = (
                geom_mask
                & (observation.depth_m > 0)
                & (observation.depth_m <= self.config.max_depth_m)
            )
            rows, cols = np.nonzero(valid)
            if len(rows) < self.config.mapping.min_detection_points:
                self.rejected_detection_reasons[index] = (
                    "insufficient_valid_depth_after_subtraction"
                )
                continue
            if trim_q > 0 and len(rows) >= 24:
                depths = observation.depth_m[rows, cols]
                low_d, high_d = np.percentile(depths, [2.0, (1.0 - trim_q) * 100.0])
                inliers = (depths >= low_d) & (depths <= high_d)
                if (
                    int(np.count_nonzero(inliers))
                    >= self.config.mapping.min_detection_points
                ):
                    rows = rows[inliers]
                    cols = cols[inliers]
            duplicate = next(
                (
                    region.index
                    for region in projected
                    if self._duplicate_mask(
                        detection,
                        region.detection,
                        left_mask=effective_masks[index],
                        right_mask=effective_masks[region.index],
                    )
                ),
                None,
            )
            if duplicate is not None:
                self.suppressed_detection_indices[index] = duplicate
                continue
            points, colors = process_cloud(
                project_pixels(observation, rows, cols),
                rgb[rows, cols].astype(float) / 255,
                self.config.mapping,
                denoise=True,
            )
            if not len(points):
                continue
            box = detection.bbox_xyxy
            projected.append(MappedDetection(index, detection, points, colors, box))
        return projected

    def _duplicate_mask(
        self,
        left: Detection,
        right: Detection,
        *,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> bool:
        intersection = int(np.count_nonzero(left_mask & right_mask))
        union = int(np.count_nonzero(left_mask | right_mask))
        return (
            intersection / union >= self.config.mapping.duplicate_mask_iou
            and feature_cosine(left.image_feature, right.image_feature)
            >= self.config.mapping.duplicate_image_similarity
        )

    def _assign_regions(
        self, regions: Sequence[MappedDetection], observation: Observation
    ) -> dict[int, int]:
        candidates: list[tuple[float, int, int]] = []
        for region in regions:
            for node in self.objects.values():
                score = self._association_score(region, node, observation)
                if score is not None:
                    candidates.append((score, region.index, node.object_id))
        assigned: dict[int, int] = {}
        used: set[int] = set()
        for _, detection_index, object_id in sorted(
            candidates, key=lambda item: (-item[0], item[1], item[2])
        ):
            if detection_index not in assigned and object_id not in used:
                assigned[detection_index] = object_id
                used.add(object_id)
        return assigned

    def _is_held_self_occlusion(
        self, region: MappedDetection, observation: Observation
    ) -> bool:
        """Return whether ``region`` is a close-up foreground view of the held item."""
        if self.held_object_id is None:
            return False
        camera_pos = observation.camera_to_world[:3, 3]
        center = region.points.mean(axis=0)
        cam_dist = float(np.linalg.norm(center - camera_pos))
        if cam_dist > 0.58:
            return False
        height = observation.depth_m.shape[0]
        _, y1, _, y2 = region.box_xyxy
        return y2 >= 0.50 * height and y1 >= 0.10 * height

    def _association_score(
        self, region: MappedDetection, node: ObjectMemory, observation: Observation
    ) -> float | None:
        if node.state == "sliced":
            return None
        if not node.geometry_valid:
            if node.object_id == self.held_object_id and self._is_held_self_occlusion(
                region, observation
            ):
                return 3.5 + (
                    0.5 if region.detection.category == node.category else 0.0
                )
            image_similarity = feature_cosine(
                region.detection.image_feature, node.image_feature
            )
            text_similarity = feature_cosine(
                region.detection.text_feature, node.text_feature
            )
            semantic_score = 0.75 * image_similarity + 0.25 * text_similarity
            if self.config.mapping.same_category_bonus > 0:
                if region.detection.category == node.category:
                    semantic_score += 0.20
                elif text_similarity < 0.85:
                    return None
            if semantic_score < self.config.mapping.reacquire_similarity:
                return None
            center = region.points.mean(axis=0)
            if node.object_id == self.held_object_id:
                distance = float(
                    np.linalg.norm(center - observation.camera_to_world[:3, 3])
                )
                return (
                    3 + semantic_score
                    if distance <= self.config.mapping.held_max_camera_distance_m
                    else None
                )
            bounds = (
                node.reacquire_bounds
                if node.reacquire_bounds is not None
                else node.bounds
            )
            gap = np.maximum(0, np.maximum(bounds[0] - center, center - bounds[1]))
            return (
                2 + semantic_score
                if np.linalg.norm(gap) <= self.config.mapping.reacquire_margin_m
                else None
            )
        if self.held_object_id is not None and self._is_held_self_occlusion(
            region, observation
        ):
            return None
        if (
            node.category not in _STATIONARY_RECEPTACLE_CATEGORIES
            and region.detection.category != node.category
        ):
            if region.detection.category in _STATIONARY_RECEPTACLE_CATEGORIES:
                return None
            if (
                int(observation.detections[region.index].mask.sum())
                > 0.25 * observation.depth_m.size
                and float(np.ptp(node.points, axis=0).max()) <= 0.45
            ):
                return None
        margin = self.config.mapping.envelope_margin_m
        if not oriented_envelopes_overlap(region.points, node.points, margin_m=margin):
            return None
        dist = self.config.mapping.association_distance_m
        if margin > 0:
            overlap = max(
                point_overlap(region.points, node.points, dist),
                point_overlap(node.points, region.points, dist),
            )
        else:
            overlap = point_overlap(region.points, node.points, dist)
        if overlap <= 0:
            return None
        image_similarity = feature_cosine(
            region.detection.image_feature, node.image_feature
        )
        score = association_similarity(
            overlap,
            image_similarity,
            self.config.mapping.phys_bias,
        )
        bonus = self.config.mapping.same_category_bonus
        if bonus > 0:
            if region.detection.category == node.category:
                score += bonus
            else:
                text_similarity = feature_cosine(
                    region.detection.text_feature, node.text_feature
                )
                if text_similarity < 0.82:
                    score -= 1.5 * bonus
        return score if score >= self.config.mapping.association_threshold else None

    def _fuse_observation(
        self, node: ObjectMemory, region: MappedDetection, frame_id: int
    ) -> None:
        points, colors = region.points, region.colors
        if node.geometry_valid:
            points, colors = np.concatenate((node.points, points)), np.concatenate(
                (node.colors, colors)
            )
        node.points, node.colors = process_cloud(
            points, colors, self.config.mapping, denoise=False
        )
        node.last_seen = frame_id
        if (
            node.object_id == self.held_object_id
            and region.detection.category != node.category
        ):
            return
        node.image_feature = fuse_features(
            node.image_feature, region.detection.image_feature, node.detection_count, 1
        )
        node.text_feature = fuse_features(
            node.text_feature, region.detection.text_feature, node.detection_count, 1
        )
        node.detection_count += 1
        evidence = node.category_evidence
        evidence[region.detection.category] = (
            evidence.get(region.detection.category, 0) + 1
        )
        # Stable lexical tie-breaking does not depend on detector arrival order.
        node.category = min(evidence, key=lambda label: (-evidence[label], label))

    def _set_slice_parent(self, child: ObjectMemory) -> None:
        if not child.category.casefold().endswith("sliced"):
            return
        base = child.category.casefold().removesuffix("sliced")
        parents = [
            node
            for node in self.objects.values()
            if node.state == "sliced"
            and node.category.casefold() == base
            and np.linalg.norm(node.centroid - child.centroid)
            <= self.config.mapping.reacquire_margin_m
        ]
        if parents:
            child.parent_object_id = min(
                parents, key=lambda node: np.linalg.norm(node.centroid - child.centroid)
            ).object_id

    def maintain_objects(self) -> None:
        """Denoise and consolidate overlapping identities without reusing IDs.

        Identities observed as separate detections in the same frame cannot be
        merged. Low-support objects remain explicitly provisional, not erased.
        """
        cutoff = self._latest_frame_id - self.config.mapping.maintenance_interval
        for node in self.objects.values():
            if node.geometry_valid and (
                self._latest_frame_id < 0 or node.last_seen >= cutoff
            ):
                node.points, node.colors = process_cloud(
                    node.points, node.colors, self.config.mapping, denoise=True
                )
        for _, first_id, second_id in self._merge_candidates():
            first_id, second_id = self.resolve_object_id(
                first_id
            ), self.resolve_object_id(second_id)
            if first_id == second_id:
                continue
            left_id, right_id = sorted((first_id, second_id))
            left, right = self.objects[left_id], self.objects[right_id]
            if self.object_to_views[left_id] & self.object_to_views[right_id]:
                continue
            if (
                feature_cosine(left.image_feature, right.image_feature)
                <= self.config.mapping.merge_image_similarity
                or feature_cosine(left.text_feature, right.text_feature)
                <= self.config.mapping.merge_text_similarity
            ):
                continue
            self._merge_objects(left, right)

    def _merge_candidates(self) -> list[tuple[float, int, int]]:
        candidates: list[tuple[float, int, int]] = []
        ids = sorted(key for key, node in self.objects.items() if node.geometry_valid)
        margin = self.config.mapping.envelope_margin_m
        dist = self.config.mapping.association_distance_m
        for position, left_id in enumerate(ids):
            for right_id in ids[position + 1 :]:
                left, right = self.objects[left_id], self.objects[right_id]
                if not oriented_envelopes_overlap(
                    left.points, right.points, margin_m=margin
                ):
                    continue
                overlap = max(
                    point_overlap(left.points, right.points, dist),
                    point_overlap(right.points, left.points, dist),
                )
                if overlap > self.config.mapping.merge_overlap_threshold:
                    candidates.append((overlap, left_id, right_id))
        return sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))

    def _merge_objects(self, survivor: ObjectMemory, retired: ObjectMemory) -> None:
        left, right = survivor.object_id, retired.object_id
        survivor.points, survivor.colors = process_cloud(
            np.concatenate((survivor.points, retired.points)),
            np.concatenate((survivor.colors, retired.colors)),
            self.config.mapping,
            denoise=True,
        )
        survivor.image_feature = fuse_features(
            survivor.image_feature,
            retired.image_feature,
            survivor.detection_count,
            retired.detection_count,
        )
        survivor.text_feature = fuse_features(
            survivor.text_feature,
            retired.text_feature,
            survivor.detection_count,
            retired.detection_count,
        )
        survivor.detection_count += retired.detection_count
        survivor.provisional = (
            survivor.detection_count < self.config.mapping.confirmation_observations
        )
        survivor.first_seen = min(survivor.first_seen, retired.first_seen)
        survivor.last_seen = max(survivor.last_seen, retired.last_seen)
        survivor.stale_through_frame = max(
            survivor.stale_through_frame, retired.stale_through_frame
        )
        survivor.geometry_epoch = max(survivor.geometry_epoch, retired.geometry_epoch)
        for name, evidence in retired.action_states.items():
            previous = survivor.action_states.get(name)
            if previous is None or evidence.frame_id > previous.frame_id:
                survivor.action_states[name] = evidence
        survivor.views = sorted(
            survivor.views + retired.views, key=lambda view: view.frame_id
        )[-self.config.max_object_views :]
        for category, count in retired.category_evidence.items():
            survivor.category_evidence[category] = (
                survivor.category_evidence.get(category, 0) + count
            )
        survivor.category = min(
            survivor.category_evidence,
            key=lambda label: (-survivor.category_evidence[label], label),
        )
        self.object_to_views[left].update(self.object_to_views.pop(right))
        self._store.retire_alias(right, left)
        for frame_id in self.object_to_views[left]:
            if frame_id not in self.frames:
                continue
            object_ids = {
                self.resolve_object_id(value)
                for value in self.view_to_objects[frame_id]
            }
            self.view_to_objects[frame_id] = object_ids
            self.frames[frame_id] = replace(
                self.frames[frame_id], object_ids=tuple(sorted(object_ids))
            )
        self.current_detection_objects = {
            index: self.resolve_object_id(value)
            for index, value in self.current_detection_objects.items()
        }
        self._record_event(
            {
                "event": "merge",
                "frame_id": self._latest_frame_id,
                "survivor": left,
                "alias": right,
            }
        )

    def select_interaction_mask(
        self,
        object_id: int,
        *,
        action: str,
        allow_spatial_fallback: bool = True,
    ) -> tuple[int, np.ndarray, str] | None:
        """Select the best current-frame binary mask for ``object_id`` and ``action``.

        For ``OpenObject`` / ``CloseObject``, returns the unsubtracted raw SAM
        mask so container doors are not hollowed out by contained child masks.
        For ``PutObject``, returns the child-subtracted effective mask so placement
        targets empty receptacle surface rather than items already sitting on it.
        """
        resolved_id = self.resolve_object_id(object_id)
        direct_indices = [
            idx
            for idx, identity in self.current_detection_objects.items()
            if self.resolve_object_id(identity) == resolved_id
        ]
        chosen_idx: int | None = None
        mask_source = "current_mapping_mask"

        if direct_indices:
            chosen_idx = max(
                direct_indices,
                key=lambda idx: (
                    int(self.current_detection_masks[idx].sum()),
                    self.current_detection_confidences.get(idx, 0.0),
                    -idx,
                ),
            )
        elif allow_spatial_fallback and resolved_id in self.objects:
            target_node = self.objects[resolved_id]
            fallback_candidates: list[tuple[int, float, int]] = []
            if action in {"PickupObject", "SliceObject"}:
                max_dist = (
                    0.55
                    if not target_node.geometry_valid
                    and target_node.reacquire_bounds is not None
                    else 0.30
                )
            else:
                max_dist = 0.68
            for idx, identity in self.current_detection_objects.items():
                cand_id = self.resolve_object_id(identity)
                if cand_id not in self.objects or cand_id == self.held_object_id:
                    continue
                cand_node = self.objects[cand_id]
                if cand_node.action_states.get(f"invalid_{action}") is not None:
                    continue
                if not _is_compatible_spatial_fallback(target_node, cand_node, action):
                    continue
                dist = float(np.linalg.norm(cand_node.centroid - target_node.centroid))
                if dist <= max_dist:
                    exact_cat_rank = (
                        0
                        if cand_node.category.casefold()
                        == target_node.category.casefold()
                        else 1
                    )
                    fallback_candidates.append((exact_cat_rank, dist, idx))
            if fallback_candidates:
                fallback_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
                chosen_idx = fallback_candidates[0][2]
                mask_source = "current_spatial_fallback_mask"

        if chosen_idx is None:
            return None

        eff_mask = self.current_detection_masks[chosen_idx]
        raw_mask = self.current_raw_detection_masks.get(chosen_idx, eff_mask)
        if action in {"OpenObject", "CloseObject", "PutObject"} and int(
            raw_mask.sum()
        ) > int(eff_mask.sum()):
            return chosen_idx, raw_mask, "current_raw_sam_mask"
        return chosen_idx, eff_mask, mask_source

    def record_interaction(
        self, action: str, object_id: int | None, *, success: bool
    ) -> None:
        """Apply feedback before integrating the post-action observation.

        Success supplies an identity/state prior, not privileged geometry. Old
        point clouds never accumulate with the new pose of a moved object.
        """
        if not isinstance(success, bool):
            raise ValueError("success must be bool")
        if object_id is not None:
            object_id = self.resolve_object_id(object_id)
        if not success:
            return
        affected = {object_id} if object_id is not None else set()
        if action == "PickupObject":
            if object_id is None:
                raise ValueError("PickupObject requires an observed object_id")
            self.held_object_id = object_id
        elif action == "PutObject":
            if self.held_object_id is not None:
                held = self.objects[self.held_object_id]
                held.reacquire_bounds = (
                    self.objects[object_id].bounds.copy()
                    if object_id is not None
                    else None
                )
                affected.add(held.object_id)
            self.held_object_id = None
        elif action in {"OpenObject", "CloseObject"} and object_id is not None:
            target = self.objects[object_id]
            affected.update(
                node.object_id
                for node in self.objects.values()
                if node.geometry_valid
                and np.linalg.norm(node.centroid - target.centroid) <= 0.8
            )
        elif action not in {"SliceObject", "ToggleObjectOn", "ToggleObjectOff"}:
            return
        state_actions = {
            "OpenObject": ("open", True),
            "CloseObject": ("open", False),
            "ToggleObjectOn": ("toggled_on", True),
            "ToggleObjectOff": ("toggled_on", False),
        }
        if object_id is not None and action in state_actions:
            name, value = state_actions[action]
            self.objects[object_id].action_states[name] = ActionState(
                value, self._latest_frame_id
            )
        spatial_change = action not in {
            "ToggleObjectOn",
            "ToggleObjectOff",
            "OpenObject",
            "CloseObject",
        }
        invalidated: set[GridCell] = set()
        for affected_id in affected:
            node = self.objects[affected_id]
            node.stale_through_frame = self._latest_frame_id
            if action in {"OpenObject", "CloseObject"}:
                node.state = (
                    "held" if affected_id == self.held_object_id else "observed"
                )
            else:
                node.state = "held" if affected_id == self.held_object_id else "changed"
            if action == "SliceObject" and affected_id == object_id:
                node.state = "sliced"
            if spatial_change or action in {"OpenObject", "CloseObject"}:
                cells = {
                    self.world_to_region(float(point[0]), float(point[1]))
                    for point in node.points
                }
                invalidated.update(cells)
                for cell in cells:
                    self._grid_stale_through[cell] = self._latest_frame_id
            if spatial_change:
                node.geometry_valid = False
                node.geometry_epoch += 1
        self.spatial_map.invalidate_region(invalidated)
        self._record_event(
            {
                "event": "interaction",
                "action": action,
                "frame_id": self._latest_frame_id,
                "object_id": object_id,
                "affected_ids": sorted(affected),
                "invalidated_regions": sorted(invalidated),
            }
        )

    def retrieve_frames(
        self,
        *,
        object_ids: Sequence[int] = (),
        cells: Sequence[GridCell] = (),
        limit: int = 3,
        include_stale: bool = False,
        exclude_frame_ids: Sequence[int] = (),
    ) -> tuple[FrameRecord, ...]:
        """Select depth-observed historical regions and complementary instance views.

        Region support comes from SpatialMap's observed depth footprint, never
        an unseen object catalog. Known object tags cannot exclude region views.
        Interactions mark old views stale; explicit history requests may read them.
        """
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        if not isinstance(include_stale, bool):
            raise ValueError("include_stale must be bool")
        for frame_id in exclude_frame_ids:
            if type(frame_id) is not int or frame_id < 0:
                raise ValueError("exclude_frame_ids must contain nonnegative integers")
        for cell in cells:
            if (
                not isinstance(cell, tuple)
                or len(cell) != 2
                or any(type(value) is not int for value in cell)
            ):
                raise ValueError(
                    "cells must contain pairs of integer coarse region coordinates"
                )
        requested_objects = {self.resolve_object_id(value) for value in object_ids}
        requested_cells = set(cells)
        candidates: set[int] = set()
        for object_id in requested_objects:
            candidates.update(self.object_to_views[object_id])
        for cell in requested_cells:
            candidates.update(self.grid_to_views.get(cell, set()))
        if not requested_objects and not requested_cells:
            candidates.update(self.frames)
        candidates.difference_update(exclude_frame_ids)
        coverage: dict[int, set[tuple[str, object]]] = {}
        for frame_id in candidates:
            if frame_id > self._latest_frame_id:
                raise RuntimeError("future frame found in historical index")
            frame = self.frames[frame_id]
            covered: set[tuple[str, object]] = {
                ("object", value)
                for value in requested_objects.intersection(frame.object_ids)
                if include_stale
                or (
                    frame_id > self.objects[value].stale_through_frame
                    and self._object_visible_in_frame(value, frame)
                )
            }
            covered.update(
                ("cell", cell)
                for cell in requested_cells.intersection(frame.cells)
                if include_stale or frame_id > self._grid_stale_through.get(cell, -1)
            )
            if covered or not requested_objects and not requested_cells:
                coverage[frame_id] = covered
        selected: list[FrameRecord] = []
        covered_targets: set[tuple[str, object]] = set()
        while coverage and len(selected) < limit:
            best = max(
                coverage,
                key=lambda index: (
                    len(coverage[index] - covered_targets),
                    self._view_novelty(self.frames[index], selected),
                    self._object_view_quality(index, requested_objects),
                    index,
                ),
            )
            selected.append(self.frames[best])
            covered_targets.update(coverage.pop(best))
        return tuple(selected)

    def _object_visible_in_frame(self, object_id: int, frame: FrameRecord) -> bool:
        node = self.objects[object_id]
        if not node.geometry_valid:
            # A held view supplies appearance without a stationary world position.
            return node.state == "held" and frame.frame_id > node.stale_through_frame
        with np.load(frame.geometry_path, allow_pickle=False) as saved:
            visible = depth_visible_points(
                node.points,
                saved["depth_m"],
                saved["intrinsics"],
                saved["camera_to_world"],
                tolerance_m=self.config.visibility_depth_tolerance_m,
            )
        return bool(visible.any())

    def _object_view_quality(self, frame_id: int, object_ids: set[int]) -> float:
        return sum(
            view.confidence * math.log1p(view.pixel_area)
            for object_id in object_ids
            for view in self.objects[object_id].views
            if view.frame_id == frame_id
        )

    @staticmethod
    def _view_novelty(frame: FrameRecord, selected: Sequence[FrameRecord]) -> float:
        if not selected:
            return 0.0
        return min(
            min(
                2.0,
                float(
                    np.linalg.norm(
                        np.array(frame.camera_position)
                        - np.array(other.camera_position)
                    )
                ),
            )
            + 1
            - float(np.array(frame.camera_direction) @ np.array(other.camera_direction))
            + 0.5 * (1 - feature_cosine(frame.frame_feature, other.frame_feature))
            for other in selected
        )

    def snapshot(self) -> dict[str, object]:
        """Expose current identities, validity, provenance and spatial evidence."""
        current_ids = set(self.current_detection_objects.values())
        return {
            "coordinate_system": "world_z_up",
            "grid_convention": "coarse region=(floor(world_x/region_size),floor(world_y/region_size))",
            "frame_count": len(self.frames),
            "latest_frame_id": self._latest_frame_id,
            "agent_position": self.agent_position,
            "held_object_id": self.held_object_id,
            "aliases": dict(self.aliases),
            "rejected_detection_reasons": dict(self.rejected_detection_reasons),
            "observed_region_count": len(self.grid_to_views),
            "spatial_map": self.spatial_map.snapshot(),
            "objects": [
                {
                    "object_id": node.object_id,
                    "category": node.category,
                    "category_evidence": dict(node.category_evidence),
                    "centroid": (
                        node.centroid.round(3).tolist() if node.geometry_valid else None
                    ),
                    "last_known_centroid": node.centroid.round(3).tolist(),
                    "geometry_valid": node.geometry_valid,
                    "geometry_epoch": node.geometry_epoch,
                    "provisional": node.provisional,
                    "active": node.state != "sliced",
                    "state": node.state,
                    "last_seen": node.last_seen,
                    "observation_count": node.detection_count,
                    "stale_through_frame": node.stale_through_frame,
                    "visible_now": node.object_id in current_ids,
                    "parent_object_id": node.parent_object_id,
                    "action_states": {
                        name: {
                            "value": evidence.value,
                            "source": "successful_action",
                            "source_frame": evidence.frame_id,
                        }
                        for name, evidence in node.action_states.items()
                    },
                    "region": (
                        self.world_to_region(*node.centroid[:2])
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
        }

    def to_scene_objects(self) -> list[SceneObject]:
        """Export current geometry and CLIP features using the existing scene model."""
        from keyframe.models.scene import SceneObject

        result: list[SceneObject] = []
        for node in self.objects.values():
            if not node.geometry_valid:
                continue
            views = [
                view for view in node.views if view.frame_id > node.stale_through_frame
            ]
            result.append(
                SceneObject(
                    obj_id=node.object_id,
                    category=node.category,
                    centroid=node.centroid,
                    pcd_np=node.points.copy(),
                    pcd_color_np=node.colors.copy(),
                    bbox_np=node.bounds,
                    clip_ft=node.image_feature.astype(np.float32),
                    image_idx=[view.frame_id for view in views],
                    class_name=[node.category for _ in views],
                    xyxy=[np.array(view.box_xyxy) for view in views],
                    num_detections=node.detection_count,
                )
            )
        return result

    def render_bev(self, path: Path) -> Path:
        """Render the current objects and separate spatial map."""
        from .rendering import render_memory_bev

        return render_memory_bev(self, path)

    def _save_geometry(self, observation: Observation) -> Path:
        directory = self.output_dir / "observations"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"frame_{observation.frame_id:06d}.npz"
        if path.exists():
            raise FileExistsError(f"observation artifact already exists: {path}")
        detections = observation.detections
        dimension = len(observation.frame_feature)
        np.savez_compressed(
            path,
            depth_m=observation.depth_m.astype(np.float32),
            intrinsics=observation.intrinsics,
            camera_to_world=observation.camera_to_world,
            agent_position=observation.agent_position,
            masks=(
                np.stack([detection.mask for detection in detections])
                if detections
                else np.empty((0, *observation.depth_m.shape), dtype=bool)
            ),
            categories=np.array(
                [detection.category for detection in detections], dtype=str
            ),
            confidences=np.array([detection.confidence for detection in detections]),
            detector_boxes=np.asarray(
                [detection.bbox_xyxy for detection in detections], dtype=np.float64
            ).reshape(-1, 4),
            image_features=(
                np.stack([detection.image_feature for detection in detections])
                if detections
                else np.empty((0, dimension))
            ),
            text_features=(
                np.stack([detection.text_feature for detection in detections])
                if detections
                else np.empty((0, dimension))
            ),
            frame_feature=observation.frame_feature,
        )
        return path.resolve()

    def _save_frame_index(
        self, frame: FrameRecord, image_shape: tuple[int, ...]
    ) -> None:
        effective_path = frame.geometry_path.with_suffix(".mapping.npz")
        indices = sorted(self.current_detection_masks)
        with effective_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                detection_indices=np.array(indices, dtype=np.int64),
                masks=(
                    np.stack([self.current_detection_masks[index] for index in indices])
                    if indices
                    else np.empty((0, *image_shape), dtype=bool)
                ),
            )
        record = {
            "frame_id": frame.frame_id,
            "rgb_path": str(frame.rgb_path),
            "geometry_path": str(frame.geometry_path),
            "object_ids": frame.object_ids,
            "detection_object_ids": self.current_detection_objects,
            "suppressed_detection_indices": self.suppressed_detection_indices,
            "effective_masks_path": str(effective_path),
            "rejected_detection_reasons": self.rejected_detection_reasons,
            "cells": frame.cells,
            "camera_position": frame.camera_position,
            "camera_direction": frame.camera_direction,
            "aliases": self.aliases,
            "object_geometry_epochs": {
                str(value): self.objects[value].geometry_epoch
                for value in frame.object_ids
            },
        }
        with frame.geometry_path.with_suffix(".json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(record, stream, allow_nan=False)

    def _record_event(self, event: dict[str, object]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "memory_events.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")


__all__ = [
    "Detection",
    "FrameRecord",
    "IncrementalSceneMemory",
    "MappingConfig",
    "MemoryConfig",
    "ObjectMemory",
    "Observation",
]
