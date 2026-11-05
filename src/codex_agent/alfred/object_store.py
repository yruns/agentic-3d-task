"""Stable object identities and typed observations for online scene memory."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Detection, FloatArray


@dataclass(frozen=True)
class MappedDetection:
    """One image detection with usable metric geometry."""

    index: int
    detection: Detection
    points: FloatArray
    colors: FloatArray
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ObjectView:
    """A source region and the geometry version observed at that moment."""

    frame_id: int
    detection_index: int
    box_xyxy: tuple[float, float, float, float]
    geometry_epoch: int
    confidence: float
    pixel_area: int


@dataclass
class ObjectMemory:
    """A persistent identity with replaceable geometry and accumulated CLIP evidence.

    A provisional identity is immediately queryable. Confidence increases through
    distinct frames; repeated same-frame detections cannot confirm an object.
    """

    object_id: int
    category: str
    points: FloatArray
    colors: FloatArray
    image_feature: FloatArray
    text_feature: FloatArray
    first_seen: int
    last_seen: int
    detection_count: int = 1
    provisional: bool = True
    geometry_valid: bool = True
    geometry_epoch: int = 0
    stale_through_frame: int = -1
    state: str = "observed"
    # Unweighted distinct-frame category votes; confidence stays on ObjectView.
    category_evidence: dict[str, int] = field(default_factory=dict)
    views: list[ObjectView] = field(default_factory=list)
    reacquire_bounds: FloatArray | None = None
    parent_object_id: int | None = None
    action_states: dict[str, ActionState] = field(default_factory=dict)

    @property
    def centroid(self) -> FloatArray:
        """Current or last-known centroid; navigation must require geometry_valid."""
        return np.asarray(self.points.mean(axis=0), dtype=np.float64)

    @property
    def bounds(self) -> FloatArray:
        """Axis-aligned metric envelope of the current or last-known cloud."""
        return np.stack((self.points.min(axis=0), self.points.max(axis=0)))


@dataclass(frozen=True)
class ActionState:
    """A successful-action state prior, explicitly distinct from visual proof."""

    value: bool
    frame_id: int


class ObjectStore:
    """Own monotonically allocated IDs; aliases survive object consolidation."""

    def __init__(self) -> None:
        self.objects: dict[int, ObjectMemory] = {}
        self.aliases: dict[int, int] = {}
        self._next_id = 0

    def resolve(self, object_id: int) -> int:
        """Resolve any previously issued ID or reject an unknown identity."""
        if type(object_id) is not int or object_id < 0:
            raise ValueError(
                f"object_id must be a nonnegative integer, got {object_id!r}"
            )
        resolved = object_id
        while resolved in self.aliases:
            resolved = self.aliases[resolved]
        if resolved not in self.objects:
            raise KeyError(f"unknown object_id={object_id}")
        return resolved

    def create(self, region: MappedDetection, frame_id: int) -> ObjectMemory:
        """Allocate a fresh ID without using a simulator identity or list position."""
        detection = region.detection
        node = ObjectMemory(
            object_id=self._next_id,
            category=detection.category,
            points=region.points.copy(),
            colors=region.colors.copy(),
            image_feature=detection.image_feature.copy(),
            text_feature=detection.text_feature.copy(),
            first_seen=frame_id,
            last_seen=frame_id,
            category_evidence={detection.category: 1},
        )
        self._next_id += 1
        self.objects[node.object_id] = node
        return node

    def retire_alias(self, retired: int, survivor: int) -> None:
        """Retire a merged identity after its evidence has moved to the survivor."""
        retired, survivor = self.resolve(retired), self.resolve(survivor)
        if retired == survivor:
            raise ValueError("cannot alias an identity to itself")
        if retired < survivor:
            raise ValueError("the oldest object ID must survive a merge")
        self.aliases[retired] = survivor
        del self.objects[retired]
