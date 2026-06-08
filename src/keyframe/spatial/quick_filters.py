"""Fast coordinate-based pre-filters and attribute filters.

``QuickFilters`` cheaply prunes candidates for view-independent relations
(vertical / distance) before the full :mod:`keyframe.spatial.checker` runs.
``AttributeFilter`` filters by color (with synonyms) and size rank.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from keyframe.models.scene import SceneObject


class FilterType(str, Enum):
    """Family of a quick filter."""

    VERTICAL = "vertical"
    DISTANCE = "distance"


#: Direction of a threshold comparison ("keep when diff > / < threshold").
Comparator = Literal["gt", "lt"]
#: Axis along which a distance filter measures separation.
FilterAxis = Literal["z", "xy", "xyz"]


@dataclass(frozen=True)
class FilterConfig:
    """Threshold configuration for a single quick filter."""

    filter_type: FilterType
    comparator: Comparator
    threshold: float
    axis: FilterAxis = "xyz"
    loose_factor: float = 1.5


# Only view-independent relations get quick filters (gravity / Euclidean
# distance are observer-invariant). Directional relations need full reasoning.
QUICK_FILTER_CONFIGS: dict[str, FilterConfig] = {
    "on_top_of": FilterConfig(
        FilterType.VERTICAL, "gt", 0.0, axis="z", loose_factor=1.0
    ),
    "above": FilterConfig(FilterType.VERTICAL, "gt", 0.1, axis="z", loose_factor=1.0),
    "below": FilterConfig(FilterType.VERTICAL, "lt", -0.1, axis="z", loose_factor=1.0),
    "near": FilterConfig(FilterType.DISTANCE, "lt", 3.0, axis="xyz", loose_factor=1.5),
    "next_to": FilterConfig(
        FilterType.DISTANCE, "lt", 1.5, axis="xyz", loose_factor=1.5
    ),
    "beside": FilterConfig(
        FilterType.DISTANCE, "lt", 1.5, axis="xyz", loose_factor=1.5
    ),
}

FILTER_ALIASES: dict[str, str] = {
    "on": "on_top_of",
    "upon": "on_top_of",
    "atop": "on_top_of",
    "over": "above",
    "beneath": "below",
    "underneath": "below",
    "under": "below",
    "nearby": "near",
    "close_to": "near",
    "adjacent": "next_to",
    "adjacent_to": "next_to",
}


class QuickFilters:
    """Fast spatial pre-filters using simple coordinate comparisons."""

    def __init__(self, use_loose_threshold: bool = True) -> None:
        self.use_loose_threshold = use_loose_threshold

    @staticmethod
    def _canonical(relation: str) -> str:
        normalized = relation.lower().replace(" ", "_")
        return FILTER_ALIASES.get(normalized, normalized)

    def has_filter(self, relation: str) -> bool:
        """Whether a quick filter is defined for ``relation``."""
        return self._canonical(relation) in QUICK_FILTER_CONFIGS

    @staticmethod
    def _centroid(obj: SceneObject) -> NDArray[np.float64]:
        if obj.centroid is not None:
            return np.asarray(obj.centroid, dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    def _anchor_center(self, anchors: list[SceneObject]) -> NDArray[np.float64]:
        if not anchors:
            return np.zeros(3, dtype=np.float64)
        return np.mean([self._centroid(a) for a in anchors], axis=0)  # type: ignore[no-any-return]

    def filter_candidates(
        self,
        candidates: list[SceneObject],
        anchors: list[SceneObject],
        relation: str,
    ) -> list[SceneObject]:
        """Return the candidates that pass the quick filter for ``relation``."""
        canonical = self._canonical(relation)
        config = QUICK_FILTER_CONFIGS.get(canonical)
        if config is None:
            return candidates
        if config.filter_type == FilterType.VERTICAL:
            return self._filter_vertical(candidates, anchors, config)
        return self._filter_distance(candidates, anchors, config)

    def _filter_vertical(
        self,
        candidates: list[SceneObject],
        anchors: list[SceneObject],
        config: FilterConfig,
    ) -> list[SceneObject]:
        anchor_z = float(self._anchor_center(anchors)[2])
        threshold = config.threshold
        if self.use_loose_threshold:
            adjust = abs(threshold) * (config.loose_factor - 1.0)
            threshold += -adjust if config.comparator == "gt" else adjust

        kept: list[SceneObject] = []
        for obj in candidates:
            diff = float(self._centroid(obj)[2]) - anchor_z
            if config.comparator == "gt" and diff > threshold:
                kept.append(obj)
            elif config.comparator == "lt" and diff < threshold:
                kept.append(obj)
        return kept

    def _filter_distance(
        self,
        candidates: list[SceneObject],
        anchors: list[SceneObject],
        config: FilterConfig,
    ) -> list[SceneObject]:
        anchor_center = self._anchor_center(anchors)
        threshold = config.threshold
        if self.use_loose_threshold:
            threshold *= config.loose_factor

        kept: list[SceneObject] = []
        for obj in candidates:
            position = self._centroid(obj)
            if config.axis == "z":
                distance = abs(float(position[2] - anchor_center[2]))
            elif config.axis == "xy":
                distance = float(np.linalg.norm(position[:2] - anchor_center[:2]))
            else:
                distance = float(np.linalg.norm(position - anchor_center))
            if config.comparator == "lt" and distance < threshold:
                kept.append(obj)
            elif config.comparator == "gt" and distance > threshold:
                kept.append(obj)
        return kept


class AttributeFilter:
    """Filter candidates by color (with synonyms) or size rank."""

    COLOR_SYNONYMS: dict[str, list[str]] = {
        "red": ["red", "crimson", "scarlet", "maroon", "ruby"],
        "blue": ["blue", "navy", "azure", "cobalt", "cyan", "teal"],
        "green": ["green", "olive", "lime", "emerald", "forest"],
        "yellow": ["yellow", "gold", "golden", "amber"],
        "orange": ["orange", "tangerine", "coral"],
        "purple": ["purple", "violet", "lavender", "magenta", "plum"],
        "pink": ["pink", "rose", "salmon", "fuchsia"],
        "brown": ["brown", "chocolate", "tan", "beige", "khaki", "coffee"],
        "black": ["black", "dark", "ebony", "charcoal"],
        "white": ["white", "ivory", "cream", "pearl"],
        "gray": ["gray", "grey", "silver", "slate"],
    }

    def __init__(self) -> None:
        self._color_lookup: dict[str, str] = {}
        for canonical, variants in self.COLOR_SYNONYMS.items():
            for variant in variants:
                self._color_lookup[variant] = canonical

    def knows_color(self, token: str) -> bool:
        """Whether ``token`` is a recognized color (or color synonym)."""
        return token.lower() in self._color_lookup

    def filter_by_color(
        self, candidates: list[SceneObject], color: str
    ) -> list[SceneObject]:
        """Keep candidates whose text mentions ``color`` (or a synonym)."""
        canonical = self._color_lookup.get(color.lower(), color.lower())
        variants = self.COLOR_SYNONYMS.get(canonical, [canonical])
        kept = [
            obj
            for obj in candidates
            if any(variant in self._searchable_text(obj) for variant in variants)
        ]
        # Never filter to empty on a soft attribute cue.
        return kept if kept else candidates

    def filter_by_size_rank(
        self,
        candidates: list[SceneObject],
        order: str,
        top_k: int = 1,
    ) -> list[SceneObject]:
        """Return the ``top_k`` largest (order='max') or smallest objects."""
        if not candidates:
            return []
        ranked = sorted(candidates, key=self._compute_size, reverse=(order == "max"))
        return ranked[:top_k]

    @staticmethod
    def _searchable_text(obj: SceneObject) -> str:
        parts = [obj.summary, obj.object_tag, obj.category]
        return " ".join(part for part in parts if part).lower()

    @staticmethod
    def _compute_size(obj: SceneObject) -> float:
        if obj.bbox_np is not None:
            points = np.asarray(obj.bbox_np, dtype=np.float64)
            if points.ndim == 2 and points.shape[1] >= 3:
                extent = points[:, :3].max(axis=0) - points[:, :3].min(axis=0)
                return float(np.prod(extent))
        if obj.pcd_np is not None:
            return float(len(obj.pcd_np))
        return 1.0
