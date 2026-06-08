"""Geometric checks for spatial relations between 3D objects.

Each relation is evaluated from object centroids and (when available) 3D
bounding boxes, returning a :class:`RelationResult` with a boolean
satisfaction flag and a continuous score in [0, 1].
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from keyframe.models.scene import SceneObject

#: Natural-language relation variants mapped to canonical relation names.
RELATION_ALIASES: dict[str, str] = {
    "on": "on_top_of",
    "upon": "on_top_of",
    "atop": "on_top_of",
    "on_top": "on_top_of",
    "on_top_of": "on_top_of",
    "resting_on": "on_top_of",
    "above": "above",
    "over": "above",
    "higher_than": "above",
    "below": "below",
    "under": "below",
    "beneath": "below",
    "underneath": "below",
    "lower_than": "below",
    "next_to": "next_to",
    "beside": "next_to",
    "near": "near",
    "by": "near",
    "close_to": "near",
    "adjacent_to": "next_to",
    "next": "next_to",
    "in_front_of": "in_front_of",
    "front": "in_front_of",
    "facing": "in_front_of",
    "before": "in_front_of",
    "behind": "behind",
    "back": "behind",
    "back_of": "behind",
    "in_back_of": "behind",
    "left_of": "left_of",
    "left": "left_of",
    "to_the_left_of": "left_of",
    "on_the_left_of": "left_of",
    "right_of": "right_of",
    "right": "right_of",
    "to_the_right_of": "right_of",
    "on_the_right_of": "right_of",
    "inside": "inside",
    "in": "inside",
    "within": "inside",
    "contained_in": "inside",
    "between": "between",
    "in_between": "between",
    "against": "against",
    "leaning_on": "against",
    "leaning_against": "against",
    "around": "around",
    "surrounding": "around",
}

# Distance/offset thresholds in meters.
_ON_MAX_HORIZONTAL = 0.5
_ON_MIN_VERTICAL = 0.0
_ON_MAX_VERTICAL = 1.0
_ON_CONTACT_TOLERANCE = 0.2
_ABOVE_MAX_HORIZONTAL = 1.0
_ABOVE_MIN_VERTICAL = 0.1
_BELOW_MAX_HORIZONTAL = 1.0
_BELOW_MAX_VERTICAL = -0.1
_NEXT_TO_MAX_DISTANCE = 1.5
_NEAR_MAX_DISTANCE = 3.0
_INSIDE_MARGIN = 0.1
_BETWEEN_MAX_DISTANCE_RATIO = 0.3
_AGAINST_MAX_DISTANCE = 0.5

# Anchors for which "on" means "attached to a vertical surface".
_VERTICAL_SURFACE_TERMS: frozenset[str] = frozenset(
    {
        "wall",
        "partition",
        "door",
        "whiteboard",
        "blackboard",
        "chalkboard",
        "bulletin board",
        "board",
        "mirror",
        "window",
    }
)

BBox = tuple[NDArray[np.float64], NDArray[np.float64]]


@dataclass(frozen=True)
class RelationResult:
    """Result of a spatial relation check."""

    satisfies: bool
    score: float


def canonical_relation(relation: str) -> str:
    """Return the canonical relation name for a natural-language relation."""
    normalized = relation.lower().replace(" ", "_")
    return RELATION_ALIASES.get(normalized, normalized)


class SpatialRelationChecker:
    """Checks geometric spatial relations between scene objects."""

    def check(
        self,
        target: SceneObject,
        anchor: SceneObject | list[SceneObject],
        relation: str,
    ) -> RelationResult:
        """Check whether ``target`` satisfies ``relation`` with ``anchor``(s)."""
        canonical = canonical_relation(relation)

        if canonical == "between":
            anchors = anchor if isinstance(anchor, list) else [anchor]
            if len(anchors) < 2:
                return RelationResult(satisfies=False, score=0.0)
            best = RelationResult(satisfies=False, score=0.0)
            best_unsatisfied = RelationResult(satisfies=False, score=0.0)
            for anchor1, anchor2 in combinations(anchors, 2):
                result = self.is_between(target, anchor1, anchor2)
                if result.satisfies and result.score >= best.score:
                    best = result
                elif result.score >= best_unsatisfied.score:
                    best_unsatisfied = result
            return best if best.satisfies else best_unsatisfied

        single_anchor = anchor[0] if isinstance(anchor, list) else anchor
        if single_anchor is None:
            return RelationResult(satisfies=False, score=0.0)

        checker = self._relation_method(canonical)
        return checker(target, single_anchor)

    def _relation_method(
        self, canonical: str
    ) -> Callable[[SceneObject, SceneObject], RelationResult]:
        methods = {
            "on_top_of": self.is_on_top_of,
            "above": self.is_above,
            "below": self.is_below,
            "next_to": self.is_next_to,
            "near": self.is_near,
            "in_front_of": self.is_in_front_of,
            "behind": self.is_behind,
            "left_of": self.is_left_of,
            "right_of": self.is_right_of,
            "inside": self.is_inside,
            "against": self.is_against,
            "around": self.is_near,
        }
        return methods.get(canonical, self.is_near)

    # ----- geometry helpers -------------------------------------------------

    @staticmethod
    def _centroid(obj: SceneObject) -> NDArray[np.float64]:
        if obj.centroid is not None:
            return np.asarray(obj.centroid, dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    @staticmethod
    def _bbox(obj: SceneObject) -> BBox | None:
        bbox = obj.bbox_np
        if bbox is None:
            return None
        array = np.asarray(bbox, dtype=np.float64)
        if array.size < 6:
            return None
        if array.ndim == 1 and array.size == 6:
            points = array.reshape(2, 3)
        elif array.ndim >= 2 and array.shape[-1] >= 3:
            points = array.reshape(-1, array.shape[-1])[:, :3]
        elif array.ndim == 1 and array.size % 3 == 0:
            points = array.reshape(-1, 3)
        else:
            return None
        if len(points) < 2 or not np.isfinite(points).all():
            return None
        return np.min(points, axis=0), np.max(points, axis=0)

    @staticmethod
    def _bbox_center(bbox: BBox) -> NDArray[np.float64]:
        return (bbox[0] + bbox[1]) / 2.0

    @staticmethod
    def _bbox_xy_gap(bbox1: BBox, bbox2: BBox) -> float:
        min1, max1 = bbox1
        min2, max2 = bbox2
        dx = max(float(min2[0] - max1[0]), float(min1[0] - max2[0]), 0.0)
        dy = max(float(min2[1] - max1[1]), float(min1[1] - max2[1]), 0.0)
        return float(np.hypot(dx, dy))

    @staticmethod
    def _bbox_3d_gap(bbox1: BBox, bbox2: BBox) -> float:
        min1, max1 = bbox1
        min2, max2 = bbox2
        gaps = np.maximum(np.maximum(min2 - max1, min1 - max2), 0.0)
        return float(np.linalg.norm(gaps))

    @staticmethod
    def _bbox_xy_overlap_ratio(bbox1: BBox, bbox2: BBox) -> float:
        min1, max1 = bbox1
        min2, max2 = bbox2
        overlap_x = max(0.0, float(min(max1[0], max2[0]) - max(min1[0], min2[0])))
        overlap_y = max(0.0, float(min(max1[1], max2[1]) - max(min1[1], min2[1])))
        intersection = overlap_x * overlap_y
        area1 = max(0.0, float((max1[0] - min1[0]) * (max1[1] - min1[1])))
        area2 = max(0.0, float((max2[0] - min2[0]) * (max2[1] - min2[1])))
        denom = min(area1, area2)
        return float(intersection / denom) if denom > 1e-6 else 0.0

    @staticmethod
    def _axis_overlap_ratio(bbox1: BBox, bbox2: BBox, axis: int) -> float:
        min1, max1 = bbox1
        min2, max2 = bbox2
        overlap = max(
            0.0, float(min(max1[axis], max2[axis]) - max(min1[axis], min2[axis]))
        )
        extent = max(1e-6, float(max1[axis] - min1[axis]))
        return float(overlap / extent)

    @staticmethod
    def _category_text(obj: SceneObject) -> str:
        parts = [obj.category, obj.object_tag, obj.summary, *obj.class_name]
        return " ".join(part for part in parts if part).lower()

    def _is_vertical_surface_anchor(self, obj: SceneObject) -> bool:
        text = self._category_text(obj)
        return any(term in text for term in _VERTICAL_SURFACE_TERMS)

    # ----- vertical relations ----------------------------------------------

    def is_on_top_of(self, target: SceneObject, anchor: SceneObject) -> RelationResult:
        """Whether ``target`` rests on (or is attached to) ``anchor``."""
        t_bbox = self._bbox(target)
        a_bbox = self._bbox(anchor)
        if t_bbox is not None and a_bbox is not None:
            if self._is_vertical_surface_anchor(anchor):
                return self._vertical_surface_attachment(t_bbox, a_bbox)

            t_min, _t_max = t_bbox
            a_min, a_max = a_bbox
            xy_gap = self._bbox_xy_gap(t_bbox, a_bbox)
            overlap_ratio = self._bbox_xy_overlap_ratio(t_bbox, a_bbox)
            vertical_gap = float(t_min[2] - a_max[2])
            anchor_xy_diag = float(np.linalg.norm(a_max[:2] - a_min[:2]))
            max_horizontal = max(_ON_MAX_HORIZONTAL, min(1.0, 0.35 * anchor_xy_diag))
            anchor_height = max(0.0, float(a_max[2] - a_min[2]))
            penetration_tolerance = (
                max(_ON_CONTACT_TOLERANCE, min(1.75, 0.85 * anchor_height))
                if overlap_ratio > 0.0
                else max(_ON_CONTACT_TOLERANCE, min(0.45, 0.45 * anchor_height))
            )

            vertical_ok = (
                _ON_MIN_VERTICAL - penetration_tolerance
                <= vertical_gap
                <= _ON_MAX_VERTICAL
            )
            supported = overlap_ratio > 0.0 or xy_gap <= max_horizontal
            if not supported or not vertical_ok:
                return RelationResult(satisfies=False, score=0.0)

            gap_score = max(0.0, 1.0 - xy_gap / (max_horizontal + 1e-6))
            support_score = max(gap_score, min(1.0, overlap_ratio))
            if vertical_gap < 0.0:
                vertical_score = max(
                    0.0, 1.0 - abs(vertical_gap) / (penetration_tolerance + 1e-6)
                )
            else:
                vertical_score = max(
                    0.0, 1.0 - abs(vertical_gap) / (_ON_MAX_VERTICAL + 1e-6)
                )
            return RelationResult(
                satisfies=True, score=0.65 * support_score + 0.35 * vertical_score
            )

        diff = self._centroid(target) - self._centroid(anchor)
        horizontal_dist = float(np.linalg.norm(diff[:2]))
        vertical_diff = float(diff[2])
        if vertical_diff < _ON_MIN_VERTICAL or horizontal_dist > _ON_MAX_HORIZONTAL:
            return RelationResult(satisfies=False, score=0.0)
        h_score = max(0.0, 1.0 - horizontal_dist / _ON_MAX_HORIZONTAL)
        v_score = min(1.0, vertical_diff / 0.3)
        return RelationResult(satisfies=True, score=0.6 * h_score + 0.4 * v_score)

    def _vertical_surface_attachment(
        self, t_bbox: BBox, a_bbox: BBox
    ) -> RelationResult:
        xy_gap = self._bbox_xy_gap(t_bbox, a_bbox)
        overlap_ratio = self._bbox_xy_overlap_ratio(t_bbox, a_bbox)
        z_overlap_ratio = self._axis_overlap_ratio(t_bbox, a_bbox, axis=2)
        max_surface_gap = 0.75
        has_surface_contact = overlap_ratio > 0.0 or xy_gap <= max_surface_gap
        has_vertical_overlap = z_overlap_ratio >= 0.25
        if not has_surface_contact or not has_vertical_overlap:
            return RelationResult(satisfies=False, score=0.0)
        proximity_score = (
            min(1.0, overlap_ratio)
            if overlap_ratio > 0.0
            else max(0.0, 1.0 - xy_gap / (max_surface_gap + 1e-6))
        )
        score = 0.55 * proximity_score + 0.45 * min(1.0, z_overlap_ratio)
        return RelationResult(satisfies=True, score=float(score))

    def is_above(self, target: SceneObject, anchor: SceneObject) -> RelationResult:
        """Whether ``target`` is above ``anchor`` (not necessarily touching)."""
        t_bbox = self._bbox(target)
        a_bbox = self._bbox(anchor)
        if t_bbox is not None and a_bbox is not None:
            vertical_diff = float(
                self._bbox_center(t_bbox)[2] - self._bbox_center(a_bbox)[2]
            )
            xy_gap = self._bbox_xy_gap(t_bbox, a_bbox)
            overlap_ratio = self._bbox_xy_overlap_ratio(t_bbox, a_bbox)
            if vertical_diff < _ABOVE_MIN_VERTICAL:
                return RelationResult(satisfies=False, score=0.0)
            if xy_gap > _ABOVE_MAX_HORIZONTAL and overlap_ratio <= 0.0:
                return RelationResult(satisfies=False, score=0.0)
            h_score = (
                min(1.0, overlap_ratio)
                if overlap_ratio > 0
                else max(0.0, 1.0 - xy_gap / (_ABOVE_MAX_HORIZONTAL + 1e-6))
            )
            return RelationResult(
                satisfies=True, score=0.5 * h_score + 0.5 * min(1.0, vertical_diff)
            )

        diff = self._centroid(target) - self._centroid(anchor)
        horizontal_dist = float(np.linalg.norm(diff[:2]))
        vertical_diff = float(diff[2])
        if (
            vertical_diff < _ABOVE_MIN_VERTICAL
            or horizontal_dist > _ABOVE_MAX_HORIZONTAL
        ):
            return RelationResult(satisfies=False, score=0.0)
        h_score = max(0.0, 1.0 - horizontal_dist / _ABOVE_MAX_HORIZONTAL)
        return RelationResult(
            satisfies=True, score=0.5 * h_score + 0.5 * min(1.0, vertical_diff)
        )

    def is_below(self, target: SceneObject, anchor: SceneObject) -> RelationResult:
        """Whether ``target`` is below ``anchor``."""
        t_bbox = self._bbox(target)
        a_bbox = self._bbox(anchor)
        if t_bbox is not None and a_bbox is not None:
            vertical_diff = float(
                self._bbox_center(t_bbox)[2] - self._bbox_center(a_bbox)[2]
            )
            xy_gap = self._bbox_xy_gap(t_bbox, a_bbox)
            overlap_ratio = self._bbox_xy_overlap_ratio(t_bbox, a_bbox)
            if vertical_diff > _BELOW_MAX_VERTICAL:
                return RelationResult(satisfies=False, score=0.0)
            if xy_gap > _BELOW_MAX_HORIZONTAL and overlap_ratio <= 0.0:
                return RelationResult(satisfies=False, score=0.0)
            h_score = (
                min(1.0, overlap_ratio)
                if overlap_ratio > 0
                else max(0.0, 1.0 - xy_gap / (_BELOW_MAX_HORIZONTAL + 1e-6))
            )
            return RelationResult(
                satisfies=True, score=0.5 * h_score + 0.5 * min(1.0, abs(vertical_diff))
            )

        diff = self._centroid(target) - self._centroid(anchor)
        horizontal_dist = float(np.linalg.norm(diff[:2]))
        vertical_diff = float(diff[2])
        if (
            vertical_diff > _BELOW_MAX_VERTICAL
            or horizontal_dist > _BELOW_MAX_HORIZONTAL
        ):
            return RelationResult(satisfies=False, score=0.0)
        h_score = max(0.0, 1.0 - horizontal_dist / _BELOW_MAX_HORIZONTAL)
        return RelationResult(
            satisfies=True, score=0.5 * h_score + 0.5 * min(1.0, abs(vertical_diff))
        )

    # ----- distance relations ----------------------------------------------

    def is_next_to(self, target: SceneObject, anchor: SceneObject) -> RelationResult:
        """Whether ``target`` is immediately next to ``anchor``."""
        t_bbox = self._bbox(target)
        a_bbox = self._bbox(anchor)
        if t_bbox is not None and a_bbox is not None:
            distance = self._bbox_xy_gap(t_bbox, a_bbox)
        else:
            distance = float(
                np.linalg.norm(self._centroid(target) - self._centroid(anchor))
            )
        if distance > _NEXT_TO_MAX_DISTANCE:
            return RelationResult(satisfies=False, score=0.0)
        return RelationResult(
            satisfies=True,
            score=max(0.0, 1.0 - distance / (_NEXT_TO_MAX_DISTANCE + 1e-6)),
        )

    def is_near(self, target: SceneObject, anchor: SceneObject) -> RelationResult:
        """Whether ``target`` is near ``anchor`` (looser than next_to)."""
        t_bbox = self._bbox(target)
        a_bbox = self._bbox(anchor)
        if t_bbox is not None and a_bbox is not None:
            distance = self._bbox_3d_gap(t_bbox, a_bbox)
        else:
            distance = float(
                np.linalg.norm(self._centroid(target) - self._centroid(anchor))
            )
        if distance > _NEAR_MAX_DISTANCE:
            return RelationResult(satisfies=False, score=0.0)
        return RelationResult(
            satisfies=True, score=max(0.0, 1.0 - distance / (_NEAR_MAX_DISTANCE + 1e-6))
        )

    # ----- world-frame directional relations -------------------------------

    def is_in_front_of(
        self, target: SceneObject, anchor: SceneObject
    ) -> RelationResult:
        """Whether ``target`` is in front of ``anchor`` (+Y forward)."""
        diff = self._centroid(target) - self._centroid(anchor)
        forward_dist = float(diff[1])
        if forward_dist <= 0:
            return RelationResult(satisfies=False, score=0.0)
        lateral_dist = abs(float(diff[0]))
        if lateral_dist > forward_dist:
            return RelationResult(satisfies=False, score=0.0)
        score = min(1.0, forward_dist / 2.0) * max(
            0.0, 1.0 - lateral_dist / forward_dist
        )
        return RelationResult(satisfies=True, score=float(score))

    def is_behind(self, target: SceneObject, anchor: SceneObject) -> RelationResult:
        """Whether ``target`` is behind ``anchor`` (-Y)."""
        diff = self._centroid(target) - self._centroid(anchor)
        if diff[1] >= 0:
            return RelationResult(satisfies=False, score=0.0)
        lateral_dist = abs(float(diff[0]))
        backward_dist = abs(float(diff[1]))
        if lateral_dist > backward_dist:
            return RelationResult(satisfies=False, score=0.0)
        score = min(1.0, backward_dist / 2.0) * max(
            0.0, 1.0 - lateral_dist / backward_dist
        )
        return RelationResult(satisfies=True, score=float(score))

    def is_left_of(self, target: SceneObject, anchor: SceneObject) -> RelationResult:
        """Whether ``target`` is to the left of ``anchor`` (-X)."""
        diff = self._centroid(target) - self._centroid(anchor)
        if diff[0] >= 0:
            return RelationResult(satisfies=False, score=0.0)
        lateral_dist = abs(float(diff[0]))
        if abs(float(diff[1])) > lateral_dist:
            return RelationResult(satisfies=False, score=0.0)
        return RelationResult(satisfies=True, score=min(1.0, lateral_dist / 2.0))

    def is_right_of(self, target: SceneObject, anchor: SceneObject) -> RelationResult:
        """Whether ``target`` is to the right of ``anchor`` (+X)."""
        diff = self._centroid(target) - self._centroid(anchor)
        if diff[0] <= 0:
            return RelationResult(satisfies=False, score=0.0)
        lateral_dist = abs(float(diff[0]))
        if abs(float(diff[1])) > lateral_dist:
            return RelationResult(satisfies=False, score=0.0)
        return RelationResult(satisfies=True, score=min(1.0, lateral_dist / 2.0))

    # ----- containment / contact / multi-object ----------------------------

    def is_inside(self, target: SceneObject, anchor: SceneObject) -> RelationResult:
        """Whether ``target`` is inside ``anchor`` (requires anchor bbox)."""
        a_bbox = self._bbox(anchor)
        if a_bbox is None:
            return RelationResult(satisfies=False, score=0.0)
        a_min, a_max = a_bbox
        t_bbox = self._bbox(target)
        if t_bbox is not None:
            t_min, t_max = t_bbox
            inside = bool(
                np.all(t_min >= a_min - _INSIDE_MARGIN)
                and np.all(t_max <= a_max + _INSIDE_MARGIN)
            )
            target_center = self._bbox_center(t_bbox)
        else:
            point = self._centroid(target)
            inside = bool(
                np.all(point >= a_min - _INSIDE_MARGIN)
                and np.all(point <= a_max + _INSIDE_MARGIN)
            )
            target_center = point
        if not inside:
            return RelationResult(satisfies=False, score=0.0)
        center = (a_min + a_max) / 2.0
        size = a_max - a_min
        normalized = np.abs(target_center - center) / (size / 2.0 + 1e-6)
        return RelationResult(
            satisfies=True, score=max(0.0, float(1.0 - np.mean(normalized)))
        )

    def is_between(
        self, target: SceneObject, anchor1: SceneObject, anchor2: SceneObject
    ) -> RelationResult:
        """Whether ``target`` lies between ``anchor1`` and ``anchor2``."""
        t_pos = self._centroid(target)
        a1_pos = self._centroid(anchor1)
        a2_pos = self._centroid(anchor2)
        line_vec = a2_pos - a1_pos
        line_len = float(np.linalg.norm(line_vec))
        if line_len < 1e-6:
            return RelationResult(satisfies=False, score=0.0)
        line_dir = line_vec / line_len
        proj_len = float(np.dot(t_pos - a1_pos, line_dir))
        if proj_len < 0 or proj_len > line_len:
            return RelationResult(satisfies=False, score=0.0)
        proj_point = a1_pos + proj_len * line_dir
        dist_to_line = float(np.linalg.norm(t_pos - proj_point))
        max_dist = line_len * _BETWEEN_MAX_DISTANCE_RATIO
        if dist_to_line > max_dist:
            return RelationResult(satisfies=False, score=0.0)
        center_score = 1.0 - abs(proj_len - line_len / 2.0) / (line_len / 2.0)
        line_score = 1.0 - dist_to_line / max_dist if max_dist > 0 else 1.0
        return RelationResult(
            satisfies=True, score=0.5 * center_score + 0.5 * line_score
        )

    def is_against(self, target: SceneObject, anchor: SceneObject) -> RelationResult:
        """Whether ``target`` is against (touching) ``anchor``."""
        distance = float(
            np.linalg.norm(self._centroid(target) - self._centroid(anchor))
        )
        if distance > _AGAINST_MAX_DISTANCE:
            return RelationResult(satisfies=False, score=0.0)
        return RelationResult(
            satisfies=True, score=max(0.0, 1.0 - distance / _AGAINST_MAX_DISTANCE)
        )
