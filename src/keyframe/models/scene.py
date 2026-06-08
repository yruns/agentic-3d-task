"""Scene object model.

``SceneObject`` is the in-memory representation of one detected 3D object,
loaded from a prepared ConceptGraph scene (``pcd_saves/*.pkl.gz``) and
optionally enriched with LLM-generated descriptions
(``enriched_objects.json``).

Only the fields used by the keyframe-selection mainline are kept: geometry
(centroid / point cloud / 3D bbox), per-detection 2D boxes (visibility
scoring), the CLIP visual feature (optional semantic fallback), class-name
votes (multi-label category matching) and enrichment text.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

# Class-name tokens that carry no semantic value and must not win the
# majority-vote that decides an object's primary category.
_NON_INFORMATIVE_CLASS_NAMES: frozenset[str] = frozenset({"", "item", "none"})


def _iter_raw(value: object) -> list[object]:
    """Normalize a raw pickle value into a Python list, or [] if not iterable."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _to_str_list(value: object) -> list[str]:
    return [str(item) for item in _iter_raw(value)]


def _to_int_list(value: object) -> list[int]:
    return [int(item) for item in _iter_raw(value)]  # type: ignore[call-overload]


def _to_float_list(value: object) -> list[float]:
    return [float(item) for item in _iter_raw(value)]  # type: ignore[arg-type]


def _to_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def _to_float_array(
    value: object, dim: int | None = None
) -> NDArray[np.float64] | None:
    """Convert raw pickle data to a float64 array, or None if absent/empty."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return None
    if dim is not None and array.shape[-1] != dim:
        return None
    return array


def _to_box_list(value: object) -> list[NDArray[np.float64]]:
    """Convert per-detection 2D boxes into a list of (4,) float64 arrays."""
    if value is None:
        return []
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1 and array.shape[0] == 4:
        return [array]
    if array.ndim == 2 and array.shape[1] == 4:
        return list(array)
    return []


def _centroid_from_raw(
    raw: dict[str, object], pcd: NDArray[np.float64] | None
) -> NDArray[np.float64] | None:
    """Resolve a centroid from point cloud, explicit centroid, or 3D bbox."""
    if pcd is not None:
        return pcd.mean(axis=0)  # type: ignore[no-any-return]
    centroid_raw = raw.get("centroid")
    if centroid_raw is not None:
        array = np.asarray(centroid_raw, dtype=np.float64).reshape(-1)
        if array.size >= 3:
            return array[:3]
    bbox = _to_float_array(raw.get("bbox_np"), dim=3)
    if bbox is not None and bbox.ndim == 2 and bbox.shape[1] >= 3:
        return bbox[:, :3].mean(axis=0)  # type: ignore[no-any-return]
    return None


def _majority_category(class_names: Sequence[str], fallback: str) -> str:
    """Pick the most common informative class name, else a stable fallback."""
    informative = [
        name
        for name in class_names
        if name and name.lower() not in _NON_INFORMATIVE_CLASS_NAMES
    ]
    if informative:
        return Counter(informative).most_common(1)[0][0]
    if class_names:
        return class_names[0]
    return fallback


class SceneObject(BaseModel):
    """A single detected 3D object in a prepared scene."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    obj_id: int
    category: str
    object_tag: str = ""

    # Geometry
    centroid: NDArray[np.float64] | None = None
    pcd_np: NDArray[np.float64] | None = None
    pcd_color_np: NDArray[np.float64] | None = None
    bbox_np: NDArray[np.float64] | None = None

    # CLIP visual feature (optional semantic fallback when category match fails)
    clip_ft: NDArray[np.float32] | None = None

    # Per-detection data (parallel lists indexed by detection order)
    image_idx: list[int] = Field(default_factory=list)
    class_name: list[str] = Field(default_factory=list)
    conf: list[float] = Field(default_factory=list)
    xyxy: list[NDArray[np.float64]] = Field(default_factory=list)

    num_detections: int = 0
    is_background: bool = False

    # LLM enrichment (enriched_objects.json / affordance file)
    summary: str = ""
    affordance_category: str = ""
    co_objects: list[str] = Field(default_factory=list)

    @property
    def clip_feature(self) -> NDArray[np.float32] | None:
        """Alias for the CLIP visual feature."""
        return self.clip_ft

    @property
    def point_cloud(self) -> NDArray[np.float64] | None:
        """Alias for the object's point cloud."""
        return self.pcd_np

    def has_detections(self) -> bool:
        """Whether the object has per-frame 2D detections."""
        return len(self.image_idx) > 0

    @classmethod
    def from_conceptgraph(cls, obj_id: int, raw: dict[str, object]) -> SceneObject:
        """Build a ``SceneObject`` from a raw ConceptGraph pickle entry."""
        class_names = _to_str_list(raw.get("class_name"))
        category = _majority_category(class_names, fallback=f"object_{obj_id}")

        pcd = _to_float_array(raw.get("pcd_np"), dim=3)
        centroid = _centroid_from_raw(raw, pcd)

        clip_raw = raw.get("clip_ft")
        clip_ft = (
            np.asarray(clip_raw, dtype=np.float32).reshape(-1)
            if clip_raw is not None
            else None
        )

        return cls(
            obj_id=obj_id,
            category=category,
            object_tag=category,
            centroid=centroid,
            pcd_np=pcd,
            pcd_color_np=_to_float_array(raw.get("pcd_color_np"), dim=3),
            bbox_np=_to_float_array(raw.get("bbox_np"), dim=3),
            clip_ft=clip_ft,
            image_idx=_to_int_list(raw.get("image_idx")),
            class_name=class_names,
            conf=_to_float_list(raw.get("conf")),
            xyxy=_to_box_list(raw.get("xyxy")),
            num_detections=_to_int(raw.get("num_detections"), 0),
            is_background=bool(raw.get("is_background", False)),
        )
