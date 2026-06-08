"""Spatial relation checking and view geometry."""

from __future__ import annotations

from keyframe.spatial.checker import (
    RELATION_ALIASES,
    RelationResult,
    SpatialRelationChecker,
    canonical_relation,
)
from keyframe.spatial.frustum import (
    frustum_overlap_l1,
    frustum_overlap_l2,
    load_scene_intrinsic,
)
from keyframe.spatial.quick_filters import (
    QUICK_FILTER_CONFIGS,
    AttributeFilter,
    FilterConfig,
    FilterType,
    QuickFilters,
)
from keyframe.spatial.viewpoint import (
    ViewerPose,
    resolve_viewer_pose,
    viewer_axes,
    viewer_frame_axis_value,
    viewer_frame_relation_score,
)

__all__ = [
    # Relation checking
    "SpatialRelationChecker",
    "RelationResult",
    "RELATION_ALIASES",
    "canonical_relation",
    # Quick filters
    "QuickFilters",
    "AttributeFilter",
    "FilterConfig",
    "FilterType",
    "QUICK_FILTER_CONFIGS",
    # Frustum
    "frustum_overlap_l1",
    "frustum_overlap_l2",
    "load_scene_intrinsic",
    # Viewpoint geometry
    "ViewerPose",
    "resolve_viewer_pose",
    "viewer_axes",
    "viewer_frame_axis_value",
    "viewer_frame_relation_score",
]
