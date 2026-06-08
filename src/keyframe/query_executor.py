"""Recursive executor for nested spatial queries.

Evaluates a :class:`~keyframe.models.hypotheses.GroundingQuery` against a set
of scene objects, bottom-up:

1. Find candidates by category (with semantic / multi-label expansion).
2. Filter by attributes (color).
3. Apply spatial constraints (quick pre-filter + full geometric check),
   honouring each constraint's execution policy (hard / soft / rank_only).
4. Apply the selection constraint (superlative / ordinal / comparative).

Viewer-frame constraints are resolved against the camera trajectory; if no
viewer pose can be resolved, the executor applies the declared policy with
neutral evidence rather than silently falling back to world coordinates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from keyframe.models.hypotheses import (
    DIRECTIONAL_RELATIONS,
    ConstraintType,
    ExecutionPolicy,
    GroundingQuery,
    QueryNode,
    ReferenceFrame,
    SelectConstraint,
    SpatialConstraint,
    ViewpointContext,
)
from keyframe.models.results import ExecutionMode, ExecutionResult
from keyframe.models.scene import SceneObject
from keyframe.spatial.checker import SpatialRelationChecker
from keyframe.spatial.quick_filters import AttributeFilter, QuickFilters
from keyframe.spatial.viewpoint import (
    ViewerPose,
    resolve_viewer_pose,
    viewer_frame_axis_value,
    viewer_frame_relation_score,
)

#: Multiplier that keeps a SOFT-policy candidate alive without zeroing it out.
SOFT_POLICY_SCORE_FLOOR: float = 1e-3

#: Category tokens that mean "no concrete category".
UNKNOWN_CATEGORY_SENTINELS: frozenset[str] = frozenset(
    {"", "unknow", "unknown", "none", "null", "n/a", "na"}
)

#: A CLIP text encoder maps a phrase to a normalized feature vector (or None).
ClipEncoder = Callable[[str], "NDArray[np.float32] | None"]

# Relations whose HARD failure is relaxed to rank-only under RECALL mode.
_RECALL_SOFTENED_ON_RELATIONS: frozenset[str] = frozenset(
    {"on", "on_top", "on_top_of", "upon", "atop", "resting_on"}
)


def _object_size(obj: SceneObject) -> float:
    """Approximate object size as its 3D bounding-box volume."""
    if obj.bbox_np is not None:
        points = np.asarray(obj.bbox_np, dtype=np.float64)
        if points.ndim == 2 and points.shape[1] >= 3:
            extent = points[:, :3].max(axis=0) - points[:, :3].min(axis=0)
            return float(np.prod(extent))
    if obj.pcd_np is not None and len(obj.pcd_np) > 0:
        points = np.asarray(obj.pcd_np, dtype=np.float64)
        extent = points.max(axis=0) - points.min(axis=0)
        return float(np.prod(extent))
    return 0.0


class QueryExecutor:
    """Executes nested grounding queries against scene objects."""

    def __init__(
        self,
        objects: list[SceneObject],
        relation_checker: SpatialRelationChecker | None = None,
        *,
        clip_features: NDArray[np.float32] | None = None,
        clip_encoder: ClipEncoder | None = None,
        use_quick_filters: bool = True,
        camera_poses: list[NDArray[np.float64]] | None = None,
    ) -> None:
        self.objects = objects
        self.relation_checker = relation_checker or SpatialRelationChecker()
        self.clip_features = clip_features
        self.clip_encoder = clip_encoder
        self.use_quick_filters = use_quick_filters
        self.camera_poses: list[NDArray[np.float64]] = list(camera_poses or [])

        self._quick_filters = QuickFilters() if use_quick_filters else None
        self._attribute_filter = AttributeFilter() if use_quick_filters else None

        self._viewpoint_contexts: dict[str, ViewpointContext] = {}
        self._viewer_pose_cache: dict[str, ViewerPose | None] = {}
        self._execution_mode = ExecutionMode.STRICT
        self._cache: dict[str, ExecutionResult] = {}

        self._category_index: dict[str, list[SceneObject]] = {}
        self._multilabel_index: dict[str, list[SceneObject]] = {}
        self._build_indices()

    # ----- indexing ---------------------------------------------------------

    def _build_indices(self) -> None:
        for obj in self.objects:
            self._category_index.setdefault(self._get_category(obj).lower(), []).append(
                obj
            )

        for obj in self.objects:
            if not obj.class_name:
                continue
            primary = self._get_category(obj).lower()
            for cls, count in Counter(obj.class_name).items():
                cls_lower = cls.lower() if cls else ""
                if not cls_lower or cls_lower == primary or count < 2:
                    continue
                self._multilabel_index.setdefault(cls_lower, []).append(obj)

    @staticmethod
    def _get_category(obj: SceneObject) -> str:
        return obj.object_tag or obj.category

    @staticmethod
    def _get_centroid(obj: SceneObject) -> NDArray[np.float64]:
        if obj.centroid is not None:
            return np.asarray(obj.centroid, dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    # ----- neutral-evidence policy handling --------------------------------

    @staticmethod
    def _neutral_select_for_policy(
        candidates: list[SceneObject],
        scores: dict[int, float],
        policy: ExecutionPolicy,
    ) -> tuple[list[SceneObject], dict[int, float]]:
        """Apply a SelectConstraint policy with no metric evidence."""
        if policy == ExecutionPolicy.HARD:
            return [], {}
        return list(candidates), {
            c.obj_id: scores.get(c.obj_id, 1.0) for c in candidates
        }

    @staticmethod
    def _neutral_evidence_for_policy(
        candidates: list[SceneObject],
        policy: ExecutionPolicy,
    ) -> tuple[list[SceneObject], dict[int, float], dict[int, float]]:
        """Apply a SpatialConstraint policy with no geometric evidence."""
        if policy == ExecutionPolicy.HARD:
            return [], {}, {}
        if policy == ExecutionPolicy.SOFT:
            return (
                list(candidates),
                {c.obj_id: SOFT_POLICY_SCORE_FLOOR for c in candidates},
                {c.obj_id: 0.0 for c in candidates},
            )
        return (
            list(candidates),
            {c.obj_id: 1.0 for c in candidates},
            {c.obj_id: 0.0 for c in candidates},
        )

    def _resolve_viewer_pose_for_constraint(
        self, constraint: SpatialConstraint | SelectConstraint
    ) -> ViewerPose | None:
        """Resolve (and cache) the viewer pose for a viewer-frame constraint."""
        ctx_id = constraint.viewpoint_context_id
        if not ctx_id:
            return None
        if ctx_id in self._viewer_pose_cache:
            return self._viewer_pose_cache[ctx_id]

        context = self._viewpoint_contexts.get(ctx_id)
        anchor_node = (
            context.facing_anchor or context.origin_anchor or context.subject_anchor
            if context is not None
            else None
        )
        if anchor_node is None:
            self._viewer_pose_cache[ctx_id] = None
            return None

        anchor_result = self._execute_node(anchor_node)
        if not anchor_result.matched_objects:
            self._viewer_pose_cache[ctx_id] = None
            return None

        all_centroids = (
            np.stack([self._get_centroid(o) for o in self.objects])
            if self.objects
            else None
        )
        pose = resolve_viewer_pose(
            anchor_result.matched_objects,
            all_object_centroids=all_centroids,
            camera_poses=self.camera_poses,
            allow_geometric_fallback=False,
        )
        self._viewer_pose_cache[ctx_id] = pose
        return pose

    # ----- top-level execution ---------------------------------------------

    def execute(
        self,
        query: GroundingQuery,
        mode: ExecutionMode | str = ExecutionMode.STRICT,
    ) -> ExecutionResult:
        """Execute a grounding query and return the matched objects."""
        self._execution_mode = (
            mode if isinstance(mode, ExecutionMode) else ExecutionMode(mode)
        )
        logger.info(
            f"[QueryExecutor] executing {query.raw_query!r} (mode={self._execution_mode.value})"
        )
        self._cache.clear()
        self._viewpoint_contexts = {ctx.id: ctx for ctx in query.viewpoint_contexts}
        self._viewer_pose_cache = {}

        result = self._execute_node(query.root)
        if query.expect_unique and len(result.matched_objects) > 1:
            best = result.best_object
            if best is not None:
                result = ExecutionResult(
                    node_id=result.node_id,
                    matched_objects=[best],
                    scores={best.obj_id: result.scores.get(best.obj_id, 1.0)},
                    soft_match_scores=result.soft_match_scores,
                )
        logger.info(f"[QueryExecutor] matched {len(result.matched_objects)} objects")
        return result

    def evaluate_node(self, node: QueryNode) -> ExecutionResult:
        """Evaluate a single query node using the current cache / viewpoint state."""
        return self._execute_node(node)

    def _execute_node(self, node: QueryNode) -> ExecutionResult:
        if node.node_id and node.node_id in self._cache:
            return self._cache[node.node_id]

        if node.open_ended and node.spatial_constraints:
            candidates = list(self.objects)
        else:
            candidates = self._find_by_categories(node.categories)

        if not candidates:
            result = ExecutionResult(node_id=node.node_id, matched_objects=[])
            if node.node_id:
                self._cache[node.node_id] = result
            return result

        if node.attributes:
            candidates = self._filter_by_attributes(candidates, node.attributes)

        scores = {obj.obj_id: 1.0 for obj in candidates}
        soft_match_scores: dict[int, dict[str, float]] = {}

        for sc_index, constraint in enumerate(node.spatial_constraints):
            candidates, constraint_scores, constraint_soft = (
                self._apply_spatial_constraint(candidates, constraint)
            )
            for obj_id, score in constraint_scores.items():
                if obj_id in scores:
                    scores[obj_id] *= score
            if constraint_soft:
                key = f"sc{sc_index}:{constraint.relation}:{constraint.execution_policy.value}"
                for obj_id, soft in constraint_soft.items():
                    soft_match_scores.setdefault(obj_id, {})[key] = soft
            if not candidates:
                break

        if candidates and node.select_constraint is not None:
            candidates, scores = self._apply_select_constraint(
                candidates, scores, node.select_constraint
            )

        result = ExecutionResult(
            node_id=node.node_id,
            matched_objects=candidates,
            scores=scores,
            soft_match_scores=soft_match_scores,
        )
        if node.node_id:
            self._cache[node.node_id] = result
        return result

    # ----- category matching -----------------------------------------------

    def _find_by_categories(self, categories: list[str]) -> list[SceneObject]:
        search = [
            category
            for category in categories
            if category.strip().lower() not in UNKNOWN_CATEGORY_SENTINELS
        ]
        if not search:
            return []

        matches: list[SceneObject] = []
        seen: set[int] = set()

        def add(objs: Iterable[SceneObject]) -> None:
            for obj in objs:
                if obj.obj_id not in seen:
                    matches.append(obj)
                    seen.add(obj.obj_id)

        for category in search:
            add(self._category_index.get(category.lower(), []))
        if matches:
            return matches

        for category in search:
            lower = category.lower()
            for indexed_category, objs in self._category_index.items():
                if lower in indexed_category or indexed_category in lower:
                    add(objs)
        if matches:
            return matches

        for category in search:
            add(self._multilabel_index.get(category.lower(), []))
        if matches:
            return matches

        if self.clip_features is not None and self.clip_encoder is not None:
            return self._find_by_clip_similarity(search[0])

        logger.warning(f"[QueryExecutor] no match for categories {categories}")
        return []

    def _find_by_clip_similarity(
        self, category: str, top_k: int = 10, min_similarity: float = 0.2
    ) -> list[SceneObject]:
        if self.clip_encoder is None or self.clip_features is None:
            return []
        text_feature = self.clip_encoder(category)
        if text_feature is None:
            return []
        similarities = self.clip_features @ text_feature
        top_indices = np.argsort(-similarities)[:top_k]
        return [
            self.objects[i] for i in top_indices if similarities[i] > min_similarity
        ]

    def _filter_by_attributes(
        self, candidates: list[SceneObject], attributes: list[str]
    ) -> list[SceneObject]:
        if not attributes or self._attribute_filter is None:
            return candidates
        filtered = candidates
        for attribute in attributes:
            attr_lower = attribute.lower()
            if self._attribute_filter.knows_color(attr_lower):
                filtered = self._attribute_filter.filter_by_color(filtered, attr_lower)
        return filtered

    # ----- spatial constraints ---------------------------------------------

    def _apply_spatial_constraint(
        self,
        candidates: list[SceneObject],
        constraint: SpatialConstraint,
    ) -> tuple[list[SceneObject], dict[int, float], dict[int, float]]:
        """Apply one spatial constraint, returning (kept, scores, soft_scores)."""
        policy = constraint.execution_policy

        anchor_objects: list[SceneObject] = []
        for anchor_node in constraint.anchors:
            anchor_objects.extend(self._execute_node(anchor_node).matched_objects)

        if not anchor_objects:
            return self._neutral_evidence_for_policy(candidates, policy)

        pre_filtered = candidates
        if (
            policy == ExecutionPolicy.HARD
            and self._quick_filters is not None
            and self._quick_filters.has_filter(constraint.relation)
        ):
            quick_filtered = self._quick_filters.filter_candidates(
                candidates, anchor_objects, constraint.relation
            )
            if quick_filtered:
                pre_filtered = quick_filtered

        wants_viewer_frame = (
            constraint.reference_frame == ReferenceFrame.VIEWER
            and constraint.relation.lower().replace(" ", "_") in DIRECTIONAL_RELATIONS
        )
        viewer_pose: ViewerPose | None = None
        if wants_viewer_frame:
            viewer_pose = self._resolve_viewer_pose_for_constraint(constraint)
            if viewer_pose is None:
                return self._neutral_evidence_for_policy(pre_filtered, policy)

        satisfied: set[int] = set()
        satisfying_scores: dict[int, float] = {}
        for candidate in pre_filtered:
            best_satisfying = 0.0
            satisfies = False
            if wants_viewer_frame and viewer_pose is not None:
                candidate_centroid = self._get_centroid(candidate)
                for anchor in anchor_objects:
                    score = viewer_frame_relation_score(
                        constraint.relation,
                        candidate_centroid,
                        self._get_centroid(anchor),
                        viewer_pose,
                    )
                    if score > 0:
                        satisfies = True
                        best_satisfying = max(best_satisfying, score)
            elif constraint.relation.lower() == "between" and len(anchor_objects) >= 2:
                result = self.relation_checker.check(
                    candidate, anchor_objects, constraint.relation
                )
                if result.satisfies:
                    satisfies = True
                    best_satisfying = result.score
            else:
                for anchor in anchor_objects:
                    result = self.relation_checker.check(
                        candidate, anchor, constraint.relation
                    )
                    if result.satisfies:
                        satisfies = True
                        best_satisfying = max(best_satisfying, result.score)
            if satisfies:
                satisfied.add(candidate.obj_id)
                satisfying_scores[candidate.obj_id] = best_satisfying

        if policy == ExecutionPolicy.HARD:
            filtered = [c for c in pre_filtered if c.obj_id in satisfied]
            relation_norm = constraint.relation.lower().replace(" ", "_")
            emptied = bool(pre_filtered and not filtered)
            if (
                self._execution_mode == ExecutionMode.RECALL
                and emptied
                and relation_norm in _RECALL_SOFTENED_ON_RELATIONS
            ):
                keep_scores = {c.obj_id: 1.0 for c in pre_filtered}
                soft_scores = {
                    c.obj_id: satisfying_scores.get(c.obj_id, 0.0) for c in pre_filtered
                }
                return list(pre_filtered), keep_scores, soft_scores
            return filtered, dict(satisfying_scores), {}

        if policy == ExecutionPolicy.SOFT:
            out_scores = {
                c.obj_id: (
                    satisfying_scores.get(c.obj_id, 0.0) or SOFT_POLICY_SCORE_FLOOR
                )
                for c in pre_filtered
            }
            return list(pre_filtered), out_scores, dict(satisfying_scores)

        # RANK_ONLY
        keep_scores = {c.obj_id: 1.0 for c in pre_filtered}
        soft_scores = {
            c.obj_id: satisfying_scores.get(c.obj_id, 0.0) for c in pre_filtered
        }
        return list(pre_filtered), keep_scores, soft_scores

    # ----- select constraints ----------------------------------------------

    def _apply_select_constraint(
        self,
        candidates: list[SceneObject],
        scores: dict[int, float],
        constraint: SelectConstraint,
    ) -> tuple[list[SceneObject], dict[int, float]]:
        if not candidates:
            return [], {}
        if constraint.constraint_type == ConstraintType.SUPERLATIVE:
            return self._apply_superlative(candidates, scores, constraint)
        if constraint.constraint_type == ConstraintType.ORDINAL:
            return self._apply_ordinal(candidates, scores, constraint)
        return candidates, scores

    def _metric_value(
        self,
        candidate: SceneObject,
        metric: str,
        ref_objects: list[SceneObject],
        viewer_pose: ViewerPose | None,
        scores: dict[int, float],
    ) -> float:
        position = self._get_centroid(candidate)
        if metric == "distance" and ref_objects:
            return min(
                float(np.linalg.norm(position - self._get_centroid(ref)))
                for ref in ref_objects
            )
        if metric == "size":
            return _object_size(candidate)
        if metric == "height":
            return float(position[2])
        if metric in ("x_position", "x"):
            if viewer_pose is not None:
                return viewer_frame_axis_value(position, viewer_pose, "right")
            return float(position[0])
        if metric in ("y_position", "y"):
            if viewer_pose is not None:
                return viewer_frame_axis_value(position, viewer_pose, "forward")
            return float(position[1])
        return scores.get(candidate.obj_id, 0.0)

    def _viewer_pose_for_axis_metric(
        self, constraint: SelectConstraint, metric: str
    ) -> tuple[ViewerPose | None, bool]:
        """Return (pose, unresolved) for a viewer-frame axis metric."""
        wants_viewer = (
            constraint.reference_frame == ReferenceFrame.VIEWER
            and metric in ("x_position", "x", "y_position", "y")
        )
        if not wants_viewer:
            return None, False
        pose = self._resolve_viewer_pose_for_constraint(constraint)
        return pose, pose is None

    def _apply_superlative(
        self,
        candidates: list[SceneObject],
        scores: dict[int, float],
        constraint: SelectConstraint,
    ) -> tuple[list[SceneObject], dict[int, float]]:
        policy = constraint.execution_policy
        metric = constraint.metric.lower()
        order = constraint.order.lower()

        viewer_pose, unresolved = self._viewer_pose_for_axis_metric(constraint, metric)
        if unresolved:
            return self._neutral_select_for_policy(candidates, scores, policy)

        ref_objects = (
            self._execute_node(constraint.reference).matched_objects
            if constraint.reference is not None
            else []
        )
        ranked = sorted(
            candidates,
            key=lambda c: self._metric_value(
                c, metric, ref_objects, viewer_pose, scores
            ),
            reverse=(order != "min"),
        )

        if policy == ExecutionPolicy.HARD:
            best = ranked[0]
            return [best], {best.obj_id: 1.0}
        return self._rank_scores(ranked, scores, policy)

    def _apply_ordinal(
        self,
        candidates: list[SceneObject],
        scores: dict[int, float],
        constraint: SelectConstraint,
    ) -> tuple[list[SceneObject], dict[int, float]]:
        if constraint.position is None:
            return candidates, scores
        policy = constraint.execution_policy
        metric = constraint.metric.lower()
        order = constraint.order.lower()

        viewer_pose, unresolved = self._viewer_pose_for_axis_metric(constraint, metric)
        if unresolved:
            return self._neutral_select_for_policy(candidates, scores, policy)

        ranked = sorted(
            candidates,
            key=lambda c: self._metric_value(c, metric, [], viewer_pose, scores),
            reverse=(order == "desc"),
        )
        position = constraint.position
        if position <= 0 or position > len(ranked):
            if policy == ExecutionPolicy.HARD:
                return [], {}
            return list(candidates), dict(scores)

        if policy == ExecutionPolicy.HARD:
            selected = ranked[position - 1]
            return [selected], {selected.obj_id: 1.0}

        total = len(ranked)
        new_scores = dict(scores) if policy == ExecutionPolicy.SOFT else {}
        for rank, candidate in enumerate(ranked):
            distance = abs(rank - (position - 1))
            rank_score = max(SOFT_POLICY_SCORE_FLOOR, 1.0 - distance / max(1, total))
            if policy == ExecutionPolicy.SOFT:
                new_scores[candidate.obj_id] = (
                    new_scores.get(candidate.obj_id, 1.0) * rank_score
                )
            else:
                new_scores[candidate.obj_id] = rank_score
        return ranked, new_scores

    @staticmethod
    def _rank_scores(
        ranked: list[SceneObject],
        scores: dict[int, float],
        policy: ExecutionPolicy,
    ) -> tuple[list[SceneObject], dict[int, float]]:
        total = len(ranked)
        new_scores = dict(scores) if policy == ExecutionPolicy.SOFT else {}
        for rank, candidate in enumerate(ranked):
            rank_score = max(SOFT_POLICY_SCORE_FLOOR, 1.0 - rank / max(1, total))
            if policy == ExecutionPolicy.SOFT:
                new_scores[candidate.obj_id] = (
                    new_scores.get(candidate.obj_id, 1.0) * rank_score
                )
            else:
                new_scores[candidate.obj_id] = rank_score
        return ranked, new_scores


def execute_query(
    query: GroundingQuery,
    objects: list[SceneObject],
    relation_checker: SpatialRelationChecker | None = None,
) -> ExecutionResult:
    """Convenience wrapper: build an executor and run a query."""
    return QueryExecutor(objects, relation_checker).execute(query)
