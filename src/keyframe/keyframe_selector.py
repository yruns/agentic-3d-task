"""Query-driven keyframe selection over a prepared 3D scene.

``KeyframeSelector`` is the production entry point. Given a prepared
ConceptGraph scene directory and a natural-language query it:

1. loads 3D objects, camera trajectory, RGB frames and an object-view
   visibility index,
2. parses the query into ranked hypotheses (LLM, scene-category constrained),
3. executes the hypotheses geometrically to ground target / anchor objects,
4. greedily selects ``k`` keyframes that jointly cover those objects.

CLIP-based semantic fallback is optional: when ``torch`` / ``open_clip`` are
not installed, category matching falls back to string / multi-label matching.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import threading
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from keyframe.lightweight_conceptgraph import load_scene_objects
from keyframe.llm.client import LLMClient
from keyframe.models.hypotheses import (
    DIRECTIONAL_RELATIONS,
    ExecutionPolicy,
    GroundingQuery,
    HypothesisKind,
    HypothesisOutputV1,
    QueryHypothesis,
    QueryNode,
    ReferenceFrame,
    SpatialConstraint,
)
from keyframe.models.results import (
    ExecutionMode,
    ExecutionResult,
    FrameMapping,
    GroundingStatus,
    HypothesisAttempt,
    HypothesisExecution,
    KeyframeResult,
    KeyframeSelectionMetadata,
)
from keyframe.models.scene import SceneObject
from keyframe.parsing.parser import QueryParser
from keyframe.query_executor import QueryExecutor
from keyframe.spatial.checker import SpatialRelationChecker

if TYPE_CHECKING:
    from keyframe.bev import BEVMarker, SceneBEVBuilder, SceneBEVConfig

#: Whether CLIP (torch + open_clip) is importable for semantic fallback.
HAS_CLIP: bool = (
    importlib.util.find_spec("open_clip") is not None
    and importlib.util.find_spec("torch") is not None
)

# Visibility-scoring constants (image size used when detections lack explicit wh).
_VIS_IMG_WIDTH = 1200
_VIS_IMG_HEIGHT = 680
_VIS_MAX_DISTANCE = 5.0

# Class-name tokens that are not useful scene categories (verbs/adjectives/parts).
_NOISE_LABELS: frozenset[str] = frozenset(
    {
        "sit",
        "lay",
        "hang",
        "open",
        "fill",
        "push",
        "take",
        "walk",
        "attach",
        "hide",
        "slide",
        "curl",
        "sleep",
        "stand",
        "make",
        "wrap",
        "sew",
        "lead",
        "roll",
        "draw",
        "writing",
        "mark",
        "couple",
        "selfie",
        "peak",
        "shine",
        "comfort",
        "mess",
        "stuff",
        "dormitory",
        "camouflage",
        "flat",
        "twin",
        "back",
        "half",
        "man",
        "woman",
        "person",
        "head",
        "nose",
        "hand",
        "foot",
        "face",
        "item",
        "object",
    }
)

_DEFAULT_INTRINSICS: NDArray[np.float64] = np.array(
    [[600.0, 0.0, 599.5], [0.0, 600.0, 339.5], [0.0, 0.0, 1.0]], dtype=np.float64
)

# Joint-coverage selection caps: cover at most this many highest-scoring target
# and anchor objects, and aim for at least this many keyframes when k allows.
_MAX_COVERAGE_TARGETS = 5
_MAX_COVERAGE_ANCHORS = 3
_MIN_KEYFRAMES = 3


class KeyframeSelector:
    """Selects query-relevant RGB keyframes from a prepared 3D scene."""

    #: Trajectory file locations probed relative to ``scene_path`` (first wins).
    _TRAJ_SEARCH_RELATIVE: tuple[str, ...] = (
        "traj.txt",
        "../raw/traj.txt",
        "../conceptgraph/traj.txt",
        "../traj.txt",
    )

    def __init__(
        self,
        scene_path: Path,
        *,
        pcd_file: Path | None = None,
        stride: int = 5,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        prefer_lightweight_pcd: bool = True,
        ensure_lightweight_pcd: bool = False,
        dataset: str = "nr3d",
        bev_config: SceneBEVConfig | None = None,
    ) -> None:
        self.scene_path = Path(scene_path)
        self.stride = stride
        self.model = model
        self.prefer_lightweight_pcd = prefer_lightweight_pcd
        self.ensure_lightweight_pcd = ensure_lightweight_pcd
        self.dataset = dataset
        self._bev_config = bev_config
        self._bev_builder: SceneBEVBuilder | None = None

        self.objects: list[SceneObject] = []
        self.object_features: NDArray[np.float32] | None = None
        self.camera_poses: list[NDArray[np.float64]] = []
        self.camera_trajectory_path: Path | None = None
        self.image_paths: list[Path] = []
        self.depth_paths: list[Path] = []
        self.intrinsics: NDArray[np.float64] = _DEFAULT_INTRINSICS.copy()
        self.scene_categories: list[str] = []

        self.object_to_views: dict[int, list[tuple[int, float]]] = {}
        self.view_to_objects: dict[int, list[tuple[int, float]]] = {}
        self.dwell_score: NDArray[np.float64] = np.array([], dtype=np.float64)

        self._llm_client = llm_client
        self._query_parser: QueryParser | None = None
        self._query_executor: QueryExecutor | None = None
        self._relation_checker: SpatialRelationChecker | None = None

        self._k_intrinsic: NDArray[np.float64] | None = None
        self._img_wh: tuple[int, int] | None = None
        self._depth_cache: dict[int, NDArray[np.float64]] = {}

        self._clip_model: object | None = None
        self._clip_tokenizer: object | None = None
        self._clip_lock = threading.Lock()

        self._load_scene(pcd_file)
        self._load_or_build_visibility_index()

    @classmethod
    def from_scene_path(
        cls,
        scene_path: str | Path,
        *,
        stride: int = 5,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        prefer_lightweight_pcd: bool = True,
        ensure_lightweight_pcd: bool = False,
        dataset: str = "nr3d",
        bev_config: SceneBEVConfig | None = None,
    ) -> KeyframeSelector:
        """Create a selector for a scene directory, auto-detecting the PCD file."""
        scene_path = Path(scene_path)
        pcd_dir = scene_path / "pcd_saves"
        pcd_file: Path | None = None
        for pattern in ("*ram*_post.pkl.gz", "*_post.pkl.gz", "*.pkl.gz"):
            matches = sorted(pcd_dir.glob(pattern))
            if matches:
                pcd_file = matches[0]
                break
        logger.info(f"[KeyframeSelector] PCD file: {pcd_file}")
        return cls(
            scene_path,
            pcd_file=pcd_file,
            stride=stride,
            llm_client=llm_client,
            model=model,
            prefer_lightweight_pcd=prefer_lightweight_pcd,
            ensure_lightweight_pcd=ensure_lightweight_pcd,
            dataset=dataset,
            bev_config=bev_config,
        )

    # ----- scene loading ----------------------------------------------------

    def _load_scene(self, pcd_file: Path | None) -> None:
        logger.info(f"[KeyframeSelector] loading scene: {self.scene_path}")
        if pcd_file is not None and pcd_file.exists():
            self.objects = load_scene_objects(
                pcd_file,
                prefer_lightweight=self.prefer_lightweight_pcd,
                ensure_lightweight=self.ensure_lightweight_pcd,
            )
            self._reassign_object_ids()
            self._build_object_features()

        enrichment_file = self.scene_path / "enriched_objects.json"
        if not enrichment_file.exists():
            raise FileNotFoundError(
                f"Enrichment file not found: {enrichment_file}. "
                "Run object enrichment to produce enriched_objects.json first."
            )
        self._load_enrichment(enrichment_file)

        self._load_camera_poses()
        if len(self.camera_poses) >= 2:
            self._compute_trajectory_stats()
        else:
            self.dwell_score = np.zeros(len(self.camera_poses), dtype=np.float64)

        self._set_image_paths()
        self.scene_categories = self._build_scene_categories()
        logger.success(
            f"[KeyframeSelector] {len(self.objects)} objects, "
            f"{len(self.camera_poses)} poses, {len(self.scene_categories)} categories"
        )

    def _reassign_object_ids(self) -> None:
        """Ensure obj_id equals list index (contiguous, used as feature row id)."""
        for index, obj in enumerate(self.objects):
            obj.obj_id = index

    def _build_object_features(self) -> None:
        features = [obj.clip_ft for obj in self.objects]
        valid = [f for f in features if f is not None]
        if not valid:
            self.object_features = None
            return
        dim = int(valid[0].shape[0])
        aligned = np.zeros((len(features), dim), dtype=np.float32)
        for index, feature in enumerate(features):
            if feature is not None and feature.shape[0] == dim:
                aligned[index] = feature
        norms = np.linalg.norm(aligned, axis=1, keepdims=True)
        non_zero = norms.squeeze(-1) > 0
        aligned[non_zero] = aligned[non_zero] / (norms[non_zero] + 1e-8)
        self.object_features = aligned

    def _load_enrichment(self, enrichment_file: Path) -> None:
        with enrichment_file.open() as handle:
            data = json.load(handle)
        entries = data.get("objects", []) if isinstance(data, dict) else []
        enrichment_by_id: dict[int, dict[str, object]] = {
            int(entry["obj_id"]): entry["enrichment"]
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("status") == "success"
            and isinstance(entry.get("enrichment"), dict)
            and "obj_id" in entry
        }
        count = 0
        for obj in self.objects:
            enrichment = enrichment_by_id.get(obj.obj_id)
            if enrichment is None:
                continue
            obj.category = str(enrichment.get("category", obj.category))
            obj.object_tag = obj.category
            obj.summary = str(enrichment.get("description", obj.summary))
            obj.affordance_category = str(
                enrichment.get("usability", obj.affordance_category)
            )
            nearby = enrichment.get("nearby_objects", obj.co_objects)
            if isinstance(nearby, list):
                obj.co_objects = [str(item) for item in nearby]
            count += 1
        logger.info(f"[KeyframeSelector] enriched {count}/{len(self.objects)} objects")

    def _build_scene_categories(self) -> list[str]:
        categories: set[str] = set()
        for obj in self.objects:
            primary = obj.object_tag or obj.category
            if primary and primary.lower() not in _NOISE_LABELS:
                categories.add(primary)
            for cls, count in Counter(obj.class_name).items():
                if count >= 2 and cls and cls.lower() not in _NOISE_LABELS:
                    categories.add(cls)
        return sorted(categories)

    # ----- camera trajectory ------------------------------------------------

    def _candidate_pack_roots(self) -> list[Path]:
        roots: list[Path] = []
        if self.scene_path.exists():
            roots.append(self.scene_path)
            parent = self.scene_path.parent
            if parent.exists() and parent != self.scene_path:
                roots.append(parent)
        return roots

    def _trajectory_search_paths(self) -> list[str]:
        paths = [
            str((self.scene_path / rel).resolve()) for rel in self._TRAJ_SEARCH_RELATIVE
        ]
        for root in self._candidate_pack_roots():
            paths.append(str(root / "pack_*/camera_trajectory.json"))
        return paths

    def _enumerate_trajectory_files(self) -> list[Path]:
        found: list[Path] = []
        seen: set[Path] = set()

        def add(candidate: Path) -> None:
            if not candidate.exists():
                return
            canonical = candidate.resolve()
            if canonical not in seen:
                seen.add(canonical)
                found.append(canonical)

        for root in self._candidate_pack_roots():
            for pack_dir in sorted(root.glob("pack_*")):
                add(pack_dir / "camera_trajectory.json")
        for rel in self._TRAJ_SEARCH_RELATIVE:
            add((self.scene_path / rel).resolve())
        return found

    def _load_camera_poses(self) -> None:
        candidates = self._enumerate_trajectory_files()
        if not candidates:
            logger.warning(
                f"[KeyframeSelector] no camera trajectory found in "
                f"{self._trajectory_search_paths()}"
            )
            return
        for traj_path in candidates:
            try:
                if traj_path.suffix.lower() == ".json":
                    all_poses = self._read_trajectory_json(traj_path)
                else:
                    all_poses = self._read_trajectory_txt(traj_path)
            except (OSError, ValueError) as error:
                logger.error(f"[KeyframeSelector] failed to parse {traj_path}: {error}")
                continue
            if not all_poses:
                continue
            self.camera_trajectory_path = traj_path
            self.camera_poses = [
                all_poses[i] for i in range(0, len(all_poses), self.stride)
            ]
            logger.info(
                f"[KeyframeSelector] loaded {len(self.camera_poses)} poses from {traj_path}"
            )
            return

    @staticmethod
    def _read_trajectory_txt(traj_path: Path) -> list[NDArray[np.float64]]:
        with traj_path.open() as handle:
            lines = handle.readlines()
        poses: list[NDArray[np.float64]] = []
        if lines and len(lines[0].split()) == 16:
            for line in lines:
                values = [float(x) for x in line.split()]
                if len(values) == 16:
                    poses.append(np.array(values, dtype=np.float64).reshape(4, 4))
        else:
            for i in range(0, len(lines) - 3, 4):
                try:
                    rows = [[float(x) for x in lines[i + r].split()] for r in range(4)]
                except (ValueError, IndexError):
                    continue
                pose = np.array(rows, dtype=np.float64)
                if pose.shape == (4, 4):
                    poses.append(pose)
        return poses

    @staticmethod
    def _read_trajectory_json(traj_path: Path) -> list[NDArray[np.float64]]:
        with traj_path.open() as handle:
            payload = json.load(handle)
        entries: list[object]
        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            named = (
                payload.get("poses")
                or payload.get("camera_poses")
                or payload.get("frames")
            )
            if named is not None:
                entries = list(named)
            elif KeyframeSelector._is_position_only(payload):
                logger.warning(
                    f"[KeyframeSelector] {traj_path} is position-only; needs full 4x4 poses"
                )
                return []
            else:
                raise ValueError(f"unrecognized trajectory payload in {traj_path}")
        else:
            raise ValueError(f"unexpected trajectory payload type in {traj_path}")

        poses: list[NDArray[np.float64]] = []
        for entry in entries:
            matrix: object = entry
            if isinstance(entry, dict):
                matrix = (
                    entry.get("world_T_cam")
                    or entry.get("pose")
                    or entry.get("matrix")
                    or entry.get("world_to_cam")
                )
            if matrix is None:
                continue
            try:
                array = np.asarray(matrix, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if array.shape == (16,):
                array = array.reshape(4, 4)
            if array.shape == (4, 4):
                poses.append(array)
        return poses

    @staticmethod
    def _is_position_only(payload: dict[str, object]) -> bool:
        if not payload:
            return False
        for key, value in payload.items():
            if not isinstance(key, str) or not key.isdigit():
                return False
            if not isinstance(value, list) or len(value) != 3:
                return False
        return True

    def _compute_trajectory_stats(self) -> None:
        translations = np.asarray(
            [pose[:3, 3] for pose in self.camera_poses], dtype=np.float64
        )
        steps = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        velocities = np.empty(len(self.camera_poses), dtype=np.float64)
        velocities[:-1] = steps
        velocities[-1] = steps[-1] if steps.size else 0.0
        smoothed = self._median_smooth(velocities, radius=2)
        self.dwell_score = 1.0 / (1.0 + 5.0 * smoothed)

    @staticmethod
    def _median_smooth(
        values: NDArray[np.float64], radius: int = 2
    ) -> NDArray[np.float64]:
        smoothed = np.empty_like(values)
        for index in range(values.shape[0]):
            start = max(0, index - radius)
            end = min(values.shape[0], index + radius + 1)
            smoothed[index] = float(np.median(values[start:end]))
        return smoothed

    # ----- raw dir / intrinsics / depth (pose-aware path) ------------------

    def _resolve_raw_dir(self) -> Path:
        for candidate in (self.scene_path / "raw", self.scene_path.parent / "raw"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"raw dir not found under {self.scene_path}")

    def _get_intrinsic(self) -> tuple[NDArray[np.float64], tuple[int, int]]:
        if self._k_intrinsic is None or self._img_wh is None:
            from keyframe.spatial.frustum import load_scene_intrinsic

            self._k_intrinsic, self._img_wh = load_scene_intrinsic(
                self._resolve_raw_dir()
            )
        return self._k_intrinsic, self._img_wh

    def _load_depth_for_view(self, view_id: int) -> NDArray[np.float64]:
        if view_id in self._depth_cache:
            return self._depth_cache[view_id]
        if view_id < 0 or view_id >= len(self.camera_poses):
            raise ValueError(f"view_id {view_id} out of range")
        from PIL import Image

        frame_id = self.map_view_to_frame(view_id)
        depth_path = self._resolve_raw_dir() / f"{frame_id:06d}-depth.png"
        if not depth_path.exists():
            raise FileNotFoundError(depth_path)
        depth = np.asarray(Image.open(depth_path), dtype=np.float64) / 1000.0
        if depth.ndim != 2 or not np.isfinite(depth).all():
            raise ValueError(f"invalid depth image: {depth_path}")
        self._depth_cache[view_id] = depth
        return depth

    def _set_image_paths(self) -> None:
        results_dir = self.scene_path / "results"
        raw_dir = self.scene_path.parent / "raw"

        images = sorted(results_dir.glob("frame*.jpg")) or sorted(
            results_dir.glob("*.jpg")
        )
        if not images and raw_dir.is_dir():
            images = sorted(raw_dir.glob("*-rgb.jpg")) or sorted(
                raw_dir.glob("*-rgb.png")
            )
        self.image_paths = [images[i] for i in range(0, len(images), self.stride)]

        depths = sorted(results_dir.glob("depth*.png"))
        if not depths and raw_dir.is_dir():
            depths = sorted(raw_dir.glob("*-depth.png"))
        self.depth_paths = [depths[i] for i in range(0, len(depths), self.stride)]

    # ----- visibility index -------------------------------------------------

    def _load_or_build_visibility_index(self) -> None:
        index_path = self.scene_path / "indices" / "visibility_index.pkl"
        if index_path.exists():
            self._load_visibility_index(index_path)
            return
        logger.warning(
            f"[KeyframeSelector] visibility index not found at {index_path}; building online"
        )
        self._build_visibility_index_online()

    def _load_visibility_index(self, index_path: Path) -> None:
        with index_path.open("rb") as handle:
            data = pickle.load(handle)
        if "object_to_views" in data:
            self.object_to_views = {
                int(k): list(v) for k, v in data["object_to_views"].items()
            }
            self.view_to_objects = {
                int(k): list(v) for k, v in data["view_to_objects"].items()
            }
        else:
            raw_index = data.get("visibility_index", {})
            self.object_to_views = {int(k): list(v) for k, v in raw_index.items()}
            self.view_to_objects = {}
            for obj_id, views in self.object_to_views.items():
                for view_id, score in views:
                    self.view_to_objects.setdefault(view_id, []).append((obj_id, score))
            for views in self.view_to_objects.values():
                views.sort(key=lambda item: item[1], reverse=True)
        logger.success(
            f"[KeyframeSelector] loaded visibility index "
            f"({len(self.object_to_views)} objects, {len(self.view_to_objects)} views)"
        )

    def _build_visibility_index_online(self) -> None:
        self.view_to_objects = {}
        for obj in self.objects:
            scores = (
                self._compute_visibility_scores(obj)
                if obj.image_idx
                else self._compute_geometric_scores(obj)
            )
            scores.sort(key=lambda item: item[1], reverse=True)
            self.object_to_views[obj.obj_id] = scores
            for view_id, score in scores:
                self.view_to_objects.setdefault(view_id, []).append((obj.obj_id, score))
        for views in self.view_to_objects.values():
            views.sort(key=lambda item: item[1], reverse=True)
        logger.success(
            f"[KeyframeSelector] built visibility index "
            f"({len(self.object_to_views)} objects, {len(self.view_to_objects)} views)"
        )

    def _compute_visibility_scores(self, obj: SceneObject) -> list[tuple[int, float]]:
        img_area = _VIS_IMG_WIDTH * _VIS_IMG_HEIGHT
        view_to_indices: dict[int, list[int]] = {}
        for detection_index, view_id in enumerate(obj.image_idx):
            view_to_indices.setdefault(view_id, []).append(detection_index)

        scores: list[tuple[int, float]] = []
        centroid = obj.centroid if obj.centroid is not None else np.zeros(3)
        for view_id, indices in view_to_indices.items():
            if view_id >= len(self.camera_poses):
                continue
            completeness = 0.0
            for idx in indices:
                if idx < len(obj.xyxy):
                    box = obj.xyxy[idx]
                    if box.shape[0] == 4:
                        x1, y1, x2, y2 = (float(v) for v in box)
                        size_score = min(1.0, (x2 - x1) * (y2 - y1) / (img_area * 0.3))
                        clipped = (
                            x1 < 10
                            or y1 < 10
                            or x2 > _VIS_IMG_WIDTH - 10
                            or y2 > _VIS_IMG_HEIGHT - 10
                        )
                        completeness = max(
                            completeness,
                            max(0.0, size_score - (0.3 if clipped else 0.0)),
                        )
            geo = 0.0
            if not np.allclose(centroid, 0.0):
                pose = self.camera_poses[view_id]
                cam_pos = pose[:3, 3]
                distance = float(np.linalg.norm(centroid - cam_pos))
                if distance <= _VIS_MAX_DISTANCE:
                    direction = (centroid - cam_pos) / (distance + 1e-8)
                    angle = max(0.0, float(np.dot(direction, -pose[:3, 2])))
                    geo = (
                        0.6 * max(0.0, 1.0 - distance / _VIS_MAX_DISTANCE) + 0.4 * angle
                    )
            quality = min(1.0, len(indices) / 3.0)
            scores.append((view_id, 0.5 * completeness + 0.3 * geo + 0.2 * quality))
        return scores

    def _compute_geometric_scores(self, obj: SceneObject) -> list[tuple[int, float]]:
        if obj.centroid is None:
            return []
        centroid = obj.centroid
        scores: list[tuple[int, float]] = []
        for view_id, pose in enumerate(self.camera_poses):
            cam_pos = pose[:3, 3]
            distance = float(np.linalg.norm(centroid - cam_pos))
            if distance > _VIS_MAX_DISTANCE:
                continue
            direction = (centroid - cam_pos) / (distance + 1e-8)
            angle = max(0.0, float(np.dot(direction, -pose[:3, 2])))
            combined = 0.6 * max(0.0, 1.0 - distance / _VIS_MAX_DISTANCE) + 0.4 * angle
            if combined > 0.1:
                scores.append((view_id, combined))
        return scores

    # ----- joint-coverage keyframe selection -------------------------------

    def get_joint_coverage_views(
        self,
        object_ids: list[int],
        max_views: int = 3,
        *,
        pose_aware: bool = False,
        frustum_overlap_threshold: float = 0.7,
        redundancy_penalty: float = 0.2,
        dwell_weight: float = 0.15,
        frustum_method: str = "l1",
    ) -> list[int]:
        """Greedily select views maximizing joint coverage of ``object_ids``."""
        if not object_ids:
            return []
        view_scores = self._view_scores_for_objects(object_ids)
        candidate_views = set(view_scores)
        if not candidate_views:
            return []
        if pose_aware:
            self._validate_frustum_method(frustum_method)

        selected: list[int] = []
        covered = dict.fromkeys(object_ids, 0.0)
        for _ in range(max_views):
            best_view: int | None = None
            best_gain = 0.0
            for view_id in candidate_views - set(selected):
                base_gain = sum(
                    max(0.0, view_scores[view_id].get(obj_id, 0.0) - covered[obj_id])
                    for obj_id in object_ids
                )
                gain = (
                    self._pose_aware_gain(
                        base_gain,
                        view_id,
                        selected,
                        frustum_overlap_threshold,
                        redundancy_penalty,
                        dwell_weight,
                        frustum_method,
                    )
                    if pose_aware
                    else base_gain
                )
                if gain > best_gain:
                    best_gain, best_view = gain, view_id
            if best_view is None:
                break
            selected.append(best_view)
            for obj_id in object_ids:
                covered[obj_id] = max(
                    covered[obj_id], view_scores[best_view].get(obj_id, 0.0)
                )
        return selected

    def _view_scores_for_objects(
        self, object_ids: list[int]
    ) -> dict[int, dict[int, float]]:
        view_scores: dict[int, dict[int, float]] = {}
        for obj_id in object_ids:
            for view_id, score in self.object_to_views.get(obj_id, []):
                view_scores.setdefault(view_id, {})[obj_id] = score
        return view_scores

    @staticmethod
    def _validate_frustum_method(frustum_method: str) -> None:
        if frustum_method not in {"l1", "l2"}:
            raise ValueError(
                f"frustum_method must be 'l1' or 'l2', got {frustum_method!r}"
            )

    def _pose_aware_gain(
        self,
        base_gain: float,
        view_id: int,
        selected: list[int],
        frustum_overlap_threshold: float,
        redundancy_penalty: float,
        dwell_weight: float,
        frustum_method: str,
    ) -> float:
        if base_gain <= 0.0:
            return 0.0
        gain = base_gain
        if selected:
            overlaps = [
                self._view_frustum_overlap(view_id, other, frustum_method)
                for other in selected
            ]
            if any(overlap > frustum_overlap_threshold for overlap in overlaps):
                gain *= redundancy_penalty
        dwell = (
            float(self.dwell_score[view_id])
            if 0 <= view_id < len(self.dwell_score)
            else 0.0
        )
        return gain * (1.0 + dwell_weight * dwell)

    def _view_frustum_overlap(
        self, anchor_view: int, neighbor_view: int, frustum_method: str
    ) -> float:
        self._validate_frustum_method(frustum_method)
        for view in (anchor_view, neighbor_view):
            if view < 0 or view >= len(self.camera_poses):
                raise ValueError(f"view {view} out of range")
        k, img_wh = self._get_intrinsic()
        pose_anchor = self.camera_poses[anchor_view]
        pose_neighbor = self.camera_poses[neighbor_view]
        if frustum_method == "l1":
            from keyframe.spatial.frustum import frustum_overlap_l1

            return frustum_overlap_l1(pose_anchor, pose_neighbor, k, img_wh)
        from keyframe.spatial.frustum import frustum_overlap_l2

        return frustum_overlap_l2(
            self._load_depth_for_view(anchor_view),
            pose_anchor,
            pose_neighbor,
            k,
            img_wh,
        )

    def pad_keyframes_to_minimum(
        self, selected: list[int], object_ids: list[int], min_count: int = 3
    ) -> list[int]:
        """Pad with highest-total-visibility views until ``min_count`` reached."""
        if len(selected) >= min_count:
            return selected
        selected = list(selected)
        selected_set = set(selected)

        candidate_views: set[int] = set()
        for obj_id in object_ids:
            for view_id, _ in self.object_to_views.get(obj_id, []):
                candidate_views.add(view_id)
        if not candidate_views:
            candidate_views = set(self.view_to_objects)

        totals = [
            (view_id, sum(score for _, score in self.view_to_objects.get(view_id, [])))
            for view_id in candidate_views - selected_set
        ]
        totals.sort(key=lambda item: item[1], reverse=True)
        for view_id, _ in totals:
            if len(selected) >= min_count:
                break
            selected.append(view_id)
        return selected

    # ----- CLIP fallback (optional) ----------------------------------------

    def _load_clip_model(self) -> None:
        if self._clip_model is not None or not HAS_CLIP:
            return
        with self._clip_lock:
            if self._clip_model is not None:
                return
            import open_clip
            import torch

            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-H-14", "laion2b_s32b_b79k"
            )
            model = model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            self._clip_model = model
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

    def _encode_text(self, text: str) -> NDArray[np.float32] | None:
        self._load_clip_model()
        model = self._clip_model
        tokenizer = self._clip_tokenizer
        if model is None or tokenizer is None:
            return None
        import torch

        tokens = tokenizer([text])  # type: ignore[operator]
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        with torch.no_grad():
            feature = model.encode_text(tokens)  # type: ignore[attr-defined]
            feature = feature / feature.norm(dim=-1, keepdim=True)
        return feature.cpu().numpy().flatten().astype(np.float32)  # type: ignore[no-any-return]

    # ----- BEV visual context (optional) -----------------------------------

    def _scene_dir(self) -> Path:
        """Resolve the scene directory (the parent of a ``conceptgraph/`` pack)."""
        if self.scene_path.name == "conceptgraph":
            return self.scene_path.parent
        return self.scene_path

    def _get_bev_builder(self) -> SceneBEVBuilder:
        if self._bev_builder is None:
            from keyframe.bev import Nr3dSceneBEVBuilder, SceneBEVConfig

            if self.dataset != "nr3d":
                raise ValueError(
                    f"No BEV builder for dataset {self.dataset!r}; "
                    "only 'nr3d' is supported"
                )
            self._bev_builder = Nr3dSceneBEVBuilder(
                self._bev_config or SceneBEVConfig()
            )
        return self._bev_builder

    def _bev_markers(self) -> list[BEVMarker]:
        from keyframe.bev import BEVMarker

        markers: list[BEVMarker] = []
        for obj in self.objects:
            if obj.centroid is None:
                continue
            category = obj.object_tag or obj.category or "object"
            markers.append(
                BEVMarker(
                    obj_id=obj.obj_id,
                    category=category,
                    position=(
                        float(obj.centroid[0]),
                        float(obj.centroid[1]),
                        float(obj.centroid[2]),
                    ),
                )
            )
        return markers

    def generate_scene_bev(self, *, use_cache: bool = True) -> Path:
        """Render (or load) the top-down scene BEV used as parser visual context.

        Requires the ``vision`` extra (opencv + pillow + plyfile) and the
        ScanNet mesh / trajectory / intrinsic assets for the scene.
        """
        builder = self._get_bev_builder()
        scene_dir = self._scene_dir()
        output_path = scene_dir / "bev_cache" / "keyframe_bev.png"
        return builder.build(
            scene_id=scene_dir.name,
            data_root=scene_dir.parent,
            markers=self._bev_markers(),
            output_path=output_path,
            use_cache=use_cache,
        )

    # ----- query parsing / execution ---------------------------------------

    def _get_query_parser(self) -> QueryParser:
        if self._query_parser is None:
            if self._llm_client is None:
                self._llm_client = LLMClient.from_toml()
            self._query_parser = QueryParser(
                self.scene_categories, self._llm_client, model=self.model
            )
        return self._query_parser

    def _get_query_executor(self) -> QueryExecutor:
        if self._query_executor is None:
            if self._relation_checker is None:
                self._relation_checker = SpatialRelationChecker()
            self._query_executor = QueryExecutor(
                self.objects,
                relation_checker=self._relation_checker,
                clip_features=self.object_features,
                clip_encoder=self._encode_text if HAS_CLIP else None,
                camera_poses=list(self.camera_poses),
            )
        return self._query_executor

    def parse_query_hypotheses(
        self,
        query: str,
        *,
        apply_viewpoint_normalize: bool = False,
        use_visual_context: bool = False,
    ) -> HypothesisOutputV1:
        """Parse a query into scene-category-sanitized ranked hypotheses.

        With ``use_visual_context=True`` a top-down scene BEV (mesh + camera
        trajectory + object labels) is rendered and passed to the parser as
        multimodal context. This requires the ``vision`` extra and the scene's
        ScanNet mesh assets.
        """
        parser = self._get_query_parser()
        scene_images: list[Path] = (
            [self.generate_scene_bev()] if use_visual_context else []
        )
        output = parser.parse(query, scene_images=scene_images)
        sanitized: list[QueryHypothesis] = []
        for hypothesis in output.hypotheses:
            grounding = self._sanitize_categories(hypothesis.grounding_query)
            if apply_viewpoint_normalize:
                grounding = self._normalize_viewpoint_policies(grounding)
            sanitized.append(
                QueryHypothesis(
                    kind=hypothesis.kind,
                    rank=hypothesis.rank,
                    grounding_query=grounding,
                    lexical_hints=hypothesis.lexical_hints,
                )
            )
        result = HypothesisOutputV1(parse_mode=output.parse_mode, hypotheses=sanitized)
        result.validate_categories(self.scene_categories)
        return result

    def execute_query(
        self,
        grounding_query: GroundingQuery,
        mode: ExecutionMode = ExecutionMode.STRICT,
    ) -> ExecutionResult:
        """Execute one grounding query against the scene."""
        return self._get_query_executor().execute(grounding_query, mode=mode)

    def execute_hypotheses(
        self,
        hypothesis_output: HypothesisOutputV1,
        *,
        hidden_categories: Sequence[str] = (),
        mode: ExecutionMode = ExecutionMode.STRICT,
    ) -> HypothesisExecution:
        """Execute hypotheses by rank, returning the first grounded one."""
        hypothesis_output.validate_categories(self.scene_categories)
        attempts: list[HypothesisAttempt] = []
        for hypothesis in hypothesis_output.ordered_hypotheses():
            grounding = hypothesis.grounding_query
            self._validate_categories_in_scene(grounding)
            self._validate_no_mask_leak(grounding, hidden_categories)
            if self._has_unknown_anchors(grounding):
                attempts.append(
                    HypothesisAttempt(
                        kind=hypothesis.kind,
                        rank=hypothesis.rank,
                        status="skipped_unknown_anchor",
                    )
                )
                continue
            result = self.execute_query(grounding, mode=mode)
            if result.is_empty:
                attempts.append(
                    HypothesisAttempt(
                        kind=hypothesis.kind, rank=hypothesis.rank, status="empty"
                    )
                )
                continue
            attempts.append(
                HypothesisAttempt(
                    kind=hypothesis.kind, rank=hypothesis.rank, status="grounded"
                )
            )
            return HypothesisExecution(
                status=_STATUS_BY_KIND.get(
                    hypothesis.kind, GroundingStatus.CONTEXT_ONLY
                ),
                result=result,
                hypothesis=hypothesis,
                attempts=attempts,
            )
        return HypothesisExecution(
            status=GroundingStatus.NO_EVIDENCE,
            result=ExecutionResult(node_id="none"),
            attempts=attempts,
        )

    def _rank_target_objects_by_score(
        self, result: ExecutionResult
    ) -> list[SceneObject]:
        scores = result.scores
        soft = result.soft_match_scores

        def sort_key(obj: SceneObject) -> tuple[float, float]:
            soft_total = sum(soft.get(obj.obj_id, {}).values())
            return (-scores.get(obj.obj_id, 0.0), -soft_total)

        return sorted(result.matched_objects, key=sort_key)

    def _force_legacy_world_hard(
        self, output: HypothesisOutputV1
    ) -> HypothesisOutputV1:
        new_hypotheses: list[QueryHypothesis] = []
        for hypothesis in output.hypotheses:
            grounding = hypothesis.grounding_query.model_copy(deep=True)
            for _node, constraint in grounding.iter_constraints():
                constraint.reference_frame = ReferenceFrame.WORLD
                constraint.viewpoint_context_id = None
                constraint.execution_policy = ExecutionPolicy.HARD
            grounding.viewpoint_contexts = []
            new_hypotheses.append(
                QueryHypothesis(
                    kind=hypothesis.kind,
                    rank=hypothesis.rank,
                    grounding_query=grounding,
                    lexical_hints=hypothesis.lexical_hints,
                )
            )
        return HypothesisOutputV1(
            parse_mode=output.parse_mode, hypotheses=new_hypotheses
        )

    def _guard_viewpoint_traj_available(self, output: HypothesisOutputV1) -> None:
        if self.camera_poses:
            return
        for hypothesis in output.hypotheses:
            for _node, constraint in hypothesis.grounding_query.iter_constraints():
                if constraint.reference_frame == ReferenceFrame.VIEWER:
                    raise RuntimeError(
                        "viewpoint_aware=True requires a camera trajectory, but none "
                        f"of {self._trajectory_search_paths()} yielded 4x4 poses."
                    )

    def _normalize_viewpoint_policies(
        self, grounding_query: GroundingQuery
    ) -> GroundingQuery:
        grounding = grounding_query.model_copy(deep=True)
        ungrounded: set[str] = set()
        for context in grounding.viewpoint_contexts:
            has_real = any(
                cat != "UNKNOW"
                for anchor in (
                    context.facing_anchor,
                    context.origin_anchor,
                    context.subject_anchor,
                )
                if anchor is not None
                for node in self._iter_nodes(anchor)
                for cat in node.categories
            )
            if not has_real:
                ungrounded.add(context.id)

        for _node, constraint in grounding.iter_constraints():
            if isinstance(constraint, SpatialConstraint):
                directional = (
                    constraint.relation.lower().replace(" ", "_")
                    in DIRECTIONAL_RELATIONS
                )
            else:
                directional = constraint.metric.lower() in {
                    "x_position",
                    "x",
                    "y_position",
                    "y",
                }

            if (
                directional
                and constraint.reference_frame == ReferenceFrame.WORLD
                and constraint.viewpoint_context_id is None
                and constraint.execution_policy == ExecutionPolicy.HARD
            ):
                constraint.reference_frame = ReferenceFrame.AMBIGUOUS
                constraint.execution_policy = ExecutionPolicy.RANK_ONLY
            elif (
                constraint.reference_frame
                in (ReferenceFrame.VIEWER, ReferenceFrame.OBJECT_LOCAL)
                and constraint.viewpoint_context_id in ungrounded
            ):
                constraint.reference_frame = ReferenceFrame.AMBIGUOUS
                constraint.viewpoint_context_id = None
                constraint.execution_policy = ExecutionPolicy.RANK_ONLY
        return grounding

    def _sanitize_categories(self, grounding_query: GroundingQuery) -> GroundingQuery:
        scene_set = set(self.scene_categories)
        grounding = grounding_query.model_copy(deep=True)
        nodes = list(self._iter_nodes(grounding.root))
        for context in grounding.viewpoint_contexts:
            for anchor in (
                context.facing_anchor,
                context.origin_anchor,
                context.subject_anchor,
            ):
                if anchor is not None:
                    nodes.extend(self._iter_nodes(anchor))
        for node in nodes:
            cleaned: list[str] = []
            for category in node.categories:
                if (
                    category in scene_set or category == "UNKNOW"
                ) and category not in cleaned:
                    cleaned.append(category)
            node.categories = cleaned or ["UNKNOW"]

        root = grounding.root
        if (
            root.categories == ["UNKNOW"]
            and root.spatial_constraints
            and any(
                "UNKNOW" not in anchor.categories
                for sc in root.spatial_constraints
                for anchor in sc.anchors
            )
        ):
            root.open_ended = True
        return grounding

    def _validate_categories_in_scene(self, grounding_query: GroundingQuery) -> None:
        scene_set = set(self.scene_categories)
        for category in grounding_query.get_all_categories():
            if category != "UNKNOW" and category not in scene_set:
                raise ValueError(
                    f"Category {category!r} is not in scene categories or UNKNOW"
                )

    def _validate_no_mask_leak(
        self, grounding_query: GroundingQuery, hidden_categories: Sequence[str]
    ) -> None:
        hidden = set(hidden_categories)
        if not hidden:
            return
        for category in grounding_query.get_all_categories():
            if category in hidden:
                raise ValueError(f"Masked category leak: {category!r}")

    def _has_unknown_anchors(self, grounding_query: GroundingQuery) -> bool:
        for node in self._iter_nodes(grounding_query.root):
            for constraint in node.spatial_constraints:
                if any("UNKNOW" in anchor.categories for anchor in constraint.anchors):
                    return True
            if (
                node.select_constraint is not None
                and node.select_constraint.reference is not None
            ):
                if "UNKNOW" in node.select_constraint.reference.categories:
                    return True
        return False

    @staticmethod
    def _iter_nodes(root: QueryNode) -> Iterator[QueryNode]:
        stack = [root]
        while stack:
            node = stack.pop()
            yield node
            for constraint in node.spatial_constraints:
                stack.extend(constraint.anchors)
            if (
                node.select_constraint is not None
                and node.select_constraint.reference is not None
            ):
                stack.append(node.select_constraint.reference)

    # ----- keyframe path resolution ----------------------------------------

    def map_view_to_frame(self, view_id: int) -> int:
        """Map a sampled view index back to the original frame index."""
        return view_id * self.stride

    def _resolve_keyframe_path(self, view_id: int) -> tuple[Path | None, int]:
        for candidate in (view_id, view_id - 1, view_id + 1, view_id - 2, view_id + 2):
            if candidate < 0:
                continue
            frame_id = self.map_view_to_frame(candidate)
            expected = self.scene_path / "results" / f"frame{frame_id:06d}.jpg"
            if expected.exists():
                return expected, candidate
            if (
                0 <= candidate < len(self.image_paths)
                and self.image_paths[candidate].exists()
            ):
                return self.image_paths[candidate], candidate
        return None, view_id

    # ----- main entry point -------------------------------------------------

    def select_keyframes_v2(
        self,
        query: str,
        k: int = 3,
        *,
        strategy: str = "joint_coverage",
        hidden_categories: Sequence[str] = (),
        pose_aware: bool = False,
        frustum_method: str = "l1",
        viewpoint_aware: bool = False,
        use_visual_context: bool = False,
    ) -> KeyframeResult:
        """Select ``k`` keyframes that ground and cover the query's objects."""
        logger.info(f"[KeyframeSelector] selecting {k} keyframes for {query!r}")

        hypothesis_output = self.parse_query_hypotheses(
            query,
            apply_viewpoint_normalize=viewpoint_aware,
            use_visual_context=use_visual_context,
        )
        if viewpoint_aware:
            self._guard_viewpoint_traj_available(hypothesis_output)
        else:
            hypothesis_output = self._force_legacy_world_hard(hypothesis_output)

        execution_mode = (
            ExecutionMode.RECALL if viewpoint_aware else ExecutionMode.STRICT
        )
        execution = self.execute_hypotheses(
            hypothesis_output, hidden_categories=hidden_categories, mode=execution_mode
        )
        first_hypothesis = hypothesis_output.ordered_hypotheses()[0]

        if execution.hypothesis is None or execution.result.is_empty:
            return KeyframeResult(
                query=query,
                target_term=first_hypothesis.grounding_query.root.category,
                anchor_term=None,
                metadata=KeyframeSelectionMetadata(
                    status=execution.status,
                    strategy=strategy,
                    execution_mode=execution_mode,
                    error="No matching objects",
                    attempts=execution.attempts,
                    hypothesis_output=hypothesis_output,
                ),
            )

        target_objects = self._rank_target_objects_by_score(execution.result)
        selected_query = execution.hypothesis.grounding_query
        anchor_objects = self._collect_anchor_objects(selected_query)

        target_ids = [obj.obj_id for obj in target_objects[:_MAX_COVERAGE_TARGETS]]
        object_ids = list(target_ids)
        object_ids.extend(obj.obj_id for obj in anchor_objects[:_MAX_COVERAGE_ANCHORS])

        keyframe_indices = self.get_joint_coverage_views(
            object_ids if strategy == "joint_coverage" else target_ids,
            max_views=k,
            pose_aware=pose_aware,
            frustum_method=frustum_method,
        )
        keyframe_indices = self.pad_keyframes_to_minimum(
            keyframe_indices, object_ids, min_count=min(k, _MIN_KEYFRAMES)
        )

        keyframe_paths, frame_mappings = self._resolve_keyframes(keyframe_indices)
        anchor_term = self._anchor_term(selected_query)

        return KeyframeResult(
            query=query,
            target_term=selected_query.root.category,
            anchor_term=anchor_term,
            keyframe_indices=keyframe_indices,
            keyframe_paths=keyframe_paths,
            target_objects=target_objects,
            anchor_objects=anchor_objects,
            metadata=KeyframeSelectionMetadata(
                status=execution.status,
                strategy=strategy,
                execution_mode=execution_mode,
                selected_hypothesis_kind=execution.hypothesis.kind,
                selected_hypothesis_rank=execution.hypothesis.rank,
                all_object_ids=object_ids,
                frame_mappings=frame_mappings,
                attempts=execution.attempts,
                hypothesis_output=hypothesis_output,
            ),
        )

    def _collect_anchor_objects(
        self, grounding_query: GroundingQuery
    ) -> list[SceneObject]:
        executor = self._get_query_executor()
        anchors: list[SceneObject] = []
        for constraint in grounding_query.root.spatial_constraints:
            for anchor_node in constraint.anchors:
                anchors.extend(executor.evaluate_node(anchor_node).matched_objects)
        return anchors

    @staticmethod
    def _anchor_term(grounding_query: GroundingQuery) -> str | None:
        constraints = grounding_query.root.spatial_constraints
        if constraints and constraints[0].anchors:
            return constraints[0].anchors[0].category
        return None

    def _resolve_keyframes(
        self, keyframe_indices: list[int]
    ) -> tuple[list[Path], list[FrameMapping]]:
        keyframe_paths: list[Path] = []
        frame_mappings: list[FrameMapping] = []
        for view_id in keyframe_indices:
            path, resolved_view = self._resolve_keyframe_path(view_id)
            if path is not None:
                keyframe_paths.append(path)
            frame_mappings.append(
                FrameMapping(
                    requested_view_id=view_id,
                    requested_frame_id=self.map_view_to_frame(view_id),
                    resolved_view_id=resolved_view,
                    resolved_frame_id=self.map_view_to_frame(resolved_view),
                    path=str(path) if path is not None else None,
                )
            )
        return keyframe_paths, frame_mappings


_STATUS_BY_KIND: dict[HypothesisKind, GroundingStatus] = {
    HypothesisKind.DIRECT: GroundingStatus.DIRECT_GROUNDED,
    HypothesisKind.PROXY: GroundingStatus.PROXY_GROUNDED,
    HypothesisKind.CONTEXT: GroundingStatus.CONTEXT_ONLY,
}


def select_keyframes(
    scene_path: str | Path,
    query: str,
    k: int = 3,
    *,
    stride: int = 5,
    llm_client: LLMClient | None = None,
    model: str | None = None,
    dataset: str = "nr3d",
    use_visual_context: bool = False,
) -> KeyframeResult:
    """Convenience wrapper: build a selector for a scene and select keyframes."""
    selector = KeyframeSelector.from_scene_path(
        scene_path,
        stride=stride,
        llm_client=llm_client,
        model=model,
        dataset=dataset,
    )
    return selector.select_keyframes_v2(
        query, k=k, use_visual_context=use_visual_context
    )
