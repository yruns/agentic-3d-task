"""3D scene grid specification and depth-verified grid-view visibility index.

Bridges top-down BEV grid coordinates ``(row, col)`` to 3D axis-aligned scene
regions and first-person camera views:

- :class:`SceneGridSpec` defines the deterministic mapping between 0-indexed
  BEV grid cells ``(row, col)`` (``row`` top-to-bottom along ``-Y``, ``col``
  left-to-right along ``+X``) and 3D world boxes ``[x_min, x_max] x [y_min, y_max]``.
- :class:`GridVisibilityIndex` maintains bidirectional indices between grid
  cells and views (``grid_to_views`` / ``view_to_grids``) as well as grid cells
  and 3D objects (``grid_to_objects`` / ``object_to_grids``), verifying region
  visibility against per-frame depth maps when available.
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from keyframe.models.scene import SceneObject

#: A 0-indexed ``(row, col)`` grid coordinate on the BEV plane.
GridCell = tuple[int, int]

#: Inclusive ``(min_row, max_row, min_col, max_col)`` range covered by a view.
GridBounds = tuple[int, int, int, int]

#: Optional callback returning a ``(H, W)`` depth map in meters for a ``view_id``.
DepthLoader = Callable[[int], "NDArray[np.float64] | None"]

GRID_INDEX_FORMAT_VERSION = "grid_visibility_index_v1"

_DEFAULT_TARGET_CELL_SIZE_M = 1.2
_DEFAULT_MIN_SIDE_CELLS = 4
_DEFAULT_MAX_SIDE_CELLS = 10
_DEFAULT_FALLBACK_ROWS = 6
_DEFAULT_FALLBACK_COLS = 6
_MIN_SPAN_M = 1.0
_BOUNDS_MARGIN_M = 0.25

_MIN_OBJECT_COVERAGE_RATIO = 0.20
_MIN_GRID_COVERAGE_RATIO = 0.25

_DEPTH_MIN_M = 0.1
_DEPTH_MAX_M = 5.0
_DEPTH_SUBSAMPLE = 8
_OCCLUSION_TOLERANCE_M = 0.35
_MIN_GRID_VISIBILITY_SCORE = 0.05


class SceneGridSpec(BaseModel):
    """Deterministic mapping between BEV ``(row, col)`` cells and 3D XY regions.

    Follows the BEV image convention shared by both :mod:`keyframe.bev.render`
    and :mod:`keyframe.bev.schematic`:

    - ``col`` increases left-to-right with world ``+X`` (``min_x -> max_x``).
    - ``row`` increases top-to-bottom with world ``-Y`` (``max_y -> min_y``).
    """

    model_config = ConfigDict(frozen=True)

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float = 0.0
    max_z: float = 2.5
    rows: int = Field(default=_DEFAULT_FALLBACK_ROWS, ge=1, le=64)
    cols: int = Field(default=_DEFAULT_FALLBACK_COLS, ge=1, le=64)

    @model_validator(mode="after")
    def _validate_bounds(self) -> SceneGridSpec:
        for name, value in (
            ("min_x", self.min_x),
            ("max_x", self.max_x),
            ("min_y", self.min_y),
            ("max_y", self.max_y),
            ("min_z", self.min_z),
            ("max_z", self.max_z),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.max_x <= self.min_x:
            raise ValueError(
                f"max_x ({self.max_x}) must be greater than min_x ({self.min_x})"
            )
        if self.max_y <= self.min_y:
            raise ValueError(
                f"max_y ({self.max_y}) must be greater than min_y ({self.min_y})"
            )
        if self.max_z <= self.min_z:
            raise ValueError(
                f"max_z ({self.max_z}) must be greater than min_z ({self.min_z})"
            )
        return self

    @property
    def shape(self) -> tuple[int, int]:
        """Grid dimensions ``(rows, cols)``."""
        return (self.rows, self.cols)

    @property
    def cell_width(self) -> float:
        """World-space X extent (meters) of one grid cell."""
        return (self.max_x - self.min_x) / self.cols

    @property
    def cell_height(self) -> float:
        """World-space Y extent (meters) of one grid cell."""
        return (self.max_y - self.min_y) / self.rows

    def is_valid_cell(self, cell: GridCell) -> bool:
        """Return whether ``(row, col)`` lies inside ``[0, rows) x [0, cols)``."""
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols

    def clamp_cell(self, row: int, col: int) -> GridCell:
        """Clamp ``(row, col)`` into valid grid indices."""
        return (
            max(0, min(self.rows - 1, int(row))),
            max(0, min(self.cols - 1, int(col))),
        )

    def sanitize_cells(
        self, cells: Sequence[GridCell], *, clamp: bool = False
    ) -> list[GridCell]:
        """Deduplicate and filter (or clamp) ``cells`` against this grid."""
        out: list[GridCell] = []
        seen: set[GridCell] = set()
        for raw_row, raw_col in cells:
            cell = (int(raw_row), int(raw_col))
            if clamp:
                cell = self.clamp_cell(cell[0], cell[1])
            elif not self.is_valid_cell(cell):
                continue
            if cell not in seen:
                seen.add(cell)
                out.append(cell)
        return out

    def world_to_cell(
        self, x: float, y: float, *, clamp: bool = False
    ) -> GridCell | None:
        """Map a world ``(x, y)`` coordinate to its ``(row, col)`` grid cell."""
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        if not clamp and (
            x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y
        ):
            return None
        raw_col = math.floor((x - self.min_x) / self.cell_width)
        raw_row = math.floor((self.max_y - y) / self.cell_height)
        if clamp or (self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y):
            return self.clamp_cell(raw_row, raw_col)
        return None

    def cell_to_world_bounds(
        self, row: int, col: int
    ) -> tuple[float, float, float, float]:
        """Return ``(x_min, y_min, x_max, y_max)`` in world meters for ``(row, col)``."""
        if not self.is_valid_cell((row, col)):
            raise ValueError(
                f"grid cell ({row}, {col}) out of bounds for shape {self.shape}"
            )
        cw = self.cell_width
        ch = self.cell_height
        x_min = self.min_x + col * cw
        x_max = x_min + cw
        y_max = self.max_y - row * ch
        y_min = y_max - ch
        return (x_min, y_min, x_max, y_max)

    def cell_center_world(
        self, row: int, col: int, *, z: float | None = None
    ) -> tuple[float, float, float]:
        """Return the 3D world center ``(cx, cy, cz)`` of ``(row, col)``."""
        x_min, y_min, x_max, y_max = self.cell_to_world_bounds(row, col)
        cz = (self.min_z + self.max_z) * 0.5 if z is None else float(z)
        return ((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, cz)

    def cells_for_footprint(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        *,
        centroid_xy: tuple[float, float] | None = None,
        min_obj_coverage: float = _MIN_OBJECT_COVERAGE_RATIO,
        min_grid_coverage: float = _MIN_GRID_COVERAGE_RATIO,
    ) -> list[GridCell]:
        """Return ``(row, col)`` cells occupied by an XY footprint.

        Applies the dual-coverage criterion so bounding-box edge slivers do not
        pollute adjacent cells: a candidate cell is kept when it contains the
        object's centroid, covers at least ``min_obj_coverage`` of the object's
        footprint (IoO), or is covered by at least ``min_grid_coverage`` of its
        own area (IoG). Returned cells are ordered by descending coverage with
        the centroid cell first.
        """
        if not all(math.isfinite(v) for v in (x_min, y_min, x_max, y_max)):
            return []
        xa, xb = min(x_min, x_max), max(x_min, x_max)
        ya, yb = min(y_min, y_max), max(y_min, y_max)
        if xb < self.min_x or xa > self.max_x or yb < self.min_y or ya > self.max_y:
            return []
        top_left = self.world_to_cell(
            max(xa, self.min_x), min(yb, self.max_y), clamp=True
        )
        bottom_right = self.world_to_cell(
            min(xb, self.max_x), max(ya, self.min_y), clamp=True
        )
        if top_left is None or bottom_right is None:
            return []

        cx, cy = (
            centroid_xy
            if centroid_xy is not None
            else (0.5 * (xa + xb), 0.5 * (ya + yb))
        )
        centroid_cell = self.world_to_cell(cx, cy, clamp=True)
        obj_area = max((xb - xa) * (yb - ya), 1e-6)
        cell_area = max(self.cell_width * self.cell_height, 1e-6)

        r0, c0 = top_left
        r1, c1 = bottom_right
        scored: list[tuple[bool, float, GridCell]] = []
        for r in range(min(r0, r1), max(r0, r1) + 1):
            for c in range(min(c0, c1), max(c0, c1) + 1):
                cell = (r, c)
                gx0, gy0, gx1, gy1 = self.cell_to_world_bounds(r, c)
                inter_w = max(0.0, min(xb, gx1) - max(xa, gx0))
                inter_h = max(0.0, min(yb, gy1) - max(ya, gy0))
                inter = inter_w * inter_h
                ioo = inter / obj_area
                iog = inter / cell_area
                is_centroid = cell == centroid_cell
                if is_centroid or ioo >= min_obj_coverage or iog >= min_grid_coverage:
                    scored.append((is_centroid, max(ioo, iog), cell))

        if not scored and centroid_cell is not None:
            return [centroid_cell]
        scored.sort(key=lambda item: (not item[0], -item[1], item[2]))
        return [cell for _, _, cell in scored]

    def to_payload(self) -> dict[str, object]:
        """Serialize grid parameters into a JSON-ready dict."""
        return {
            "min_x": round(self.min_x, 4),
            "max_x": round(self.max_x, 4),
            "min_y": round(self.min_y, 4),
            "max_y": round(self.max_y, 4),
            "min_z": round(self.min_z, 4),
            "max_z": round(self.max_z, 4),
            "rows": self.rows,
            "cols": self.cols,
        }

    @classmethod
    def from_scene_geometry(
        cls,
        objects: Sequence[SceneObject],
        camera_poses: Sequence[NDArray[np.float64]] = (),
        *,
        target_cell_size_m: float = _DEFAULT_TARGET_CELL_SIZE_M,
        min_side_cells: int = _DEFAULT_MIN_SIDE_CELLS,
        max_side_cells: int = _DEFAULT_MAX_SIDE_CELLS,
        fixed_shape: tuple[int, int] | None = None,
    ) -> SceneGridSpec:
        """Build a :class:`SceneGridSpec` enclosing scene objects and valid camera poses."""
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []

        for obj in objects:
            if obj.bbox_np is not None:
                arr = np.asarray(obj.bbox_np, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 3:
                    finite = arr[np.isfinite(arr[:, :3]).all(axis=1), :3]
                    if finite.size:
                        xs.extend(
                            (float(finite[:, 0].min()), float(finite[:, 0].max()))
                        )
                        ys.extend(
                            (float(finite[:, 1].min()), float(finite[:, 1].max()))
                        )
                        zs.extend(
                            (float(finite[:, 2].min()), float(finite[:, 2].max()))
                        )
                        continue
            if obj.centroid is not None:
                c = np.asarray(obj.centroid, dtype=np.float64).reshape(-1)
                if c.size >= 3 and np.isfinite(c[:3]).all():
                    xs.append(float(c[0]))
                    ys.append(float(c[1]))
                    zs.append(float(c[2]))

        for raw_pose in camera_poses:
            pose = np.asarray(raw_pose, dtype=np.float64)
            if pose.shape == (4, 4) and np.isfinite(pose).all():
                xs.append(float(pose[0, 3]))
                ys.append(float(pose[1, 3]))
                zs.append(float(pose[2, 3]))

        if not xs or not ys:
            return cls(
                min_x=-3.0,
                max_x=3.0,
                min_y=-3.0,
                max_y=3.0,
                min_z=0.0,
                max_z=2.5,
                rows=fixed_shape[0] if fixed_shape else _DEFAULT_FALLBACK_ROWS,
                cols=fixed_shape[1] if fixed_shape else _DEFAULT_FALLBACK_COLS,
            )

        min_x = min(xs) - _BOUNDS_MARGIN_M
        max_x = max(xs) + _BOUNDS_MARGIN_M
        min_y = min(ys) - _BOUNDS_MARGIN_M
        max_y = max(ys) + _BOUNDS_MARGIN_M
        if max_x - min_x < _MIN_SPAN_M:
            mid_x = 0.5 * (min_x + max_x)
            min_x, max_x = mid_x - 0.5 * _MIN_SPAN_M, mid_x + 0.5 * _MIN_SPAN_M
        if max_y - min_y < _MIN_SPAN_M:
            mid_y = 0.5 * (min_y + max_y)
            min_y, max_y = mid_y - 0.5 * _MIN_SPAN_M, mid_y + 0.5 * _MIN_SPAN_M

        min_z = min(zs) if zs else 0.0
        max_z = max(zs) if zs else 2.5
        if max_z - min_z < 0.5:
            max_z = min_z + 2.0

        if fixed_shape is not None:
            rows, cols = fixed_shape
        else:
            step = max(target_cell_size_m, 0.25)
            cols = int(
                np.clip(round((max_x - min_x) / step), min_side_cells, max_side_cells)
            )
            rows = int(
                np.clip(round((max_y - min_y) / step), min_side_cells, max_side_cells)
            )

        return cls(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_z=min_z,
            max_z=max_z,
            rows=rows,
            cols=cols,
        )


def compute_grid_bounds(cells: Sequence[GridCell]) -> GridBounds | None:
    """Return ``(min_row, max_row, min_col, max_col)`` for a non-empty cell sequence."""
    if not cells:
        return None
    rows = [int(r) for r, _ in cells]
    cols = [int(c) for _, c in cells]
    return (min(rows), max(rows), min(cols), max(cols))


class GridVisibilityIndex(BaseModel):
    """Bidirectional index between BEV grid cells, 3D objects, and camera views."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid_spec: SceneGridSpec
    grid_to_views: dict[GridCell, list[tuple[int, float]]] = Field(default_factory=dict)
    view_to_grids: dict[int, list[tuple[GridCell, float]]] = Field(default_factory=dict)
    object_to_grids: dict[int, list[GridCell]] = Field(default_factory=dict)
    grid_to_objects: dict[GridCell, list[int]] = Field(default_factory=dict)
    stride: int = 1
    depth_verified_views: int = 0

    def covered_grids_for_view(
        self, view_id: int, *, min_score: float = _MIN_GRID_VISIBILITY_SCORE
    ) -> list[GridCell]:
        """Return the ``(row, col)`` cells visible in ``view_id`` ordered by score."""
        return [
            cell
            for cell, score in self.view_to_grids.get(view_id, [])
            if score >= min_score
        ]

    def grid_bounds_for_view(
        self, view_id: int, *, min_score: float = _MIN_GRID_VISIBILITY_SCORE
    ) -> GridBounds | None:
        """Return ``(min_row, max_row, min_col, max_col)`` covered by ``view_id``."""
        return compute_grid_bounds(
            self.covered_grids_for_view(view_id, min_score=min_score)
        )

    def views_for_grids(
        self, cells: Sequence[GridCell]
    ) -> dict[int, dict[GridCell, float]]:
        """Return ``{view_id: {(row, col): score}}`` for the requested grid cells."""
        valid = self.grid_spec.sanitize_cells(cells)
        view_scores: dict[int, dict[GridCell, float]] = {}
        for cell in valid:
            for view_id, score in self.grid_to_views.get(cell, []):
                view_scores.setdefault(view_id, {})[cell] = score
        return view_scores

    def grids_for_objects(self, object_ids: Sequence[int]) -> list[GridCell]:
        """Return deduplicated ``(row, col)`` cells occupied by ``object_ids``."""
        out: list[GridCell] = []
        seen: set[GridCell] = set()
        for obj_id in object_ids:
            for cell in self.object_to_grids.get(int(obj_id), []):
                if cell not in seen:
                    seen.add(cell)
                    out.append(cell)
        return out

    def to_cache_payload(self) -> dict[str, object]:
        """Serialize into a pickle-friendly dict."""
        return {
            "format_version": GRID_INDEX_FORMAT_VERSION,
            "stride": self.stride,
            "depth_verified_views": self.depth_verified_views,
            "grid_spec": self.grid_spec.model_dump(),
            "grid_to_views": {
                (int(r), int(c)): [(int(v), float(s)) for v, s in views]
                for (r, c), views in self.grid_to_views.items()
            },
            "view_to_grids": {
                int(v): [((int(r), int(c)), float(s)) for (r, c), s in cells]
                for v, cells in self.view_to_grids.items()
            },
            "object_to_grids": {
                int(oid): [(int(r), int(c)) for r, c in cells]
                for oid, cells in self.object_to_grids.items()
            },
            "grid_to_objects": {
                (int(r), int(c)): [int(oid) for oid in oids]
                for (r, c), oids in self.grid_to_objects.items()
            },
        }

    @classmethod
    def from_cache_payload(cls, payload: Mapping[str, object]) -> GridVisibilityIndex:
        """Deserialize and validate a cached :class:`GridVisibilityIndex`."""
        if payload.get("format_version") != GRID_INDEX_FORMAT_VERSION:
            raise ValueError(
                f"unsupported grid index format_version: {payload.get('format_version')!r}"
            )
        raw_spec = payload.get("grid_spec")
        if not isinstance(raw_spec, Mapping):
            raise ValueError("grid_spec missing or invalid in cache payload")
        grid_spec = SceneGridSpec.model_validate(dict(raw_spec))

        raw_g2v = payload.get("grid_to_views", {})
        grid_to_views: dict[GridCell, list[tuple[int, float]]] = {}
        if isinstance(raw_g2v, Mapping):
            for key, views in raw_g2v.items():
                if isinstance(key, tuple) and len(key) == 2 and isinstance(views, list):
                    cell = (int(key[0]), int(key[1]))
                    grid_to_views[cell] = [(int(v), float(s)) for v, s in views]

        raw_v2g = payload.get("view_to_grids", {})
        view_to_grids: dict[int, list[tuple[GridCell, float]]] = {}
        if isinstance(raw_v2g, Mapping):
            for vid, cells in raw_v2g.items():
                if isinstance(cells, list):
                    view_to_grids[int(vid)] = [
                        ((int(c[0]), int(c[1])), float(s)) for c, s in cells
                    ]

        raw_o2g = payload.get("object_to_grids", {})
        object_to_grids: dict[int, list[GridCell]] = {}
        if isinstance(raw_o2g, Mapping):
            for oid, cells in raw_o2g.items():
                if isinstance(cells, list):
                    object_to_grids[int(oid)] = [(int(c[0]), int(c[1])) for c in cells]

        raw_g2o = payload.get("grid_to_objects", {})
        grid_to_objects: dict[GridCell, list[int]] = {}
        if isinstance(raw_g2o, Mapping):
            for key, oids in raw_g2o.items():
                if isinstance(key, tuple) and len(key) == 2 and isinstance(oids, list):
                    grid_to_objects[(int(key[0]), int(key[1]))] = [
                        int(oid) for oid in oids
                    ]

        stride_raw = payload.get("stride", 1)
        depth_raw = payload.get("depth_verified_views", 0)
        return cls(
            grid_spec=grid_spec,
            grid_to_views=grid_to_views,
            view_to_grids=view_to_grids,
            object_to_grids=object_to_grids,
            grid_to_objects=grid_to_objects,
            stride=int(stride_raw) if isinstance(stride_raw, (int, float)) else 1,
            depth_verified_views=(
                int(depth_raw) if isinstance(depth_raw, (int, float)) else 0
            ),
        )


def build_grid_visibility_index(
    grid_spec: SceneGridSpec,
    objects: Sequence[SceneObject],
    camera_poses: Sequence[NDArray[np.float64]],
    intrinsic: NDArray[np.float64],
    img_wh: tuple[int, int],
    *,
    object_to_views: Mapping[int, Sequence[tuple[int, float]]] | None = None,
    depth_loader: DepthLoader | None = None,
    stride: int = 1,
    max_distance: float = _DEPTH_MAX_M,
) -> GridVisibilityIndex:
    """Build the bidirectional ``grid <-> view`` and ``grid <-> object`` indices.

    For each view with a finite 4x4 camera pose:

    1. When ``depth_loader(view_id)`` returns a valid depth map, region visibility
       is verified geometrically with depth:
       - valid depth pixels are subsampled and backprojected into 3D world space
         to measure observed surface hits per ``(row, col)`` cell;
       - 3D sample points inside each cell are projected into the camera frustum
         and checked against the depth buffer (``z_cam <= d_obs + tolerance``)
         to reject cells occluded by foreground walls or furniture.
    2. When no depth map is available for a view, frustum projection of cell
       sample points is combined with object-level visibility propagation
       (``object_to_views`` + ``object_to_grids``).
    """
    object_to_grids, grid_to_objects = _build_object_grid_mapping(grid_spec, objects)
    cell_samples, sample_cells = _build_cell_sample_points(grid_spec)
    cell_centers = _build_cell_centers_map(grid_spec)

    grid_to_views_map: dict[GridCell, dict[int, float]] = {}
    view_to_grids_map: dict[int, dict[GridCell, float]] = {}
    depth_verified_count = 0

    for view_id, raw_pose in enumerate(camera_poses):
        pose = np.asarray(raw_pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            continue

        depth: NDArray[np.float64] | None = None
        if depth_loader is not None:
            try:
                depth = depth_loader(view_id)
            except (OSError, ValueError):
                depth = None

        if depth is not None and depth.ndim == 2 and np.isfinite(depth).all():
            cell_scores = _score_view_cells_with_depth(
                grid_spec,
                pose,
                intrinsic,
                img_wh,
                depth,
                cell_samples,
                sample_cells,
                cell_centers,
                max_distance=max_distance,
            )
            depth_verified_count += 1
        else:
            cell_scores = _score_view_cells_frustum_only(
                pose,
                intrinsic,
                img_wh,
                cell_samples,
                sample_cells,
                cell_centers,
                max_distance=max_distance,
            )

        for cell, score in cell_scores.items():
            if score >= _MIN_GRID_VISIBILITY_SCORE:
                grid_to_views_map.setdefault(cell, {})[view_id] = score
                view_to_grids_map.setdefault(view_id, {})[cell] = score

    # Supplement with object-visibility evidence so any view known to observe an
    # object also indexes that object's grid cell(s) (unless depth explicitly
    # proved the view has zero visibility and depth was available).
    if object_to_views:
        for obj_id, views in object_to_views.items():
            cells = object_to_grids.get(int(obj_id), [])
            if not cells:
                continue
            for view_id, obj_score in views:
                if view_id < 0 or (camera_poses and view_id >= len(camera_poses)):
                    continue
                boost = min(1.0, float(obj_score) * 0.85)
                if boost < _MIN_GRID_VISIBILITY_SCORE:
                    continue
                for cell in cells:
                    prev = grid_to_views_map.setdefault(cell, {}).get(view_id, 0.0)
                    merged = max(prev, boost) if prev > 0.0 else boost
                    grid_to_views_map[cell][view_id] = merged
                    view_to_grids_map.setdefault(view_id, {})[cell] = merged

    grid_to_views: dict[GridCell, list[tuple[int, float]]] = {
        cell: sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        for cell, scores in grid_to_views_map.items()
        if scores
    }
    view_to_grids: dict[int, list[tuple[GridCell, float]]] = {
        view_id: sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        for view_id, scores in view_to_grids_map.items()
        if scores
    }

    return GridVisibilityIndex(
        grid_spec=grid_spec,
        grid_to_views=grid_to_views,
        view_to_grids=view_to_grids,
        object_to_grids=object_to_grids,
        grid_to_objects=grid_to_objects,
        stride=stride,
        depth_verified_views=depth_verified_count,
    )


def load_grid_visibility_index(index_path: Path) -> GridVisibilityIndex:
    """Load a precomputed :class:`GridVisibilityIndex` from ``index_path``.

    Raises:
        FileNotFoundError: If ``index_path`` does not exist.
        ValueError: If the pickle payload is malformed or has an unsupported version.
    """
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Grid visibility index not found: {index_path}. "
            "Run offline index building (scripts/datasets/build_grid_indices.py) "
            "to produce indices/grid_visibility_index.pkl first."
        )
    with index_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{index_path} must contain a mapping payload")
    return GridVisibilityIndex.from_cache_payload(payload)


def save_grid_visibility_index(
    grid_index: GridVisibilityIndex, index_path: Path
) -> Path:
    """Atomically write a :class:`GridVisibilityIndex` to ``index_path``."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_name(f".{index_path.name}.tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(
            grid_index.to_cache_payload(), handle, protocol=pickle.HIGHEST_PROTOCOL
        )
    tmp_path.replace(index_path)
    return index_path


# ----- internal geometry helpers --------------------------------------------


def _build_object_grid_mapping(
    grid_spec: SceneGridSpec, objects: Sequence[SceneObject]
) -> tuple[dict[int, list[GridCell]], dict[GridCell, list[int]]]:
    object_to_grids: dict[int, list[GridCell]] = {}
    grid_to_objects: dict[GridCell, list[int]] = {}

    for obj in objects:
        cells: list[GridCell] = []
        centroid_xy: tuple[float, float] | None = None
        if obj.centroid is not None:
            c = np.asarray(obj.centroid, dtype=np.float64).reshape(-1)
            if c.size >= 2 and np.isfinite(c[:2]).all():
                centroid_xy = (float(c[0]), float(c[1]))
        if obj.bbox_np is not None:
            arr = np.asarray(obj.bbox_np, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 3:
                finite = arr[np.isfinite(arr[:, :3]).all(axis=1), :3]
                if finite.size:
                    cells = grid_spec.cells_for_footprint(
                        float(finite[:, 0].min()),
                        float(finite[:, 1].min()),
                        float(finite[:, 0].max()),
                        float(finite[:, 1].max()),
                        centroid_xy=centroid_xy,
                    )
        if not cells and centroid_xy is not None:
            cell = grid_spec.world_to_cell(centroid_xy[0], centroid_xy[1], clamp=True)
            if cell is not None:
                cells = [cell]
        if cells:
            object_to_grids[obj.obj_id] = cells
            for cell in cells:
                grid_to_objects.setdefault(cell, []).append(obj.obj_id)

    return object_to_grids, grid_to_objects


def _build_cell_sample_points(
    grid_spec: SceneGridSpec,
) -> tuple[NDArray[np.float64], list[GridCell]]:
    """Build 3D sample points (2x2 XY x 2 Z levels = 8 points per cell)."""
    points: list[tuple[float, float, float]] = []
    cells: list[GridCell] = []
    z_span = grid_spec.max_z - grid_spec.min_z
    z_levels = (
        grid_spec.min_z + 0.25 * z_span,
        grid_spec.min_z + 0.55 * z_span,
    )
    for row in range(grid_spec.rows):
        for col in range(grid_spec.cols):
            x_min, y_min, x_max, y_max = grid_spec.cell_to_world_bounds(row, col)
            dx = x_max - x_min
            dy = y_max - y_min
            for fx in (0.3, 0.7):
                for fy in (0.3, 0.7):
                    for z in z_levels:
                        points.append((x_min + fx * dx, y_min + fy * dy, z))
                        cells.append((row, col))
    return np.asarray(points, dtype=np.float64), cells


def _build_cell_centers_map(
    grid_spec: SceneGridSpec,
) -> dict[GridCell, NDArray[np.float64]]:
    return {
        (row, col): np.asarray(grid_spec.cell_center_world(row, col), dtype=np.float64)
        for row in range(grid_spec.rows)
        for col in range(grid_spec.cols)
    }


def _score_view_cells_with_depth(
    grid_spec: SceneGridSpec,
    pose: NDArray[np.float64],
    intrinsic: NDArray[np.float64],
    img_wh: tuple[int, int],
    depth: NDArray[np.float64],
    cell_samples: NDArray[np.float64],
    sample_cells: Sequence[GridCell],
    cell_centers: Mapping[GridCell, NDArray[np.float64]],
    *,
    max_distance: float,
) -> dict[GridCell, float]:
    """Compute depth-verified visibility scores for each grid cell from one view."""
    depth_h, depth_w = depth.shape
    if depth_h == 0 or depth_w == 0:
        return {}
    orig_w, orig_h = img_wh
    sx = depth_w / max(orig_w, 1)
    sy = depth_h / max(orig_h, 1)
    fx = float(intrinsic[0, 0]) * sx
    fy = float(intrinsic[1, 1]) * sy
    cx = float(intrinsic[0, 2]) * sx
    cy = float(intrinsic[1, 2]) * sy
    if abs(fx) < 1e-6 or abs(fy) < 1e-6:
        return {}

    # 1. Backproject subsampled valid depth pixels into world XY -> grid cells.
    us = np.arange(0, depth_w, _DEPTH_SUBSAMPLE, dtype=np.float64)
    vs = np.arange(0, depth_h, _DEPTH_SUBSAMPLE, dtype=np.float64)
    grid_u, grid_v = np.meshgrid(us, vs, indexing="xy")
    d_sub = depth[grid_v.astype(np.int64), grid_u.astype(np.int64)]
    valid_mask = (d_sub > _DEPTH_MIN_M) & (d_sub <= max_distance)
    surface_hits: dict[GridCell, int] = {}
    total_valid_rays = int(np.count_nonzero(valid_mask))
    if total_valid_rays > 0:
        u_val = grid_u[valid_mask]
        v_val = grid_v[valid_mask]
        z_val = d_sub[valid_mask]
        x_cam = (u_val - cx) * z_val / fx
        y_cam = (v_val - cy) * z_val / fy
        pts_cam = np.stack([x_cam, y_cam, z_val], axis=1)
        pts_world = (pose[:3, :3] @ pts_cam.T).T + pose[:3, 3]

        cols_idx = np.floor(
            (pts_world[:, 0] - grid_spec.min_x) / grid_spec.cell_width
        ).astype(np.int64)
        rows_idx = np.floor(
            (grid_spec.max_y - pts_world[:, 1]) / grid_spec.cell_height
        ).astype(np.int64)
        in_grid = (
            (rows_idx >= 0)
            & (rows_idx < grid_spec.rows)
            & (cols_idx >= 0)
            & (cols_idx < grid_spec.cols)
        )
        for r, c in zip(rows_idx[in_grid].tolist(), cols_idx[in_grid].tolist()):
            cell = (int(r), int(c))
            surface_hits[cell] = surface_hits.get(cell, 0) + 1

    # 2. Project 3D cell sample points into the camera and check depth occlusion.
    try:
        world_to_cam = np.linalg.inv(pose)
    except np.linalg.LinAlgError:
        return {}
    cam_pts = (world_to_cam[:3, :3] @ cell_samples.T).T + world_to_cam[:3, 3]
    z_cam = cam_pts[:, 2]
    in_front = (z_cam > _DEPTH_MIN_M) & (z_cam <= max_distance)
    safe_z = np.where(in_front, z_cam, 1.0)
    u_proj = np.rint(fx * cam_pts[:, 0] / safe_z + cx).astype(np.int64)
    v_proj = np.rint(fy * cam_pts[:, 1] / safe_z + cy).astype(np.int64)
    in_image = (
        in_front
        & (u_proj >= 0)
        & (u_proj < depth_w)
        & (v_proj >= 0)
        & (v_proj < depth_h)
    )

    unoccluded_counts: dict[GridCell, int] = {}
    for idx in np.nonzero(in_image)[0].tolist():
        d_obs = float(depth[v_proj[idx], u_proj[idx]])
        # Visible if observed depth is valid and does not occlude the sample point.
        if d_obs > _DEPTH_MIN_M and float(z_cam[idx]) <= d_obs + _OCCLUSION_TOLERANCE_M:
            cell = sample_cells[idx]
            unoccluded_counts[cell] = unoccluded_counts.get(cell, 0) + 1

    # 3. Combine depth-verified evidence with camera distance/angle geometry.
    candidate_cells = set(surface_hits) | set(unoccluded_counts)
    scores: dict[GridCell, float] = {}
    expected_rays_per_cell = max(
        4.0, total_valid_rays / max(1, grid_spec.rows * grid_spec.cols)
    )
    for cell in candidate_cells:
        hit_score = min(1.0, surface_hits.get(cell, 0) / expected_rays_per_cell)
        unocc_score = min(1.0, unoccluded_counts.get(cell, 0) / 8.0)
        depth_evidence = 0.65 * hit_score + 0.35 * unocc_score
        if depth_evidence <= 0.0:
            continue
        geo = _cell_geometric_factor(pose, cell_centers[cell], max_distance)
        combined = round(0.65 * depth_evidence + 0.35 * geo, 4)
        if combined >= _MIN_GRID_VISIBILITY_SCORE:
            scores[cell] = combined
    return scores


def _score_view_cells_frustum_only(
    pose: NDArray[np.float64],
    intrinsic: NDArray[np.float64],
    img_wh: tuple[int, int],
    cell_samples: NDArray[np.float64],
    sample_cells: Sequence[GridCell],
    cell_centers: Mapping[GridCell, NDArray[np.float64]],
    *,
    max_distance: float,
) -> dict[GridCell, float]:
    """Fallback frustum visibility when per-frame depth PNGs are absent."""
    try:
        world_to_cam = np.linalg.inv(pose)
    except np.linalg.LinAlgError:
        return {}
    width, height = img_wh
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    cam_pts = (world_to_cam[:3, :3] @ cell_samples.T).T + world_to_cam[:3, 3]
    z_cam = cam_pts[:, 2]
    in_front = (z_cam > _DEPTH_MIN_M) & (z_cam <= max_distance)
    safe_z = np.where(in_front, z_cam, 1.0)
    u_proj = fx * cam_pts[:, 0] / safe_z + cx
    v_proj = fy * cam_pts[:, 1] / safe_z + cy
    in_frustum = (
        in_front
        & (u_proj >= 0.0)
        & (u_proj < width)
        & (v_proj >= 0.0)
        & (v_proj < height)
    )

    counts: dict[GridCell, int] = {}
    for idx in np.nonzero(in_frustum)[0].tolist():
        cell = sample_cells[idx]
        counts[cell] = counts.get(cell, 0) + 1

    scores: dict[GridCell, float] = {}
    for cell, count in counts.items():
        frustum_fraction = min(1.0, count / 8.0)
        geo = _cell_geometric_factor(pose, cell_centers[cell], max_distance)
        combined = round(0.55 * frustum_fraction + 0.45 * geo, 4)
        if combined >= _MIN_GRID_VISIBILITY_SCORE:
            scores[cell] = combined
    return scores


def _cell_geometric_factor(
    pose: NDArray[np.float64],
    cell_center: NDArray[np.float64],
    max_distance: float,
) -> float:
    cam_pos = pose[:3, 3]
    delta = cell_center - cam_pos
    distance = float(np.linalg.norm(delta))
    if distance > max_distance or distance < 1e-6:
        return 0.0
    forward = -pose[:3, 2]
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-6:
        return 0.0
    cos_angle = max(0.0, float(np.dot(delta / distance, forward / forward_norm)))
    dist_score = max(0.0, 1.0 - distance / max_distance)
    return 0.6 * dist_score + 0.4 * cos_angle


__all__ = [
    "GRID_INDEX_FORMAT_VERSION",
    "GridCell",
    "GridBounds",
    "DepthLoader",
    "SceneGridSpec",
    "GridVisibilityIndex",
    "compute_grid_bounds",
    "build_grid_visibility_index",
    "load_grid_visibility_index",
    "save_grid_visibility_index",
]
