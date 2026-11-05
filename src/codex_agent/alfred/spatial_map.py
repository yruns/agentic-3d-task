"""Observation-only occupancy and coverage in stable, metric world grids."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .geometry import GridCell, Observation, project_pixels

CellState = Literal["unknown", "free", "occupied"]


@dataclass(frozen=True)
class SpatialMapConfig:
    """Metric map parameters; radius is an explicit, uncalibrated planning input."""

    fine_cell_size_m: float = 0.05
    region_size_m: float = 0.5
    robot_radius_m: float = 0.20
    clearance_m: float = 0.025
    floor_height_m: float = 0.0
    floor_tolerance_m: float = 0.10
    obstacle_min_height_m: float = 0.12
    obstacle_max_height_m: float = 1.8
    max_depth_m: float = 8.0
    occupied_evidence: float = 0.9
    clear_evidence: float = 0.5
    max_evidence: float = 3.0
    max_raster_cells: int = 2_000_000

    def __post_init__(self) -> None:
        for name in (
            "fine_cell_size_m",
            "region_size_m",
            "robot_radius_m",
            "floor_tolerance_m",
            "obstacle_min_height_m",
            "obstacle_max_height_m",
            "max_depth_m",
            "occupied_evidence",
            "clear_evidence",
            "max_evidence",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}={value!r} must be finite and positive")
        if not math.isfinite(self.clearance_m) or self.clearance_m < 0:
            raise ValueError(f"clearance_m={self.clearance_m!r} must be nonnegative")
        if not math.isfinite(self.floor_height_m):
            raise ValueError("floor_height_m must be finite")
        if self.obstacle_min_height_m >= self.obstacle_max_height_m:
            raise ValueError(
                "obstacle_min_height_m must be below obstacle_max_height_m"
            )
        ratio = self.region_size_m / self.fine_cell_size_m
        if not math.isclose(ratio, round(ratio)) or ratio < 1:
            raise ValueError(
                "region_size_m must be an integer multiple of fine_cell_size_m"
            )
        primitive_cells = 0.25 / self.fine_cell_size_m
        if primitive_cells < 1 or not math.isclose(
            primitive_cells, round(primitive_cells)
        ):
            raise ValueError(
                "fine_cell_size_m must divide the 0.25 m movement primitive"
            )
        if self.max_evidence < 0.5:
            raise ValueError("max_evidence must reach the 0.5 occupied threshold")
        if type(self.max_raster_cells) is not int or self.max_raster_cells < 1:
            raise ValueError("max_raster_cells must be a positive integer")


@dataclass
class RegionCoverage:
    """Visibility evidence, independent of whether the ground is traversable."""

    observations: int = 0
    last_seen_frame: int = -1
    direction_bins: set[int] = field(default_factory=set)
    visits: int = 0


@dataclass
class ExplorationProgress:
    """An episode-owned frontier commitment and bounded panoramic observation."""

    goal: GridCell | None = None
    region: GridCell | None = None
    heading_degrees: float = 0.0
    scan_remaining: int = 0
    scanning: bool = False
    initial_scan_remaining: int = 4
    attempts: dict[GridCell, int] = field(default_factory=dict)


@dataclass(frozen=True)
class NavigationRaster:
    """Dense navigation window; array row increases with world +Y."""

    origin: GridCell
    traversable: NDArray[np.bool_]
    resolution_m: float

    def __post_init__(self) -> None:
        if len(self.origin) != 2 or any(
            type(value) is not int for value in self.origin
        ):
            raise ValueError("raster origin must contain two integer indices")
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0:
            raise ValueError("raster resolution_m must be finite and positive")
        if (
            self.traversable.dtype != np.bool_
            or self.traversable.ndim != 2
            or not self.traversable.size
        ):
            raise ValueError("traversable must be a nonempty boolean 2D raster")

    def index(self, cell: GridCell) -> tuple[int, int] | None:
        row, col = cell[1] - self.origin[1], cell[0] - self.origin[0]
        if (
            0 <= row < self.traversable.shape[0]
            and 0 <= col < self.traversable.shape[1]
        ):
            return row, col
        return None

    def contains(self, cell: GridCell) -> bool:
        index = self.index(cell)
        return index is not None and bool(self.traversable[index])

    def cell(self, row: int, col: int) -> GridCell:
        return self.origin[0] + col, self.origin[1] + row


def _disk_offsets(radius_m: float, resolution_m: float) -> tuple[GridCell, ...]:
    radius = math.ceil(radius_m / resolution_m)
    return tuple(
        (x, y)
        for x in range(-radius, radius + 1)
        for y in range(-radius, radius + 1)
        if math.hypot(x, y) * resolution_m <= radius_m + 1e-9
    )


class SpatialMap:
    """Own sparse evidence for one episode; never derive free space from labels.

    All valid depth pixels contribute before endpoint voxel deduplication. Rays
    stop at measured surfaces. Body-height ray clearance and floor endpoints
    supply negative occupancy evidence; body-height endpoints supply positive
    evidence. Physical collisions persist until traversal or explicit region
    invalidation. Coverage records rays independently of these occupancy votes.
    """

    def __init__(self, config: SpatialMapConfig | None = None) -> None:
        self.config = config if config is not None else SpatialMapConfig()
        self._evidence: dict[GridCell, float] = {}
        self._seen: set[GridCell] = set()
        self._collisions: set[GridCell] = set()
        self._visited: set[GridCell] = set()
        self.coverage: dict[GridCell, RegionCoverage] = {}
        self.exploration = ExplorationProgress()
        self.agent_position: tuple[float, float, float] | None = None
        self.camera_direction = (0.0, 1.0, 0.0)
        self.latest_frame_id = -1
        self.revision = 0

    def world_to_cell(self, x: float, y: float) -> GridCell:
        return self._world_index(x, y, self.config.fine_cell_size_m)

    def world_to_region(self, x: float, y: float) -> GridCell:
        return self._world_index(x, y, self.config.region_size_m)

    @staticmethod
    def _world_index(x: float, y: float, size: float) -> GridCell:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"world coordinates {(x, y)!r} must be finite")
        # Trigonometric 90-degree turns produce values such as -4e-17 instead
        # of zero. Snap only sub-nanocell roundoff before flooring world IDs.
        return math.floor(round(x / size, 9)), math.floor(round(y / size, 9))

    def cell_center(self, cell: GridCell) -> tuple[float, float]:
        size = self.config.fine_cell_size_m
        return (cell[0] + 0.5) * size, (cell[1] + 0.5) * size

    def region_center(self, region: GridCell) -> tuple[float, float]:
        size = self.config.region_size_m
        return (region[0] + 0.5) * size, (region[1] + 0.5) * size

    def region_for_cell(self, cell: GridCell) -> GridCell:
        return self.world_to_region(*self.cell_center(cell))

    @property
    def known_cells(self) -> frozenset[GridCell]:
        return frozenset(self._evidence) | frozenset(self._collisions)

    @property
    def seen_cells(self) -> frozenset[GridCell]:
        return frozenset(self._seen)

    @property
    def collision_cells(self) -> frozenset[GridCell]:
        return frozenset(self._collisions)

    def cell_state(self, cell: GridCell) -> CellState:
        if cell in self._collisions or self._evidence.get(cell, 0.0) >= 0.5:
            return "occupied"
        if self._evidence.get(cell, 0.0) <= -0.25:
            return "free"
        return "unknown"

    def update(
        self, observation: Observation, ignored_mask: NDArray[np.bool_] | None = None
    ) -> tuple[GridCell, ...]:
        """Integrate one new RGB-D observation and return its visible coarse regions."""
        if observation.frame_id <= self.latest_frame_id:
            raise ValueError("observation.frame_id must increase monotonically")
        valid = (observation.depth_m > 0) & (
            observation.depth_m <= self.config.max_depth_m
        )
        if ignored_mask is not None:
            if ignored_mask.shape != valid.shape or ignored_mask.dtype != np.bool_:
                raise ValueError(
                    "ignored_mask must be boolean and match depth dimensions"
                )
            valid &= ~ignored_mask
        rows, cols = np.nonzero(valid)
        points = project_pixels(observation, rows, cols)
        # Reduce repeated measurements only after every valid pixel was projected.
        if len(points):
            vcoords = np.floor(points / self.config.fine_cell_size_m).astype(np.int64)
            vbias = 500_000
            vkeys = (
                ((vcoords[:, 0] + vbias) << 40)
                | ((vcoords[:, 1] + vbias) << 20)
                | (vcoords[:, 2] + vbias)
            )
            _, retained = np.unique(vkeys, return_index=True)
            points = points[retained]
        seen, cleared, occupied = self._ray_evidence(
            observation.camera_to_world[:3, 3], points
        )
        for cell in cleared - occupied:
            self._evidence[cell] = max(
                -self.config.max_evidence,
                self._evidence.get(cell, 0.0) - self.config.clear_evidence,
            )
        for cell in occupied:
            self._evidence[cell] = min(
                self.config.max_evidence,
                self._evidence.get(cell, 0.0) + self.config.occupied_evidence,
            )
        previous_position = self.agent_position
        self.agent_position = observation.agent_position
        direction = observation.camera_to_world[:3, 2]
        self.camera_direction = (
            float(direction[0]),
            float(direction[1]),
            float(direction[2]),
        )
        self._record_traversal(observation.agent_position, previous_position)
        seen.add(self.world_to_cell(*observation.agent_position[:2]))
        self._seen.update(seen)
        regions = {self.region_for_cell(cell) for cell in seen}
        heading = math.atan2(self.camera_direction[0], self.camera_direction[1])
        direction_bin = int(round(heading / (math.pi / 4))) % 8
        for region in regions:
            coverage = self.coverage.setdefault(region, RegionCoverage())
            coverage.observations += 1
            coverage.last_seen_frame = observation.frame_id
            coverage.direction_bins.add(direction_bin)
        agent_region = self.world_to_region(*observation.agent_position[:2])
        if (
            previous_position is None
            or self.world_to_region(*previous_position[:2]) != agent_region
        ):
            self.coverage.setdefault(agent_region, RegionCoverage()).visits += 1
        self.latest_frame_id = observation.frame_id
        self.revision += 1
        return tuple(sorted(regions))

    def _ray_evidence(
        self, camera: NDArray[np.float64], points: NDArray[np.float64]
    ) -> tuple[set[GridCell], set[GridCell], set[GridCell]]:
        if not len(points):
            return set(), set(), set()
        size = self.config.fine_cell_size_m
        raw_heights = points[:, 2] - self.config.floor_height_m
        # Adaptive local floor-plane calibration: when monocular depth (DA3)
        # underestimates range by 8-20%, a flat horizontal floor plane appears
        # elevated to z=+0.12..+0.25m. Detect a flat horizontal floor mode
        # (tight p50-p10 height spread across >=100 low surface cells) so real
        # floor is marked clear rather than false obstacle walls.
        floor_offset = 0.0
        low_heights = raw_heights[(raw_heights >= -0.20) & (raw_heights <= 0.30)]
        if len(low_heights) >= 100:
            p10, p20, p50 = np.percentile(low_heights, [10.0, 20.0, 50.0])
            if (p50 - p10) <= 0.09 and -0.15 <= p20 <= 0.26:
                floor_offset = float(p20)

        adj_heights = raw_heights - floor_offset
        end_cells = np.floor(np.round(points[:, :2] / size, 9)).astype(np.int64)
        bias = 1_000_000

        def _pack(cx: NDArray[np.int64], cy: NDArray[np.int64]) -> NDArray[np.int64]:
            return ((cx + bias) << 32) | (cy + bias)

        def _unpack_set(keys: NDArray[np.int64]) -> set[GridCell]:
            if not len(keys):
                return set()
            unique_keys = np.unique(keys)
            cx = (unique_keys >> 32) - bias
            cy = (unique_keys & 0xFFFFFFFF) - bias
            return set(zip(cx.tolist(), cy.tolist(), strict=True))

        end_keys = _pack(end_cells[:, 0], end_cells[:, 1])
        is_occ = (adj_heights >= self.config.obstacle_min_height_m) & (
            adj_heights <= self.config.obstacle_max_height_m
        )
        is_floor = np.abs(adj_heights) <= self.config.floor_tolerance_m
        occupied = _unpack_set(end_keys[is_occ])
        floor_clear = _unpack_set(end_keys[is_floor])

        # Deduplicate endpoints by (cell_x, cell_y, 10cm_height_bin) before ray
        # marching; rays terminating in the same 5cm cell at the same height
        # band traverse identical 2D grid cells.
        h_bin = np.floor(adj_heights / 0.10).astype(np.int64)
        dedup_keys = (end_keys << 8) ^ (h_bin & 0xFF)
        _, unique_idx = np.unique(dedup_keys, return_index=True)
        ray_points = points[unique_idx]

        seen_keys_list: list[NDArray[np.int64]] = [end_keys]
        clear_keys_list: list[NDArray[np.int64]] = []
        for start in range(0, len(ray_points), 512):
            endpoints = ray_points[start : start + 512]
            lengths = np.linalg.norm(endpoints - camera, axis=1)
            steps = max(2, math.ceil(float(lengths.max()) / (size * 0.5)))
            fractions = np.linspace(0, 1, steps, endpoint=False)
            rays = (
                camera[None, None, :]
                + (endpoints[:, None, :] - camera) * fractions[None, :, None]
            )
            cells = np.floor(np.round(rays[..., :2] / size, 9)).astype(np.int64)
            rkeys = _pack(cells[..., 0], cells[..., 1])
            seen_keys_list.append(rkeys.ravel())
            heights = (rays[..., 2] - self.config.floor_height_m) - floor_offset
            body = (heights >= self.config.obstacle_min_height_m) & (
                heights <= self.config.obstacle_max_height_m
            )
            clear_keys_list.append(rkeys[body])

        seen = _unpack_set(np.concatenate(seen_keys_list))
        clear = (
            _unpack_set(np.concatenate(clear_keys_list)) if clear_keys_list else set()
        ) | floor_clear
        return seen, clear, occupied

    def _record_traversal(
        self,
        position: tuple[float, float, float],
        previous: tuple[float, float, float] | None,
    ) -> None:
        # Only an ordinary ALFRED primitive-sized displacement grants swept
        # footprint evidence. A reset/relocalization jump must not clear a corridor.
        distance = math.dist(position[:2], previous[:2]) if previous else math.inf
        if previous is not None and distance <= 0.25 + self.config.fine_cell_size_m:
            steps = max(2, math.ceil(distance / self.config.fine_cell_size_m) + 1)
            positions = np.linspace(previous[:2], position[:2], steps)
        else:
            positions = np.asarray([position[:2]])
        offsets = _disk_offsets(
            self.config.robot_radius_m, self.config.fine_cell_size_m
        )
        for point in positions:
            center = self.world_to_cell(float(point[0]), float(point[1]))
            self._visited.add(center)
            for dx, dy in offsets:
                cell = center[0] + dx, center[1] + dy
                self._evidence[cell] = -self.config.max_evidence
                self._collisions.discard(cell)

    def record_collision(self, cell: GridCell) -> None:
        """Record a failed-motion endpoint in fine grid coordinates."""
        if len(cell) != 2 or any(type(value) is not int for value in cell):
            raise ValueError("collision cell must contain two integer indices")
        self._collisions.add(cell)
        self.revision += 1

    def invalidate_region(self, cells: Collection[GridCell]) -> None:
        """Discard current geometry after interaction, retaining historical coverage."""
        regions = set(cells)
        affected = {
            cell for cell in self.known_cells if self.region_for_cell(cell) in regions
        }
        for cell in affected:
            self._evidence.pop(cell, None)
        self._collisions.difference_update(affected)
        self.revision += 1

    def navigation_raster(self) -> NavigationRaster:
        """Block unknown centers; inflate known obstacles by the robot footprint.

        Unknown neighbors do not invalidate an observed free center. This keeps
        exploration feasible with the 60-degree FOV and 90-degree turn lattice;
        it does not assert that every point of the swept body was observed.
        """
        if self.agent_position is None:
            raise ValueError("navigation requires at least one observation")
        cells = self.known_cells
        padding = (
            math.ceil(
                (self.config.robot_radius_m + self.config.clearance_m)
                / self.config.fine_cell_size_m
            )
            + 1
        )
        min_x, max_x = (
            min(x for x, _ in cells) - padding,
            max(x for x, _ in cells) + padding,
        )
        min_y, max_y = (
            min(y for _, y in cells) - padding,
            max(y for _, y in cells) + padding,
        )
        shape = max_y - min_y + 1, max_x - min_x + 1
        if shape[0] * shape[1] > self.config.max_raster_cells:
            raise ValueError(f"navigation raster {shape!r} exceeds max_raster_cells")
        free = np.zeros(shape, dtype=bool)
        occupied = np.zeros(shape, dtype=bool)
        for cell in cells:
            if self.cell_state(cell) == "free":
                free[cell[1] - min_y, cell[0] - min_x] = True
            elif self.cell_state(cell) == "occupied":
                occupied[cell[1] - min_y, cell[0] - min_x] = True
        inflated = occupied.copy()
        for dx, dy in _disk_offsets(
            self.config.robot_radius_m + self.config.clearance_m,
            self.config.fine_cell_size_m,
        ):
            shifted = np.zeros_like(occupied)
            source_y = slice(max(0, dy), min(shape[0], shape[0] + dy))
            source_x = slice(max(0, dx), min(shape[1], shape[1] + dx))
            target_y = slice(max(0, -dy), min(shape[0], shape[0] - dy))
            target_x = slice(max(0, -dx), min(shape[1], shape[1] - dx))
            shifted[target_y, target_x] = occupied[source_y, source_x]
            inflated |= shifted
        allowed = free & ~inflated
        # The current physical stance is known valid even when its safety margin
        # extends beyond currently reconstructed floor. No adjacent cell is exempt.
        start = self.world_to_cell(*self.agent_position[:2])
        allowed[start[1] - min_y, start[0] - min_x] = True
        allowed.setflags(write=False)
        return NavigationRaster((min_x, min_y), allowed, self.config.fine_cell_size_m)

    def frontier_cells(self) -> frozenset[GridCell]:
        """Observed free boundary cells adjacent to occupancy-unknown space."""
        return frozenset(
            cell
            for cell in self.known_cells
            if self.cell_state(cell) == "free"
            and any(
                self.cell_state((cell[0] + dx, cell[1] + dy)) == "unknown"
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
        )

    def snapshot(self) -> dict[str, object]:
        """Compact policy-facing statistics; fine geometry stays in memory/BEV."""
        return {
            "fine_cell_size_m": self.config.fine_cell_size_m,
            "region_size_m": self.config.region_size_m,
            "free_cell_count": sum(
                self.cell_state(cell) == "free" for cell in self.known_cells
            ),
            "occupied_cell_count": sum(
                self.cell_state(cell) == "occupied" for cell in self.known_cells
            ),
            "seen_cell_count": len(self._seen),
            "visited_cell_count": len(self._visited),
            "unknown_is_traversable": False,
            "obstacle_inflation_radius_m": self.config.robot_radius_m
            + self.config.clearance_m,
            "unobserved_footprint_neighbors_require_collision_feedback": True,
            "regions": [
                {
                    "region": region,
                    "observations": coverage.observations,
                    "direction_bins": sorted(coverage.direction_bins),
                    "last_seen_frame": coverage.last_seen_frame,
                    "visits": coverage.visits,
                }
                for region, coverage in sorted(self.coverage.items())
            ],
        }
