"""Reachable frontier selection and persistent, bounded exploration actions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from .geometry import GridCell
from .navigation import (
    MOVE_DISTANCE_M,
    compute_fmm_distances,
    navigation_step,
    rotation_towards,
)

if TYPE_CHECKING:
    from .memory import IncrementalSceneMemory
    from .spatial_map import SpatialMap


@dataclass(frozen=True)
class FrontierCandidate:
    """A reachable observation stance near a free/unknown boundary."""

    region: GridCell
    stance_cell: GridCell
    world_position: tuple[float, float]
    heading_degrees: float
    information_gain_m2: float
    path_cost_m: float
    revisit_count: int
    score: float


def _information_gain(spatial_map: SpatialMap, stance: GridCell) -> float:
    """Estimate visible unknown area using 32 rays stopped by known obstacles."""
    unknown: set[GridCell] = set()
    radius_cells = max(1, round(2.0 / spatial_map.config.fine_cell_size_m))
    for angle in np.linspace(0, 2 * math.pi, 32, endpoint=False):
        for distance in range(1, radius_cells + 1):
            cell = (
                stance[0] + round(math.cos(float(angle)) * distance),
                stance[1] + round(math.sin(float(angle)) * distance),
            )
            state = spatial_map.cell_state(cell)
            if state == "occupied":
                break
            if state == "unknown":
                unknown.add(cell)
    return len(unknown) * spatial_map.config.fine_cell_size_m**2


def suggest_frontiers(
    memory: IncrementalSceneMemory, *, limit: int = 5
) -> tuple[FrontierCandidate, ...]:
    """Rank feasible stances by novel area, FMM distance and observation revisits."""
    if type(limit) is not int or limit < 1:
        raise ValueError(f"limit={limit!r} must be a positive integer")
    spatial_map = memory.spatial_map
    if spatial_map.agent_position is None:
        return ()
    raster = spatial_map.navigation_raster()
    start = spatial_map.world_to_cell(*spatial_map.agent_position[:2])
    current_coverage = spatial_map.coverage.get(spatial_map.region_for_cell(start))
    costs = compute_fmm_distances(raster, {start})
    stride = round(MOVE_DISTANCE_M / raster.resolution_m)
    if stride < 1 or not math.isclose(stride * raster.resolution_m, MOVE_DISTANCE_M):
        raise ValueError("fine_cell_size_m must divide the primitive movement length")
    rows, cols = np.nonzero(raster.traversable & np.isfinite(costs))
    reachable = [
        (raster.cell(int(row), int(col)), float(costs[row, col]))
        for row, col in zip(rows, cols, strict=True)
    ]
    reachable = [
        (cell, cost)
        for cell, cost in reachable
        if (cell[0] - start[0]) % stride == 0 and (cell[1] - start[1]) % stride == 0
    ]
    boundaries: dict[GridCell, list[GridCell]] = {}
    for cell in spatial_map.frontier_cells():
        boundaries.setdefault(spatial_map.region_for_cell(cell), []).append(cell)
    candidates: list[FrontierCandidate] = []
    max_standoff = (
        spatial_map.config.robot_radius_m + spatial_map.config.clearance_m + 0.3
    )
    for region, cells in sorted(boundaries.items()):
        center = np.mean(cells, axis=0)
        possible = [
            (
                float(np.linalg.norm(np.asarray(cell) - center)) * raster.resolution_m,
                cost,
                cell,
            )
            for cell, cost in reachable
        ]
        nearby = [entry for entry in possible if entry[0] <= max_standoff]
        if not nearby:
            continue
        _, cost, stance = min(nearby)
        if (
            cost <= MOVE_DISTANCE_M * 0.45
            and current_coverage is not None
            and len(current_coverage.direction_bins) >= 4
        ):
            # Four discrete headings cannot fill the remaining diagonal FOV
            # gaps from this same stance. Translation is needed for new evidence.
            continue
        gain = _information_gain(spatial_map, stance)
        coverage = spatial_map.coverage.get(region)
        revisits = spatial_map.exploration.attempts.get(region, 0)
        direction_count = len(coverage.direction_bins) if coverage else 0
        score = gain / (1.0 + cost) / (1.0 + revisits + direction_count / 8.0)
        if gain <= 0 or revisits >= 3:
            continue
        heading = (
            math.degrees(
                math.atan2(float(center[0] - stance[0]), float(center[1] - stance[1]))
            )
            % 360
        )
        candidates.append(
            FrontierCandidate(
                region,
                stance,
                spatial_map.cell_center(stance),
                heading,
                gain,
                cost,
                revisits,
                score,
            )
        )
    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.path_cost_m,
            candidate.region,
        )
    )
    unique: list[FrontierCandidate] = []
    used_stances: set[GridCell] = set()
    for candidate in candidates:
        if candidate.stance_cell not in used_stances:
            unique.append(candidate)
            used_stances.add(candidate.stance_cell)
        if len(unique) == limit:
            break
    return tuple(unique)


def exploration_feedback(memory: IncrementalSceneMemory) -> dict[str, object]:
    """Describe observed coverage and feasible frontiers, without hidden map area."""
    spatial_map = memory.spatial_map
    return {
        "covered_regions": len(spatial_map.coverage),
        "coverage_is_not_traversability": True,
        "locked_stance_cell": spatial_map.exploration.goal,
        "frontiers": [asdict(candidate) for candidate in suggest_frontiers(memory)],
        "exhaustion_scope": "currently observed free-space component; unseen space is not proven absent",
    }


def next_exploration_action(
    memory: IncrementalSceneMemory, *, yaw_degrees: float, horizon_degrees: float
) -> tuple[str | None, str]:
    """Keep one frontier commitment until arrival/invalidation, then observe it.

    Arrival triggers a bounded four-turn panorama at a downward pitch. When
    no stance is reachable, one bounded initial panorama obtains more geometry;
    exhaustion is explicit and never replaced by a repeated no-op action.
    """
    spatial_map = memory.spatial_map
    progress = spatial_map.exploration
    # Two passes permit replacing an invalid target without a no-op model turn.
    for _ in range(2):
        if progress.goal is not None and not progress.scanning:
            action, feedback = navigation_step(
                spatial_map, {progress.goal}, yaw_degrees=yaw_degrees
            )
            if action is not None:
                return (
                    action,
                    "Exploring toward the locked frontier stance. " + feedback,
                )
            if feedback.startswith("Reached"):
                progress.scanning = True
                progress.scan_remaining = 4
            else:
                if progress.region is not None:
                    progress.attempts[progress.region] = 3
                progress.goal = None
        if progress.goal is not None and progress.scanning:
            if abs(horizon_degrees - 30.0) > 1:
                return (
                    "LookDown" if horizon_degrees < 30 else "LookUp"
                ), "Observing near-floor clearance at the frontier stance."
            if progress.scan_remaining:
                progress.scan_remaining -= 1
                return (
                    "RotateRight",
                    "Adding a new view from the locked frontier stance.",
                )
            progress.goal = None
            progress.region = None
            progress.scanning = False
        candidates = suggest_frontiers(memory)
        if candidates:
            target = candidates[0]
            progress.goal = target.stance_cell
            progress.region = target.region
            progress.heading_degrees = target.heading_degrees
            progress.attempts[target.region] = (
                progress.attempts.get(target.region, 0) + 1
            )
            rotation = rotation_towards(yaw_degrees, target.heading_degrees)
            if target.path_cost_m <= MOVE_DISTANCE_M * 0.45 and rotation is not None:
                return (
                    rotation,
                    "Orienting toward the selected frontier before observing.",
                )
            continue
        if progress.initial_scan_remaining:
            if abs(horizon_degrees - 30.0) > 1:
                return (
                    "LookDown" if horizon_degrees < 30 else "LookUp"
                ), "Acquiring near-floor geometry before frontier navigation."
            progress.initial_scan_remaining -= 1
            return (
                "RotateRight",
                "No reachable frontier yet; acquiring a bounded panoramic view.",
            )
        return (
            None,
            "No currently reachable frontier remains; unobserved space is still unknown.",
        )
    return None, "Current frontier cannot be expanded with a safe primitive action."
