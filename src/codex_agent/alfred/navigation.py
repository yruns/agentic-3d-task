"""Local routing through the incremental map, with no simulator navigation oracle."""

from __future__ import annotations

import heapq
import math
from collections.abc import Collection

from .geometry import GridCell
from .memory import IncrementalSceneMemory


def shortest_observed_path(
    start: GridCell, goals: Collection[GridCell], free_cells: Collection[GridCell]
) -> tuple[GridCell, ...]:
    """Find a four-connected shortest path, returning empty for unreachable goals."""
    allowed = set(free_cells) | {start}
    destinations = set(goals) & allowed
    if not destinations:
        return ()
    frontier: list[tuple[int, GridCell]] = [(0, start)]
    distance = {start: 0}
    previous: dict[GridCell, GridCell] = {}
    while frontier:
        cost, cell = heapq.heappop(frontier)
        if cost != distance[cell]:
            continue
        if cell in destinations:
            path = [cell]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return tuple(reversed(path))
        for delta_x, delta_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = cell[0] + delta_x, cell[1] + delta_y
            if neighbor in allowed and cost + 1 < distance.get(neighbor, math.inf):
                distance[neighbor] = cost + 1
                previous[neighbor] = cell
                heapq.heappush(frontier, (cost + 1, neighbor))
    return ()


def next_navigation_action(
    memory: IncrementalSceneMemory,
    *,
    yaw_degrees: float,
    horizon_degrees: float,
    object_id: int | None,
    cell: GridCell | None,
    camera_height_m: float = 1.576,
) -> tuple[str | None, str]:
    """Replan one primitive from current memory, or describe why routing stopped."""
    if memory.agent_position is None:
        raise ValueError("navigation requires an integrated observation")
    if (object_id is None) == (cell is None):
        raise ValueError("navigation requires exactly one object_id or cell")
    position = memory.agent_position
    start = memory.world_to_cell(position[0], position[1])
    free = memory.free_cells - memory.occupied_cells
    target = None
    if object_id is not None:
        if object_id not in memory.objects:
            return None, f"Object {object_id} has not been observed."
        node = memory.objects[object_id]
        if not node.geometry_valid or memory.held_object_id == object_id:
            return None, "Object location is stale or held; obtain a new observation."
        target = node.centroid
        size = memory.config.cell_size_m
        goals = {
            candidate
            for candidate in free | {start}
            if math.hypot(
                (candidate[0] + 0.5) * size - target[0],
                (candidate[1] + 0.5) * size - target[1],
            )
            <= 1.0
        }
    else:
        assert cell is not None
        goals = {cell}
    path = shortest_observed_path(start, goals, free)
    if not path:
        return (
            None,
            "No route through observed free cells; explore or select another approach.",
        )
    if len(path) > 1:
        dx, dy = path[1][0] - start[0], path[1][1] - start[1]
        heading = math.degrees(math.atan2(dx, dy)) % 360
        rotation = _rotation_towards(yaw_degrees, heading)
        return (
            rotation or "MoveAhead",
            "Following the current observed free-space route.",
        )
    if target is not None:
        target_dx, target_dy = float(target[0] - position[0]), float(
            target[1] - position[1]
        )
        rotation = _rotation_towards(
            yaw_degrees, math.degrees(math.atan2(target_dx, target_dy))
        )
        if rotation:
            return rotation, "Facing the remembered object vicinity."
        distance = math.hypot(target_dx, target_dy)
        desired_pitch = math.degrees(
            math.atan2(camera_height_m - float(target[2]), max(distance, 0.01))
        )
        desired_pitch = max(-30.0, min(60.0, desired_pitch))
        if abs(horizon_degrees - desired_pitch) > 10:
            return (
                "LookDown" if desired_pitch > horizon_degrees else "LookUp"
            ), "Adjusting camera pitch."
    return (
        None,
        "Reached observed approach; verify current target mask and interaction distance.",
    )


def _rotation_towards(current: float, desired: float) -> str | None:
    difference = (desired - current + 180) % 360 - 180
    if abs(difference) <= 45:
        return None
    return "RotateRight" if difference > 0 else "RotateLeft"
