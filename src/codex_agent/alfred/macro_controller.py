"""High-level macro action controller for ALFRED episodes.

Codex never issues atomic primitive movements (``MoveAhead``, ``RotateLeft``,
``RotateRight``, ``LookUp``, ``LookDown``) directly. Instead, Codex invokes
semantic macro operations (``navigate_to``, ``explore_frontier``,
``scan_surroundings``, ``interact_object``) which this controller executes
closed-loop against the simulator and perception worker, updating
:class:`IncrementalSceneMemory` on every intermediate frame and returning only a
compact :func:`compute_scene_delta` summary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .delta import (
    RevisionWatermark,
    capture_watermark,
    compute_scene_delta,
    format_visible_objects_now,
)
from .exploration import next_exploration_action, suggest_frontiers
from .geometry import GridCell
from .navigation import (
    MOVE_DISTANCE_M,
    compute_fmm_distances,
    navigation_step,
    next_navigation_action,
    rotation_towards,
)
from .object_mapping import feature_cosine
from .object_store import ActionState

if TYPE_CHECKING:
    from .execution import EpisodeExecutor

_ACTION_MAP: dict[str, str] = {
    "pickup": "PickupObject",
    "PickupObject": "PickupObject",
    "put": "PutObject",
    "PutObject": "PutObject",
    "open": "OpenObject",
    "OpenObject": "OpenObject",
    "close": "CloseObject",
    "CloseObject": "CloseObject",
    "toggle_on": "ToggleObjectOn",
    "ToggleObjectOn": "ToggleObjectOn",
    "toggle_off": "ToggleObjectOff",
    "ToggleObjectOff": "ToggleObjectOff",
    "slice": "SliceObject",
    "SliceObject": "SliceObject",
}

_OPENABLE_CATEGORIES = frozenset(
    {"Fridge", "Microwave", "Cabinet", "Drawer", "Safe", "LaundryHamper"}
)


def _box_distance_2d(point_xy: np.ndarray, bounds: np.ndarray) -> float:
    nearest = np.clip(point_xy, bounds[0, :2], bounds[1, :2])
    return float(np.linalg.norm(point_xy - nearest))


def _is_fov_aligned(
    stance_xy: tuple[float, float] | np.ndarray, target_xy: np.ndarray
) -> bool:
    """Return whether ``target_xy`` falls within the +/-22 deg cardinal camera FOV cone."""
    dx = abs(float(target_xy[0] - stance_xy[0]))
    dy = abs(float(target_xy[1] - stance_xy[1]))
    major = max(dx, dy)
    minor = min(dx, dy)
    return major <= 0.15 or (minor / major) <= math.tan(math.radians(22.0))


class MacroActionController:
    """Drive multi-step navigation, scanning, exploration, and object interactions."""

    def __init__(self, executor: EpisodeExecutor) -> None:
        self.executor = executor

    def _adjust_pitch_to(self, target_horizon: float) -> tuple[int, int]:
        """Step camera pitch toward ``target_horizon`` in 15-degree increments."""
        clamped = max(-30.0, min(60.0, round(target_horizon / 15.0) * 15.0))
        steps = 0
        failed = 0
        for _ in range(6):
            if not self.executor.can_step():
                break
            current = self.executor.current_observation.horizon_degrees
            if abs(current - clamped) <= 7.5:
                break
            action = "LookDown" if clamped > current else "LookUp"
            ok = self.executor.step_primitive(action, object_id=None)
            steps += 1
            if not ok:
                failed += 1
                break
        return steps, failed

    def _orient_towards_point(
        self, target_xyz: np.ndarray, *, camera_height_m: float = 1.576
    ) -> tuple[int, int]:
        """Rotate horizontally and adjust pitch to center ``target_xyz`` in view."""
        steps = 0
        failed = 0
        pos = self.executor.memory.agent_position
        if pos is None:
            return 0, 0
        dx = float(target_xyz[0] - pos[0])
        dy = float(target_xyz[1] - pos[1])
        horiz_dist = math.hypot(dx, dy)
        if horiz_dist > 1e-3:
            desired_yaw = math.degrees(math.atan2(dx, dy)) % 360.0
            for _ in range(2):
                if not self.executor.can_step():
                    break
                rot = rotation_towards(
                    self.executor.current_observation.yaw_degrees, desired_yaw
                )
                if rot is None:
                    break
                ok = self.executor.step_primitive(rot, object_id=None)
                steps += 1
                if not ok:
                    failed += 1
                    break

        desired_pitch = math.degrees(
            math.atan2(camera_height_m - float(target_xyz[2]), max(0.20, horiz_dist))
        )
        p_steps, p_failed = self._adjust_pitch_to(desired_pitch)
        return steps + p_steps, failed + p_failed

    def scan_surroundings(
        self,
        *,
        pitches: Sequence[float] = (30.0, 0.0),
    ) -> dict[str, object]:
        """Perform a 360-degree panoramic scan at the requested camera pitches."""
        before = capture_watermark(self.executor.memory)
        steps = 0
        failed = 0
        valid_pitches = tuple(pitches) if pitches else (30.0,)
        for pitch in valid_pitches[:2]:
            if not self.executor.can_step():
                break
            p_steps, p_failed = self._adjust_pitch_to(float(pitch))
            steps += p_steps
            failed += p_failed
            for _ in range(4):
                if not self.executor.can_step():
                    break
                ok = self.executor.step_primitive("RotateRight", object_id=None)
                steps += 1
                if not ok:
                    failed += 1
                    break

        return self._build_delta(
            before,
            macro_action=f"scan_surroundings(pitches={list(valid_pitches[:2])})",
            status="completed",
            message=f"Completed panoramic scan across pitches {list(valid_pitches[:2])} in {steps} primitive steps.",
            primitive_steps=steps,
            failed_steps=failed,
        )

    def explore_frontier(
        self,
        *,
        region: GridCell | None = None,
        stop_on_categories: Sequence[str] = (),
        max_steps: int = 20,
    ) -> dict[str, object]:
        """Navigate to a reachable frontier and scan new directions."""
        before = capture_watermark(self.executor.memory)
        steps = 0
        failed = 0
        stop_cats = {
            cat.strip().casefold() for cat in stop_on_categories if cat.strip()
        }
        memory = self.executor.memory

        if region is not None:
            candidates = suggest_frontiers(memory, limit=25)
            chosen = next(
                (cand for cand in candidates if cand.region == region),
                None,
            )
            if chosen is None and candidates:
                chosen = min(
                    candidates,
                    key=lambda cand: math.hypot(
                        cand.region[0] - region[0], cand.region[1] - region[1]
                    ),
                )
            if chosen is not None:
                memory.spatial_map.exploration.goal = chosen.stance_cell
                memory.spatial_map.exploration.region = chosen.region
                memory.spatial_map.exploration.heading_degrees = chosen.heading_degrees
                memory.spatial_map.exploration.scanning = False

        status = "completed"
        message = "Explored reachable frontier and updated scene map."
        bounded_steps = max(1, min(36, max_steps))

        for _ in range(bounded_steps):
            if not self.executor.can_step():
                status = "budget_reached"
                message = (
                    "Action or failure budget reached during frontier exploration."
                )
                break
            if stop_cats:
                matched_id = self._find_newly_visible_category(before, stop_cats)
                if matched_id is not None:
                    node = memory.objects[matched_id]
                    o_steps, o_fail = self._orient_towards_point(node.centroid)
                    steps += o_steps
                    failed += o_fail
                    status = "target_discovered"
                    message = (
                        f"Discovered requested category {node.category!r} "
                        f"(object_id={matched_id}) and oriented camera toward it."
                    )
                    break

            obs = self.executor.current_observation
            action, feedback = next_exploration_action(
                memory,
                yaw_degrees=obs.yaw_degrees,
                horizon_degrees=obs.horizon_degrees,
            )
            message = feedback
            if action is None:
                break
            ok = self.executor.step_primitive(action, object_id=None)
            steps += 1
            if not ok:
                failed += 1
                memory.spatial_map.exploration.goal = None
                memory.spatial_map.exploration.scanning = False

        return self._build_delta(
            before,
            macro_action=(
                f"explore_frontier(region={list(region) if region else None}, "
                f"stop_on={sorted(stop_cats)})"
            ),
            status=status,
            message=message,
            primitive_steps=steps,
            failed_steps=failed,
        )

    def navigate_to(
        self,
        *,
        object_id: int | None = None,
        region: GridCell | None = None,
        max_steps: int = 36,
    ) -> dict[str, object]:
        """Navigate to an observed object or coarse grid region and frame the target."""
        if (object_id is None) == (region is None):
            raise ValueError(
                "navigate_to requires exactly one of object_id or region=[x, y]"
            )
        before = capture_watermark(self.executor.memory)
        steps, failed, status, message = self._execute_navigation(
            object_id=object_id,
            region=region,
            max_steps=max(1, min(48, max_steps)),
        )
        target_desc = (
            f"object_id={object_id}"
            if object_id is not None
            else f"region={list(region)}"  # type: ignore[arg-type]
        )
        return self._build_delta(
            before,
            macro_action=f"navigate_to({target_desc})",
            status=status,
            message=message,
            primitive_steps=steps,
            failed_steps=failed,
        )

    def _execute_navigation(
        self,
        *,
        object_id: int | None,
        region: GridCell | None,
        max_steps: int = 36,
    ) -> tuple[int, int, str, str]:
        memory = self.executor.memory
        steps = 0
        failed = 0
        resolved_id: int | None = None
        if object_id is not None:
            try:
                resolved_id = memory.resolve_object_id(object_id)
            except KeyError:
                return 0, 0, "failed", f"Object #{object_id} has never been observed."
            if resolved_id == memory.held_object_id:
                return (
                    0,
                    0,
                    "already_held",
                    f"Object #{resolved_id} is already held in hand.",
                )

        last_feedback = "Starting FMM navigation."
        for _ in range(max_steps):
            if not self.executor.can_step():
                return (
                    steps,
                    failed,
                    "budget_reached",
                    "Action or failure budget reached during navigation.",
                )
            if resolved_id is not None:
                resolved_id = memory.resolve_object_id(resolved_id)
            obs = self.executor.current_observation
            action, feedback = self._next_adaptive_nav_action(
                resolved_id=resolved_id,
                region=region,
                yaw_degrees=obs.yaw_degrees,
                horizon_degrees=obs.horizon_degrees,
            )
            last_feedback = feedback
            if action is None:
                break
            ok = self.executor.step_primitive(action, object_id=None)
            steps += 1
            if not ok:
                failed += 1
                if failed >= 3:
                    break

        if resolved_id is not None and resolved_id in memory.objects:
            resolved_id = memory.resolve_object_id(resolved_id)
            node = memory.objects[resolved_id]
            o_steps, o_fail = self._auto_frame_object(resolved_id)
            steps += o_steps
            failed += o_fail
            resolved_id = memory.resolve_object_id(resolved_id)
            node = memory.objects[resolved_id]
            mask_match = memory.select_interaction_mask(
                resolved_id, action="PickupObject"
            )
            pos = np.asarray((memory.agent_position or (0.0, 0.0, 0.0))[:2])
            box_dist = _box_distance_2d(pos, node.bounds)
            center_dist = float(np.linalg.norm(node.centroid[:2] - pos))
            if mask_match is not None and (box_dist <= 1.05 or center_dist <= 1.35):
                return (
                    steps,
                    failed,
                    "arrived_and_framed",
                    f"Arrived within reach of #{resolved_id} ({node.category}, box_dist={box_dist:.2f}m, center_dist={center_dist:.2f}m) and verified interaction mask in current view.",
                )
            if box_dist <= 1.05 or center_dist <= 1.35:
                return (
                    steps,
                    failed,
                    "arrived_unverified_mask",
                    f"Reached vicinity of #{resolved_id} ({node.category}, box_dist={box_dist:.2f}m), but target mask was not detected in current frame.",
                )
            return (
                steps,
                failed,
                "partial_approach",
                f"Approached to box_dist={box_dist:.2f}m (center_dist={center_dist:.2f}m) of #{resolved_id} ({node.category}); {last_feedback}",
            )

        return steps, failed, "completed", last_feedback

    def _next_adaptive_nav_action(
        self,
        *,
        resolved_id: int | None,
        region: GridCell | None,
        yaw_degrees: float,
        horizon_degrees: float,
    ) -> tuple[str | None, str]:
        """Plan FMM step toward target; progressively reveal floor when target is far."""
        memory = self.executor.memory
        if resolved_id is None:
            assert region is not None
            return self._next_region_nav_action(
                region=region,
                yaw_degrees=yaw_degrees,
                horizon_degrees=horizon_degrees,
            )

        node = memory.objects[resolved_id]
        spatial_map = memory.spatial_map
        position = spatial_map.agent_position
        if position is None:
            return None, "Navigation requires an integrated observation."

        pos_xy = np.asarray(position[:2], dtype=np.float64)
        bounds = node.bounds
        target_xy = node.centroid[:2]
        current_box_dist = _box_distance_2d(pos_xy, bounds)
        current_aligned = _is_fov_aligned(pos_xy, target_xy)
        is_large_appliance = node.category in {"Fridge", "Cabinet", "Microwave"}
        min_stance = 0.45 if is_large_appliance else 0.25
        max_stance = 0.95 if is_large_appliance else 0.85
        if min_stance <= current_box_dist <= max_stance and current_aligned:
            return None, "Reached an observed interaction stance near target."

        raster = spatial_map.navigation_raster()
        start = spatial_map.world_to_cell(*position[:2])
        costs_from_start = compute_fmm_distances(raster, {start})
        stride = round(MOVE_DISTANCE_M / raster.resolution_m)
        rows, cols = np.nonzero(raster.traversable & np.isfinite(costs_from_start))
        reachable_cells = [
            raster.cell(int(row), int(col)) for row, col in zip(rows, cols, strict=True)
        ]
        lattice_cells = [
            cell
            for cell in reachable_cells
            if (cell[0] - start[0]) % stride == 0 and (cell[1] - start[1]) % stride == 0
        ]
        if not lattice_cells:
            lattice_cells = reachable_cells

        close_goals = {
            cell
            for cell in lattice_cells
            if min_stance
            <= _box_distance_2d(np.asarray(spatial_map.cell_center(cell)), bounds)
            <= max_stance
        }
        if not close_goals and is_large_appliance:
            close_goals = {
                cell
                for cell in lattice_cells
                if 0.25
                <= _box_distance_2d(np.asarray(spatial_map.cell_center(cell)), bounds)
                <= 0.95
            }
        aligned_close = {
            cell
            for cell in close_goals
            if _is_fov_aligned(spatial_map.cell_center(cell), target_xy)
        }
        if aligned_close:
            action, feedback = navigation_step(
                spatial_map, aligned_close, yaw_degrees=yaw_degrees
            )
            if action is not None or feedback.startswith("Reached"):
                return action, feedback

        # Progressive approach: if an aligned 0.25-0.85m stance is still in unobserved space,
        # advance to the reachable lattice cell that improves (distance + alignment penalty).
        align_penalty = 0.32
        scored = [
            (
                _box_distance_2d(np.asarray(spatial_map.cell_center(cell)), bounds)
                + (
                    0.0
                    if _is_fov_aligned(spatial_map.cell_center(cell), target_xy)
                    else align_penalty
                ),
                _box_distance_2d(np.asarray(spatial_map.cell_center(cell)), bounds),
                cell,
            )
            for cell in lattice_cells
        ]
        if scored:
            min_score = min(item[0] for item in scored)
            cur_score = current_box_dist + (0.0 if current_aligned else align_penalty)
            if min_score < cur_score - 0.10:
                progressive_goals = {
                    cell for sc, _, cell in scored if sc <= min_score + 0.15
                }
                prog_action, _ = navigation_step(
                    spatial_map, progressive_goals, yaw_degrees=yaw_degrees
                )
                if prog_action is not None:
                    return (
                        prog_action,
                        f"Advancing along FMM toward #{resolved_id} (target score={min_score:.2f}m).",
                    )

        if close_goals:
            if 0.25 <= current_box_dist <= 0.85:
                return None, "Reached an observed interaction stance near target."
            action, feedback = navigation_step(
                spatial_map, close_goals, yaw_degrees=yaw_degrees
            )
            if action is not None or feedback.startswith("Reached"):
                return action, feedback

        # At the boundary of currently known free floor, orient toward the target and
        # look down (30-45 deg) so the depth sensor clears the floor ahead!
        dx = float(node.centroid[0] - position[0])
        dy = float(node.centroid[1] - position[1])
        desired_yaw = math.degrees(math.atan2(dx, dy)) % 360.0
        rot = rotation_towards(yaw_degrees, desired_yaw)
        if rot is not None:
            return rot, "Orienting toward target to observe floor clearance ahead."
        if horizon_degrees < 30.0 - 1e-3:
            return (
                "LookDown",
                "Tilting camera down to observe floor clearance toward target.",
            )
        return None, "No further free-floor path toward target from current stance."

    def _next_region_nav_action(
        self,
        *,
        region: GridCell,
        yaw_degrees: float,
        horizon_degrees: float,
    ) -> tuple[str | None, str]:
        memory = self.executor.memory
        spatial_map = memory.spatial_map
        position = spatial_map.agent_position
        if position is None:
            return None, "Navigation requires an integrated observation."
        current_region = spatial_map.world_to_region(position[0], position[1])
        if current_region == region:
            return None, f"Reached target region {list(region)}."

        action, feedback = next_navigation_action(
            memory,
            yaw_degrees=yaw_degrees,
            horizon_degrees=horizon_degrees,
            object_id=None,
            cell=region,
        )
        if action is not None or feedback.startswith("Reached"):
            return action, feedback

        # Progressive fallback toward the target region's center
        target_xy = np.asarray(spatial_map.region_center(region), dtype=np.float64)
        current_dist = float(np.linalg.norm(np.asarray(position[:2]) - target_xy))
        raster = spatial_map.navigation_raster()
        start = spatial_map.world_to_cell(*position[:2])
        costs_from_start = compute_fmm_distances(raster, {start})
        stride = round(MOVE_DISTANCE_M / raster.resolution_m)
        rows, cols = np.nonzero(raster.traversable & np.isfinite(costs_from_start))
        lattice_cells = [
            raster.cell(int(row), int(col))
            for row, col in zip(rows, cols, strict=True)
            if (int(col) + raster.origin[0] - start[0]) % stride == 0
            and (int(row) + raster.origin[1] - start[1]) % stride == 0
        ]
        scored = [
            (
                float(
                    np.linalg.norm(
                        np.asarray(spatial_map.cell_center(cell)) - target_xy
                    )
                ),
                cell,
            )
            for cell in lattice_cells
        ]
        if scored:
            min_dist = min(item[0] for item in scored)
            if min_dist < current_dist - 0.12:
                prog_goals = {cell for dist, cell in scored if dist <= min_dist + 0.15}
                prog_action, _ = navigation_step(
                    spatial_map, prog_goals, yaw_degrees=yaw_degrees
                )
                if prog_action is not None:
                    return (
                        prog_action,
                        f"Advancing toward region {list(region)} (remaining={min_dist:.2f}m).",
                    )
        return None, feedback

    def _auto_frame_object(
        self, resolved_id: int, *, action: str = "PickupObject"
    ) -> tuple[int, int]:
        """Orient camera toward ``resolved_id`` and probe pitches without oscillation."""
        memory = self.executor.memory
        if resolved_id not in memory.objects:
            return 0, 0
        steps = 0
        failed = 0
        node = memory.objects[resolved_id]
        pos = memory.agent_position
        horiz_dist = 0.5
        if pos is not None:
            dx = float(node.centroid[0] - pos[0])
            dy = float(node.centroid[1] - pos[1])
            horiz_dist = math.hypot(dx, dy)
            if horiz_dist > 1e-3:
                desired_yaw = math.degrees(math.atan2(dx, dy)) % 360.0
                for _ in range(2):
                    if not self.executor.can_step():
                        break
                    rot = rotation_towards(
                        self.executor.current_observation.yaw_degrees, desired_yaw
                    )
                    if rot is None:
                        break
                    ok = self.executor.step_primitive(rot, object_id=None)
                    steps += 1
                    if not ok:
                        failed += 1
                        break

        resolved_id = memory.resolve_object_id(resolved_id)
        if memory.select_interaction_mask(resolved_id, action=action) is not None:
            return steps, failed

        node = memory.objects[resolved_id]
        desired_pitch = math.degrees(
            math.atan2(1.576 - float(node.centroid[2]), max(0.20, horiz_dist))
        )
        clamped_desired = max(-30.0, min(60.0, round(desired_pitch / 15.0) * 15.0))
        candidate_pitches = [clamped_desired]
        for delta in (15.0, -15.0, 30.0, -30.0):
            cand_p = max(-30.0, min(60.0, clamped_desired + delta))
            if cand_p not in candidate_pitches:
                candidate_pitches.append(cand_p)

        for target_p in candidate_pitches[:3]:
            if not self.executor.can_step():
                break
            p_steps, p_fail = self._adjust_pitch_to(target_p)
            steps += p_steps
            failed += p_fail
            resolved_id = memory.resolve_object_id(resolved_id)
            if memory.select_interaction_mask(resolved_id, action=action) is not None:
                return steps, failed
        return steps, failed

    def interact_object(
        self,
        *,
        action: str,
        object_id: int,
    ) -> dict[str, object]:
        """Execute an atomic or compound interaction with auto-approach and micro-recovery."""
        before = capture_watermark(self.executor.memory)
        memory = self.executor.memory
        try:
            resolved_id = memory.resolve_object_id(object_id)
        except KeyError:
            return self._build_delta(
                before,
                macro_action=f"interact_object(action={action!r}, object_id={object_id})",
                status="failed",
                message=f"Object #{object_id} has never been observed in scene memory.",
                primitive_steps=0,
                failed_steps=0,
            )

        normalized = action.strip()
        if normalized in {"clean_in_sink", "heat_in_microwave", "cool_in_fridge"}:
            steps, failed, ok, msg = self._run_compound_interaction(
                normalized, resolved_id
            )
            return self._build_delta(
                before,
                macro_action=f"interact_object(action={normalized!r}, object_id={resolved_id})",
                status="completed" if ok else "failed",
                message=msg,
                primitive_steps=steps,
                failed_steps=failed,
            )

        if normalized not in _ACTION_MAP:
            raise ValueError(f"Unsupported interaction action: {action!r}")
        sim_action = _ACTION_MAP[normalized]
        steps, failed, ok, msg = self._execute_single_interaction(
            sim_action, resolved_id
        )
        return self._build_delta(
            before,
            macro_action=f"interact_object(action={sim_action!r}, object_id={resolved_id})",
            status="completed" if ok else "failed",
            message=msg,
            primitive_steps=steps,
            failed_steps=failed,
        )

    def _execute_single_interaction(
        self,
        sim_action: str,
        resolved_id: int,
        *,
        allow_navigation: bool = True,
    ) -> tuple[int, int, bool, str]:
        memory = self.executor.memory
        steps = 0
        failed = 0

        pos = np.asarray((memory.agent_position or (0.0, 0.0, 0.0))[:2])
        node = memory.objects[resolved_id]
        box_dist = _box_distance_2d(pos, node.bounds)
        aligned = _is_fov_aligned(pos, node.centroid[:2])
        mask_info = memory.select_interaction_mask(resolved_id, action=sim_action)
        if allow_navigation and (
            mask_info is None or box_dist > 0.90 or (box_dist > 0.55 and not aligned)
        ):
            nav_steps, nav_fail, _, _ = self._execute_navigation(
                object_id=resolved_id, region=None, max_steps=32
            )
            steps += nav_steps
            failed += nav_fail
            resolved_id = memory.resolve_object_id(resolved_id)

        if not self.executor.can_step():
            return (
                steps,
                failed,
                False,
                "Action or failure budget exhausted before interaction.",
            )

        if memory.select_interaction_mask(resolved_id, action=sim_action) is None:
            o_steps, o_fail = self._auto_frame_object(resolved_id, action=sim_action)
            steps += o_steps
            failed += o_fail
            resolved_id = memory.resolve_object_id(resolved_id)

        mask_sel = memory.select_interaction_mask(resolved_id, action=sim_action)
        if mask_sel is None:
            alt_hint = self._format_alternative_candidates_hint(resolved_id, sim_action)
            return (
                steps,
                failed,
                False,
                f"Target #{resolved_id} ({memory.objects[resolved_id].category}) has no detection mask in current view after framing.{alt_hint}",
            )

        det_idx = mask_sel[0]
        det_cat: str | None = None
        if self.executor._predicted is not None and 0 <= det_idx < len(
            self.executor._predicted.detections
        ):
            det_cat = self.executor._predicted.detections[det_idx].category

        ok = self.executor.step_primitive(sim_action, object_id=resolved_id)
        steps += 1
        if ok:
            self.executor.completed_macros.add(sim_action)
            resolved_id = memory.resolve_object_id(resolved_id)
            node = memory.objects[resolved_id]
            if (
                sim_action == "OpenObject"
                and node.category in _OPENABLE_CATEGORIES
                and self.executor.can_step()
                and self.executor.current_observation.horizon_degrees < 30.0 - 1e-3
            ):
                p_steps, p_fail = self._adjust_pitch_to(30.0)
                steps += p_steps
                failed += p_fail
                resolved_id = memory.resolve_object_id(resolved_id)
                node = memory.objects[resolved_id]
            top_ev = dict(
                sorted(node.category_evidence.items(), key=lambda kv: (-kv[1], kv[0]))[
                    :3
                ]
            )
            mismatch_note = ""
            if (
                sim_action == "PickupObject"
                and det_cat is not None
                and det_cat.casefold() != node.category.casefold()
            ):
                mismatch_note = (
                    f" (note: current-frame detector labeled mask {det_cat!r}, "
                    f"evidence={top_ev})"
                )
            elif len(top_ev) > 1:
                mismatch_note = f" (evidence={top_ev})"
            return (
                steps,
                failed,
                True,
                f"{sim_action} on #{resolved_id} ({node.category}) succeeded.{mismatch_note}",
            )
        failed += 1
        first_err = self.executor.current_observation.last_action_error

        resolved_id = memory.resolve_object_id(resolved_id)
        node = memory.objects[resolved_id]
        is_invalid_target_err = any(
            phrase in first_err
            for phrase in (
                "must have the property CanPickup",
                "NOT a receptacle",
                "object not found: object",
            )
        )
        if is_invalid_target_err:
            node.action_states[f"invalid_{sim_action}"] = ActionState(
                True, memory._latest_frame_id
            )
            if (
                sim_action == "PickupObject"
                and "must have the property CanPickup" in first_err
                and self.executor.can_step()
            ):
                alt_pick = self._find_pickupable_near(
                    resolved_id, expected_id=resolved_id
                )
                if alt_pick is not None and alt_pick != resolved_id:
                    if (
                        memory.select_interaction_mask(alt_pick, action="PickupObject")
                        is not None
                    ):
                        retry_ok = self.executor.step_primitive(
                            "PickupObject", object_id=alt_pick
                        )
                        steps += 1
                        if retry_ok:
                            self.executor.completed_macros.add("PickupObject")
                            alt_node = memory.objects[
                                memory.resolve_object_id(alt_pick)
                            ]
                            return (
                                steps,
                                failed,
                                True,
                                f"PickupObject on #{alt_node.object_id} ({alt_node.category}) succeeded after #{resolved_id} was non-pickupable.",
                            )
                        failed += 1
            alt_hint = self._format_alternative_candidates_hint(resolved_id, sim_action)
            return (
                steps,
                failed,
                False,
                f"{sim_action} on #{resolved_id} ({node.category}) failed: {first_err}.{alt_hint}",
            )

        if (
            sim_action == "PutObject"
            and node.category in _OPENABLE_CATEGORIES
            and not node.action_states.get("open", None)
            and self.executor.can_step()
        ):
            if (
                memory.select_interaction_mask(resolved_id, action="OpenObject")
                is not None
            ):
                open_ok = self.executor.step_primitive(
                    "OpenObject", object_id=resolved_id
                )
                steps += 1
                if open_ok and self.executor.can_step():
                    self.executor.completed_macros.add("OpenObject")
                    put_ok = self.executor.step_primitive(
                        "PutObject", object_id=resolved_id
                    )
                    steps += 1
                    if put_ok:
                        self.executor.completed_macros.add("PutObject")
                        return (
                            steps,
                            failed,
                            True,
                            f"Auto-opened #{resolved_id} ({node.category}) and succeeded with PutObject.",
                        )
                    failed += 1
                elif not open_ok:
                    failed += 1

        if self.executor.can_step() and "Bad interact mask" not in first_err:
            pos = np.asarray((memory.agent_position or (0.0, 0.0, 0.0))[:2])
            box_dist = _box_distance_2d(pos, node.bounds)
            cur_h = self.executor.current_observation.horizon_degrees
            if sim_action == "PutObject" and (
                "No valid positions" in first_err
                or node.category
                in {
                    "Fridge",
                    "Cabinet",
                    "Shelf",
                    "Microwave",
                    "CounterTop",
                    "DiningTable",
                    "Desk",
                    "SinkBasin",
                }
            ):
                adj_actions = (
                    ["LookUp", "LookDown", "RotateLeft"]
                    if cur_h >= 30.0
                    else ["LookDown", "LookUp", "RotateRight"]
                )
            elif box_dist > 0.50:
                adj_actions = ["MoveAhead"]
            else:
                adj_actions = ["LookDown" if cur_h < 30.0 else "LookUp"]
            for adj_action in adj_actions:
                if not self.executor.can_step():
                    break
                adj_ok = self.executor.step_primitive(adj_action, object_id=None)
                steps += 1
                if not adj_ok:
                    failed += 1
                    continue
                resolved_id = memory.resolve_object_id(resolved_id)
                if (
                    self.executor.can_step()
                    and memory.select_interaction_mask(resolved_id, action=sim_action)
                    is not None
                ):
                    retry_ok = self.executor.step_primitive(
                        sim_action, object_id=resolved_id
                    )
                    steps += 1
                    if retry_ok:
                        self.executor.completed_macros.add(sim_action)
                        return (
                            steps,
                            failed,
                            True,
                            f"{sim_action} on #{resolved_id} ({node.category}) succeeded after micro-adjustment ({adj_action}).",
                        )
                    failed += 1

        if "Bad interact mask" in first_err:
            node.action_states[f"invalid_{sim_action}"] = ActionState(
                True, memory._latest_frame_id
            )
        alt_hint = self._format_alternative_candidates_hint(resolved_id, sim_action)
        return (
            steps,
            failed,
            False,
            f"{sim_action} on #{resolved_id} ({node.category}) failed: {first_err or self.executor.current_observation.last_action_error}.{alt_hint}",
        )

    def _run_compound_interaction(
        self, macro_name: str, receptacle_id: int
    ) -> tuple[int, int, bool, str]:
        memory = self.executor.memory
        held_id = memory.held_object_id
        if held_id is None:
            return (
                0,
                0,
                False,
                f"{macro_name} requires holding an object in hand first (held_object_id is None).",
            )
        total_steps = 0
        total_failed = 0

        if macro_name == "clean_in_sink":
            put_frame = memory._latest_frame_id
            s, f, ok, msg = self._execute_single_interaction("PutObject", receptacle_id)
            total_steps += s
            total_failed += f
            if not ok and self.executor.can_step():
                alt_sink = self._find_nearest_category(
                    receptacle_id,
                    ("SinkBasin", "BathtubBasin"),
                    exclude_ids={memory.resolve_object_id(receptacle_id)},
                )
                if alt_sink is not None and alt_sink != memory.resolve_object_id(
                    receptacle_id
                ):
                    s2, f2, ok, msg = self._execute_single_interaction(
                        "PutObject", alt_sink, allow_navigation=False
                    )
                    total_steps += s2
                    total_failed += f2
                    if ok:
                        receptacle_id = alt_sink
            if not ok:
                return (
                    total_steps,
                    total_failed,
                    False,
                    f"clean_in_sink failed at PutObject: {msg}",
                )

            faucet_id = self._find_nearest_category(receptacle_id, ("Faucet",))
            if (
                faucet_id is None
                and self.executor.can_step()
                and self.executor.current_observation.horizon_degrees >= 15.0
            ):
                up_ok = self.executor.step_primitive("LookUp", object_id=None)
                total_steps += 1
                if not up_ok:
                    total_failed += 1
                faucet_id = self._find_nearest_category(receptacle_id, ("Faucet",))
            if faucet_id is None:
                faucet_id = self._find_nearest_category(
                    receptacle_id, ("Faucet", "Sink", "SinkBasin")
                )
            if faucet_id is None:
                return (
                    total_steps,
                    total_failed,
                    False,
                    "Placed object in sink, but no Faucet was detected nearby to toggle.",
                )
            for toggle_act in ("ToggleObjectOn", "ToggleObjectOff"):
                s, f, ok, msg = self._execute_single_interaction(
                    toggle_act, faucet_id, allow_navigation=False
                )
                total_steps += s
                total_failed += f
                if not ok:
                    return (
                        total_steps,
                        total_failed,
                        False,
                        f"clean_in_sink failed at {toggle_act}(#{faucet_id}): {msg}",
                    )

            target_pickup = memory.resolve_object_id(held_id)
            s, f, ok, msg = self._execute_single_interaction(
                "PickupObject", target_pickup, allow_navigation=False
            )
            total_steps += s
            total_failed += f
            if not ok and self.executor.can_step():
                alt_pick = self._find_pickupable_near(
                    receptacle_id, expected_id=held_id, min_frame_id=put_frame
                )
                if alt_pick is not None and alt_pick != target_pickup:
                    s2, f2, ok, msg = self._execute_single_interaction(
                        "PickupObject", alt_pick, allow_navigation=False
                    )
                    total_steps += s2
                    total_failed += f2
            if ok:
                self.executor.completed_macros.add("clean_in_sink")
            return (
                total_steps,
                total_failed,
                ok,
                (
                    f"Cleaned object in sink and picked #{memory.held_object_id or target_pickup} back up."
                    if ok
                    else f"Cleaned object in sink, but failed to pick #{target_pickup} back up: {msg}"
                ),
            )

        put_frame = memory._latest_frame_id
        if macro_name == "heat_in_microwave":
            seq = [
                ("OpenObject", receptacle_id, False),
                ("PutObject", receptacle_id, True),
                ("CloseObject", receptacle_id, True),
                ("ToggleObjectOn", receptacle_id, True),
                ("ToggleObjectOff", receptacle_id, True),
                ("OpenObject", receptacle_id, True),
                ("PickupObject", held_id, True),
            ]
            fallback_cats: tuple[str, ...] = ("Microwave",)
        else:  # cool_in_fridge
            seq = [
                ("OpenObject", receptacle_id, False),
                ("PutObject", receptacle_id, True),
                ("CloseObject", receptacle_id, True),
                ("OpenObject", receptacle_id, True),
                ("PickupObject", held_id, True),
            ]
            fallback_cats = ("Fridge", "Cabinet", "Shelf")

        arrived_at_appliance = False
        current_receptacle_id = receptacle_id
        for sim_act, target_id, mandatory in seq:
            actual_target = (
                current_receptacle_id if target_id == receptacle_id else target_id
            )
            resolved_target = memory.resolve_object_id(actual_target)
            if (
                sim_act == "OpenObject"
                and not mandatory
                and memory.objects[resolved_target].action_states.get("open")
                is not None
                and memory.objects[resolved_target].action_states["open"].value
            ):
                continue
            s, f, ok, msg = self._execute_single_interaction(
                sim_act,
                resolved_target,
                allow_navigation=not arrived_at_appliance,
            )
            total_steps += s
            total_failed += f
            if ok:
                arrived_at_appliance = True
            elif sim_act == "PickupObject" and self.executor.can_step():
                alt_pick = self._find_pickupable_near(
                    current_receptacle_id,
                    expected_id=held_id,
                    min_frame_id=put_frame,
                )
                if alt_pick is not None and alt_pick != resolved_target:
                    s2, f2, ok, msg = self._execute_single_interaction(
                        "PickupObject", alt_pick, allow_navigation=False
                    )
                    total_steps += s2
                    total_failed += f2
            elif (
                sim_act in {"OpenObject", "PutObject", "CloseObject", "ToggleObjectOn"}
                and self.executor.can_step()
            ):
                alt_app = self._find_nearest_category(
                    current_receptacle_id,
                    fallback_cats,
                    exclude_ids={resolved_target},
                    max_distance_m=0.85,
                )
                if alt_app is not None and alt_app != resolved_target:
                    s2, f2, ok, msg = self._execute_single_interaction(
                        sim_act, alt_app, allow_navigation=False
                    )
                    total_steps += s2
                    total_failed += f2
                    if ok:
                        current_receptacle_id = alt_app
                        arrived_at_appliance = True
            if not ok and mandatory:
                return (
                    total_steps,
                    total_failed,
                    False,
                    f"{macro_name} stopped at {sim_act}(#{resolved_target}): {msg}",
                )

        self.executor.completed_macros.add(macro_name)
        return (
            total_steps,
            total_failed,
            True,
            f"Completed {macro_name} with receptacle #{current_receptacle_id} and retrieved item #{memory.held_object_id or memory.resolve_object_id(held_id)}.",
        )

    def _find_pickupable_near(
        self,
        anchor_id: int,
        *,
        expected_id: int | None = None,
        min_frame_id: int = -1,
    ) -> int | None:
        """Find a currently visible pickupable object inside or near ``anchor_id``."""
        from .memory import (
            _COMPATIBLE_FALLBACK_FAMILIES,
            _STATIONARY_RECEPTACLE_CATEGORIES,
        )

        memory = self.executor.memory
        anchor = memory.objects[memory.resolve_object_id(anchor_id)]
        expected_node = (
            memory.objects.get(memory.resolve_object_id(expected_id))
            if expected_id is not None
            else None
        )
        candidates: list[tuple[int, int, float, float, int]] = []
        for obj_id in memory.current_detection_objects.values():
            resolved = memory.resolve_object_id(obj_id)
            if resolved not in memory.objects:
                continue
            node = memory.objects[resolved]
            if (
                node.category in _STATIONARY_RECEPTACLE_CATEGORIES
                or node.category == "Faucet"
                or node.action_states.get("invalid_PickupObject") is not None
            ):
                continue
            dist = float(np.linalg.norm(node.centroid - anchor.centroid))
            if dist > 0.70:
                continue
            id_rank = (
                0 if (expected_node and resolved == expected_node.object_id) else 1
            )
            cat_rank = 1
            vis_sim = 0.0
            if expected_node is not None:
                vis_sim = feature_cosine(
                    node.image_feature, expected_node.image_feature
                )
                same_cat = (
                    node.category.casefold() == expected_node.category.casefold()
                    or expected_node.category in node.category_evidence
                    or node.category in expected_node.category_evidence
                    or any(
                        expected_node.category in fam and node.category in fam
                        for fam in _COMPATIBLE_FALLBACK_FAMILIES
                    )
                )
                if same_cat:
                    cat_rank = 0
                elif node.first_seen >= min_frame_id and min_frame_id >= 0:
                    cat_rank = 1
                elif vis_sim < 0.65:
                    # Reject pre-existing unrelated objects (e.g. a Tomato already
                    # sitting in the sink when retrieving a Spatula).
                    continue
            candidates.append((id_rank, cat_rank, -vis_sim, dist, resolved))
        if candidates:
            candidates.sort()
            return candidates[0][4]
        return None

    def _find_nearest_category(
        self,
        anchor_id: int,
        categories: Sequence[str],
        *,
        exclude_ids: set[int] | None = None,
        max_distance_m: float = 1.5,
    ) -> int | None:
        memory = self.executor.memory
        anchor = memory.objects[memory.resolve_object_id(anchor_id)]
        excluded = exclude_ids or set()
        for cat in categories:
            matches = [
                node
                for node in memory.objects.values()
                if node.object_id not in excluded
                and node.category.casefold() == cat.casefold()
                and float(np.linalg.norm(node.centroid - anchor.centroid))
                <= max_distance_m
            ]
            if matches:
                best = min(
                    matches,
                    key=lambda node: float(
                        np.linalg.norm(node.centroid - anchor.centroid)
                    ),
                )
                return best.object_id
        return None

    def _format_alternative_candidates_hint(
        self, failed_id: int, sim_action: str
    ) -> str:
        from .memory import _COMPATIBLE_FALLBACK_FAMILIES

        memory = self.executor.memory
        if failed_id not in memory.objects:
            return ""
        target = memory.objects[failed_id]
        agent_pos = np.asarray(memory.agent_position or (0.0, 0.0, 0.0))
        compatible_cats = {target.category.casefold()}
        for fam in _COMPATIBLE_FALLBACK_FAMILIES:
            if target.category in fam:
                compatible_cats.update(c.casefold() for c in fam)
        alts: list[tuple[int, float, str]] = []
        for oid, node in sorted(memory.objects.items()):
            if (
                oid == failed_id
                or oid == memory.held_object_id
                or node.action_states.get(f"invalid_{sim_action}") is not None
            ):
                continue
            has_cat = node.category.casefold() in compatible_cats or any(
                ev.casefold() in compatible_cats for ev in node.category_evidence
            )
            if not has_cat:
                continue
            exact = 0 if node.category.casefold() == target.category.casefold() else 1
            dist = round(float(np.linalg.norm(node.centroid - agent_pos)), 2)
            reg = memory.world_to_region(
                float(node.centroid[0]), float(node.centroid[1])
            )
            alts.append(
                (exact, dist, f"#{oid} {node.category}@[{reg[0]},{reg[1]}]({dist}m)")
            )
        if not alts:
            return ""
        alts.sort()
        formatted = ", ".join(item[2] for item in alts[:4])
        return f" Alternative candidates in memory: [{formatted}] — try another candidate instead of repeating #{failed_id}."

    def _find_newly_visible_category(
        self, before: RevisionWatermark, stop_cats: set[str]
    ) -> int | None:
        memory = self.executor.memory
        for det_idx, obj_id in memory.current_detection_objects.items():
            resolved = memory.resolve_object_id(obj_id)
            if resolved in memory.objects:
                node = memory.objects[resolved]
                labels = [
                    node.category.casefold(),
                    *(ev.casefold() for ev in node.category_evidence),
                ]
                matched = any(
                    sc == lbl or (len(sc) >= 3 and (sc in lbl or lbl in sc))
                    for sc in stop_cats
                    for lbl in labels
                )
                if matched and (
                    resolved not in before.objects
                    or int(memory.current_detection_masks[det_idx].sum()) >= 40
                ):
                    return resolved
        return None

    def _build_delta(
        self,
        before: RevisionWatermark,
        *,
        macro_action: str,
        status: str,
        message: str,
        primitive_steps: int,
        failed_steps: int,
    ) -> dict[str, object]:
        annotated_path, bev_path = self.executor.render_latest_artifacts()
        obs = self.executor.current_observation
        return compute_scene_delta(
            before,
            self.executor.memory,
            macro_action=macro_action,
            status=status,
            message=message,
            primitive_steps=primitive_steps,
            failed_steps=failed_steps,
            yaw_degrees=obs.yaw_degrees,
            horizon_degrees=obs.horizon_degrees,
            annotated_rgb_path=annotated_path,
            bev_path=bev_path,
            remaining_actions=self.executor.config.max_actions - self.executor.actions,
            remaining_failures=self.executor.config.max_failures
            - self.executor.failures,
            remaining_decisions=max(
                0, self.executor.config.max_decisions - self.executor.decisions
            ),
        )


__all__ = ["MacroActionController", "format_visible_objects_now"]
