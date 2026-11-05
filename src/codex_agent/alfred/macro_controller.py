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
    navigation_step,
    next_navigation_action,
    rotation_towards,
)

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
        max_steps: int = 32,
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
        max_steps: int = 32,
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
                # If a forward move was blocked, allow FMM to replan around the new collision cell.
                if failed >= 3:
                    break

        if resolved_id is not None and resolved_id in memory.objects:
            node = memory.objects[resolved_id]
            o_steps, o_fail = self._auto_frame_object(resolved_id)
            steps += o_steps
            failed += o_fail
            mask_match = memory.select_interaction_mask(
                resolved_id, action="PickupObject"
            )
            pos = memory.agent_position or (0.0, 0.0, 0.0)
            dist = float(np.linalg.norm(node.centroid[:2] - np.asarray(pos[:2])))
            if mask_match is not None:
                return (
                    steps,
                    failed,
                    "arrived_and_framed",
                    f"Arrived near #{resolved_id} ({node.category}, horizontal_dist={dist:.2f}m) and verified interaction mask in current view.",
                )
            return (
                steps,
                failed,
                "arrived_unverified_mask" if dist <= 1.25 else "partial_approach",
                f"{last_feedback} (horizontal_dist={dist:.2f}m to #{resolved_id} {node.category}).",
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
        """Plan FMM step with two-stage adaptive standoff when close floor is unobserved."""
        memory = self.executor.memory
        if resolved_id is None:
            return next_navigation_action(
                memory,
                yaw_degrees=yaw_degrees,
                horizon_degrees=horizon_degrees,
                object_id=None,
                cell=region,
            )

        node = memory.objects[resolved_id]
        spatial_map = memory.spatial_map
        position = spatial_map.agent_position
        if position is None:
            return None, "Navigation requires an integrated observation."

        raster = spatial_map.navigation_raster()
        rows, cols = np.nonzero(raster.traversable)
        available = [
            raster.cell(int(row), int(col)) for row, col in zip(rows, cols, strict=True)
        ]
        bounds = node.bounds

        def _cells_in_band(low_m: float, high_m: float) -> set[GridCell]:
            selected: set[GridCell] = set()
            for candidate in available:
                point = np.asarray(spatial_map.cell_center(candidate))
                nearest = np.clip(point, bounds[0, :2], bounds[1, :2])
                distance = float(np.linalg.norm(point - nearest))
                if low_m <= distance <= high_m:
                    selected.add(candidate)
            return selected

        close_goals = _cells_in_band(0.30, 0.85)
        action, feedback = navigation_step(
            spatial_map, close_goals, yaw_degrees=yaw_degrees
        )
        if action is not None or feedback.startswith("Reached"):
            return action, feedback

        # Two-stage fallback: if the 0.30-0.85m band is still unobserved/unreachable,
        # step to the closest reachable cell in 0.85-1.65m and look down to reveal the floor.
        mid_goals = _cells_in_band(0.85, 1.65)
        if mid_goals:
            start = spatial_map.world_to_cell(*position[:2])
            if start in mid_goals and horizon_degrees < 30.0:
                return (
                    "LookDown",
                    "Tilting camera down to observe floor clearance in front of target.",
                )
            mid_action, mid_feedback = navigation_step(
                spatial_map, mid_goals, yaw_degrees=yaw_degrees
            )
            if mid_action is not None:
                return (
                    mid_action,
                    "Approaching intermediate stance to reveal floor near target.",
                )
            if mid_feedback.startswith("Reached") and horizon_degrees < 30.0:
                return (
                    "LookDown",
                    "Reached intermediate stance; tilting camera down to observe floor clearance.",
                )

        return action, feedback

    def _auto_frame_object(self, resolved_id: int) -> tuple[int, int]:
        """Orient camera toward ``resolved_id`` and probe pitch if mask is missing."""
        memory = self.executor.memory
        if resolved_id not in memory.objects:
            return 0, 0
        node = memory.objects[resolved_id]
        steps, failed = self._orient_towards_point(node.centroid)
        if (
            memory.select_interaction_mask(resolved_id, action="PickupObject")
            is not None
        ):
            return steps, failed

        # Try one adjacent pitch adjustment (+15 or -15 deg) if target mask was not detected.
        current_h = self.executor.current_observation.horizon_degrees
        probe_action = "LookDown" if current_h < 30.0 else "LookUp"
        if self.executor.can_step():
            ok = self.executor.step_primitive(probe_action, object_id=None)
            steps += 1
            if not ok:
                failed += 1
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
        self, sim_action: str, resolved_id: int
    ) -> tuple[int, int, bool, str]:
        memory = self.executor.memory
        steps = 0
        failed = 0

        # Auto-approach if object is not currently visible or is too far away (> 1.15m).
        pos = memory.agent_position or (0.0, 0.0, 0.0)
        node = memory.objects[resolved_id]
        horiz_dist = float(np.linalg.norm(node.centroid[:2] - np.asarray(pos[:2])))
        mask_info = memory.select_interaction_mask(resolved_id, action=sim_action)
        if mask_info is None or horiz_dist > 1.25:
            nav_steps, nav_fail, _, _ = self._execute_navigation(
                object_id=resolved_id, region=None, max_steps=24
            )
            steps += nav_steps
            failed += nav_fail
            resolved_id = memory.resolve_object_id(resolved_id)

        # Attempt 1
        if not self.executor.can_step():
            return (
                steps,
                failed,
                False,
                "Action or failure budget exhausted before interaction.",
            )

        if memory.select_interaction_mask(resolved_id, action=sim_action) is None:
            o_steps, o_fail = self._auto_frame_object(resolved_id)
            steps += o_steps
            failed += o_fail

        if memory.select_interaction_mask(resolved_id, action=sim_action) is None:
            return (
                steps,
                failed,
                False,
                f"Target #{resolved_id} ({memory.objects[resolved_id].category}) has no detection mask in current view after framing.",
            )

        ok = self.executor.step_primitive(sim_action, object_id=resolved_id)
        steps += 1
        if ok:
            return (
                steps,
                failed,
                True,
                f"{sim_action} on #{resolved_id} ({memory.objects[resolved_id].category}) succeeded.",
            )
        failed += 1
        first_err = self.executor.current_observation.last_action_error

        # 1-Step Micro-Recovery: if PutObject failed on a closed openable container, try OpenObject first!
        node = memory.objects[resolved_id]
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
                    put_ok = self.executor.step_primitive(
                        "PutObject", object_id=resolved_id
                    )
                    steps += 1
                    if put_ok:
                        return (
                            steps,
                            failed,
                            True,
                            f"Auto-opened #{resolved_id} ({node.category}) and succeeded with PutObject.",
                        )
                    failed += 1
                elif not open_ok:
                    failed += 1

        # Distance / pitch micro-adjustment retry if target was out of reach or poorly angled
        if self.executor.can_step():
            pos = memory.agent_position or (0.0, 0.0, 0.0)
            horiz_dist = float(np.linalg.norm(node.centroid[:2] - np.asarray(pos[:2])))
            adj_action = (
                "MoveAhead"
                if horiz_dist > 0.65
                else (
                    "LookDown"
                    if self.executor.current_observation.horizon_degrees < 30.0
                    else "LookUp"
                )
            )
            adj_ok = self.executor.step_primitive(adj_action, object_id=None)
            steps += 1
            if not adj_ok:
                failed += 1
            elif (
                self.executor.can_step()
                and memory.select_interaction_mask(resolved_id, action=sim_action)
                is not None
            ):
                retry_ok = self.executor.step_primitive(
                    sim_action, object_id=resolved_id
                )
                steps += 1
                if retry_ok:
                    return (
                        steps,
                        failed,
                        True,
                        f"{sim_action} on #{resolved_id} ({node.category}) succeeded after micro-adjustment ({adj_action}).",
                    )
                failed += 1

        return (
            steps,
            failed,
            False,
            f"{sim_action} on #{resolved_id} ({node.category}) failed: {first_err or self.executor.current_observation.last_action_error}",
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
            s, f, ok, msg = self._execute_single_interaction("PutObject", receptacle_id)
            total_steps += s
            total_failed += f
            if not ok:
                return (
                    total_steps,
                    total_failed,
                    False,
                    f"clean_in_sink failed at PutObject: {msg}",
                )

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
                s, f, ok, msg = self._execute_single_interaction(toggle_act, faucet_id)
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
                "PickupObject", target_pickup
            )
            total_steps += s
            total_failed += f
            return (
                total_steps,
                total_failed,
                ok,
                (
                    "Cleaned object in sink and picked it back up."
                    if ok
                    else f"Cleaned object in sink, but failed to pick #{target_pickup} back up: {msg}"
                ),
            )

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
        else:  # cool_in_fridge
            seq = [
                ("OpenObject", receptacle_id, False),
                ("PutObject", receptacle_id, True),
                ("CloseObject", receptacle_id, True),
                ("OpenObject", receptacle_id, True),
                ("PickupObject", held_id, True),
            ]

        for sim_act, target_id, mandatory in seq:
            resolved_target = memory.resolve_object_id(target_id)
            if (
                sim_act == "OpenObject"
                and not mandatory
                and memory.objects[resolved_target].action_states.get("open")
                is not None
                and memory.objects[resolved_target].action_states["open"].value
            ):
                continue
            s, f, ok, msg = self._execute_single_interaction(sim_act, resolved_target)
            total_steps += s
            total_failed += f
            if not ok and mandatory:
                return (
                    total_steps,
                    total_failed,
                    False,
                    f"{macro_name} stopped at {sim_act}(#{resolved_target}): {msg}",
                )

        return (
            total_steps,
            total_failed,
            True,
            f"Completed {macro_name} with receptacle #{receptacle_id} and retrieved item #{memory.resolve_object_id(held_id)}.",
        )

    def _find_nearest_category(
        self, anchor_id: int, categories: Sequence[str]
    ) -> int | None:
        memory = self.executor.memory
        anchor = memory.objects[memory.resolve_object_id(anchor_id)]
        for cat in categories:
            matches = [
                node
                for node in memory.objects.values()
                if node.category.casefold() == cat.casefold()
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

    def _find_newly_visible_category(
        self, before: RevisionWatermark, stop_cats: set[str]
    ) -> int | None:
        memory = self.executor.memory
        for det_idx, obj_id in memory.current_detection_objects.items():
            resolved = memory.resolve_object_id(obj_id)
            if resolved in memory.objects:
                node = memory.objects[resolved]
                if node.category.casefold() in stop_cats and (
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
        )


__all__ = ["MacroActionController", "format_visible_objects_now"]
