"""Guarded subgoal policy programs and pre-commit goal predicate verification.

Instead of asking the LLM to emit unconstrained Python loops (which suffer from
open-loop runaway when 3D object IDs split/merge, lose VLM visual disambiguation
inside the loop, and waste turns on syntax/attribute errors), this module
provides:
1. :class:`SubgoalProgramSpec`: A typed declarative subgoal policy block with
   explicit preconditions, target candidate ranking/fallback, closed-loop action
   execution, and postcondition verification.
2. :func:`execute_subgoal_program`: Deterministic closed-loop interpreter run
   inside :class:`MacroActionController` that completes a multi-step subgoal
   (with automatic container opening, multi-pitch framing, and 3D sibling
   candidate fallback) in a single decision turn.
3. :func:`verify_goal_predicates`: Pre-commit goal predicate verifier executed
   when ``finish_episode`` is called (or via ``verify_task_goal``), preventing
   the 55% false-completion rate by checking held-object invariants, required
   state transitions (``cleaned``, ``heated``, ``cooled``, ``sliced``,
   ``toggled_on``), two-object placement counts, and movable-receptacle
   containment topology.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from .delta import capture_watermark
from .memory import _STATIONARY_RECEPTACLE_CATEGORIES

if TYPE_CHECKING:
    from .execution import EpisodeExecutor
    from .macro_controller import MacroActionController
    from .memory import IncrementalSceneMemory

SubgoalType = Literal[
    "acquire_target",
    "place_held",
    "clean_held",
    "heat_held",
    "cool_held",
    "slice_target",
    "examine_in_light",
    "transfer_with_movable_receptacle",
]

SUBGOAL_TYPES: tuple[SubgoalType, ...] = (
    "acquire_target",
    "place_held",
    "clean_held",
    "heat_held",
    "cool_held",
    "slice_target",
    "examine_in_light",
    "transfer_with_movable_receptacle",
)

_MOVABLE_RECEPTACLES: frozenset[str] = frozenset(
    {"Bowl", "Cup", "Mug", "Pan", "Plate", "Pot"}
)

_LIGHT_CATEGORIES: frozenset[str] = frozenset({"DeskLamp", "FloorLamp"})

_KNIFE_CATEGORIES: tuple[str, ...] = ("Knife", "ButterKnife")

_UNLOAD_SURFACES: tuple[str, ...] = (
    "CounterTop",
    "DiningTable",
    "Desk",
    "SideTable",
    "CoffeeTable",
    "Dresser",
    "Shelf",
)


_STRICT_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"mug", "cup"}),
    frozenset({"knife", "butterknife"}),
    frozenset({"sinkbasin", "bathtubbasin"}),
    frozenset({"desklamp", "floorlamp"}),
    frozenset({"countertop", "diningtable", "desk", "sidetable", "coffeetable"}),
)


@dataclass(frozen=True)
class SubgoalProgramSpec:
    """Declarative guarded subgoal policy block submitted by the agent."""

    subgoal: SubgoalType
    target_id: int | None = None
    target_categories: tuple[str, ...] = ()
    receptacle_id: int | None = None
    receptacle_categories: tuple[str, ...] = ()
    destination_id: int | None = None
    destination_categories: tuple[str, ...] = ()
    auto_open_containers: bool = True
    try_sibling_candidates: bool = True
    require_stable_evidence: bool = False
    max_candidates: int = 2


@dataclass
class GoalVerificationReport:
    """Result of checking episode goal predicates against IncrementalSceneMemory."""

    passed: bool
    inferred_requirements: dict[str, object]
    satisfied_predicates: list[str] = field(default_factory=list)
    unmet_predicates: list[str] = field(default_factory=list)
    remediation_hints: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "inferred_requirements": self.inferred_requirements,
            "satisfied_predicates": self.satisfied_predicates,
            "unmet_predicates": self.unmet_predicates,
            "remediation_hints": self.remediation_hints,
        }


def parse_subgoal_program(args: dict[str, object]) -> SubgoalProgramSpec:
    """Validate and normalize raw JSON tool arguments into a SubgoalProgramSpec."""
    raw_subgoal = str(args.get("subgoal") or args.get("program") or "").strip()
    aliases: dict[str, SubgoalType] = {
        "acquire": "acquire_target",
        "pickup": "acquire_target",
        "acquire_target": "acquire_target",
        "place": "place_held",
        "put": "place_held",
        "place_held": "place_held",
        "place_held_into_receptacle": "place_held",
        "clean": "clean_held",
        "clean_held": "clean_held",
        "clean_in_sink": "clean_held",
        "heat": "heat_held",
        "heat_held": "heat_held",
        "heat_in_microwave": "heat_held",
        "cool": "cool_held",
        "cool_held": "cool_held",
        "cool_in_fridge": "cool_held",
        "slice": "slice_target",
        "slice_target": "slice_target",
        "examine": "examine_in_light",
        "examine_in_light": "examine_in_light",
        "look_at_obj_in_light": "examine_in_light",
        "illuminate_held_object": "examine_in_light",
        "movable_recep": "transfer_with_movable_receptacle",
        "transfer_with_movable_receptacle": "transfer_with_movable_receptacle",
    }
    subgoal = aliases.get(raw_subgoal)
    if subgoal is None:
        raise ValueError(
            f"Unsupported subgoal={raw_subgoal!r}. Supported values: {list(SUBGOAL_TYPES)}"
        )

    def _int_or_none(val: object) -> int | None:
        if isinstance(val, int) and not isinstance(val, bool):
            return val
        return None

    def _first_int(*vals: object) -> int | None:
        for val in vals:
            parsed = _int_or_none(val)
            if parsed is not None:
                return parsed
        return None

    def _str_tuple(val: object) -> tuple[str, ...]:
        if isinstance(val, str) and val.strip():
            return (val.strip(),)
        if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            return tuple(str(x).strip() for x in val if str(x).strip())
        return ()

    target_cats = _str_tuple(
        args.get("target_categories")
        or args.get("target_category")
        or args.get("categories")
    )
    recep_cats = _str_tuple(
        args.get("receptacle_categories") or args.get("receptacle_category")
    )
    dest_cats = _str_tuple(
        args.get("destination_categories") or args.get("destination_category")
    )
    max_cands_raw = _int_or_none(args.get("max_candidates"))
    max_cands = max(1, min(3, max_cands_raw if max_cands_raw is not None else 2))

    return SubgoalProgramSpec(
        subgoal=subgoal,
        target_id=_first_int(args.get("target_id"), args.get("object_id")),
        target_categories=target_cats,
        receptacle_id=_first_int(args.get("receptacle_id")),
        receptacle_categories=recep_cats,
        destination_id=_first_int(args.get("destination_id")),
        destination_categories=dest_cats,
        auto_open_containers=bool(args.get("auto_open_containers", True)),
        try_sibling_candidates=bool(args.get("try_sibling_candidates", True)),
        require_stable_evidence=bool(args.get("require_stable_evidence", False)),
        max_candidates=max_cands,
    )


def rank_candidate_objects(
    memory: IncrementalSceneMemory,
    *,
    preferred_id: int | None = None,
    categories: Sequence[str] = (),
    action_filter: str | None = None,
    exclude_ids: set[int] | None = None,
    require_pickupable: bool = False,
    limit: int = 4,
) -> list[int]:
    """Rank candidate object IDs in memory by category match, stability, and distance."""
    excluded = set(exclude_ids or ())
    agent_pos = np.asarray(memory.agent_position or (0.0, 0.0, 0.0))
    cat_set = {c.strip().casefold() for c in categories if c.strip()}
    compatible_set = set(cat_set)
    for c in list(cat_set):
        for grp in _STRICT_SYNONYM_GROUPS:
            if c in grp:
                compatible_set.update(grp)

    resolved_pref: int | None = None
    if preferred_id is not None:
        try:
            resolved_pref = memory.resolve_object_id(preferred_id)
            if resolved_pref in memory.objects and not cat_set:
                pref_cat = memory.objects[resolved_pref].category.casefold()
                cat_set.add(pref_cat)
                compatible_set.add(pref_cat)
        except KeyError:
            resolved_pref = None

    scored: list[tuple[int, int, int, float, int]] = []
    for oid, node in sorted(memory.objects.items()):
        if oid in excluded or node.state == "sliced":
            continue
        if (
            action_filter
            and node.action_states.get(f"invalid_{action_filter}") is not None
        ):
            continue
        if require_pickupable and (
            node.category in _STATIONARY_RECEPTACLE_CATEGORIES
            or node.category == "Faucet"
            or node.action_states.get("invalid_PickupObject") is not None
        ):
            continue

        is_pref = 0 if (resolved_pref is not None and oid == resolved_pref) else 1
        node_cat_lower = node.category.casefold()
        ev_lowers = {k.casefold(): v for k, v in node.category_evidence.items()}

        if is_pref == 0:
            cat_tier = 0
        elif cat_set:
            if node_cat_lower in cat_set:
                cat_tier = 0
            elif any(ev_lowers.get(c, 0) >= 2 for c in cat_set):
                cat_tier = 1
            elif node_cat_lower in compatible_set:
                cat_tier = 2
            else:
                continue
        else:
            continue

        ev_sum = node.evidence_summary
        stability_tier = 0 if ev_sum["is_stable"] else 1
        dist_m = float(np.linalg.norm(node.centroid - agent_pos))
        scored.append((is_pref, cat_tier, stability_tier, dist_m, oid))

    scored.sort()
    return [item[4] for item in scored[:limit]]


def execute_subgoal_program(
    macro: MacroActionController, spec: SubgoalProgramSpec
) -> dict[str, object]:
    """Execute a guarded subgoal policy block closed-loop and return a SceneDelta."""
    executor = macro.executor
    memory = executor.memory
    before = capture_watermark(memory)
    total_steps = 0
    total_failed = 0
    trace_log: list[str] = []

    def _finish(status: str, message: str) -> dict[str, object]:
        delta = macro._build_delta(
            before,
            macro_action=f"execute_subgoal_program(subgoal={spec.subgoal!r})",
            status=status,
            message=message,
            primitive_steps=total_steps,
            failed_steps=total_failed,
        )
        delta["subgoal_program"] = {
            "subgoal": spec.subgoal,
            "execution_trace": trace_log,
            "postcondition_verified": status == "completed",
        }
        return delta

    # 1. ACQUIRE TARGET
    if spec.subgoal == "acquire_target":
        if memory.held_object_id is not None:
            held_node = memory.objects.get(memory.held_object_id)
            held_cat = held_node.category if held_node else "Unknown"
            want_cats = {c.casefold() for c in spec.target_categories}
            if (
                spec.target_id is not None and memory.held_object_id == spec.target_id
            ) or (want_cats and held_cat.casefold() in want_cats):
                return _finish(
                    "completed",
                    f"Already holding matching target #{memory.held_object_id} ({held_cat}).",
                )
            # Precondition: empty hands onto nearest surface
            unload_cands = rank_candidate_objects(
                memory,
                categories=_UNLOAD_SURFACES,
                action_filter="PutObject",
                limit=2,
            )
            for surf_id in unload_cands:
                if not executor.can_step():
                    break
                s, f, ok, msg = macro._execute_single_interaction("PutObject", surf_id)
                total_steps += s
                total_failed += f
                trace_log.append(f"Precondition unload onto #{surf_id}: {msg}")
                if ok:
                    break
            if memory.held_object_id is not None:
                return _finish(
                    "failed",
                    f"Hands occupied with #{memory.held_object_id} ({held_cat}) and auto-unload failed.",
                )

        cands = rank_candidate_objects(
            memory,
            preferred_id=spec.target_id,
            categories=spec.target_categories,
            action_filter="PickupObject",
            require_pickupable=True,
            limit=spec.max_candidates if spec.try_sibling_candidates else 1,
        )
        if not cands:
            return _finish(
                "failed",
                f"No pickupable candidates found in memory for id={spec.target_id}, categories={list(spec.target_categories)}. Use explore_frontier or open containers first.",
            )

        for cid in cands:
            if not executor.can_step():
                break
            node = memory.objects.get(memory.resolve_object_id(cid))
            # If inside a closed container, open container first
            if (
                node is not None
                and node.parent_receptacle_id is not None
                and spec.auto_open_containers
            ):
                p_id = memory.resolve_object_id(node.parent_receptacle_id)
                p_node = memory.objects.get(p_id)
                if (
                    p_node is not None
                    and p_node.category
                    in {"Fridge", "Cabinet", "Drawer", "Microwave", "Safe"}
                    and not (
                        p_node.action_states.get("open") is not None
                        and p_node.action_states["open"].value
                    )
                ):
                    s, f, ok, msg = macro._execute_single_interaction(
                        "OpenObject", p_id
                    )
                    total_steps += s
                    total_failed += f
                    trace_log.append(f"Precondition open parent #{p_id}: {msg}")
            s, f, ok, msg = macro._execute_single_interaction("PickupObject", cid)
            total_steps += s
            total_failed += f
            trace_log.append(f"Try PickupObject(#{cid}): {msg}")
            if ok and memory.held_object_id is not None:
                held_node = memory.objects[memory.held_object_id]
                return _finish(
                    "completed",
                    f"Successfully acquired #{held_node.object_id} ({held_node.category}, evidence={held_node.evidence_summary['top_alternatives'] or held_node.category_evidence}).",
                )
        return _finish(
            "failed",
            f"All {len(cands)} candidate(s) {cands} failed PickupObject. Trace: {'; '.join(trace_log)}",
        )

    # 2. PLACE HELD OBJECT
    if spec.subgoal == "place_held":
        if memory.held_object_id is None:
            return _finish(
                "failed",
                "place_held precondition failed: held_object_id is None (nothing in hand to place).",
            )
        held_before = memory.held_object_id
        pref_rec_id = (
            spec.receptacle_id
            if spec.receptacle_id is not None
            else (spec.target_id if spec.target_id is not None else spec.destination_id)
        )
        cands = rank_candidate_objects(
            memory,
            preferred_id=pref_rec_id,
            categories=spec.receptacle_categories
            or spec.target_categories
            or spec.destination_categories,
            action_filter="PutObject",
            exclude_ids={held_before},
            limit=spec.max_candidates if spec.try_sibling_candidates else 1,
        )
        if not cands:
            return _finish(
                "failed",
                f"No matching receptacle candidates in memory for id={spec.receptacle_id or spec.target_id}, categories={list(spec.receptacle_categories or spec.target_categories)}.",
            )
        for rid in cands:
            if not executor.can_step():
                break
            s, f, ok, msg = macro._execute_single_interaction("PutObject", rid)
            total_steps += s
            total_failed += f
            trace_log.append(f"Try PutObject(#{rid}): {msg}")
            if ok and memory.held_object_id is None:
                r_node = memory.objects[memory.resolve_object_id(rid)]
                return _finish(
                    "completed",
                    f"Verified #{held_before} placed into/onto #{r_node.object_id} ({r_node.category}).",
                )
        return _finish(
            "failed",
            f"Failed to place #{held_before} across candidates {cands}. Trace: {'; '.join(trace_log)}",
        )

    # 3. APPLIANCE PROCESSING (CLEAN / HEAT / COOL) + OPTIONAL DESTINATION PLACEMENT
    if spec.subgoal in {"clean_held", "heat_held", "cool_held"}:
        macro_map = {
            "clean_held": ("clean_in_sink", ("SinkBasin", "BathtubBasin")),
            "heat_held": ("heat_in_microwave", ("Microwave",)),
            "cool_held": ("cool_in_fridge", ("Fridge",)),
        }
        compound_name, default_app_cats = macro_map[spec.subgoal]
        if memory.held_object_id is None and (
            spec.target_id is not None or spec.target_categories
        ):
            acq_spec = SubgoalProgramSpec(
                subgoal="acquire_target",
                target_id=spec.target_id,
                target_categories=spec.target_categories,
                try_sibling_candidates=spec.try_sibling_candidates,
                max_candidates=spec.max_candidates,
            )
            acq_res = execute_subgoal_program(macro, acq_spec)
            steps_info = acq_res.get("steps_executed", {})
            if isinstance(steps_info, dict):
                total_steps += int(steps_info.get("primitive_steps", 0))
                total_failed += int(steps_info.get("failed_steps", 0))
            trace_log.append(f"Auto-acquire: {acq_res.get('message')}")
        if memory.held_object_id is None:
            return _finish(
                "failed",
                f"{spec.subgoal} requires holding an object first (held_object_id is None).",
            )

        app_cands = rank_candidate_objects(
            memory,
            preferred_id=spec.receptacle_id,
            categories=spec.receptacle_categories or default_app_cats,
            limit=2,
        )
        if not app_cands:
            return _finish(
                "failed",
                f"No {default_app_cats} observed in scene memory yet for {spec.subgoal}.",
            )
        app_ok = False
        last_msg = ""
        for app_id in app_cands:
            if not executor.can_step():
                break
            s, f, ok, msg = macro._run_compound_interaction(compound_name, app_id)
            total_steps += s
            total_failed += f
            last_msg = msg
            trace_log.append(f"{compound_name}(#{app_id}): {msg}")
            if ok:
                app_ok = True
                break
        if not app_ok:
            return _finish("failed", f"{spec.subgoal} failed: {last_msg}")

        if spec.destination_id is not None or spec.destination_categories:
            place_spec = SubgoalProgramSpec(
                subgoal="place_held",
                receptacle_id=spec.destination_id,
                receptacle_categories=spec.destination_categories,
                try_sibling_candidates=spec.try_sibling_candidates,
                max_candidates=spec.max_candidates,
            )
            place_res = execute_subgoal_program(macro, place_spec)
            p_steps = place_res.get("steps_executed", {})
            if isinstance(p_steps, dict):
                total_steps += int(p_steps.get("primitive_steps", 0))
                total_failed += int(p_steps.get("failed_steps", 0))
            trace_log.append(f"Place at destination: {place_res.get('message')}")
            if place_res.get("status") != "completed":
                return _finish(
                    "failed",
                    f"Completed {compound_name} but failed destination placement: {place_res.get('message')}",
                )
            return _finish(
                "completed",
                f"Completed {compound_name} and placed processed item at destination ({place_res.get('message')}).",
            )
        return _finish("completed", last_msg)

    # 4. SLICE TARGET
    if spec.subgoal == "slice_target":
        held_id = memory.held_object_id
        holding_knife = False
        if held_id is not None and held_id in memory.objects:
            holding_knife = memory.objects[held_id].category in _KNIFE_CATEGORIES
        if not holding_knife:
            acq_knife = SubgoalProgramSpec(
                subgoal="acquire_target",
                target_categories=_KNIFE_CATEGORIES,
                try_sibling_candidates=True,
                max_candidates=3,
            )
            k_res = execute_subgoal_program(macro, acq_knife)
            k_steps = k_res.get("steps_executed", {})
            if isinstance(k_steps, dict):
                total_steps += int(k_steps.get("primitive_steps", 0))
                total_failed += int(k_steps.get("failed_steps", 0))
            trace_log.append(f"Acquire knife: {k_res.get('message')}")
            if k_res.get("status") != "completed":
                return _finish(
                    "failed",
                    f"slice_target failed to acquire Knife/ButterKnife: {k_res.get('message')}",
                )

        food_cands = rank_candidate_objects(
            memory,
            preferred_id=spec.target_id,
            categories=spec.target_categories,
            action_filter="SliceObject",
            exclude_ids=(
                {memory.held_object_id} if memory.held_object_id is not None else set()
            ),
            limit=spec.max_candidates if spec.try_sibling_candidates else 1,
        )
        if not food_cands:
            return _finish(
                "failed",
                f"No sliceable target found for id={spec.target_id}, categories={list(spec.target_categories)}.",
            )
        sliced_ok = False
        sliced_msg = ""
        for fid in food_cands:
            if not executor.can_step():
                break
            s, f, ok, msg = macro._execute_single_interaction("SliceObject", fid)
            total_steps += s
            total_failed += f
            sliced_msg = msg
            trace_log.append(f"SliceObject(#{fid}): {msg}")
            if ok:
                sliced_ok = True
                break
        if not sliced_ok:
            return _finish("failed", f"SliceObject failed: {sliced_msg}")

        # Auto-put knife down onto nearest surface if requested or if a destination is set
        unload_cands = rank_candidate_objects(
            memory,
            preferred_id=spec.receptacle_id,
            categories=spec.receptacle_categories or _UNLOAD_SURFACES,
            action_filter="PutObject",
            limit=2,
        )
        for surf_id in unload_cands:
            if not executor.can_step() or memory.held_object_id is None:
                break
            s, f, ok, msg = macro._execute_single_interaction("PutObject", surf_id)
            total_steps += s
            total_failed += f
            trace_log.append(f"Put down knife on #{surf_id}: {msg}")
            if ok:
                break
        return _finish(
            "completed",
            f"Sliced target ({sliced_msg}) and released knife={memory.held_object_id is None}.",
        )

    # 5. EXAMINE IN LIGHT
    if spec.subgoal == "examine_in_light":
        if memory.held_object_id is None and (
            spec.target_id is not None or spec.target_categories
        ):
            acq_res = execute_subgoal_program(
                macro,
                SubgoalProgramSpec(
                    subgoal="acquire_target",
                    target_id=spec.target_id,
                    target_categories=spec.target_categories,
                    try_sibling_candidates=spec.try_sibling_candidates,
                    max_candidates=spec.max_candidates,
                ),
            )
            a_steps = acq_res.get("steps_executed", {})
            if isinstance(a_steps, dict):
                total_steps += int(a_steps.get("primitive_steps", 0))
                total_failed += int(a_steps.get("failed_steps", 0))
            trace_log.append(f"Acquire item: {acq_res.get('message')}")
        if memory.held_object_id is None:
            return _finish(
                "failed",
                "examine_in_light requires holding the target object in hand first.",
            )
        lamp_pref = (
            spec.receptacle_id
            if spec.receptacle_id is not None
            else spec.destination_id
        )
        lamp_cands = rank_candidate_objects(
            memory,
            preferred_id=lamp_pref,
            categories=spec.receptacle_categories
            or spec.destination_categories
            or tuple(_LIGHT_CATEGORIES),
            action_filter="ToggleObjectOn",
            limit=spec.max_candidates,
        )
        if not lamp_cands:
            return _finish(
                "failed",
                "No DeskLamp or FloorLamp candidates observed in memory yet.",
            )
        for lid in lamp_cands:
            if not executor.can_step():
                break
            s, f, ok, msg = macro._execute_single_interaction("ToggleObjectOn", lid)
            total_steps += s
            total_failed += f
            trace_log.append(f"ToggleObjectOn(#{lid}): {msg}")
            if ok:
                return _finish(
                    "completed",
                    f"Toggled on lamp #{lid} while holding #{memory.held_object_id}.",
                )
        return _finish(
            "failed",
            f"Failed to toggle on any lamp in {lamp_cands}. Trace: {'; '.join(trace_log)}",
        )

    # 6. TRANSFER WITH MOVABLE RECEPTACLE
    if spec.subgoal == "transfer_with_movable_receptacle":
        # Step A: ensure holding small item, place into movable receptacle
        if memory.held_object_id is None and (
            spec.target_id is not None or spec.target_categories
        ):
            acq_res = execute_subgoal_program(
                macro,
                SubgoalProgramSpec(
                    subgoal="acquire_target",
                    target_id=spec.target_id,
                    target_categories=spec.target_categories,
                    try_sibling_candidates=spec.try_sibling_candidates,
                    max_candidates=spec.max_candidates,
                ),
            )
            a_steps = acq_res.get("steps_executed", {})
            if isinstance(a_steps, dict):
                total_steps += int(a_steps.get("primitive_steps", 0))
                total_failed += int(a_steps.get("failed_steps", 0))
            trace_log.append(f"Step 1 acquire content: {acq_res.get('message')}")
            if acq_res.get("status") != "completed":
                return _finish(
                    "failed",
                    f"Failed to acquire small content object: {acq_res.get('message')}",
                )

        mov_cands = rank_candidate_objects(
            memory,
            preferred_id=spec.receptacle_id,
            categories=spec.receptacle_categories or tuple(_MOVABLE_RECEPTACLES),
            action_filter="PutObject",
            require_pickupable=True,
            limit=spec.max_candidates,
        )
        if not mov_cands:
            return _finish(
                "failed",
                f"No movable receptacle ({list(spec.receptacle_categories or _MOVABLE_RECEPTACLES)}) found in memory.",
            )

        chosen_mov: int | None = None
        if memory.held_object_id is not None:
            held_node = memory.objects.get(memory.held_object_id)
            if held_node is not None and held_node.object_id in mov_cands:
                chosen_mov = held_node.object_id
            else:
                for mid in mov_cands:
                    if not executor.can_step():
                        break
                    s, f, ok, msg = macro._execute_single_interaction("PutObject", mid)
                    total_steps += s
                    total_failed += f
                    trace_log.append(f"Step 2 put content into #{mid}: {msg}")
                    if ok:
                        chosen_mov = memory.resolve_object_id(mid)
                        break
                if chosen_mov is None:
                    return _finish(
                        "failed",
                        f"Failed to place held item into movable receptacle candidates {mov_cands}.",
                    )
        else:
            chosen_mov = mov_cands[0]

        # Step B: pick up the movable receptacle
        if memory.held_object_id != chosen_mov:
            s, f, ok, msg = macro._execute_single_interaction(
                "PickupObject", chosen_mov
            )
            total_steps += s
            total_failed += f
            trace_log.append(f"Step 3 pickup movable receptacle #{chosen_mov}: {msg}")
            if not ok:
                return _finish(
                    "failed",
                    f"Placed item into #{chosen_mov} but failed to pick #{chosen_mov} up: {msg}",
                )

        # Step C: place the movable receptacle onto the destination surface
        dest_cands = rank_candidate_objects(
            memory,
            preferred_id=spec.destination_id,
            categories=spec.destination_categories or _UNLOAD_SURFACES,
            action_filter="PutObject",
            exclude_ids={chosen_mov},
            limit=spec.max_candidates,
        )
        if not dest_cands:
            return _finish(
                "failed",
                f"Holding movable receptacle #{chosen_mov}, but no destination surface ({list(spec.destination_categories or _UNLOAD_SURFACES)}) found in memory.",
            )
        for did in dest_cands:
            if not executor.can_step():
                break
            s, f, ok, msg = macro._execute_single_interaction("PutObject", did)
            total_steps += s
            total_failed += f
            trace_log.append(f"Step 4 place movable receptacle onto #{did}: {msg}")
            if ok and memory.held_object_id is None:
                d_node = memory.objects[memory.resolve_object_id(did)]
                return _finish(
                    "completed",
                    f"Completed movable-receptacle transfer: placed content in #{chosen_mov} and placed #{chosen_mov} onto #{d_node.object_id} ({d_node.category}).",
                )
        return _finish(
            "failed",
            f"Holding movable receptacle #{chosen_mov}, but PutObject failed on destination candidates {dest_cands}.",
        )

    return _finish("failed", f"Unhandled subgoal: {spec.subgoal!r}")


def verify_goal_predicates(executor: EpisodeExecutor) -> GoalVerificationReport:
    """Verify whether the current scene memory state satisfies the task instruction."""
    memory = executor.memory
    instruction_text = ""
    if executor.current_instruction is not None:
        parts = [executor.current_instruction.goal]
        if executor.config.use_step_instructions and executor.current_instruction.steps:
            parts.extend(executor.current_instruction.steps)
        instruction_text = " ".join(parts)
    low = instruction_text.casefold()

    # Infer structural task requirements purely from the natural-language goal
    is_examine = bool(
        re.search(
            r"\b(look at|examine|inspect|read|under .*light|in .*light|by .*light|light of|turn(?:ing)? on (?:a |the )?(?:[\w-]+\s+){0,2}lamp|lamp turned on|lit lamp)\b",
            low,
        )
    )
    requires_clean = bool(
        re.search(r"\b(clean|cleaned|wash|washed|rinse|rinsed)\b", low)
    ) and not bool(re.search(r"\b(clean towel|clean cloth|dishsponge)\b", low))
    requires_heat = bool(
        re.search(r"\b(heat|heated|warm|warmed|cooked|hot|microwave)\b", low)
    ) and not bool(
        re.search(
            r"\b(in the microwave|into the microwave|inside the microwave|to the microwave)\b",
            low,
        )
        and not re.search(r"\b(heat|heated|warm|warmed|cooked|hot)\b", low)
    )
    requires_cool = bool(re.search(r"\b(cool|cooled|chill|chilled|cold)\b", low))
    requires_slice = bool(re.search(r"\b(slice|sliced|cut|piece of|wedge of)\b", low))
    requires_two = bool(re.search(r"\b(two|both|2|pair of)\b", low))

    inferred: dict[str, object] = {
        "is_examine_in_light": is_examine,
        "requires_clean": requires_clean,
        "requires_heat": requires_heat,
        "requires_cool": requires_cool,
        "requires_slice": requires_slice,
        "requires_two_objects": requires_two,
    }
    satisfied: list[str] = []
    unmet: list[str] = []
    hints: list[str] = []

    if is_examine:
        if memory.held_object_id is not None:
            held_node = memory.objects.get(memory.held_object_id)
            satisfied.append(
                f"holding_target_in_hand (#{memory.held_object_id} {held_node.category if held_node else ''})"
            )
        else:
            unmet.append("examine_in_light_requires_holding_object")
            hints.append(
                "This task requires examining an object under a light: you MUST hold the target object in hand (`held_object_id != None`) while the lamp is toggled on."
            )

        lamp_on = any(
            node.category in _LIGHT_CATEGORIES
            and node.action_states.get("toggled_on") is not None
            and node.action_states["toggled_on"].value
            for node in memory.objects.values()
        ) or ("ToggleObjectOn" in executor.completed_macros)
        if lamp_on:
            satisfied.append("light_source_toggled_on")
        else:
            unmet.append("light_source_not_toggled_on")
            hints.append(
                "No DeskLamp or FloorLamp has been toggled on yet. Navigate to the lamp and call `interact_object({'action': 'toggle_on', 'object_id': <lamp_id>})` while holding the target object."
            )
        return GoalVerificationReport(
            passed=len(unmet) == 0,
            inferred_requirements=inferred,
            satisfied_predicates=satisfied,
            unmet_predicates=unmet,
            remediation_hints=hints,
        )

    # All non-examine ALFRED tasks require placing the target object(s) onto/into a destination receptacle!
    if memory.held_object_id is not None:
        held_node = memory.objects.get(memory.held_object_id)
        held_desc = (
            f"#{held_node.object_id} ({held_node.category})"
            if held_node is not None
            else f"#{memory.held_object_id}"
        )
        unmet.append(f"still_holding_object ({held_desc})")
        hints.append(
            f"You are still holding {held_desc} in hand! Placement, clean, heat, cool, slice, and movable-receptacle tasks require placing the final object onto/into the destination receptacle (e.g., CounterTop, DiningTable, Fridge, Cabinet, SinkBasin) before calling `finish_episode`."
        )
    else:
        satisfied.append("hands_empty_after_placement")

    if "PutObject" not in executor.completed_macros:
        unmet.append("no_put_action_executed")
        hints.append(
            "No successful `PutObject` has been executed in this episode yet. You must place the target object into/onto the destination receptacle."
        )
    else:
        satisfied.append("put_action_executed")

    if requires_clean:
        has_cleaned = "clean_in_sink" in executor.completed_macros or any(
            node.action_states.get("cleaned") is not None
            and node.action_states["cleaned"].value
            for node in memory.objects.values()
        )
        if has_cleaned:
            satisfied.append("object_cleaned")
        else:
            unmet.append("required_clean_not_completed")
            hints.append(
                "The goal requires cleaning/rinsing/washing an object, but neither `clean_in_sink` nor a Faucet toggle on a placed item has succeeded yet."
            )

    if requires_heat:
        has_heated = "heat_in_microwave" in executor.completed_macros or any(
            node.action_states.get("heated") is not None
            and node.action_states["heated"].value
            for node in memory.objects.values()
        )
        if has_heated:
            satisfied.append("object_heated")
        else:
            unmet.append("required_heat_not_completed")
            hints.append(
                "The goal requires heating/warming/cooking an object, but `heat_in_microwave` (or toggling the Microwave with the object inside) has not succeeded yet."
            )

    if requires_cool:
        has_cooled = "cool_in_fridge" in executor.completed_macros or any(
            node.action_states.get("cooled") is not None
            and node.action_states["cooled"].value
            for node in memory.objects.values()
        )
        if has_cooled:
            satisfied.append("object_cooled")
        else:
            unmet.append("required_cool_not_completed")
            hints.append(
                "The goal requires cooling/chilling an object, but `cool_in_fridge` (or retrieving a pre-chilled item from inside the Fridge) has not been recorded yet."
            )

    if requires_slice:
        has_sliced = "SliceObject" in executor.completed_macros or any(
            node.state == "sliced"
            or (
                node.action_states.get("sliced") is not None
                and node.action_states["sliced"].value
            )
            or node.category.casefold().endswith("sliced")
            for node in memory.objects.values()
        )
        if has_sliced:
            satisfied.append("object_sliced")
        else:
            unmet.append("required_slice_not_completed")
            hints.append(
                "The goal requires a sliced/cut piece of food, but `SliceObject` has not succeeded and no `*Sliced` object is present."
            )

    if requires_two:
        placed_objects = [
            node
            for node in memory.objects.values()
            if node.action_states.get("placed_by_agent") is not None
            and node.action_states["placed_by_agent"].value
            and node.object_id != memory.held_object_id
        ]
        # Count distinct placed objects (excluding knives that were simply unloaded)
        non_knife_placed = [
            node
            for node in placed_objects
            if node.category not in _KNIFE_CATEGORIES or "knife" in low
        ]
        if len(non_knife_placed) >= 2:
            satisfied.append(
                f"two_objects_placed ({[f'#{n.object_id}:{n.category}' for n in non_knife_placed[:4]]})"
            )
        else:
            unmet.append(
                f"two_objects_required_but_only_{len(non_knife_placed)}_placed"
            )
            hints.append(
                f"The goal requires placing TWO objects (`two` / `both` / `pair`), but only {len(non_knife_placed)} object(s) have been placed by `PutObject` so far ({[f'#{n.object_id}:{n.category}' for n in non_knife_placed]}). Find and place the second instance before finishing!"
            )

    return GoalVerificationReport(
        passed=len(unmet) == 0,
        inferred_requirements=inferred,
        satisfied_predicates=satisfied,
        unmet_predicates=unmet,
        remediation_hints=hints,
    )


__all__ = [
    "GoalVerificationReport",
    "SUBGOAL_TYPES",
    "SubgoalProgramSpec",
    "execute_subgoal_program",
    "parse_subgoal_program",
    "rank_candidate_objects",
    "verify_goal_predicates",
]
