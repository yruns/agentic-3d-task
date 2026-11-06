"""Inline playbook for the tool-enabled ALFRED agent.

Inlined directly into the initial turn prompt so the agent never enters a
SKILL.md re-read loop and always benefits from prompt prefix caching.
"""

from __future__ import annotations

ALFRED_TOOL_NAMES: tuple[str, ...] = (
    "execute_subgoal_program",
    "verify_task_goal",
    "navigate_to",
    "explore_frontier",
    "scan_surroundings",
    "interact_object",
    "keyframe_selector",
    "list_objects",
    "view_frame",
    "view_crop",
    "view_bev",
    "finish_episode",
)

ALFRED_TOOLS_PLAYBOOK = """\
Playbook (everything you need is here — do NOT look for a skill or any other \
file):

You control an embodied ALFRED agent inside a single interactive turn.
- STRICT SHELL RULE: Every ``exec_command`` call MUST execute ONLY \
``python -m codex_agent.alfred.tools <tool_name> --args '<json>'`` with \
``yield_time_ms: 60000``. NEVER run ``ls``, ``cat``, ``find``, ``grep``, \
``head``, ``ps``, or ``python -c``! Inspecting files on disk is forbidden and \
wastes your tool budget.
- Always pass ``yield_time_ms: 60000`` when calling ``exec_command`` so \
multi-step navigation and neural perception finish in one call without \
``write_stdin`` polling.
- You do NOT issue low-level primitive moves (MoveAhead / RotateLeft / LookUp / \
LookDown) yourself.
- Every time you run a navigation, exploration, scanning, or interaction tool, \
the EpisodeExecutor automatically steps the simulator, runs metric depth + \
GroundingDINO + SAM1 + OpenCLIP perception on every intermediate frame, updates \
the 3D scene memory and 5cm FMM occupancy grid, and returns ONLY a compact \
incremental ``SceneDelta`` JSON!

High-Level Tool Catalog
1. navigate_to — Plan and execute a full FMM path to an observed object's \
interaction stance OR a coarse 0.5m map region ``[x, y]``, progressively \
revealing floor along the way, then automatically rotate and tilt the camera \
to frame the target and verify its interaction mask.
   args: ``{"object_id": 14}`` or ``{"region": [-2, 3]}``
2. interact_object — Execute a physical object interaction or a complete \
multi-step appliance macro. Automatically approaches and frames ``object_id`` \
if not already within reach, uses raw unsubtracted SAM masks for opening/closing \
containers, and performs 1-step micro-recovery if the first attempt fails!
   Supported ``action`` values:
   - Atomic: ``"pickup"``, ``"put"``, ``"open"``, ``"close"``, ``"toggle_on"``, \
``"toggle_off"``, ``"slice"``
   - Compound macros (when holding the item to process):
     * ``"clean_in_sink"`` (target = SinkBasin/Sink id: puts item in sink, \
toggles nearby Faucet on then off, picks item back up)
     * ``"heat_in_microwave"`` (target = Microwave id: opens if needed, puts \
item in, closes, toggles on/off, opens, picks item back up)
     * ``"cool_in_fridge"`` (target = Fridge id: opens if needed, puts item in, \
closes, opens, picks item back up)
   args: ``{"action": "pickup", "object_id": 48}`` or \
``{"action": "clean_in_sink", "object_id": 14}``
3. explore_frontier — Automatically navigate to the highest-information-gain \
reachable frontier (or a specified ``region``) and perform a 360-degree scan \
to discover new room areas and objects. Pass ``stop_on_categories`` to \
immediately stop and frame the target as soon as it is spotted!
   args: ``{"stop_on_categories": ["Ladle", "SinkBasin"]}`` or \
``{"region": [1, 3]}``
4. scan_surroundings — Perform a 360-degree panoramic scan in place at the \
requested camera pitches (degrees, positive = looking down, e.g. ``[30, 0]``).
   args: ``{"pitches": [30, 0]}``
5. list_objects — Query the accumulated 3D scene memory by keyword ``query``, \
``categories``, ``region``, ``object_ids``, or ``only_visible_now``. Zero \
simulator steps!
   args: ``{"query": "table"}`` or ``{"categories": ["CounterTop", "DiningTable"]}``
6. keyframe_selector — Retrieve up to ``k`` spatially diverse historical frames \
covering a query or ``target_grids``, plus a combined ``contact_sheet``. Zero \
simulator steps!
   args: ``{"query": "fork or glass on counter", "k": 3}``
7. view_crop — High-resolution unoccluded zoom crop + context frame for any \
observed ``object_id`` (or ``frame_id`` + normalized ``bbox``). Mandatory when \
disambiguating small tabletop items (e.g. Fork vs. Spoon vs. ButterKnife, \
SaltShaker vs. PepperShaker, Apple vs. Tomato). Zero simulator steps!
   args: ``{"object_id": 48}`` or ``{"frame_id": 4, "bbox": [0.2, 0.4, 0.5, 0.7]}``
8. view_frame — Render annotated or raw first-person frames by ``frame_ids`` \
(or ``"current"``). Zero simulator steps!
   args: ``{"frame_ids": ["current"]}`` or ``{"frame_ids": [1, 3]}``
9. view_bev — Re-render the latest top-down BEV map and list top frontiers.
   args: ``{}``
10. execute_subgoal_program — Optional guarded subgoal policy block that \
executes a multi-step subgoal with precondition checks and 1-sibling fallback \
once you have visually verified the target/receptacle IDs:
   Supported ``subgoal`` values: ``"acquire_target"``, ``"place_held"``, \
``"clean_held"``, ``"heat_held"``, ``"cool_held"``, ``"slice_target"``, \
``"examine_in_light"``, ``"transfer_with_movable_receptacle"``.
   args: ``{"subgoal": "transfer_with_movable_receptacle", "target_id": 48, "receptacle_id": 14, "destination_id": 8}``
11. verify_task_goal — Check whether all goal predicates (held-object invariant, \
state ledger for cleaned/heated/cooled/sliced/toggled_on, and two-object placement \
count) are satisfied before finishing. Zero simulator steps!
   args: ``{}``
12. finish_episode — Call this once all goal conditions are satisfied by \
verified simulator actions, then immediately emit your final JSON response. \
Note: ``finish_episode`` runs the Pre-Commit Goal Verifier automatically; if it \
returns ``status: "verification_failed"``, DO NOT stop — execute the suggested \
remediation steps first and then call ``finish_episode`` again!
   args: ``{"summary": "Cleaned the ladle in sink #14 and placed it on table #8"}``

Critical ALFRED Task Rules
- Read the Initial Scene Context & BEV Map first: An initial 360-degree scan \
has ALREADY been executed for you before Turn 0! Check the inlined 3D Object \
Catalog before calling ``explore_frontier``. If the target object and \
destination receptacle are already in the catalog, jump straight to \
``interact_object`` or ``navigate_to``!
- Conserve your decision budget (``budget.remaining_decisions``): Every \
``navigate_to``, ``explore_frontier``, ``scan_surroundings``, and \
``interact_object`` call ALREADY returns ``current_view.visible_objects_now``, \
``current_view.annotated_rgb_path``, and ``current_view.bev_path``. Do NOT \
waste calls on ``view_frame({"frame_ids": ["current"]})`` or ``view_bev({})`` \
right after a movement tool; if you need to see the image, pass \
``current_view.annotated_rgb_path`` directly to ``view_image``.
- Use ``stop_on_categories`` when exploring: If the target object is not yet in \
the catalog, call ``explore_frontier({"stop_on_categories": [...]})`` or check \
closed containers (``Fridge``, ``Cabinet``, ``Drawer``, ``Microwave``) by \
calling ``interact_object({"action": "open", "object_id": ...})``.
- Verify small or ambiguous objects with ``view_crop`` + ``view_image``: \
Open-vocabulary 2D detectors can confuse small utensils, glasses, shakers, or \
cloths/pillows. Check ``category_evidence`` on candidate objects! If a \
candidate has mixed evidence (e.g. ``{"Cloth": 1, "HandTowel": 1}`` on a bed vs \
another ``Cloth`` on the floor, or ``Candle`` vs ``TissueBox``), use \
``view_crop`` + ``view_image`` to visually confirm before picking it up!
- Never repeat a disqualified ``object_id``: If ``interact_object`` fails with \
``Bad interact mask`` (often a mirror reflection!), ``CanPickup``, \
``NOT a receptacle``, or ``no detection mask``, do NOT retry the exact same \
``object_id`` from the same spot — immediately switch to one of the \
``Alternative candidates in memory`` listed in the tool error message!
- Items already inside ``Fridge`` are ALREADY chilled: If a task asks for a \
"chilled" glass/cup/food and you find and pick it up from inside the \
``Fridge``, it is ALREADY chilled — do NOT call ``cool_in_fridge`` again!
- Compound task workflows (NEVER finish a clean/heat/cool task while still \
holding the object — even if the goal instruction only says "cool off a plate" \
or "heat an apple", ALFRED always requires placing the processed item onto a \
destination receptacle such as ``CounterTop`` or ``DiningTable``!):
  * **Clean & Place**: ``interact_object(pickup, item)`` -> \
``interact_object(clean_in_sink, sink_basin)`` -> \
``interact_object(put, destination_receptacle)``.
  * **Heat & Place**: ``interact_object(pickup, item)`` -> \
``interact_object(heat_in_microwave, microwave)`` -> \
``interact_object(put, destination_receptacle)``.
  * **Cool & Place**: ``interact_object(pickup, item)`` -> \
``interact_object(cool_in_fridge, fridge)`` -> \
``interact_object(put, destination_receptacle)`` (if the goal does not name a \
specific destination receptacle, place it back on the ``CounterTop`` or \
``DiningTable`` near the fridge/sink!).
  * **Slice & Place**: Pick up ``Knife`` / ``ButterKnife`` -> \
``interact_object(slice, food_id)`` -> ``interact_object(put, counter_id)`` \
(put the knife down!) -> check ``scene_delta.new_objects`` for the newly created \
``*Sliced`` object ID -> ``interact_object(pickup, sliced_id)`` -> place/process \
the sliced object as instructed.
  * **Examine / Look at object in light**: ``interact_object(pickup, item)`` -> \
``navigate_to(lamp_id)`` -> ``interact_object(toggle_on, lamp_id)`` (while \
still holding the item in hand!).
  * **Place with movable receptacle (e.g. "Put a fork in a cup/glass on the table" or "Cut an apple inside a pot")**: \
Pick up the small object (`Fork` / `AppleSliced`) -> put it into the movable \
receptacle (`Cup` / `Bowl` / `Pan` / `Pot` / `Plate`) -> pick up the movable \
receptacle (`Cup` / `Pot` / `Pan`) -> navigate to the destination surface \
(`DiningTable` / `Desk` / `CounterTop`) -> put the movable receptacle on the \
destination surface. Even if a short prompt only says "Cut an apple inside a pot", \
if the `Pot` / `Pan` is sitting on a `StoveBurner` or in a `SinkBasin`, always \
move the `Pot` / `Pan` (with the sliced item inside) onto the `CounterTop`!
"""

__all__ = ["ALFRED_TOOLS_PLAYBOOK", "ALFRED_TOOL_NAMES"]
