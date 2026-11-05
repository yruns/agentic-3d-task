"""Inline playbook for the tool-enabled ALFRED agent.

Inlined directly into the initial turn prompt so the agent never enters a
SKILL.md re-read loop and always benefits from prompt prefix caching.
"""

from __future__ import annotations

ALFRED_TOOL_NAMES: tuple[str, ...] = (
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
- You do NOT issue low-level primitive moves (MoveAhead / RotateLeft / LookUp / \
LookDown) yourself.
- Instead, you invoke HIGH-LEVEL MACRO TOOLS via ``python -m codex_agent.alfred.tools``.
- Every time you run a navigation, exploration, scanning, or interaction tool, \
the EpisodeExecutor automatically steps the simulator, runs metric depth + \
GroundingDINO + SAM1 + OpenCLIP perception on every intermediate frame, updates \
the 3D scene memory and 5cm FMM occupancy grid, and returns ONLY a compact \
incremental ``SceneDelta`` JSON!

High-Level Tool Catalog
1. navigate_to — Plan and execute a full FMM path to an observed object's \
interaction stance OR a coarse 0.5m map region ``[x, y]``, then automatically \
rotate and tilt the camera to frame the target and verify its interaction mask.
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
8. view_frame — Renderannotated or raw first-person frames by ``frame_ids`` \
(or ``"current"``). Zero simulator steps!
   args: ``{"frame_ids": ["current"]}`` or ``{"frame_ids": [1, 3]}``
9. view_bev — Re-render the latest top-down BEV map and list top frontiers.
   args: ``{}``
10. finish_episode — Call this once all goal conditions are satisfied by \
verified simulator actions, then immediately emit your final JSON response.
   args: ``{"summary": "Cleaned the ladle in sink #14 and placed it on table #8"}``

Critical ALFRED Task Rules
- Read the Initial Scene Context & BEV Map first: An initial 360-degree scan \
has ALREADY been executed for you before Turn 0! Check the inlined 3D Object \
Catalog before calling ``explore_frontier``. If the target object and \
destination receptacle are already in the catalog, jump straight to \
``navigate_to`` / ``interact_object``!
- Use ``stop_on_categories`` when exploring: If the target object is not yet in \
the catalog, call ``explore_frontier({"stop_on_categories": [...]})`` or check \
closed containers (``Fridge``, ``Cabinet``, ``Drawer``, ``Microwave``) by \
navigating to them and calling ``interact_object({"action": "open", ...})``.
- Verify small or ambiguous objects with ``view_crop`` + ``view_image``: \
Open-vocabulary 2D detectors can confuse small utensils or shakers. Before \
picking up a small item, check its high-res crop if multiple candidates exist.
- Compound task workflows:
  * **Clean & Place**: ``interact_object(pickup, item)`` -> \
``interact_object(clean_in_sink, sink_basin)`` -> \
``interact_object(put, destination_receptacle)``.
  * **Heat & Place**: ``interact_object(pickup, item)`` -> \
``interact_object(heat_in_microwave, microwave)`` -> \
``interact_object(put, destination_receptacle)``.
  * **Cool & Place**: ``interact_object(pickup, item)`` -> \
``interact_object(cool_in_fridge, fridge)`` -> \
``interact_object(put, destination_receptacle)``.
  * **Slice & Place**: Pick up ``Knife`` / ``ButterKnife`` -> \
``interact_object(slice, food_id)`` -> ``interact_object(put, counter_id)`` \
(put the knife down!) -> check ``scene_delta.new_objects`` for the newly created \
``*Sliced`` object ID -> ``interact_object(pickup, sliced_id)`` -> place/process \
the sliced object as instructed.
  * **Examine / Look at object in light**: ``interact_object(pickup, item)`` -> \
``navigate_to(lamp_id)`` -> ``interact_object(toggle_on, lamp_id)`` (while \
still holding the item in hand!).
  * **Place with movable receptacle (e.g. "Put a fork in a cup on the table")**: \
Pick up the small object (`Fork`) -> put it into the movable receptacle (`Cup` \
/ `Bowl` / `Pan` / `Plate`) -> pick up the movable receptacle (`Cup`) -> \
navigate to the destination surface (`DiningTable`) -> put the movable \
receptacle on the destination surface.
"""

__all__ = ["ALFRED_TOOLS_PLAYBOOK", "ALFRED_TOOL_NAMES"]
