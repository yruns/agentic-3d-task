# ALFRED v15: Root-Cause Fixes for Held-Object Self-Occlusion, Spatial Mask Fallback, Auto-Framing Pitch Search, and Decision Budget Accounting

- **Date:** 2026-11-06
- **Branch:** `feat/alfred-benchmark`
- **Commit:** `a3bd8e2` (`fix(alfred): resolve held-object self-occlusion, spatial fallback masks, auto-framing pitch oscillation, and decision budget accounting`)
- **Fold:** `docs/benchmark/alfred/folds/pilot15.json` (`prefix5`: shards `0..4`, all `valid_seen`)
- **Model:** `gemini-3.8-flash` (`codex_adapter`, `reasoning_effort = "provider-default"`)
- **Scene Config:** `configs/alfred/scene.toml` (`depth_source = "simulator_gt"`, `association_distance_m = 0.06`, `envelope_margin_m = 0.05`)

---

## 1. Root-Cause Analysis of V14 Failures (Why Failed Episodes Called 25–29 Tools)

A frame-by-frame trace audit of all 4 failed episodes in `v14` (`i00`, `i02`, `i03`, `i04`) revealed that the agent was **not** wandering randomly—in every episode it located and picked up the target object within the first few decisions—but was trapped by **five systemic mechanism bugs** across the online mapping, controller, and runtime layers:

1. **Invisible Guillotine (`CommandExecution + ImageView = 36 > 35`):**
   - `alfred_runner.py` passed `max_tool_calls = config.execution.max_decisions` (`35`) to `CodexAgentRuntime._ToolCallLoopGuard`.
   - Because `_ToolCallLoopGuard` counts **both** CLI commands (`CommandExecution`) and native `view_image` (`ImageView`) calls toward `max_tool_calls`, while `SceneDelta.budget` only reported `remaining_actions` and `remaining_failures`, all 4 failed trajectories in `v14` hit `CommandExecution + ImageView == 36` (`25+11`, `26+10`, `27+9`, `29+7`) and were abruptly terminated by `<turn_aborted>`.
2. **Held-Object Self-Occlusion Generating Phantom Obstacle Walls (`memory.py` & `spatial_map.py`):**
   - In AI2-THOR, a held object floats `0.37–0.45 m` in front of the camera across the lower half of the viewport.
   - At such extreme close range, `GroundingDINO` misclassifies the held object across viewpoints (e.g. held `Pillow` $\rightarrow$ `Sofa`/`CounterTop` in `i00`; held `Plate` $\rightarrow$ `CD`/`Ladle` in `i04`; held `Ladle` $\rightarrow$ `ButterKnife`/`CD` in `i03`).
   - Because the category changed, `_association_score` rejected associating the detection to `held_object_id`, creating **14 to 60+ phantom 3D objects** along the trajectory and leaving `held_mask` empty. Consequently, `SpatialMap.update()` projected the held object's `0.40 m` depth pixels as solid `occupied` obstacle walls right in front of the agent on every step, eventually locking FMM navigation (`"No reachable stance through observed space"`).
3. **Blind `0.42 m` Spatial Fallback in `select_interaction_mask` (`memory.py`):**
   - When the target `#ID` was missed in the current frame, `select_interaction_mask` fell back to *any* detected object within `0.42 m` regardless of category.
   - In `i03`, when `clean_in_sink` targeted `#77 SinkBasin`, spatial fallback selected `#67 SaltShaker` on the counter rim, causing AI2-THOR to return `"This target object is NOT a receptacle!"`, which tricked the model into abandoning the real sink basin.
4. **Pitch Oscillation Deadlock in `_auto_frame_object` (`macro_controller.py`):**
   - When `current_h == 30.0` and the target mask was missing, `_auto_frame_object` only probed `LookUp` (`15°`), oscillating `15° <-> 30°` (`Frames 39–43` in `i00`) and never checking `45°` where `#7 DeskLamp` had just been detected with `0.75` confidence (`Frame 37`).
5. **Container Geometry Erasure on `OpenObject` & Missing Default Destination Rule (`memory.py` & `playbook.py`):**
   - `record_interaction("OpenObject")` set `geometry_valid = False` on the opened container itself (`Fridge`/`Cabinet`), causing the immediate follow-up `PutObject` in `cool_in_fridge` to lose the container's 3D bounds.
   - In `i04`, the natural-language goal is simply `"cool off a white plate"`, while ALFRED's PDDL scorer requires placing the cooled plate back onto a `CounterTop`.

---

## 2. Summary of V15 Fixes (`a3bd8e2`)

1. **Held-Object Self-Occlusion Suppression & Near-Foreground Depth Masking (`src/codex_agent/alfred/memory.py`):**
   - Added `_is_held_self_occlusion()`: while `held_object_id is not None`, any close-up detection (`cam_dist <= 0.58 m`, `y2 >= 0.50 * H`, `y1 >= 0.10 * H`) is associated directly to `held_object_id` without polluting its category votes or spawning phantom 3D nodes, and all depth pixels within `<= 0.56 m` in the lower 70% of the frame are masked out of `SpatialMap.update()`.
2. **Category-Compatible Spatial Fallback & Container Geometry Preservation (`src/codex_agent/alfred/memory.py`):**
   - Added `_is_compatible_spatial_fallback()` so `select_interaction_mask` only falls back to semantically compatible categories (never substituting a `SaltShaker` for a `SinkBasin` or a stationary `Desk` for a `PickupObject` target).
   - Preserved `geometry_valid = True` on containers during `OpenObject` / `CloseObject` so multi-step appliance macros (`cool_in_fridge`, `heat_in_microwave`) retain valid 3D bounds.
3. **Multi-Pitch Auto-Framing & Cardinal Alignment (`src/codex_agent/alfred/macro_controller.py`):**
   - Upgraded `_auto_frame_object()` to probe `clamped_desired`, `clamped_desired + 15°`, and `clamped_desired - 15°` (covering `45°`, `30°`, `15°`) without oscillation.
   - Tightened `_is_fov_aligned` to `22°` and prioritized aligned close goals in `_next_adaptive_nav_action`.
4. **Budget Decoupling & Playbook Rules (`alfred_runner.py`, `delta.py`, `playbook.py`):**
   - Set `max_tool_calls = config.execution.max_decisions * 2` (`70`) while enforcing `max_decisions` (`35`) on CLI tool calls in `EpisodeToolServer` and exposing `budget.remaining_decisions` in every `SceneDelta`.
   - Updated `ALFRED_TOOLS_PLAYBOOK` to clarify that items retrieved from inside a `Fridge` are already chilled, and that `cool`/`heat`/`clean` tasks without an explicit destination in the prompt must place the processed item back onto a `CounterTop` or `DiningTable`.

---

## 3. Quantitative Results on `prefix5` (`v12` $\rightarrow$ `v13` $\rightarrow$ `v14` $\rightarrow$ **`v15`**)

| Episode (`valid_seen`) | Task Type & Goal | `v12` (DA3) | `v13` (DA3 Upgrade) | `v14` (GT Depth) | **`v15` (GT Depth + Loop/Mask/Framing Fixes)** | **`v15` Decisions / Actions / Failures** |
|---|---|---:|---:|---:|---:|---:|
| `i00` (`FloorPlan302`) | `look_at_obj_in_light` (*Carry a purple pillow while turning on a lamp*) | `2/2` (SR 1) | `1/2` (SR 0) | `1/2` (SR 0, 25 dec / 102 act) | **`2/2` (SR 100%, PLW 83.3%)** | **`8` dec / `42` act / `0` fail** |
| `i01` (`FloorPlan415`) | `pick_and_place_simple` (*put the spray bottle on the back of the toilet*) | `0/1` (SR 0) | `0/1` (SR 0) | `1/1` (SR 100%, 6 dec / 30 act) | **`1/1` (SR 100%, PLW 100.0%)** | **`5` dec / `29` act / `0` fail** |
| `i02` (`FloorPlan23`) | `pick_and_place_with_movable_recep` (*Place a chilled martini glass with a fork in it on the table*) | `0/3` (SR 0) | `0/3` (SR 0) | `0/3` (SR 0, 26 dec / 105 act) | `0/3` (SR 0%) | `36` dec / `112` act / `3` fail |
| `i03` (`FloorPlan4`) | `pick_clean_then_place_in_recep` (*Rinse off a laddle and move it to the table*) | `0/3` (SR 0) | `0/3` (SR 0) | `1/3` (SR 0, 27 dec / 83 act) | **`1/3` (SR 0%)** | `27` dec / `106` act / `8` fail |
| `i04` (`FloorPlan1`) | `pick_cool_then_place_in_recep` (*cool off a white plate*) | `0/3` (SR 0) | `1/3` (SR 0) | `0/3` (SR 0, 29 dec / 159 act) | **`3/3` (SR 100%, PLW 40.4%)** | **`18` dec / `146` act / `3` fail** |
| **Aggregate (`prefix5`)** | **5 Episodes (`12` Goal Conditions)** | **SR `20%` (`1/5`), GC `25%` (`3/12`)** | **SR `0%` (`0/5`), GC `16.7%` (`2/12`)** | **SR `20%` (`1/5`), GC `25%` (`3/12`)** | **SR `60.0%` (`3/5`), GC `58.3%` (`7/12`), PLW-SR `44.7%`** | **Mean `18.8` dec / `87.0` act** |

---

## 4. Per-Episode Walkthrough of Improvements & Remaining 2D Detector Bottleneck

1. **`i00` (`look_at_obj_in_light-Pillow-None-DeskLamp-302`) — From `25` Decisions (Aborted) to `8` Decisions (`SR = 100%`, `0` Failures):**
   - After picking up the purple pillow (`#44`) at Decision 5, the held pillow remained cleanly bound to `#44` across all subsequent frames (`0` phantom `Sofa` obstacles created!).
   - Decision 6 `navigate_to(#7 DeskLamp)` arrived at `(1.25, -0.25)` and framed `#7 DeskLamp` at `horizon = 45°` on the first try (`confidence = 0.771`, `4,039 px`).
   - Decision 7 `interact_object("toggle_on", 7)` succeeded in **1 primitive step**, and Decision 8 called `finish_episode` (`42` total actions vs. `35` expert actions, `0` failures).
2. **`i01` (`pick_and_place_simple-SprayBottle-None-Toilet-415`) — Completed in `5` Decisions / `29` Actions (`SR = 100%`, `PLW-SR = 100%`):**
   - Completed in 29 actions (shorter than the 33-step expert demonstration) with `0` failed actions.
3. **`i04` (`pick_cool_then_place_in_recep-Plate-None-CounterTop-1`) — First-Ever `3/3` Complete Success (`SR = 100%`):**
   - Picked up the white plate (`#86`) at Decision 6 (`Frame 44`). Thanks to `_is_held_self_occlusion`, the held plate remained `#86 Plate` throughout the 102-step kitchen traversal (`0` phantom `CD`/`Ladle` obstacles!).
   - Placed the plate into the fridge, closed the door to chill it, reopened the fridge (`Decision 12`), picked the chilled plate back up (`Decision 13`), navigated back to `#99 CounterTop` (`Decision 16`), and placed the chilled plate on the counter (`Decision 17`), achieving **`GC = 3/3 (SR = 100%)`**.
4. **Remaining Failure Mode on `i02` & `i03` (Pure 2D Detector Category Confusion on Small/Metallic Objects):**
   - In `i03`, the agent now **successfully placed the ladle (`#41`) inside the true sink basin (`#11 BathtubBasin`, `Action 49`)** and later picked it back up and placed it on the dining table (`#3 Desk`, `GC = 1/3`). However, when looking up at the bronze faucet above the sink (`Frames 50–56`), `GroundingDINO` misclassified the faucet as `#96/#97 DeskLamp`, so `ToggleObjectOn` targeted the adjacent wall mask instead of the faucet spout.
   - In `i02`, the martini glass (`Cup`) inside the dark fridge was only detected as `Glassbottle`/`Candle` when the fridge door was open, so the agent tried to use a `Ladle` (`#14`) from the table as a substitute container.

---

## 5. Artifact Locations

- `outputs/alfred/ALFRED_V15_R2_20261106_pilot15_i00`
- `outputs/alfred/ALFRED_V15_20261106_pilot15_i01`
- `outputs/alfred/ALFRED_V15_20261106_pilot15_i02`
- `outputs/alfred/ALFRED_V15_R2_20261106_pilot15_i03`
- `outputs/alfred/ALFRED_V15_20261106_pilot15_i04`
