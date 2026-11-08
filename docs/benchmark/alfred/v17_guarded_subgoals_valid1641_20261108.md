# ALFRED v17: Guarded Subgoals, Pre-Commit Verifier, Full-Validation (`1,641 / 1,641`) Results, and Joint Good-Case / Bad-Case Root-Cause Audit

- **Date:** 2026-11-08
- **Branch:** `feat/alfred-benchmark`
- **Executed Code Commit:** `3cc517ed71ba4acfc9ca8d29457dd91490a3cf37` (`tracked_worktree_drift = ""`, `executed_code_sha256 = "06e503d5fb73e6268cf1fb316f19aa64259c0c82e7b113f5bf0d4675c8e3da7b"`)
- **Fold:** `docs/benchmark/alfred/folds/valid_full1641.json` (`820 valid_seen + 821 valid_unseen = 1,641` official `oct21.json` validation trajectories)
- **Canonical Iteration Subset:** `docs/benchmark/alfred/folds/valid_strat150.json` (`75 valid_seen + 75 valid_unseen = 150` unique stratified trajectories)
- **Run ID:** `ALFRED_V17_S42_20261107_valid1641` (`42` shards `s00..s41`, `0` errors)
- **Model:** `gemini-3.8-flash` (`codex_adapter`, `reasoning_effort = "provider-default"`, 5-AK load-balanced pool)
- **Scene Config:** `configs/alfred/scene.toml` (`depth_source = "simulator_gt"`, `association_distance_m = 0.06`, `envelope_margin_m = 0.05`)
- **Budgets:** `max_actions = 250`, `max_failures = 10`, `max_decisions = 35` (`max_tool_calls = 70`)

---

## 1. Executive Summary

`v17` evaluates the first closed-loop **Guarded Subgoal Policy Program (`execute_subgoal_program`)**, **Pre-Commit Goal Predicate Verifier (`verify_task_goal` / `verify_goal_predicates`)**, and **Container–Content Topology & State Ledger** across all **1,641 / 1,641** official ALFRED validation episodes (`0` errors):

- **Overall (`1,641 / 1,641` scored, `0` errors):**
  - **Success Rate (SR):** **`24.80%` (`407 / 1,641`)** — **`+0.30 pp`** (`+5` net episodes) over `v16` (`24.50%`, `402 / 1,641`)
  - **Goal-Condition Success Rate (GC):** **`35.54%` (`1,503 / 4,229`)** — **`+1.84 pp`** (`+78` goal conditions) over `v16` (`33.70%`, `1,425 / 4,229`)
  - **PLW-SR:** **`12.98%`** (`+0.15 pp`) ｜ **PLW-GC:** **`21.72%`** (**`+1.34 pp`** over `v16` `20.39%`)
- **`valid_unseen` (`821 / 821` scored) — Significant Generalization Gain:**
  - **SR:** **`22.29%` (`183 / 821`)** — **`+3.65 pp`** (`+30` net episodes; `76` `0->1` gains vs. `46` `1->0` regressions) over `v16` (`18.64%`, `153 / 821`)
  - **GC:** **`34.76%` (`737 / 2,120`)** — **`+4.29 pp`** (`+91` goal conditions) over `v16` (`30.47%`, `646 / 2,120`)
  - **PLW-SR:** **`10.95%`** (`+1.57 pp`) ｜ **PLW-GC:** **`18.91%`** (`+1.37 pp`)
- **`valid_seen` (`820 / 820` scored) — Split Trade:**
  - **SR:** **`27.32%` (`224 / 820`)** — **`-3.05 pp`** (`-25` net episodes; `65` `0->1` gains vs. `90` `1->0` regressions, concentrated in `pick_heat` and `pick_cool`) vs. `v16` (`30.37%`, `249 / 820`)
  - **GC:** **`36.32%` (`766 / 2,109`)** (`-0.62 pp`) ｜ **PLW-SR:** **`14.84%`** (`-1.15 pp`) ｜ **PLW-GC:** **`24.31%`** (**`+1.31 pp`**)
- **Canonical `valid_strat150` Stratified Iteration Subset (`150 / 150`):**
  - **Combined (`150 / 150`):** **SR `25.33%` (`38 / 150`)**, **GC `36.43%` (`141 / 387`)**, **PLW-SR `13.37%`**, **PLW-GC `22.62%`**
  - **`valid_seen` (`75 / 75`):** **SR `28.00%` (`21 / 75`)**, **GC `36.98%` (`71 / 192`)**, **PLW-SR `15.14%`**, **PLW-GC `25.39%`**
  - **`valid_unseen` (`75 / 75`):** **SR `22.67%` (`17 / 75`)**, **GC `35.90%` (`70 / 195`)**, **PLW-SR `11.44%`**, **PLW-GC `19.61%`**

---

## 2. Official Full-Validation Results (`1,641 / 1,641`) & Head-to-Head Comparison vs. `v16`

### 2.1 Split-Level Metrics (`v16` vs. `v17`)

| Split / Cohort | `v16` SR (%) | **`v17` SR (%)** | $\Delta$ SR | `v16` GC (%) | **`v17` GC (%)** | $\Delta$ GC | `v16` PLW-SR / PLW-GC | **`v17` PLW-SR / PLW-GC** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`valid_seen` (`820`)** | `30.37%` (`249/820`) | **27.32%** (`224/820`) | `-3.05 pp` | `36.94%` (`779/2,109`) | **36.32%** (`766/2,109`) | `-0.62 pp` | `15.99% / 23.00%` | **14.84% / 24.31%** |
| **`valid_unseen` (`821`)** | `18.64%` (`153/821`) | **22.29%** (`183/821`) | **`+3.65 pp`** | `30.47%` (`646/2,120`) | **34.76%** (`737/2,120`) | **`+4.29 pp`** | `9.38% / 17.54%` | **10.95% / 18.91%** |
| **Overall (`valid_full1641`)** | `24.50%` (`402/1,641`) | **24.80%** (`407/1,641`) | **`+0.30 pp`** | `33.70%` (`1,425/4,229`) | **35.54%** (`1,503/4,229`) | **`+1.84 pp`** | `12.83% / 20.39%` | **12.98% / 21.72%** |
| **`valid_strat150` (`150`)** | `24.67%` (`37/150`) | **25.33%** (`38/150`) | **`+0.67 pp`** | `33.59%` (`129/384`) | **36.43%** (`141/387`) | **`+2.84 pp`** | `12.84% / 20.38%` | **13.37% / 22.62%** |

---

### 2.2 Breakdown by Task Type (`1,641` Episodes)

| Task Type (`task_type`) | Evals | `v16` SR (%) | **`v17` SR (%)** | $\Delta$ SR | `v16` GC (%) | **`v17` GC (%)** | $\Delta$ GC | **`v17` `valid_seen` SR / GC** | **`v17` `valid_unseen` SR / GC** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pick_two_obj_and_place` | `205` | `26.34%` (`54`) | **31.71%** (`65`) | **`+5.37 pp`** | `38.27%` | **43.81%** | **`+5.54 pp`** | **38.71%** (`48/124`) / **51.15%** | **20.99%** (`17/81`) / **33.85%** |
| `pick_and_place_simple` | `242` | `36.36%` (`88`) | **41.32%** (`100`) | **`+4.96 pp`** | `36.36%` | **41.32%** | **`+4.96 pp`** | **46.48%** (`66/142`) / **46.48%** | **34.00%** (`34/100`) / **34.00%** |
| `pick_clean_then_place_in_recep` | `225` | `17.78%` (`40`) | **22.22%** (`50`) | **`+4.44 pp`** | `34.29%` | **40.03%** | **`+5.74 pp`** | **25.00%** (`28/112`) / **40.69%** | **19.47%** (`22/113`) / **39.37%** |
| `pick_and_place_with_movable_recep` | `224` | `6.25%` (`14`) | **8.48%** (`19`) | **`+2.23 pp`** | `15.17%` | **17.84%** | **`+2.67 pp`** | **6.96%** (`8/115`) / **18.44%** | **10.09%** (`11/109`) / **17.13%** |
| `look_at_obj_in_light` | `267` | `49.44%` (`132`) | **46.44%** (`124`) | `-3.00 pp` | `65.36%` | **67.04%** | **`+1.69 pp`** | **52.13%** (`49/94`) / **69.68%** | **43.35%** (`75/173`) / **65.61%** |
| `pick_heat_then_place_in_recep` | `243` | `17.28%` (`42`) | **12.35%** (`30`) | `-4.94 pp` | `30.94%` | **31.79%** | **`+0.85 pp`** | **13.08%** (`14/107`) / **34.51%** | **11.76%** (`16/136`) / **29.58%** |
| `pick_cool_then_place_in_recep` | `235` | `13.62%` (`32`) | **8.09%** (`19`) | `-5.53 pp` | `27.76%` | **23.35%** | `-4.41 pp` | **8.73%** (`11/126`) / **23.02%** | **7.34%** (`8/109`) / **23.73%** |

---

### 2.3 Episode Transition Matrix (`v16` $\rightarrow$ `v17` across `1,641` Episodes)

| Cohort / Task Type | Total | **`0 -> 1` (Good Cases)** | **`1 -> 0` (Regressions)** | **`1 -> 1` (Stable Pass)** | **`0 -> 0` (Persistent Fail)** | Net $\Delta$ Episodes |
|---|---:|---:|---:|---:|---:|---:|
| **All `valid_full1641`** | **`1,641`** | **`141`** | **`136`** | **`266`** | **`1,098`** | **`+5`** |
| `valid_seen` | `820` | `65` | `90` | `159` | `506` | `-25` |
| `valid_unseen` | `821` | `76` | `46` | `107` | `592` | **`+30`** |
| `pick_and_place_simple` | `242` | `30` | `18` | `70` | `124` | **`+12`** |
| `pick_two_obj_and_place` | `205` | `25` | `14` | `40` | `126` | **`+11`** |
| `pick_clean_then_place_in_recep` | `225` | `27` | `17` | `23` | `158` | **`+10`** |
| `pick_and_place_with_movable_recep` | `224` | `15` | `10` | `4` | `195` | **`+5`** |
| `look_at_obj_in_light` | `267` | `21` | `29` | `103` | `114` | `-8` |
| `pick_heat_then_place_in_recep` | `243` | `13` | `25` | `17` | `188` | `-12` |
| `pick_cool_then_place_in_recep` | `235` | `10` | `23` | `9` | `193` | `-13` |
| **Non-Sliced Tasks** | `1,400` | `136` | `130` | `264` | `870` | **`+6`** (`28.14%` $\rightarrow$ `28.57%`) |
| **Sliced Tasks (`*Sliced*`)** | `241` | `5` | `6` | `2` | `228` | `-1` (`3.32%` $\rightarrow$ `2.90%`) |

---

### 2.4 Execution, Verifier, Subgoal & Resource Accounting

- **Stop Reasons (`1,641` episodes):**
  - `policy_stop`: `1,530` (`93.24%`, vs. `1,085` / `66.12%` in `v16`)
  - `decision_budget`: `56` (`3.41%`, vs. `532` / `32.42%` in `v16`)
  - `failure_budget`: `53` (`3.23%`, vs. `24` / `1.46%` in `v16`)
  - `action_budget`: `2` (`0.12%`, vs. `0` in `v16`)
- **`finish_episode` Verifier Statistics:**
  - Total `finish_episode` calls: **`1,673`** (`1,543` accepted as `episode_marked_complete`, `130` rejected as `verification_failed` across `109` distinct episodes).
  - Among the `109` episodes where `finish_episode` was rejected by the verifier, **`12` episodes (`11.01%`) subsequently recovered and achieved full terminal success (`SR = 1`)**.
- **`execute_subgoal_program` Statistics:**
  - Total calls: **`683`** across `1,641` episodes (`place_held: 237`, `acquire_target: 106`, `examine_in_light: 86`, `cool_held: 60`, `heat_held: 47`, `transfer_with_movable_receptacle: 47`, `clean_held: 34`, `slice_target: 16`, unparsed/schema errors: `50`).
- **Per-Episode Averages:**
  - Decisions: **`23.48`** (vs. `18.41` in `v16`) ｜ Primitive Actions: **`101.71`** (vs. `91.77` in `v16`) ｜ Simulator Action Failures: **`4.41`** (vs. `3.39` in `v16`)
  - Perception Time: **`41.18 s`** ｜ Policy Wall Time: **`1,646.40 s`**
  - Token Usage (`1,598 / 1,641` single-turn episodes): **`1,266,548` input tokens** / **`10,915` output tokens** per episode (`2,023,943,955` total input tokens, `17,441,617` total output tokens).

---

## 3. Deep-Dive Good-Case & Bad-Case Analysis (Joint Audit with `claude-analyst`)

Following completion of the evaluation, an independent deep-dive trajectory and source-code audit was conducted jointly with `claude-analyst` (`Claude Code Opus 5.5 --effort max` via `super-orchestrator`), cross-referencing all `1,641` `attempt.json`, `tool_calls.jsonl`, `actions.jsonl`, `memory.json`, and `simulator/terminal_evaluation.json` ground-truth states against `src/codex_agent/alfred/`.

### 3.1 Where the `141` Good Cases (`v16=0 -> v17=1`) Came From

1. **Instance-Counting Pre-Commit Verification on Two-Object Tasks (`pick_two_obj_and_place`: `+5.37 pp` SR, `+5.54 pp` GC):**
   - `verify_goal_predicates` (`subgoal_program.py:935-960`) includes an explicit instance-counting rule (`two_objects_required_but_only_1_placed`) that checks `len(placed_objects) < 2`.
   - **Concrete Good Case (`ALFRED_V17_..._s35/episodes/0011`, `valid_seen/pick_two_obj_and_place-Knife-None-Drawer-18/trial_T20190907_065232_689984#3`, `v16: 0/2 -> v17: 2/2`):** At decision 17, the agent called `finish_episode` after placing only one knife. The verifier rejected completion (`verification_failed`), forcing the agent to acquire a second knife (`#62`) at decision 23 and place it into Drawer `#68` via `execute_subgoal_program(place_held)` at decision 24, achieving `2/2` terminal success.
2. **Closed-Loop Sibling Candidate Re-Resolution After Atomic Mask/Door Failures (`pick_and_place_simple`: `+4.96 pp`, `pick_clean`: `+4.44 pp`, `movable_recep`: `+2.23 pp`):**
   - When an atomic `interact_object(put)` or `navigate_to` fails due to a stale door state or missing mask, `execute_subgoal_program(place_held)` and `execute_subgoal_program(acquire_target)` re-rank 3D sibling candidates and automatically open closed receptacles.
   - **Concrete Good Case (`ALFRED_V17_..._s30/episodes/0035`, `valid_unseen/pick_heat_then_place_in_recep-Cup-None-Cabinet-10/trial_T20190907_083429_887065#2`, `v16: 1/3 -> v17: 3/3`):** After heating the cup in Microwave `#96`, atomic `put(#99 Cabinet)` failed at decision 18 with `Receptacle is CLOSED`. At decision 20, `execute_subgoal_program(place_held)` resolved sibling Cabinet `#146`, opened it, and placed `#143`, completing `3/3` goal conditions.
   - **Concrete Good Case (`ALFRED_V17_..._s30/episodes/0020`, `valid_unseen/pick_clean_then_place_in_recep-Bowl-None-Cabinet-10/trial_T20190909_061130_844814#2`, `v16: 1/3 -> v17: 3/3`):** After `clean_in_sink(#26)` at decision 8 and a failed mask arrival at Cabinet `#21` (`arrived_unverified_mask`), `scan_surroundings` + `execute_subgoal_program(place_held)` at decision 13 located detectable sibling Cabinet `#111` and placed the cleaned bowl (`3/3`).
   - **Concrete Good Case (`ALFRED_V17_..._s10/episodes/0012`, `valid_seen/pick_and_place_simple-Pencil-None-Desk-310/trial_T20190909_113054_894334#2`, `v16: 0/1 -> v17: 1/1`):** After 9 decisions of `no detection mask in current view after framing` around `#41`, `execute_subgoal_program(acquire_target)` at decision 18 acquired `#41` via multi-pitch framing and placed it on Desk `#24` at decision 19 (`1/1`).

---

### 3.2 Six Root Causes of Bad Cases (`136` Regressions `1->0` and `1,098` Persistent Failures `0->0`)

#### Root Cause 1 (P0): State Ledger Fabricates `cooled` and `cleaned` States from Geometric Proximity / Intent Rather Than Verified Execution (`memory.py`)
- **Code Locations:** `src/codex_agent/alfred/memory.py:329-335`, `704-745` (**line 743**), and `1040-1062`.
- **Mechanism:**
  1. `_infer_spatial_containment()` (`memory.py:704-745`), which runs on **every fused observation frame**, marks any object whose 3D centroid falls within a `Fridge` AABB (`+0.10 m` XY, `+0.25 m` Z margin) as `obj.action_states.add("cooled")` (`line 743`). Consequently, **any object pre-existing inside or near a Fridge (or merely observed when the Fridge door opens) is immediately marked `cooled = True` without the agent ever placing it in the Fridge and closing the door**.
  2. Cross-checking the memory ledger against `terminal_evaluation.json` ground truth (`cooled_objects`) reveals that **67% (`133 / 199`) of cool-task episodes where the ledger claims `cooled = True` have zero cooled objects in AI2-THOR ground truth**. Both the LLM (via `verify_task_goal`) and the `finish_episode` verifier see `object_cooled = True` and terminate the episode without cooling the target! This directly caused `pick_cool_then_place_in_recep` to drop from `13.62%` (`32/235`) in `v16` to **`8.09%` (`19/235`)** in `v17`.
  3. Similarly, in `record_interaction` (`memory.py:1040-1062`), `ToggleObjectOn` on a `Faucet` marks **every object within `0.75 m` of the faucet** as `cleaned = True`—including the object **still held in the agent's hand** before it has ever been placed into the `SinkBasin` (e.g., `s14/episodes/0012`, where `ButterKnife` is marked `cleaned=True` in hand at frame 29 and 19 surrounding kitchen objects are marked `cleaned=True`, while GT `cleaned_objects = []`).

#### Root Cause 2 (P0): Pre-Commit Verifier Blind Spots, Category/Instance Agnosticism, and Exhausted-Budget Bypass (`subgoal_program.py`, `episode_server.py`)
- **Code Locations:** `src/codex_agent/alfred/subgoal_program.py:782-987` and `src/codex_agent/alfred/episode_server.py:654-680`.
- **Mechanism:**
  1. **Category- and Instance-Agnostic Predicates:** `verify_goal_predicates` checks only whether *any* `PutObject` occurred in the episode (`subgoal_program.py:875-881`, satisfied even by unloading a knife after slicing) and whether *any* object in memory has `heated`/`cooled`/`cleaned` (`lines 885-926`), without checking that the **same object instance** that was state-changed is the **target category** placed into the **destination receptacle category**.
  2. **Examine-in-Light Blindness:** For `look_at_obj_in_light`, `subgoal_program.py:829-853` checks only `held_object_id is not None` (ignoring whether the held object matches the target category—55 failed episodes finished holding the wrong object) and checks whether *any* lamp is toggled on (ignoring `DeskLamp` vs. `FloorLamp` mismatch—21 failed episodes toggled a `DeskLamp` for a `FloorLamp` goal—and ignoring whether the lamp is visible in the final frame—16 failed episodes).
  3. **Instruction Regex Underspecification (~19% of State Tasks):** For `~19%` of heat/cool/clean episodes, human instructions omit explicit thermal/cleaning keywords (e.g., *"Get an apple and place it on the counter"* in `s17/episodes/0018`, or *"Put a piece of lettuce in the fridge"* for a `clean` task), so regex-only goal inference (`subgoal_program.py:794-826`) never activates the state predicate.
  4. **Silent Completion on Budget Exhaustion:** In `episode_server.py:654-660`, when `can_remediate` is `False` (because `failures == 10` or `decisions == 35` or `finish_rejections >= 2`), `finish_episode` still returns `status = "episode_marked_complete"`, converting 466 verifier-failing episodes into `policy_stop` (`93.24%` of all episodes).

#### Root Cause 3 (P0): `PutObject` Physical Bottleneck (`60.4%` Failure Rate), Unsubtracted Container Masks, and Retry Failure Burn (`memory.py`, `macro_controller.py`)
- **Code Locations:** `src/codex_agent/alfred/memory.py:182-189`, `893-969` (**lines 965-968**), and `src/codex_agent/alfred/macro_controller.py:745-770`, `896-930`.
- **Mechanism:**
  1. Across the benchmark, `PutObject` is attempted `4,399+` times and fails **`60.4%` of the time** (`1,571` `"No valid positions to place object found"`, `386` `"Receptacle is CLOSED"`, `217` `"NOT a receptacle"`, `215` `"No valid Receptacle found"`).
  2. **Raw Mask Override on `PutObject` (`memory.py:965-968`):** Although `select_interaction_mask` documents that `PutObject` should receive the child-subtracted `effective_mask` (so placement targets empty receptacle surface rather than already-placed objects inside the container), lines `965-968` overwrite `chosen_mask` with `raw_mask` whenever `raw_mask` has more pixels!
  3. **Cross-Category Spatial Fallback (`memory.py:182-189`):** `_is_compatible_spatial_fallback` allows `PutObject` on a `Fridge` to accept a `Cabinet`, `Shelf`, or `CounterTop` mask (`78.6%` failure rate on spatial-fallback masks; `105 / 246` Fridge `PutObject` failures used a non-Fridge spatial fallback mask).
  4. **Micro-Adjustment Retry Loop & Pre-Emptive `OpenObject` Burn (`macro_controller.py:745-770`, `896-930`):** In `v17`, pre-emptive `OpenObject` before `PutObject` into `Fridge`/`Microwave`/`Safe` and an unguarded 3-step pitch/yaw retry loop in `macro_controller.py:896-930` burn up to **3–4 simulator failures per `PutObject` call** without backing up (`MoveBack`) when standing too close. On `valid_seen` `pick_heat` and `pick_cool`, this burned the 10-failure budget during final Fridge/Cabinet placement (`s11/episodes/0012`), directly explaining why `valid_seen` `pick_heat` dropped from `29` to `14` successes.

#### Root Cause 4 (P1): Sliced-Object Pipeline Collapse (`2.90%` SR on `241` `*Sliced*` Episodes vs. `28.57%` on Non-Sliced)
- **Code Locations:** `src/codex_agent/alfred/object_mapping.py:145-149`, `src/codex_agent/alfred/subgoal_program.py:242`, and `src/codex_agent/alfred/macro_controller.py`.
- **Mechanism:**
  1. **`unverified_sliced_category` Hard Filter (`object_mapping.py:145-149`):** Any detection ending in `"Sliced"` with GroundingDINO confidence `< 0.50` is unconditionally discarded before 3D fusion. Because small food slices (`PotatoSliced`, `LettuceSliced`, `TomatoSliced`, `AppleSliced`) rarely exceed `0.50` confidence, **61.3% of sliced tasks never register a `*Sliced` node in memory**.
  2. **Candidate Ranking Exclusion (`subgoal_program.py:242`):** `rank_candidate_objects` skips any node with `"sliced" in obj.action_states`, making freshly sliced objects unfindable by category in subsequent subgoals!
  3. **Atomic `SliceObject` Without Knife Precondition:** Atomic `interact_object(action="SliceObject")` does not auto-acquire a `Knife`/`ButterKnife` (89 failures with `"Agent should be holding a knife before slicing"`), and 59 episodes sliced the wrong object due to drifted interaction masks.

#### Root Cause 5 (P1): `look_at_obj_in_light` Lamp-Type Confusion & Final-Frame Visibility (`subgoal_program.py`, `memory.py`)
- **Code Locations:** `src/codex_agent/alfred/subgoal_program.py:646-690`, `838-853`.
- **Mechanism:**
  - `_LIGHT_CATEGORIES` treats `DeskLamp` and `FloorLamp` as interchangeable, causing 21 episodes to toggle a `DeskLamp` when the PDDL goal requires a `FloorLamp`. Another 16 episodes toggled the correct lamp but finished with the lamp outside the camera frustum (`visible = False` in AI2-THOR).

#### Root Cause 6 (P1): Late Invocation & Schema Friction of Fixed-Template `execute_subgoal_program` (`subgoal_program.py`, `playbook.py`)
- **Code Locations:** `src/codex_agent/alfred/subgoal_program.py:283-520` and `src/codex_agent/alfred/playbook.py`.
- **Mechanism:**
  - Because `playbook.py` frames `execute_subgoal_program` as an optional helper alongside atomic tools, Codex invokes it at **median decision 21 of 35**—as a last resort after atomic actions have already exhausted most of the failure budget. When `can_step()` is already `False`, the interpreter exits its loop on the first check and returns an empty trace (`"Failed to place #10 across candidates [...]. Trace: "`). Furthermore, 50 calls failed pre-execution due to missing `target_categories` validation or hallucinated subgoal names.

---

## 4. Action Plan for `v18` (Validated on `valid_strat150` Before Full Certification)

1. **Strict Verified State Ledger (`memory.py`):**
   - Remove `obj.action_states.add("cooled")` from `_infer_spatial_containment` (`memory.py:743`).
   - Record `cooled` only when an object was placed into a `Fridge` (`placed_by_agent`) AND `CloseObject` succeeded on that `Fridge` while the object remained inside.
   - Restrict `cleaned` on `ToggleObjectOn(Faucet)` strictly to objects inside the paired `SinkBasin` (`parent_receptacle_id`), never marking `held_object_id` or nearby counter objects.
2. **Instance- & Destination-Aware Pre-Commit Verifier (`subgoal_program.py`, `episode_server.py`):**
   - Extract target object categories, destination receptacle categories, and specific lamp type (`DeskLamp` vs. `FloorLamp`) from the instruction and memory catalog.
   - Require the **same object instance** to carry both `placed_by_agent` into the destination receptacle category AND the required state (`cleaned`/`heated`/`cooled`/`sliced`).
   - For `look_at_obj_in_light`, verify that `held_object_id` matches the target category AND a toggled lamp of the target type is **currently visible** in the final frame.
   - When `can_step()` is `False` (budget exhausted), record the true budget termination reason (`failure_budget` / `decision_budget`) rather than accepting a false `episode_marked_complete`.
3. **`PutObject` Mask Subtraction, Exact Receptacle Fallback & Back-Off Recovery (`memory.py`, `macro_controller.py`):**
   - Restrict `raw_mask` preference in `select_interaction_mask` (`memory.py:965-968`) to `OpenObject`/`CloseObject` so `PutObject` always uses the child-subtracted `effective_mask` (targeting uncluttered receptacle surface).
   - Require exact category match (`cand_cat == target_cat`) when using spatial-fallback masks for `PutObject` on containers (`Fridge`, `Cabinet`, `Drawer`, `Microwave`, `SinkBasin`).
   - Break the micro-adjustment retry loop after 1 failed adjustment and add a `0.25 m` step-back (`MoveBack`) stance recovery when `PutObject` or `OpenObject` fails at close range.
4. **Sliced-Object Retention & Candidate Ranking Fix (`object_mapping.py`, `subgoal_program.py`):**
   - Lower the `unverified_sliced_category` threshold (`object_mapping.py:145-149`) from `0.50` to `0.30` (or `0.25` after a `SliceObject` action in the episode) and remove the `"sliced" in obj.action_states` exclusion in `rank_candidate_objects` (`subgoal_program.py:242`).

---

## 5. Artifact Locations

- Merged run summary & metadata: `outputs/alfred/ALFRED_V17_S42_20261107_valid1641/{summary.json,run.json}`
- Per-shard directories (`42` shards): `outputs/alfred/ALFRED_V17_S42_20261107_valid1641_s00` .. `outputs/alfred/ALFRED_V17_S42_20261107_valid1641_s41`
- Benchmark SQLite archive: `docs/benchmark/alfred/runs.sqlite` (`run_id = "ALFRED_V17_S42_20261107_valid1641"` and `ALFRED_V17_S42_20261107_valid1641_s00..s41`)
- Independent Claude Code audit report: `~/.super-orchestrator/alfred-v17-review/findings/claude_badcase_goodcase_analysis.md`
