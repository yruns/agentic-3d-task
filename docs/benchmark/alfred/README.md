# ALFRED benchmark archive

This track evaluates incremental scene memory during interactive ALFRED task
execution. [Implementation and setup](../../codex_agent/alfred.md) describe the
actor's observations and external dependencies.

**Latest full-validation run (`v17`, `1,641 / 1,641`): SR 24.80% (`407 / 1,641`, `+0.30 pp`), GC 35.54% (`1,503 / 4,229`, `+1.84 pp`), `valid_seen` SR 27.32% (`224 / 820`) / GC 36.32%, and `valid_unseen` SR 22.29% (`183 / 821`, `+3.65 pp`) / GC 34.76% (`+4.29 pp`).** On the canonical 150-episode stratified iteration subset [`valid_strat150`](valid_strat150_protocol_20261108.md) (`75 seen + 75 unseen`), `v17` scores **SR 25.33% (`38/150`), GC 36.43% (`141/387`), PLW-SR 13.37%, PLW-GC 22.62% (`valid_seen` SR 28.00% / GC 36.98%; `valid_unseen` SR 22.67% / GC 35.90%)**, and `v16` scores **SR 24.67% (`37/150`), GC 33.59%, PLW-SR 12.84%, PLW-GC 20.38%**. See [v17 full validation & good-case/bad-case audit](v17_guarded_subgoals_valid1641_20261108.md) and [v16 full validation results](v16_full_valid1641_20261107.md).

**Earlier online-mapping prefix5 (`v15`): 5/5 scores, SR 3/5 (60.00%) and GC 7/12 (58.33%).** See [v15 root-cause fixes](v15_root_cause_fixes_prefix5_20261106.md) and [v10 results and two replays](v10_prefix5_20261105.md).
The remaining planned tasks in pilot15 were cancelled; pilot15 is not a completed cohort.

## Protocol

ALFRED's official `oct21.json` has 820 `valid_seen` and 821 `valid_unseen`
annotation entries. An episode is identified by `(split, task, repeat_idx)`;
different annotations of the same trajectory are distinct official entries.
The seven task types are simple placement, placement using a movable receptacle,
two-object placement, cleaning, heating, cooling, and examining an object under
a light. [Official data/code](https://github.com/askforalfred/alfred).

For **rapid architectural iteration, policy-program optimization, and thesis
ablations**, the canonical high-fidelity fold is **`folds/valid_strat150.json`**
(**150 unique physical trajectories: `75 valid_seen + 75 valid_unseen`**; see
[valid_strat150 protocol and verification](valid_strat150_protocol_20261108.md)).
`valid_strat150.json` uses exact Hamilton proportional allocation across all 14
`(split, task_type)` cells and aligns room distributions (`Kitchen`, `LivingRoom`,
`Bedroom`, `Bathroom`), slicing requirements (`*Sliced*`), gold action lengths,
and `(v16, v17)` empirical transition regimes so that all headline metrics
(`SR`, `GC`, `PLW-SR`, `PLW-GC`) match the full 1,641-episode validation
benchmark within `< 0.35 pp` at `10.9×` lower evaluation cost. Full milestone
releases are certified on `folds/valid_full1641.json`. Report seen/unseen
separately.

Default input is H (high-level goal), with H+L recorded separately. The
research setting uses predicted or simulator depth and instance masks, motion
odometry, yaw/pitch proprioception, and action-success feedback. It cannot be
compared directly to strict RGB-only leaderboard submissions. Gold plans and
simulator object metadata remain evaluator-only. No LLM judge is used: terminal
success and goal conditions come from the pinned official ALFRED scorer.

## Metrics and accounting

For episode `i`, let `s_i` be terminal success, `c_i / n_i` completed/total goal
conditions, `g_i` expert low-action count, and `a_i` attempted agent low-actions.
The efficiency factor is `e_i = g_i / max(g_i, a_i)`.

```text
SR    = sum(s_i) / number_of_scored_episodes
GC    = sum(c_i) / sum(n_i)
PLWSR = sum(g_i * e_i * s_i) / sum(g_i)
PLWGC = sum(g_i * e_i * c_i / n_i) / sum(g_i)
```

GC uses condition totals, not a mean of episode fractions. Path-weighted metrics
use expert-length weights, as in the
[official evaluator](https://github.com/askforalfred/alfred/blob/master/models/eval/eval_task.py).
The local counter includes attempted failed actions; Stop is not a simulator
action. The historical evaluator misses the final failed action when its failure
limit triggers before incrementing its counter. Zero-action Stop uses efficiency
1 rather than the historical zero-division failure.

Always report scored/requested coverage, infrastructure errors, action/failure/
decision budgets, stop reasons, model calls, tokens, perception/policy time, and
artifact locations alongside task metrics. Errors without an authoritative
terminal score do not become synthetic zero-goal episodes. Every invocation,
including partial or failed runs, needs a dated record and SQLite ingestion via
`codex_agent.alfred.results.ingest_run`; `runs.sqlite` stores run provenance and
every episode attempt without overwriting old run IDs.

## Timeline

| Record | Date | Scope | Result |
|---|---|---|---|
| [v17 guarded subgoals & verifier full validation 1641](v17_guarded_subgoals_valid1641_20261108.md) | 2026-11-08 | Full `oct21` validation (`1,641/1,641`), guarded subgoal policy interpreter (`execute_subgoal_program`), pre-commit goal verifier (`verify_task_goal`), container–content topology & state ledger, and joint `claude-analyst` good-case/bad-case root-cause audit | **`1,641/1,641` scored (`0` errors): Overall SR 24.80% (`407/1,641`, `+0.30 pp`), GC 35.54% (`1,503/4,229`, `+1.84 pp`), PLW-SR 12.98%, PLW-GC 21.72%; `valid_unseen` SR 22.29% (`183/821`, `+3.65 pp`), GC 34.76% (`+4.29 pp`); `valid_seen` SR 27.32% (`224/820`), GC 36.32%; `valid_strat150` SR 25.33% (`38/150`), GC 36.43%** |
| [valid_strat150 canonical iteration fold](valid_strat150_protocol_20261108.md) | 2026-11-08 | Frozen 150-trajectory stratified fold (`75 valid_seen + 75 valid_unseen`, `folds/valid_strat150.json`) for rapid iteration and ablation | **Matches `valid_full1641` across `v16` (`24.67%` vs `24.50%` SR; `33.59%` vs `33.70%` GC) and `v17` (`25.33%` vs `24.80%` SR; `36.43%` vs `35.54%` GC) at `10.9×` speedup** |
| [v16 full validation 1641](v16_full_valid1641_20261107.md) | 2026-11-07 | Full official `oct21` validation (`820 valid_seen + 821 valid_unseen = 1,641`), resilient gateway pre-commit lease rejection + retries, 11-slot multi-GPU perception server, and 84-worker evaluation (`gemini-3.8-flash`) | **`1,641/1,641` scored (`0` errors): Overall SR 24.50% (`402/1,641`), GC 33.70% (`1,425/4,229`), PLW-SR 12.83%, PLW-GC 20.39%; `valid_seen` SR 30.37% (`249/820`), GC 36.94%; `valid_unseen` SR 18.64% (`153/821`), GC 30.47%; `valid_strat150` SR 24.67% (`37/150`)** |
| [v15 root-cause fixes prefix5](v15_root_cause_fixes_prefix5_20261106.md) | 2026-11-06 | Held-object self-occlusion suppression, category-compatible spatial fallback, multi-pitch auto-framing, container geometry preservation, and CLI/SDK budget decoupling on `prefix5` | **5/5 scored, SR 3/5 (60.00%, `i00`, `i01`, `i04` solved), GC 7/12 (58.33%), PLW-SR 44.7%** |
| [v14 simulator GT depth RGB-D prefix5](v14_simulator_gt_depth_prefix5_20261105.md) | 2026-11-05 | `depth_source = "simulator_gt"` RGB-D mode (`0.34s/f`), `6cm` 3D association on `prefix5` (`gemini-3.8-flash`) | **5/5 scored, SR 1/5 (20.00%, `i01` solved in 30 steps), GC 3/12 (25.00%)**; 3D XY error **0.19–0.24m**, Free/Occupied ratio **+183% (`0.66` $\rightarrow$ `1.87`)** |
| [v13 mapping & perception upgrade prefix5](v13_mapping_perception_upgrade_20261105.md) | 2026-11-05 | Batched `bfloat16` perception (`0.38s/f`), CLIP semantic gate, `12cm` monocular association + mask erosion, adaptive floor calibration & vectorized raycast (`18ms/f`) | **Step latency 2.4× faster (`0.85–1.13s/step`), 3D object fragmentation reduced by 66.4% (`247.8` $\rightarrow$ `83.3` nodes/50f), free/occupied floor ratio +89%** |
| [v12 macro tools + 85 categories prefix5](v12_macro_tools_85cats_prefix5_20261105.md) | 2026-11-05 | Single-turn Codex macro-tool loop, 85 task categories, GPU-resident perception, AABB/OBB caching on `prefix5` (`gemini-3.8-flash`) | **5/5 scored, SR 1/5 (20.00%), GC 3/12 (25.00%), PLW-SR 7.68%, PLW-GC 14.34%**; perception **4.6× faster (0.73 s/frame)** |
| [v10 online mapping prefix5](v10_prefix5_20261105.md) | 2026-11-05 | User reduced the run to original indices 0–4, all seen; two visualized episodes | 5/5 scored, SR 0/5, GC 1/12; 5/5 official rechecks agree; four excluded cancellations separately archived |
| [v9 online mapping](v9_online_mapping_20261105.md) | 2026-11-05 | New mapper, train-only depth comparison, frozen pilot15 | Setup, train-depth comparison and full gates passed; task results in v10 |

See the [local leaderboard](leaderboard.md). Before launching evaluation,
`AGENTS.md` requires a clean tracked tree and recorded branch, HEAD and executed
code hash; new executable files must also be committed. The reviewed
implementation was evaluated from a clean committed tree. The completed
cohort retains the first terminal score for each frozen identity; scored task
failures were not rerun or replaced to improve the result. Future comparisons
require a new version record and the same frozen fold.
