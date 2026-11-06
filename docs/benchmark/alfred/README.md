# ALFRED benchmark archive

This track evaluates incremental scene memory during interactive ALFRED task
execution. [Implementation and setup](../../codex_agent/alfred.md) describe the
actor's observations and external dependencies.

**Latest full-validation run (`1,641 / 1,641`): SR 24.50% (`402 / 1,641`), GC 33.70% (`1,425 / 4,229`), `valid_seen` SR 30.37% (`249 / 820`) / GC 36.94%, and `valid_unseen` SR 18.64% (`153 / 821`) / GC 30.47%.** On the matched `pilot50` subset (`50/50`), `v16` reaches **SR 32.00% (`16/50`), `valid_seen` SR 44.00% (`11/25`), `valid_unseen` SR 20.00% (`5/25`)** (`3.2×` over `v8`). See [v16 full validation results and bottleneck analysis](v16_full_valid1641_20261107.md).

**Earlier online-mapping prefix5 (`v15`): 5/5 scores, SR 3/5 (60.00%) and GC 7/12 (58.33%).** See [v15 root-cause fixes](v15_root_cause_fixes_prefix5_20261106.md) and [v10 results and two replays](v10_prefix5_20261105.md).
The remaining planned tasks in pilot15 were cancelled; pilot15 is not a completed cohort.

**The historical integration pilot has 50/50 official terminal scores: five successes,
SR 10.00%, and GC 20/126 = 15.87%.** [v8](v8_pilot50_complete_20261105.md)
combines the unchanged 41 scores from v6 with nine newly completed tasks under
the same configuration. All 50 saved terminal scores passed independent
rechecking. [v7](v7_recovery_failures_20261105.md) preserves the unsuccessful
recovery attempts separately; none became synthetic zero-score tasks.

The new online OpenEQA mapping experiment uses a separate frozen **pilot15**
(7 seen + 8 unseen); see [v9 protocol and status](v9_online_mapping_20261105.md).
Old v8 scores are not scores of the new implementation.

## Protocol

ALFRED's official `oct21.json` has 820 `valid_seen` and 821 `valid_unseen`
annotation entries. An episode is identified by `(split, task, repeat_idx)`;
different annotations of the same trajectory are distinct official entries.
The seven task types are simple placement, placement using a movable receptacle,
two-object placement, cleaning, heating, cooling, and examining an object under
a light. [Official data/code](https://github.com/askforalfred/alfred).

The first local pilot contains **50 trajectories: 25 seen + 25 unseen**, with
all seven task types represented in each split. `create_pilot_manifest` defines
the deterministic protocol: seed 0, SHA-256 ordering of `seed:episode_id`, one
annotation per unique trajectory, then round-robin selection across sorted task
types. Thus the first four types receive four entries and the last three receive
three entries per split when sufficient candidates exist. The manifest retains
the official splits-file hash and each selected trajectory's hash.

The frozen file is `folds/pilot50.json`. It is the first matched fold for
this project's ALFRED comparisons, **not an official ALFRED subset**, and its
balanced task distribution is not the full-split distribution. Preserve it for
future comparisons; do not resample after seeing results. Report seen/unseen
separately. Any combined pilot number is descriptive, not an official headline.

Default input is H (high-level goal), with H+L recorded separately. The initial
research setting uses predicted depth and instance masks, motion odometry,
yaw/pitch proprioception, and action-success feedback. It cannot be compared
directly to strict RGB-only leaderboard submissions. Gold plans and simulator
object metadata remain evaluator-only. No LLM judge is used: terminal success
and goal conditions come from the pinned official ALFRED scorer.

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
| [v16 full validation 1641](v16_full_valid1641_20261107.md) | 2026-11-07 | Full official `oct21` validation (`820 valid_seen + 821 valid_unseen = 1,641`), resilient gateway pre-commit lease rejection + retries, 11-slot multi-GPU perception server, and 84-worker evaluation (`gemini-3.8-flash`) | **`1,641/1,641` scored (`0` errors): Overall SR 24.50% (`402/1,641`), GC 33.70% (`1,425/4,229`), PLW-SR 12.83%, PLW-GC 20.39%; `valid_seen` SR 30.37% (`249/820`), GC 36.94%; `valid_unseen` SR 18.64% (`153/821`), GC 30.47%; matched `pilot50` SR 32.00% (`16/50`)** |
| [v15 root-cause fixes prefix5](v15_root_cause_fixes_prefix5_20261106.md) | 2026-11-06 | Held-object self-occlusion suppression, category-compatible spatial fallback, multi-pitch auto-framing, container geometry preservation, and CLI/SDK budget decoupling on `prefix5` | **5/5 scored, SR 3/5 (60.00%, `i00`, `i01`, `i04` solved), GC 7/12 (58.33%), PLW-SR 44.7%** |
| [v14 simulator GT depth RGB-D prefix5](v14_simulator_gt_depth_prefix5_20261105.md) | 2026-11-05 | `depth_source = "simulator_gt"` RGB-D mode (`0.34s/f`), `6cm` 3D association on `prefix5` (`gemini-3.8-flash`) | **5/5 scored, SR 1/5 (20.00%, `i01` solved in 30 steps), GC 3/12 (25.00%)**; 3D XY error **0.19–0.24m**, Free/Occupied ratio **+183% (`0.66` $\rightarrow$ `1.87`)** |
| [v13 mapping & perception upgrade prefix5](v13_mapping_perception_upgrade_20261105.md) | 2026-11-05 | Batched `bfloat16` perception (`0.38s/f`), CLIP semantic gate, `12cm` monocular association + mask erosion, adaptive floor calibration & vectorized raycast (`18ms/f`) | **Step latency 2.4× faster (`0.85–1.13s/step`), 3D object fragmentation reduced by 66.4% (`247.8` $\rightarrow$ `83.3` nodes/50f), free/occupied floor ratio +89%** |
| [v12 macro tools + 85 categories prefix5](v12_macro_tools_85cats_prefix5_20261105.md) | 2026-11-05 | Single-turn Codex macro-tool loop, 85 task categories, GPU-resident perception, AABB/OBB caching on `prefix5` (`gemini-3.8-flash`) | **5/5 scored, SR 1/5 (20.00%), GC 3/12 (25.00%), PLW-SR 7.68%, PLW-GC 14.34%**; perception **4.6× faster (0.73 s/frame)** |
| [v10 online mapping prefix5](v10_prefix5_20261105.md) | 2026-11-05 | User reduced the run to original indices0–4, all seen; two visualized episodes | 5/5 scored, SR0/5, GC1/12; 5/5 official rechecks agree; four excluded cancellations separately archived |
| [v9 online mapping](v9_online_mapping_20261105.md) | 2026-11-05 | New mapper, train-only depth comparison, frozen pilot15 | Setup, train-depth comparison and full gates passed; task results in v10 |
| [v8 completed pilot50](v8_pilot50_complete_20261105.md) | 2026-11-05 | Original41 + nine unchanged-configuration recoveries | 50/50 scored, 5 successes; SR 10.00%, GC 15.87%; independent scores 50/50 agree |
| [v7 failed recovery rounds](v7_recovery_failures_20261105.md) | 2026-11-05 | Two recovery rounds; three unscored attempts each | No new scores; nine historical error attempts retained across five identities |
| [v6 partial pilot and upstream outage](v6_upstream_outage_20261105.md) | 2026-11-05 | Frozen50, predicted perception, H input | 41/50 scored, 5 successes; 3 upstream errors and 6 pending; 41/41 independent score checks agree |
| [v5 loopback routing](v5_loopback_proxy_20261105.md) | 2026-11-05 | Physical scalar-wire rollout and first pilot launch | Transient local health 502; pilot interrupted unscored; run-local proxy bypass verified |
| [v4 schema compatibility](v4_schema_compatibility_20261105.md) | 2026-11-05 | Same-code medium retry and controlled command probes | Medium retry 0 actions; scalar wire format passes 7/7 command families |
| [v3 repaired contract](v3_smoke1_20261105.md) | 2026-11-05 | Same prefix1; nested commands and progressive retrieval | 1/1 scored; early Stop, 0 actions, SR 0 and GC 0/2 |
| [v2 first rollout](v2_smoke1_20261105.md) | 2026-11-05 | Frozen-fold prefix1; predicted perception and LLM actions | Decision-contract error after 10 actions; 0/1 scored |
| [v1 setup](v1_setup_20261105.md) | 2026-11-05 | Simulator lifecycle, camera calibration, predicted perception, pilot protocol | Infrastructure checks passed; 0 scored episodes |

See the [local leaderboard](leaderboard.md). Before launching evaluation,
`AGENTS.md` requires a clean tracked tree and recorded branch, HEAD and executed
code hash; new executable files must also be committed. The reviewed
implementation was evaluated from a clean committed tree. The completed
cohort retains the first terminal score for each frozen identity; scored task
failures were not rerun or replaced to improve the result. Future comparisons
require a new version record and the same frozen fold.
