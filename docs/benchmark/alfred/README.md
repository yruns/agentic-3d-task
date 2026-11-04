# ALFRED benchmark archive

This track evaluates incremental scene memory during interactive ALFRED task
execution. [Implementation and setup](../../codex_agent/alfred.md) describe the
actor's observations and external dependencies.

**The first scored prefix diagnostic failed by early policy termination.**
It reached official terminal scoring with zero physical actions; pilot50 has
not run yet. A simulator reset or neural forward pass is not a task success.

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

The intended frozen file is `folds/pilot50.json`. It is the first matched fold for
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
| [v4 schema compatibility](v4_schema_compatibility_20261105.md) | 2026-11-05 | Same-code medium retry and controlled command probes | Medium retry 0 actions; scalar wire format passes 7/7 command families |
| [v3 repaired contract](v3_smoke1_20261105.md) | 2026-11-05 | Same prefix1; nested commands and progressive retrieval | 1/1 scored; early Stop, 0 actions, SR 0 and GC 0/2 |
| [v2 first rollout](v2_smoke1_20261105.md) | 2026-11-05 | Frozen-fold prefix1; predicted perception and LLM actions | Decision-contract error after 10 actions; 0/1 scored |
| [v1 setup](v1_setup_20261105.md) | 2026-11-05 | Simulator lifecycle, camera calibration, predicted perception, pilot protocol | Infrastructure checks passed; 0 scored episodes |

See the [local leaderboard](leaderboard.md). Before launching evaluation,
`AGENTS.md` requires a clean tracked tree and recorded branch, HEAD and executed
code hash; new executable files must also be committed. The reviewed
implementation is prepared for a clean-source launch. Evaluation status remains
pending until an actual agent run and its logs are available.
