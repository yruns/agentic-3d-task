# NR3D v10 skill10 - minimal prompt explicit skill smoke

This smoke reruns the first 10 strat600 samples after moving the NR3D tool
workflow into `.agents/skills/nr3d-codex-tools/SKILL.md`. In explicit Skill mode
the task prompt only carries task data, including `scene_dir`; it no longer
inlines the tool playbook or repeats CLI invocation instructions.

> **Status: COMPLETE - 10/10.** Finished 2026-07-01 (`exit=0`).
> Raw artifacts: `tmp/nr3d_tools_skill10_minprompt_20260701/`.
> Derived metrics: `assets/v10_skill10_minprompt_20260701_leaderboard.json`.
>
> **Takeaway:** minimal-prompt Skill mode matched the previous v8/v9 first-10
> behavior: **8/10 = 80.00%**, no permanent failures, no tool-loop interrupts,
> no turn timeouts, and no `SKILL.md` loop matches beyond the launch-time git
> status line.

## Setup

| Item | Value |
|---|---|
| Scope | First 10 samples from `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Head at launch | `9b262f4` |
| Worktree | Dirty: current Skill-mode prompt/skill/test edits |
| Mode | `--tools --skill-path .agents/skills/nr3d-codex-tools/SKILL.md` |
| Prompt | No inline playbook, no `Tool context`, no CLI invocation block |
| Workers | 10 |
| Reasoning summary | `auto` |
| Turn timeout | 900 s |
| Adapter route | Responses API via `codex_modelhub_adapter` |

## Result

| Metric | Value |
|---|---:|
| Samples | 10/10 |
| Status | 10 completed, 0 failed |
| Acc@0.25 | **80.00%** |
| Acc@0.50 | **80.00%** |
| Mean IoU | 0.8000 |
| Cache hit rate | 1.0000 |
| Mean cache ratio | 0.9725 |
| Reasoning summaries | 9/10 |
| Business summaries | 10/10 |
| Latency | median 88.2 s, mean 98.0 s, max 190.6 s |
| Route | 92 `/v1/responses`; 0 chat/crawl |
| Loop signals | 0 max-tool-call interrupts, 0 repeated-call interrupts, 0 turn timeouts |

First-10 comparison:

| Run | Correct | Wrong samples |
|---|---:|---|
| v8 first10 | 8/10 | `scene0500_00::25::13471`, `scene0608_00::9::17679` |
| v9 skill20 first10 | 8/10 | `scene0500_00::25::13471`, `scene0608_00::9::17679` |
| v10 skill10 minimal prompt | **8/10** | `scene0500_00::25::13471`, `scene0608_00::9::17679` |

Wrong samples:

| Sample | Selected | Reason |
|---|---:|---|
| `scannet/scene0500_00::25::13471` | 26 | Same window/chalkboard semantic ambiguity as prior runs. |
| `scannet/scene0608_00::9::17679` | 8 | Same ottoman/loveseat semantic ambiguity as prior runs. |

## Interpretation

Moving CLI usage and tool workflow instructions into the Skill did not regress
the first-10 smoke. Compared with the previous explicit-Skill prompt, the
minimal prompt preserves accuracy on this slice and improves the cache signature
on the smoke (`1.000/0.973` vs v9 first-20 `0.850/0.822`). This is a small
sample, so the next gate remains a larger run before treating Skill mode as a
default.

## Reproduction Assets

- Launch script: `docs/benchmark/nr3d/assets/run_v10_skill10_minprompt_20260701.sh`
- Raw output: `tmp/nr3d_tools_skill10_minprompt_20260701/`
- Derived metrics:
  `docs/benchmark/nr3d/assets/v10_skill10_minprompt_20260701_leaderboard.json`
