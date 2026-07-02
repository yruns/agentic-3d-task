# NR3D v11 skill600 - minimal prompt explicit skill

This full run validates explicit Codex Skill mode after moving the NR3D tool
workflow into `.agents/skills/nr3d-codex-tools/SKILL.md`. In this mode the task
prompt carries task data and `scene_dir`, but does **not** inline the tool
playbook, `Tool context`, or CLI invocation instructions.

> **Status: COMPLETE - 600/600.** Finished 2026-07-01 (`exit=0`).
> Raw artifacts: `tmp/nr3d_tools_skill600_minprompt_20260701/`.
> Derived metrics: `assets/v11_skill600_minprompt_20260701_leaderboard.json`.
>
> **Takeaway:** minimal-prompt Skill mode lands at **507/600 = 84.50%**, exactly
> matching v8 overall while preserving a stronger cache signature
> (`0.992/0.943`). No adapter errors, no encrypted-state errors, and no Skill
> re-read loop were observed. Two samples ended as model-level absent decisions.

## Setup

| Item | Value |
|---|---|
| Scope | Canonical strat600, `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Head at launch | `9b262f4` |
| Worktree | Dirty: current Skill-mode prompt/skill/test/docs edits |
| Mode | `--tools --skill-path .agents/skills/nr3d-codex-tools/SKILL.md` |
| Prompt | No inline playbook, no `Tool context`, no CLI invocation block |
| Workers | 40 |
| Reasoning summary | `auto` |
| Turn timeout | 900 s |
| Adapter route | Responses API via `codex_modelhub_adapter` |

## Result

| Slice | n | Correct | Acc@0.25 |
|---|---:|---:|---:|
| **Overall** | 600 | **507** | **84.50%** |
| Easy | 290 | 259 | 89.31% |
| Hard | 310 | 248 | 80.00% |
| **View-Dep** | 211 | 167 | 79.15% |
| View-Indep | 389 | 340 | 87.40% |

Health:

| Check | Value |
|---|---:|
| Sample records | 600/600 |
| Status | 598 completed, 2 failed |
| Wrong or failed | 93 |
| Reasoning summaries | 569/600 |
| Business summaries | 600/600 |
| Cache hit rate | 0.9917 |
| Mean cache ratio | 0.9434 |
| Latency | median 193.5 s, mean 230.9 s, p90 390.0 s, max 891.7 s |
| Route | 5842 `/v1/responses`; 0 chat/crawl |
| Upstream errors | 0 HTTP 400; 0 HTTP 5xx; 0 `invalid_encrypted_content` |
| Loop/guard signals | 7 turn-timeout finalizations; 0 max-tool-call interrupts; 0 repeated-call interrupts |

Comparison:

| Run | Correct | Overall | View-Dep | Cache |
|---|---:|---:|---:|---:|
| v7 responses passthrough | 516/600 | 86.00% | 82.46% | 0.987 / 0.930 |
| v8 fail-closed encrypted state | 507/600 | 84.50% | 77.25% | 0.990 / 0.930 |
| **v11 minimal-prompt Skill** | **507/600** | **84.50%** | **79.15%** | **0.992 / 0.943** |

Failed samples:

| Sample | Reason |
|---|---|
| `scannet/scene0435_00::21::13508` | The model judged the bathroom entrance as an open doorway and selected absent. |
| `scannet/scene0169_00::26::28815` | The model judged the smaller cabinet absent as a standalone proposal. |

## Interpretation

The full run clears the Skill-mode regression gate against v8: same Overall
accuracy, better cache ratio, fewer failed statuses than v8, and no transport or
encrypted-state errors. View-Dep sits between v8 and v7, which is consistent
with normal run variance rather than an obvious prompt/adapter regression.

The remaining caution is latency: minimal-prompt Skill mode produced seven
turn-timeout finalizations and a p90 near 390 s. Those finalizations completed
from gathered evidence, but Skill mode is not obviously faster than inline
playbook mode at 40 workers.

## Reproduction Assets

- Launch script: `docs/benchmark/nr3d/assets/run_v11_skill600_minprompt_20260701.sh`
- Raw output: `tmp/nr3d_tools_skill600_minprompt_20260701/`
- Derived metrics:
  `docs/benchmark/nr3d/assets/v11_skill600_minprompt_20260701_leaderboard.json`
