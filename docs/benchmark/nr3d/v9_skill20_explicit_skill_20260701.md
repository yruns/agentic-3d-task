# NR3D v9 skill20 - explicit Codex skill diagnostic

This diagnostic reruns the first 20 strat600 samples in tool mode while
explicitly attaching `.agents/skills/nr3d-codex-tools/SKILL.md` through
`SkillInput`. It tests whether the historical SKILL.md re-read loop still appears
when using Codex's skill progressive-disclosure path.

> **Status: COMPLETE - 20/20.** Finished 2026-07-01 (`exit=0`).
> Raw artifacts: `tmp/nr3d_tools_skill20_explicit_skill_20260701/`.
> Derived metrics: `assets/v9_skill20_explicit_skill_20260701_leaderboard.json`.
>
> **Takeaway:** explicit Skill mode completed without the old loop signals on this
> 20-sample smoke: **17/20 = 85.00%**, no permanent failures, no tool-loop
> interrupts, no turn timeouts, and no `SKILL.md` loop matches in the run logs.

## Setup

| Item | Value |
|---|---|
| Scope | First 20 samples from `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` |
| Head at launch | `9b262f4` |
| Worktree | Dirty: explicit SkillInput enablement patch under `src/codex_agent/nr3d/grounding.py` |
| Mode | `--tools --skill-path .agents/skills/nr3d-codex-tools/SKILL.md` |
| Workers | 20 |
| Reasoning summary | `auto` |
| Turn timeout | 900 s |
| Adapter route | Responses API via `codex_modelhub_adapter` |

## Result

| Slice | n | v7 correct | v8 correct | skill20 correct | skill20 Acc |
|---|---:|---:|---:|---:|---:|
| **Overall** | 20 | 16 | 16 | **17** | **85.00%** |
| Easy | 11 | 10 | 10 | 10 | 90.91% |
| Hard | 9 | 6 | 6 | 7 | 77.78% |
| **View-Dep** | 11 | 7 | 7 | 8 | 72.73% |
| View-Indep | 9 | 9 | 9 | 9 | 100.00% |

## Health

| Check | Value |
|---|---:|
| Sample records | 20/20 |
| Status | 20 completed, 0 failed |
| Reasoning summaries | 19/20 |
| Business summaries | 20/20 |
| Cache hit rate | 0.8500 |
| Mean cache ratio | 0.8224 |
| Latency | median 121.1 s, max 399.3 s |
| Route | 203 `/v1/responses`; 0 chat/crawl |
| Upstream errors | 0 HTTP 400; 1 transient HTTP 503; 0 `invalid_encrypted_content` |
| Loop signals | 0 max-tool-call interrupts, 0 repeated-call interrupts, 0 turn timeouts |

Wrong samples:

| Sample | selected | Reason |
|---|---:|---|
| `scannet/scene0164_00::14::23053` | 15 | Semantic confusion among cabinets near the fridge. |
| `scannet/scene0500_00::25::13471` | 26 | Semantic confusion among the left-wall/chalkboard windows. |
| `scannet/scene0608_00::9::17679` | 8 | Semantic confusion among similar ottomans near sofas/chairs. |

## Interpretation

The old Skill-mode failure did not reproduce in this 20-sample run. The run
finished quickly, produced all per-sample records, and did not hit the runtime's
tool-loop guard. This suggests the current Codex skill path can be viable for
small NR3D tool runs when the prompt uses the attached skill as the workflow
source instead of advertising a separate file while also telling the model to
avoid skills.

The main caution is cache: Skill mode was materially lower than v8 on this smoke
(`cache_hit_rate = 0.8500`, `mean_cache_ratio = 0.8224`). That needs a larger
run before using Skill mode as the default path.

## Reproduction Assets

- Launch script: `docs/benchmark/nr3d/assets/run_v9_skill20_explicit_skill_20260701.sh`
- Raw output: `tmp/nr3d_tools_skill20_explicit_skill_20260701/`
- Derived metrics:
  `docs/benchmark/nr3d/assets/v9_skill20_explicit_skill_20260701_leaderboard.json`
