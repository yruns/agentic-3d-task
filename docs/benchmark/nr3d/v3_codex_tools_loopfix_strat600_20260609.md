# NR3D v3 — `CodexAgentRuntime` tools + loop fix, strat600 (RUNNING)

Third **canonical strat600** evaluation of the Codex Agent SDK runtime
(`src/codex_agent/`). Same in-turn evidence tools as
[v2](v2_codex_tools_grounding_strat600_20260609.md), but with the **`SKILL.md`
re-read loop fixed** (the loop is what collapsed v2 throughput and halted it at
169/600). This run is the first full strat600 with the loop fix, so it is the
first apples-to-apples comparison against the v1 prompt-only floor on the full
fold.

> **Status: RUNNING.** Launched 2026-06-09. Results, per-tier breakdown, and the
> cross-version comparison are filled in when the run completes. Raw artifacts:
> `tmp/nr3d_tools_case600_v3_loopfix/`.

## What changed vs v2

The loop root-cause + fix is documented in
[`docs/codex_agent/skill_loop_and_reasoning_dropped_20260609.md` §Resolution](../../codex_agent/skill_loop_and_reasoning_dropped_20260609.md#resolution-landed--verified--2026-06-09).
Landed in commit `d58328f`:

1. **Inline tool playbook into the prompt; no on-disk skill in tools mode** — no
   advertised `SKILL.md` path to re-read.
2. **In-turn loop guard** — stream the turn, cap total/repeated tool actions,
   interrupt, tear down the app-server if the interrupt is ignored, finalize from
   gathered evidence. Tools defaults: `max_tool_calls=30`, `max_repeated_tool_calls=4`.
3. **Downscale viewed frames/BEV to ≤768 px** (boxes reported in scaled coords).
4. **`model_context_window` override** (~855 K observed) — stops codex using the
   conservative 258 400 default for the custom model.
5. **Reasoning ON** (unchanged from v2).

10-case pilot before this run: 0/10 loopers (v2 had ~4/10), **0 real `SKILL.md`
reads** (v2 tail: 66–91 % of shell calls), 2–14 tool calls/case (v2 loopers:
80–224), Acc@0.25 0.80.

## Pre-run

| Item | Value |
|---|---|
| Repo / branch | `agentic-3d-task` / `master` |
| Run-time code commit | `d58328f` (loop fix) |
| Head commit at launch | this version-doc commit on top of `d58328f` (doc-only delta; no worktree drift) |
| Fold | canonical `v9_3_strat600` (`tmp/nr3d_case600/sample_ids.json`, 600 ids — same file as v2) |
| Fold tiers | Easy/Hard + View-Dep/View-Indep flags carried in the fold dicts |
| Data root | `/Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet` |
| Pack | `pack_nr3d_v9_catalog_first` |
| Backend | `gpt-5.4-2026-03-05` via Codex `modelhub_adapter` (`127.0.0.1:8787`; chat-completions routing for `gpt-5.4*`) |
| AK pool | weighted `.modelhub_upstreams.toml` pool (`gpt54_a:gpt54_b:gpt54_c = 5:1:5`) |
| Sandbox | `workspace_write` + `network_access=true` |
| Skill | none (playbook inlined into the prompt) |
| Loop guard | `max_tool_calls=30`, `max_repeated_tool_calls=4` (tools defaults) |
| Turn budget | none (`--turn-timeout 0`); the tool-call cap is the loop backstop |
| Image budget | viewed frames/BEV ≤768 px |
| `model_context_window` | `900000` (`CODEX_AGENT_MODEL_CONTEXT_WINDOW`) |
| Reasoning effort | model default (ON) |
| Concurrency | **40 workers**, `sample_retries = 2` |
| Judge | none — deterministic oriented 3D IoU vs GT 9-DOF box |
| Date | 2026-06-09 |

### CLI

```bash
export CODEX_AGENT_MODEL_CONTEXT_WINDOW=900000
PYTHONPATH=src python -m codex_agent.cli.run_nr3d \
  --sample-ids tmp/nr3d_case600/sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --output-dir tmp/nr3d_tools_case600_v3_loopfix \
  --pack-name pack_nr3d_v9_catalog_first \
  --workers 40 --sample-retries 2 --tools --turn-timeout 0
```

## Results

_To be filled in when the run completes._

| Slice | n | Acc@0.25 | Acc@0.50 | mean IoU |
|---|---:|---:|---:|---:|
| Overall | | | | |
| Easy | | | | |
| Hard | | | | |
| View-Dep | | | | |
| View-Indep | | | | |

### Loop-health (vs v2)

_To be filled in: % shell calls that are doc reads, tool calls/case, loopers._
