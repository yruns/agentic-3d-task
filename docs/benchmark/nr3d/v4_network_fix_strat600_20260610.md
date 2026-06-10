# NR3D v4 — sandbox network fix (keyframe_selector live), strat600

Fourth **strat600** evaluation of the Codex Agent SDK runtime
(`src/codex_agent/`). Identical to
[v3](v3_codex_tools_loopfix_strat600_20260609.md) except for one fix: the
runtime no longer drops the workspace-write sandbox's **network access**, so the
`keyframe_selector` tool (the only tool that makes a network LLM call — its
Stage-1 query parser) actually runs instead of failing instantly and falling
back to `select_by_proposal`.

> **Status: RUNNING.** Launched 2026-06-10. Raw artifacts will land in
> `tmp/nr3d_tools_case600_v4_netfix/` (`summary.json` + `per_sample/`).
>
> **Hypothesis:** v3 achieved 78.33 % Overall with `keyframe_selector`
> effectively dead (network blocked on every tool turn). Restoring it should
> help most on the free-form / view-dependent queries the tool was built for —
> or, if the catalog selectors already cover those cases, leave the number flat
> while adding latency. This run measures that delta against the v3 baseline on
> the identical fold.

## What changed vs v3

The sandbox-network root cause + fix is described inline in the commit. In
short: `Thread.turn(sandbox=Sandbox.workspace_write)` made the SDK send a
per-turn `WorkspaceWriteSandboxPolicy` with `network_access` defaulted to
`false`, which **overrode** the `sandbox_workspace_write.network_access=true`
from `config.toml`. The probe was decisive:

```
per-turn sandbox=ENUM  -> turn_context.network_access = False   (overrides config)
per-turn sandbox=None  -> turn_context.network_access = True     (config applies; mode stays workspace-write)
```

Fix (commit `28d117a`): set the sandbox **mode** once on the thread, omit the
per-turn `sandbox` override so the session config governs. Verified end-to-end
on scene0549: `turn_context.network_access=true`, `keyframe_selector` parses +
executes (no `Connection error`), `mark_frame_with_bbox` still writes its PNG
(workspace-write + writable roots intact). Everything else (inline playbook,
loop guard `max_tool_calls=30`/`max_repeated_tool_calls=4`, image ≤768 px,
`model_context_window=900000`, reasoning ON) is unchanged from v3.

## Pre-run

| Item | Value |
|---|---|
| Repo / branch | `agentic-3d-task` / `master` |
| Run-time code commit | `28d117a` (sandbox network fix) |
| Head commit at launch | this version-doc commit on top of `28d117a` (doc-only delta; no worktree drift) |
| Fold | `tmp/nr3d_case600/sample_ids.json` (600 ids — **identical file to v3/v2**) |
| Fold tiers | Easy/Hard + View-Dep/View-Indep flags carried in the fold dicts |
| Data root | `/Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet` |
| Pack | `pack_nr3d_v9_catalog_first` |
| Backend | `gpt-5.4-2026-03-05` via Codex `modelhub_adapter` (`127.0.0.1:8787`) |
| AK pool | weighted `.modelhub_upstreams.toml` pool (`gpt54_a:gpt54_b:gpt54_c = 5:1:5`) |
| Sandbox | `workspace_write` + `network_access=true` (**now actually applied**) |
| Skill | none (playbook inlined into the prompt) |
| Loop guard | `max_tool_calls=30`, `max_repeated_tool_calls=4` (tools defaults) |
| Turn budget | none (`--turn-timeout 0`); resume tail with a wall-clock cap if it wedges (see v3 op-note) |
| `model_context_window` | `900000` (`CODEX_AGENT_MODEL_CONTEXT_WINDOW`) |
| Reasoning effort | model default (ON) |
| Concurrency | **40 workers**, `sample_retries = 2` |
| Run home | not kept (`keep_run_home=false`) — avoids the v3 43 GB run-home bloat |
| Judge | none — deterministic oriented 3D IoU vs GT 9-DOF box |
| Date | 2026-06-10 |

### CLI

```bash
export CODEX_AGENT_MODEL_CONTEXT_WINDOW=900000
PYTHONPATH=src python -m codex_agent.cli.run_nr3d \
  --sample-ids tmp/nr3d_case600/sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --output-dir tmp/nr3d_tools_case600_v4_netfix \
  --pack-name pack_nr3d_v9_catalog_first \
  --workers 40 --sample-retries 2 --tools --turn-timeout 0
```

## Results

_Pending run completion. Per-tier table (Overall / Easy / Hard / View-Dep /
View-Indep) + cross-version comparison vs v3 will be filled here, computed from
the per-sample IoU and the fold's tier flags._
