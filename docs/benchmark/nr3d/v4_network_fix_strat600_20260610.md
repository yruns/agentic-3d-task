# NR3D v4 — sandbox network fix (keyframe_selector live), strat600

Fourth **strat600** evaluation of the Codex Agent SDK runtime
(`src/codex_agent/`). Identical to
[v3](v3_codex_tools_loopfix_strat600_20260609.md) except for one fix: the
runtime no longer drops the workspace-write sandbox's **network access**, so the
`keyframe_selector` tool (the only tool that makes a network LLM call — its
Stage-1 query parser) actually runs instead of failing instantly and falling
back to `select_by_proposal`.

> **Status: COMPLETE — 600/600.** Launched + finished 2026-06-10 (`V4_EXIT_CODE=0`,
> ~57 min, no tail wedge). Raw artifacts: `tmp/nr3d_tools_case600_v4_netfix/`
> (`summary.json` + `per_sample/`).
>
> **One-line takeaway:** restoring `keyframe_selector` network access is
> **accuracy-neutral** on this fold — Overall is **identical to v3 (78.33 %,
> 470/600)** and every tier delta is a ≤6-case shuffle (≤2.84 pp) **inside** the
> strat600 variance bands. The fix is a correctness fix (the tool now actually
> runs and the silent `select_by_proposal` fallback is gone), but the
> catalog-first offline selectors already cover what `keyframe_selector` adds —
> it is not an accuracy lever for NR3D grounding here. Latency is comparable
> (median 114 s vs v3 119 s); 0 permanent failures.

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

Deterministic oriented-3D-IoU vs GT 9-DOF box. `Acc@0.25 == Acc@0.50` (pool is
`source = gt`, so IoU is bimodal and `mean IoU ≈ Acc`). `mean IoU = 0.786`,
`cache_hit_rate = 0.967`, `mean_cache_ratio = 0.572`.

| Slice | n | v4 Acc | correct | v3 Acc | Δ vs v3 | strat600 band | within band? |
|---|---:|---:|---:|---:|---:|---:|:--:|
| **Overall** | 600 | **78.33%** | 470 | 78.33% | **0.00** | ±2.3 pp | ✅ |
| Easy | 290 | 82.76% | 240 | 84.83% | −2.07 | ±2.7 pp | ✅ |
| Hard | 310 | 74.19% | 230 | 72.26% | +1.94 | ±3.5 pp | ✅ |
| **View-Dep** | 211 | **68.25%** | 144 | 71.09% | −2.84 | ±4.5 pp | ✅ |
| View-Indep | 389 | 83.80% | 326 | 82.26% | +1.55 | ±2.9 pp | ✅ |

Per-tier the run moved exactly **±6 cases** in each partition (Easy −6 / Hard +6;
View-Dep −6 / View-Indep +6), netting to **0 change in Overall**. That symmetric
±6 shuffle is the signature of ordinary run-to-run model nondeterminism (cases
flipping right↔wrong between two runs on the same fold), not a systematic effect
of `keyframe_selector`. Every delta is inside the strat600 90 % variance band.

`status=failed` (model answered `proposal_id=-1`, "target absent") on **11/600**
(1.8 %), down from v3's 20/600; **0 permanent failures**.

### Tool / network health

- The fix is live: `keyframe_selector`'s parser LLM call now reaches the network
  (verified pre-run on scene0549: `turn_context.network_access=true`, parser
  succeeds, `mark_frame_with_bbox` still writes). During the run the parser
  endpoint was exercised enough to draw **48 transient `429`s**, all absorbed by
  `sample_retries=2` (0 permanent failures) — direct evidence the tool was
  actually being called, not skipped.
- **Latency:** median **113.9 s** (v3 118.5 s), p90 278 s (v3 264 s), max 788 s
  (v3 992 s). Comparable — the real parser calls did not inflate median latency,
  because the model invokes `keyframe_selector` on only a subset of cases and the
  catalog selectors dominate the rest.

### Cross-version comparison (same strat600 fold)

| Version | keyframe_selector | Overall | View-Dep | Notes |
|---|---|---:|---:|---|
| v1 | n/a (prompt-only) | 63.83% | 51.66% | prompt-only floor (600/600) |
| v3 | **dead** (network blocked) | 78.33% | 71.09% | loop-fix run; tool silently fell back |
| **v4** | **live** (network fixed) | **78.33%** | **68.25%** | this run (600/600) |

### Interpretation

v3's 78.33 % was achieved with `keyframe_selector` **silently disabled** — so the
catalog-first selectors (`select_by_proposal`, `compare_proposals_spatial`,
`compare_candidates_to_anchors`, `inspect_proposal`) + first-person frame
annotation already carried the full result. Turning the language→frame retrieval
tool back on neither helps nor hurts on this fold (Δ Overall 0.00; all tiers in
band). Practical implications:

1. **Keep the fix** — it is a correctness fix (no silent capability loss, no
   wasted instant-fail calls, network now available to any future tool that
   needs it).
2. **`keyframe_selector` is redundant for NR3D grounding on this fold.** It could
   be made opt-in / de-emphasised in the playbook to cut its extra parser-LLM
   load (the 429 contention) without an accuracy cost.
3. The −2.84 pp View-Dep dip is within the ±4.5 pp band and is mirrored by a
   +1.55 pp View-Indep gain (the ±6-case shuffle), so it is **not** evidence that
   retrieved frames mislead view-dependent reasoning — confirming that would need
   a multi-seed run, not this single fold.

> **Caveat:** per-case `keyframe_selector` usage was not logged (this run used
> `keep_run_home=false` to avoid the v3 43 GB run-home bloat), so the exact
> fraction of cases that invoked it is inferred from the 48 parser-429s rather
> than counted. A small `keep_run_home=true` sample would quantify it if needed.
