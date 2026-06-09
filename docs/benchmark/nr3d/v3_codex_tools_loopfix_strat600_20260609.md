# NR3D v3 — `CodexAgentRuntime` tools + loop fix, strat600 (COMPLETE 600/600)

Third **canonical strat600** evaluation of the Codex Agent SDK runtime
(`src/codex_agent/`). Same in-turn evidence tools as
[v2](v2_codex_tools_grounding_strat600_20260609.md), but with the **`SKILL.md`
re-read loop fixed** (the loop is what collapsed v2 throughput and halted it at
169/600). This run is the first full strat600 with the loop fix, so it is the
first apples-to-apples comparison against the v1 prompt-only floor on the full
fold.

> **Status: COMPLETE — 600/600.** Launched + finished 2026-06-09. Raw artifacts:
> `tmp/nr3d_tools_case600_v3_loopfix/` (`summary.json` + `per_sample/`).
>
> **One-line takeaway:** with the loop fixed, the in-turn tools lift **every
> slice well above the v1 prompt-only floor** — Overall **63.8 % → 78.3 %**
> (+14.5 pp), and the tools' target slices most of all: **View-Dep 51.7 % →
> 71.1 % (+19.4 pp)**, Hard 55.2 % → 72.3 % (+17.1 pp). All deltas dwarf the
> strat600 ±2.3 pp (Overall) / ±4.5 pp (View-Dep) variance band. The v2 loop is
> gone: median **5** tool calls/case (v2 loopers: 80–224), **0** runaway loops.

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

Deterministic oriented-3D-IoU vs GT 9-DOF box. `Acc@0.25 == Acc@0.50` because
the pool is `source = gt` (a correct pick → IoU≈1, wrong → ~0), so IoU is
bimodal and `mean IoU ≈ Acc`.

| Slice | n | Acc@0.25 | Acc@0.50 | mean IoU | v1 prompt-only | Δ vs v1 |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | 600 | **78.33%** | 78.33% | 0.786 | 63.83% | **+14.50** |
| Easy | 290 | 84.83% | 84.83% | 0.849 | 73.10% | +11.73 |
| Hard | 310 | 72.26% | 72.26% | 0.727 | 55.16% | +17.10 |
| **View-Dep** | 211 | **71.09%** | 71.09% | 0.714 | 51.66% | **+19.43** |
| View-Indep | 389 | 82.26% | 82.26% | 0.825 | 70.44% | +11.82 |

Every delta exceeds the strat600 90 % variance band (Overall ±2.3 pp,
Easy/V-Indep ±2.7–2.9 pp, Hard ±3.5 pp, View-Dep ±4.5 pp), so all gains are
real. The largest lifts are on **View-Dep (+19.4 pp)** and **Hard (+17.1 pp)** —
exactly the slices the first-person frame / co-visible-frame / spatial-compare
tools were built for, and exactly what v2 *failed* to deliver because the loop
drowned the tool use.

`cache_hit_rate = 0.928`, `mean_cache_ratio = 0.618` (prefix caching healthy).
`status=failed` (model answered "target absent", proposal_id=-1) on 20/600
(3.3 %) — counted as wrong.

### Cross-version comparison (same strat600 fold)

| Version | Tools | Loop | Overall | View-Dep | Notes |
|---|---|---|---:|---:|---|
| v1 | no | n/a | 63.83% | 51.66% | prompt-only floor (600/600) |
| v2 | yes | **looping** | 63.3%* | 50.0%* | *partial 169/600; tools drowned by SKILL.md loop |
| **v3** | **yes** | **fixed** | **78.33%** | **71.09%** | this run (600/600) |

### Loop-health (vs v2)

| | v2 (looping) | **v3 (fixed)** |
|---|---|---|
| tool calls / case | 80–224 (slow tail) | **median 5, max 23** (guard-bounded) |
| `SKILL.md` re-reads | **66–91 % of shell calls** (median 58×/case) | **one-off peeks only** — ~9 reads in 50 sampled cases (~18 % of cases read the *still-on-disk* legacy `.agents/skills/nr3d-codex-tools/SKILL.md` once); **no re-read loop** |
| guard interrupts | n/a | **23/600 (3.8 %)** — each bounded to ≤ the cap and finalized from evidence |
| permanent failures | — | **0** |

The loop is gone: the inline playbook means the model never *needs* the file,
and the tool-call guard caps any residual rut. The remaining one-off `sed
SKILL.md` peeks happen because the legacy skill file still exists in the
workspace and codex's coding-agent prior makes it hunt for a skill; they are
single reads (not loops), do not hurt accuracy, and would be eliminated by
removing/relocating that legacy file (the playbook is the production source of
truth). See
[`docs/codex_agent/skill_loop_and_reasoning_dropped_20260609.md` §Resolution](../../codex_agent/skill_loop_and_reasoning_dropped_20260609.md#resolution-landed--verified--2026-06-09).

### Operational note — tail wedge + resume

The first launch completed **568/600** then stalled: the final 32 workers
wedged on **hung upstream calls** (the app-server's SSE stream produced no
events, so the *action*-based guard could not fire, and the default
`turn_timeout_s=0` left no wall-clock backstop). All 32 stalled within the same
~1 min window → a transient upstream/adapter incident, not bad cases. Fix
applied: killed the wedged session (568 checkpoints are written atomically and
survived), then **resumed** (`tmp/run_v3_resume.sh`, 12 workers,
`CODEX_AGENT_TURN_TIMEOUT_S=420`) — the runner skips cached cases and the 32
completed cleanly (0 guard interrupts, 0 failures), giving the full 600.

**Follow-up (robustness, not yet landed):** a silent upstream hang should be
bounded by a *process-level* turn kill (tear down the per-turn app-server when a
wall-clock deadline passes), since the SDK `interrupt()` does not unblock a
hung HTTP read. The action-based guard handles event-emitting loops; the
wall-clock + process-kill path handles event-less hangs. Tracking for a follow-up
iteration; it does not affect these numbers (run-time code uniform at `d58328f`).
