# NR3D benchmark archive (agentic-3d-task)

Process archive for NR3D evaluations in `agentic-3d-task`. Mirrors the
3DVLMReasoning `docs/benchmark/nr3d/` layout. Two tracks live here:

- **Stage-1 coverage** (`keyframe` selector): does the top-K keyframe set contain
  the GT target id? Metric `hit@K`.
- **Codex agent VG** (`codex_agent.CodexAgentRuntime`): does the agent select the
  correct proposal? Metric oriented 3D IoU / Acc@τ.

## Version timeline

| Version | Date | Track | Scope | Headline |
|---|---|---|---|---|
| [v1_keyframe_targetid_coverage_strat600_20260608](v1_keyframe_targetid_coverage_strat600_20260608.md) | 2026-06-08 | Stage-1 coverage | `select_keyframes_v2` target-id frame coverage, strat600, k=3, vp=True, CLIP off | **hit@3 = 80.17 %** (agentic); 78.00 % (3DVLM control); migration faithful (90.5 % outcome agreement) |
| [v1_codex_runtime_grounding_smoke_20260608](v1_codex_runtime_grounding_smoke_20260608.md) | 2026-06-08 | Codex agent VG | `CodexAgentRuntime` migration smoke, 10-case (first 10 of strat600), catalog-first prompt-only, gpt-5.4 multi-AK from `configs/llm.toml` | **mean IoU 0.70→0.80, Acc@0.50 = 0.70→0.80**, 10/10 end-to-end, 0 errors (2 runs) |
| [v1_codex_runtime_grounding_strat600_20260608](v1_codex_runtime_grounding_strat600_20260608.md) | 2026-06-08 | Codex agent VG | `CodexAgentRuntime` full **canonical strat600** (600), catalog-first prompt-only, gpt-5.4 multi-AK from `configs/llm.toml`, 50 workers | **Acc@0.25 = 63.83%** (Easy 73.1 / Hard 55.2 / V-Dep 51.7 / V-Indep 70.4); 600/600 clean, 0 infra errors, 15.8 min |
| [v2_codex_tools_grounding_strat600_20260609](v2_codex_tools_grounding_strat600_20260609.md) | 2026-06-09 | Codex agent VG | `CodexAgentRuntime` + 9 in-turn evidence tools + `view_image`, strat600, gpt-5.4, 40 workers — **PARTIAL 169/600 (halted: throughput collapse)** | **Acc@0.25 = 63.31%** on 169 (Easy 70.6 / Hard 56.0 / V-Dep 50.0 / V-Indep 71.0) — **flat vs v1 prompt-only floor.** Trace analysis: spatial/co-visible tools *are* used (V-Dep 19/22), but **91% of slow-tail shell calls are `SKILL.md` re-reads** (median 58×/case) — a re-read loop `project_doc_max_bytes=0` didn't stop. Fix v3: inline skill into prompt + strip ambient skills + turn hard-cap |
| [v3_codex_tools_loopfix_strat600_20260609](v3_codex_tools_loopfix_strat600_20260609.md) | 2026-06-09 | Codex agent VG | `CodexAgentRuntime` + 9 tools + **loop fix** (inline playbook, tool-call guard, ≤768px images, model_context_window, reasoning ON), strat600, gpt-5.4, 40 workers — **COMPLETE 600/600** | **Acc@0.25 = 78.33%** (Easy 84.8 / Hard 72.3 / **V-Dep 71.1** / V-Indep 82.3) — **+14.5 pp Overall, +19.4 pp View-Dep vs v1.** Loop gone: median 5 tool calls/case (was 80–224), 23/600 guard interrupts, 0 failures. Tools finally pay off where designed (V-Dep/Hard). Code `d58328f` |
| [v4_network_fix_strat600_20260610](v4_network_fix_strat600_20260610.md) | 2026-06-10 | Codex agent VG | v3 + **sandbox network fix** so `keyframe_selector` (its parser LLM call) actually runs instead of silently falling back, strat600, gpt-5.4, 40 workers — **COMPLETE 600/600** | **Acc@0.25 = 78.33%** (Easy 82.8 / Hard 74.2 / V-Dep 68.3 / V-Indep 83.8) — **identical Overall to v3 (470/600); every tier Δ a ≤6-case shuffle inside the variance band.** Net: the fix is a correctness fix; the catalog-first selectors already cover the fold, so working `keyframe_selector` is accuracy-neutral here. 0 permanent failures; 48 transient 429s absorbed. Code `28d117a` |

## Codex agent VG — per-tier accuracy (canonical strat600)

`Acc@0.25` (== `Acc@0.50`, pool is `source = gt`), oriented 3D IoU vs GT 9-DOF
box, gpt-5.4 via ModelHub adapter. v1 = catalog-first **prompt-only**; v3 =
v1 + 9 in-turn evidence tools + the SKILL.md-loop fix (but `keyframe_selector`
silently dead — sandbox blocked its network call); v4 = v3 + the sandbox network
fix so `keyframe_selector` actually runs. All three are full 600/600 on the same
fold. (v2 added the tools but looped, so it only ran 169/600 flat at the v1 floor
— see the timeline.)

| Slice | n | v1 (prompt-only) | v3 (tools; kf dead) | **v4** (tools; kf live) | Δ v4−v3 | Δ v4−v1 |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | 600 | 63.83% | 78.33% | **78.33%** | **0.00** | +14.50 |
| Easy | 290 | 73.10% | 84.83% | 82.76% | −2.07 | +9.66 |
| Hard | 310 | 55.16% | 72.26% | 74.19% | +1.94 | +19.03 |
| **View-Dep** | 211 | 51.66% | 71.09% | **68.25%** | −2.84 | +16.59 |
| View-Indep | 389 | 70.44% | 82.26% | 83.80% | +1.55 | +13.36 |

The **tools** are the win (v1 → v3/v4: +14.5 pp Overall, ~+16–19 pp on Hard /
View-Dep). Whether `keyframe_selector` is live (v4) or dead (v3) makes **no
measurable difference**: Overall is identical (470/600), and every v4−v3 tier
delta is a ≤6-case shuffle well inside the strat600 90 % bands (Overall ±2.3,
Hard ±3.5, View-Dep ±4.5 pp) — i.e. run-to-run nondeterminism, not a
`keyframe_selector` effect. The catalog-first selectors + first-person frame
annotation already carry the result; the language→frame retrieval tool is
redundant on this fold. Full runs:
[v3](v3_codex_tools_loopfix_strat600_20260609.md) ·
[v4](v4_network_fix_strat600_20260610.md).

## Metric

`hit@K`: fraction of fold samples where the selector's top-K returned frames
contain at least one frame in which the GT `target_id` is depth-visible
(`conceptgraph/indices/visibility_index.pkl`). Canonical fold:
`v9_3_strat600_sample_ids.json` (600, stratified on `is_easy × is_view_dep`).

This is a Stage-1 tool-coverage metric (evidence-entry-point recall), distinct
from the agent-level ReferIt3D classification accuracy.

## Comparison snapshot (strat600, k=3, vp=True)

| Selector | CLIP | hit@3 |
|---|---|---:|
| agentic `keyframe` (this repo) | off | **80.17** |
| 3DVLM `query_scene` control | off | 78.00 |
| 3DVLM `query_scene` v16 (documented) | on | 71.17 |
| 3DVLM `query_scene` v11 (documented, no-vp) | on | 62.17 |
