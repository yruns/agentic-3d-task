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
