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
| [v5_summary_responses_strat600_20260611](v5_summary_responses_strat600_20260611.md) | 2026-06-11 | Codex agent VG | v4 + **reasoning effort `""`→`medium`** + **reasoning summaries on** (`--reasoning-summary auto`, which forces the upstream **/responses** API), on the **in-repo vendored adapter**, strat600, gpt-5.4, 40 workers — **COMPLETE 600/600** | **Acc@0.25 = 85.33% (512/600), +7.00 pp over v4** — uniform gain across every tier (Easy 89.3 / Hard 81.6 / **V-Dep 81.0** / V-Indep 87.7), all far outside the variance bands → **real, not noise.** Summaries captured 491/600 (81.8%); only 2 hard failures. The +7 pp was first attributed to effort, but **[v6](v6_effort_ablation_chat_strat600_20260611.md) isolated it**: ≈+2 pp effort + **≈+5 pp the /responses path** (the dominant lever). Code `56e2d24` |
| [v6_effort_ablation_chat_strat600_20260611](v6_effort_ablation_chat_strat600_20260611.md) | 2026-06-11 | Codex agent VG | **Ablation** — effort `medium` on the **chat** path (summaries off) to split v5's +7 pp, strat600, gpt-5.4, 40 workers — **COMPLETE 600/600** | **Acc@0.25 = 80.33% (482/600)** (Easy 84.5 / Hard 76.5 / **V-Dep 72.5** / V-Indep 84.6). **Decomposes the v5 gain:** medium effort on chat = **+2.0 pp over v4** (near-noise); switching chat→/responses at fixed effort = **+5.0 pp** (v5−v6, dominant, biggest on V-Dep +8.5). **The `/responses` path — cross-turn reasoning-state carryover — is the real lever, not the effort param.** Chat-path cache signature 0.965/0.577 ≈ v4. Code `56e2d24` |
| [v7_responses_passthrough_strat600_20260630](v7_responses_passthrough_strat600_20260630.md) | 2026-06-30 | Codex agent VG | Responses passthrough adapter regression, strat600 — **COMPLETE 600/600** | **Acc@0.25 = 86.00% (516/600)**; View-Dep 82.46%; cache 0.987/0.930. Migration gate pass; v5-level behavior preserved. |
| [v8_failclosed_encrypted_state_strat600_20260701](v8_failclosed_encrypted_state_strat600_20260701.md) | 2026-07-01 | Codex agent VG | Fail-closed encrypted-state adapter regression, strat600 — **COMPLETE 600/600** | **Acc@0.25 = 84.50% (507/600)**; View-Dep 77.25%; cache 0.990/0.930. Gate pass, but tighter; one encrypted-state rejection surfaced. |
| [v8_flip41_repeat_20260701](v8_flip41_repeat_20260701.md) | 2026-07-01 | Codex agent VG diagnostic | Rerun of 41 v7/v8 correctness-flip samples — **COMPLETE 41/41** | **21/41 correct**, between v7's 25/41 and v8's 16/41. Supports run-to-run agent/model variance over deterministic adapter regression. |

## Codex agent VG — per-tier accuracy (canonical strat600)

`Acc@0.25` (== `Acc@0.50`, pool is `source = gt`), oriented 3D IoU vs GT 9-DOF
box, gpt-5.4 via ModelHub adapter, all full 600/600 on the same fold. v1 =
catalog-first **prompt-only**; v3/v4 = v1 + 9 in-turn evidence tools (+ loop fix,
+ sandbox network); v6 = v4 + **effort `medium` on the chat path**; v5 = v6 + the
**`/responses` path** (effort `medium` + summaries on); v7 validates that the
simplified responses-passthrough adapter preserves v5-level behavior; v8 removes
the encrypted-state sanitizer/retry fallback and lets mismatched encrypted state
fail closed. Columns below are ordered as floor -> tools -> effort -> path ->
adapter checkpoints. (v2 looped, ran only 169/600 at the v1 floor; v3 ≈ v4
Overall — see timeline.)

| Slice | n | v1 (floor) | v4 (tools) | v6 (+effort, chat) | v5 (+/responses) | v7 (passthrough) | **v8** (fail-closed) | v8-v7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 600 | 63.83% | 78.33% | 80.33% | 85.33% | 86.00% | **84.50%** | -1.50 |
| Easy | 290 | 73.10% | 82.76% | 84.48% | 89.31% | 90.34% | **88.62%** | -1.72 |
| Hard | 310 | 55.16% | 74.19% | 76.45% | 81.61% | 81.94% | **80.65%** | -1.29 |
| **View-Dep** | 211 | 51.66% | 68.25% | 72.51% | 81.04% | 82.46% | **77.25%** | -5.21 |
| View-Indep | 389 | 70.44% | 83.80% | 84.58% | 87.66% | 87.92% | **88.43%** | +0.51 |

Five checkpoints/levers, each isolated on this fold:

1. **Tools** (v1 → v4): **+14.5 pp Overall**, ~+16–19 pp on Hard / View-Dep.
   `keyframe_selector` live (v4) vs dead (v3) makes no measurable difference — the
   catalog-first selectors + first-person frame annotation carry it.
2. **Reasoning effort `""`→`medium`** (v4 → v6, chat path): **+2.0 pp Overall** —
   small, at the edge of the ±2.3 pp band; only View-Dep (+4.26) clearly moves.
3. **`/responses` path** (v6 → v5, effort fixed at medium): **+5.0 pp Overall,
   +8.53 pp View-Dep** — the dominant lever. `/responses` carries the model's
   encrypted reasoning state across the agent's multi-turn tool loop (the chat path
   drops it), so this inspect→rank→decide agent keeps its chain-of-thought between
   tool calls. (First attributed to effort in the v5 draft; the
   [v6 ablation](v6_effort_ablation_chat_strat600_20260611.md) corrected it.)
4. **Adapter simplification** (v5 → v7): **+0.67 pp Overall, +1.42 pp View-Dep**,
   both inside the strat600 bands. Read this as a regression gate pass: the
   responses-passthrough adapter preserves v5 behavior while improving cache ratio
   slightly (`0.930` vs `0.914`) and summary capture (`542/600` vs `491/600`).
5. **Encrypted-state fail-closed** (v7 → v8): **-1.50 pp Overall, -5.21 pp
   View-Dep**. v8 still clears the adapter gate, but it exposes one real
   `invalid_encrypted_content` turn failure and lands closer to the View-Dep
   threshold. Do not read this as a reason to restore sanitized fallback; the
   fallback can break reasoning-state/prompt-cache correspondence.

Full runs: [v3](v3_codex_tools_loopfix_strat600_20260609.md) ·
[v4](v4_network_fix_strat600_20260610.md) ·
[v5](v5_summary_responses_strat600_20260611.md) ·
[v6](v6_effort_ablation_chat_strat600_20260611.md) ·
[v7](v7_responses_passthrough_strat600_20260630.md) ·
[v8](v8_failclosed_encrypted_state_strat600_20260701.md).

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
