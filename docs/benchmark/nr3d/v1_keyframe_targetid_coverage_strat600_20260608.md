# NR3D v1 — `keyframe` selector target-id frame coverage (strat600)

Tool-coverage audit of the migrated `keyframe` package (`agentic-3d-task`),
mirroring the 3DVLMReasoning `select_by_text` coverage audits (v11..v16).
**Not** an agent leaderboard run.

## Question

> If `KeyframeSelector.select_keyframes_v2` is called once with the raw NR3D
> query and `k=3`, do the returned keyframes include at least one frame where
> the GT `target_id` is depth-visible (per
> `conceptgraph/indices/visibility_index.pkl` → `object_to_views[target_id]`)?

`hit@K = 1` iff `pred_top_K ∩ gt_frames ≠ ∅`. This is the upper bound on what
the selector can hand a downstream VLM agent as its first move. Method is byte
-identical to 3DVLM `scripts/audit_select_by_text_nr3d.py` (same fold, same
visibility index, same metric); the runner is
`tmp/nr3d_coverage/threaded_coverage_runner.py`.

## Pre-run

| Item | Value |
|---|---|
| Primary repo / branch / commit | `agentic-3d-task` / `master` / `f1fe9ad` |
| Control repo / branch / commit | `3DVLMReasoning` / `feat/intro-codex-agent-sdk` / `3317051` |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` (canonical strat600) |
| Fold size | 600 (Easy 290 / Hard 310 ; View-Dep 211 / View-Indep 389) |
| Data root | `3DVLMReasoning/data/nr3d/scannet` (shared, 119 scenes) |
| Pack | `pack_nr3d_v9_catalog_first` |
| Call shape | `select_keyframes_v2(query, k=3, hidden_categories=[], use_visual_context=False, viewpoint_aware=True)` |
| LLM | `gemini-2.5-pro` via ModelHub host (`configs/llm.toml`, 6 active AKs, QPM≈2550) |
| CLIP | **disabled** in both runs (agentic venv has no torch/open_clip; control patches `HAS_CLIP=False`) for a fair head-to-head |
| Judge | none — deterministic coverage against depth-aware visibility index |
| Date | 2026-06-08 |

## Headline coverage (agentic `keyframe`, run A2)

`runA2_threaded` — agentic selector, no-CLIP, ModelHub gemini, threaded @64, 0 errors.

| Slice | n | hit@1 | hit@2 | hit@3 | empty |
|---|---:|---:|---:|---:|---:|
| Overall | 600 | 64.33 | 77.00 | **80.17** | 28 |
| Easy | 290 | — | — | **85.52** | — |
| Hard | 310 | — | — | **75.16** | — |
| View-Dep | 211 | — | — | **80.09** | — |
| View-Indep | 389 | — | — | **80.21** | — |

Headline answer: **481 / 600 = 80.17 %** of strat600 samples have at least one
returned keyframe (k=3) containing the GT target id.

A second independent run with the previous `gpt/openapi` gemini backend
(sequential + low-concurrency cleanup, run A) scored **79.83 %** — i.e. the
number is stable to ±0.34 pp across LLM gateways.

## Comparison vs documented 3DVLM `select_by_text` audits

Same fold, same call shape, same metric. The documented rows are CLIP-enabled
on older `query_scene` commits.

| Run | Selector | CLIP | hit@3 | Easy | Hard | V-Dep | V-Indep | empty |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **A2 (this)** | agentic `keyframe` | off | **80.17** | 85.52 | 75.16 | 80.09 | 80.21 | 28 |
| **B2 control (this)** | 3DVLM `query_scene` @`3317051` | off | **78.00** | 85.52 | 70.97 | 75.83 | 79.18 | 29 |
| v16 (documented) | 3DVLM `query_scene` @`43aed24` | on | 71.17 | 78.28 | 64.52 | 70.14 | 71.72 | 81 |
| v15 (documented) | 3DVLM `query_scene` @`9c62e51` | on | 69.17 | 73.79 | 64.84 | 73.93 | 66.58 | 106 |
| v13 (documented) | 3DVLM `query_scene` @`f8e6d22` | on | 69.00 | — | — | 69.67 | — | — |
| v11 (documented, no-vp) | 3DVLM `query_scene` @`45a2dae` | on | 62.17 | — | — | 50.71 | — | — |

## Migration faithfulness (A2 vs B2, same 600 samples)

| Metric | Value |
|---|---|
| hit@3 outcome agreement (both hit or both miss) | **543 / 600 = 90.5 %** |
| identical predicted frame set | 409 / 600 = 68.2 % |
| agentic-only hit | 35 |
| control-only hit | 22 |
| net | +13 samples (+2.17 pp) to agentic |

The agentic `keyframe` selector reproduces the 3DVLM `query_scene` selector's
target-id coverage to within **+2.17 pp** on the same fold/backend with CLIP
off — inside the strat600 90 % variance band (±2.3 pp on Overall). The +13-net
edge comes from 35 recoveries vs 22 regressions, plus residual LLM parse
nondeterminism (temperature 0 gemini still varies). **Conclusion: the migration
is behaviorally faithful** (and marginally better on Hard / View-Dep).

## Why both current runs beat the documented v16 (71.17 %)

Both current-code runs (78–80 %) sit ~7–9 pp above the documented v16. The two
differences vs v16 are confounded and not separated here:

1. **CLIP off vs on.** v16 ran with CLIP soft category matching; A2/B2 use the
   string/multi-label category path.
2. **Code evolution.** v16 is `query_scene` @`43aed24` (2026-05-24); the control
   is current HEAD `3317051`, which has had executor / spatial-checker / parser
   changes since.

Because A2 (agentic) ≈ B2 (current 3DVLM), the gap is attributable to the
CLIP-off path and/or post-v16 `query_scene` evolution — **not** to the
migration. A clean CLIP ablation (current `query_scene` with CLIP on vs off)
would isolate (1); it was not run here.

## Concurrency findings (ModelHub gemini endpoint)

`weight=QPM` in `configs/llm.toml` totals ~2550 QPM, but the gemini vendor
account is best-effort (`普通账户`, no committed quota) and returns HTTP 429
`-4302 厂商资源不足` once too many requests are in flight at once.

| Workers | Outcome |
|---|---|
| 196 | **59 % error** — cascade begins after ~50–75 concurrent in-flight requests; key rotation can't recover when all keys saturate simultaneously |
| 64 | **0 % error** (both A2 and B2, 600 samples each) |
| 46 | 0 % final error (some transient 429s recovered by 6-key rotation) |

The runner architecture is a single process + `ThreadPoolExecutor` (network-
bound LLM calls), per-scene selectors loaded once and shared, with a
**thread-local `QueryExecutor` per selector** (the executor is stateful and is
relied on to carry viewpoint state within one `select_keyframes_v2` call, so it
must not be shared across threads, but reusing it within one thread is
identical to the original sequential behavior). 600 samples complete in
~3.5–7 min at 64 workers with 0 errors.

Practical guidance: for this gemini endpoint keep concurrency ≤ ~64; pushing to
196 is counterproductive (error ≫ 10 %). Headroom for "increase until error <
10 %" is bounded by the vendor capacity, not the configured per-key QPM.

## Raw artifacts

- Agentic A2 (full per-sample): `tmp/nr3d_coverage/runA2_threaded.json`
- Agentic A (old backend, merged): `tmp/nr3d_coverage/runA_agentic_vp/MERGED_final.json`
- Control B2 (full per-sample): `3DVLMReasoning/tmp/nr3d_coverage_ctrl/runB2_threaded.json`
- Trimmed durable summaries: `assets/keyframe_targetid_coverage_strat600_20260608_summaries.json`
- Runner (agentic): `tmp/nr3d_coverage/threaded_coverage_runner.py`
- Runner (control): `3DVLMReasoning/tmp/nr3d_coverage_ctrl/threaded_control_runner.py`
- Audit helpers / metric: `tmp/nr3d_coverage/audit_keyframe_coverage_nr3d.py`

## Caveats

- `tmp/` is gitignored; the durable record is this doc + the trimmed summary
  asset. Re-derive per-sample numbers from the raw JSONs above.
- CLIP-off was chosen for a fair agentic-vs-3DVLM head-to-head; it differs from
  the documented v16 CLIP-on setup (see attribution section).
- strat600 variance band: Overall ±2.3 pp (90 %); deltas under that should be
  confirmed on the full 7805 before being called real.
