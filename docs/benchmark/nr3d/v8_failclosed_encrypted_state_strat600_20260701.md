# NR3D v8 - fail-closed encrypted state, strat600

This strat600 run validates the Responses adapter after removing the
`invalid_encrypted_content` sanitizer/retry fallback. Encrypted reasoning state
is now passed through intact; if ModelHub rejects it, the current Codex turn
fails instead of retrying with stripped state.

> **Status: COMPLETE - 600/600.** Finished 2026-07-01 (`exit=0`).
> Raw artifacts:
> `tmp/nr3d_tools_case600_v8_failclosed_encrypted_state_20260701/`.
> Derived metrics:
> `assets/v8_failclosed_encrypted_state_strat600_20260701_leaderboard.json`.
>
> **Takeaway:** v8 still passes the adapter gate with **Acc@0.25 = 84.50%
> (507/600)**, **View-Dep = 77.25% (163/211)**, `cache_hit_rate = 0.9900`,
> and `mean_cache_ratio = 0.9301`. It is a tighter pass than v7 because one real
> encrypted-state rejection is now surfaced as a hard turn failure.

## Result

| Slice | n | v8 Acc | correct | v7 Acc | Delta vs v7 | v5 Acc |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | 600 | **84.50%** | 507 | 86.00% | -1.50 | 85.33% |
| Easy | 290 | 88.62% | 257 | 90.34% | -1.72 | 89.31% |
| Hard | 310 | 80.65% | 250 | 81.94% | -1.29 | 81.61% |
| **View-Dep** | 211 | **77.25%** | 163 | 82.46% | -5.21 | 81.04% |
| View-Indep | 389 | 88.43% | 344 | 87.92% | +0.51 | 87.66% |

## Gate And Health

| Check | Value | Result |
|---|---:|:--:|
| Overall Acc@0.25 | 84.50% | pass |
| View-Dep Acc@0.25 | 77.25% | pass |
| Cache hit rate | 0.9900 | pass |
| Mean cache ratio | 0.9301 | pass |
| Sample records | 600/600 | pass |
| Status | 596 completed, 4 failed | pass |
| Route | 4,998 `/v1/responses`; 0 chat/crawl | pass |

Additional health signals:

- Reasoning summaries: 545/600.
- Failures: 1 `invalid_encrypted_content` infrastructure failure, plus 3 semantic
  target-absent decisions.
- Latency: median 166.3 s, p90 359.5 s, max 878.5 s.

## Conclusion

The fail-closed adapter keeps the key v5/v7 properties: native Responses API,
healthy prompt caching, and reasoning summary capture around 90%. The accuracy
drop is concentrated in View-Dep, but the single encrypted-state failure only
explains one wrong sample. The correct next debugging target is upstream/state
stability and run-to-run variance, not restoring the sanitizer fallback.

## Reproduction Assets

- Launch script:
  `docs/benchmark/nr3d/assets/run_v8_failclosed_encrypted_state_20260701.sh`
- Sample fold: `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json`
- Data root: `/Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet`
- Pack: `pack_nr3d_v9_catalog_first`
- Raw output: `tmp/nr3d_tools_case600_v8_failclosed_encrypted_state_20260701/`
- Derived metrics:
  `docs/benchmark/nr3d/assets/v8_failclosed_encrypted_state_strat600_20260701_leaderboard.json`
