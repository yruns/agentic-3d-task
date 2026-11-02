# NR3D v7 - responses passthrough adapter, strat600

This strat600 run validates the simplified ModelHub adapter path:

```text
Codex SDK localhost /v1/responses -> ModelHub /responses
```

The adapter forwards the SDK Responses request body unchanged by default, with
request-body mutation left behind an explicit legacy compatibility flag.

> **Status: COMPLETE - 600/600.** Finished 2026-06-30 (`exit=0`).
> Raw artifacts: `tmp/nr3d_tools_case600_v7_responses_passthrough_20260630/`.
> Derived metrics:
> `assets/v7_responses_passthrough_strat600_20260630_leaderboard.json`.
>
> **Takeaway:** v7 preserves v5-level behavior with **Acc@0.25 = 86.00%
> (516/600)**, **View-Dep = 82.46% (174/211)**, `cache_hit_rate = 0.9867`,
> and `mean_cache_ratio = 0.9297`.

## Result

| Slice | n | v7 Acc | correct | v5 Acc | Delta vs v5 |
|---|---:|---:|---:|---:|---:|
| **Overall** | 600 | **86.00%** | 516 | 85.33% | +0.67 |
| Easy | 290 | 90.34% | 262 | 89.31% | +1.03 |
| Hard | 310 | 81.94% | 254 | 81.61% | +0.32 |
| **View-Dep** | 211 | **82.46%** | 174 | 81.04% | +1.42 |
| View-Indep | 389 | 87.92% | 342 | 87.66% | +0.26 |

The v7-v5 deltas are inside the strat600 comparison band, so this is a
regression-gate pass rather than a new quality claim.

## Health

| Check | Value |
|---|---:|
| Sample records | 600/600 |
| Status | 598 completed, 2 failed |
| Reasoning summaries | 542/600 |
| Cache hit rate | 0.9867 |
| Mean cache ratio | 0.9297 |
| Route | 4,837 `/v1/responses`; 0 chat/crawl |
| Hard failures | 2 semantic target-absent decisions |

## Conclusion

The simplified Responses passthrough adapter is sufficient for this path. It
keeps View-Dep accuracy high, preserves prompt caching, and keeps reasoning
summary capture in the expected range. The old chat/completion conversion and
lossy local compaction logic are not needed for the normal Responses route.

## Reproduction Assets

- Sample fold: `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json`
- Data root: `/Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet`
- Pack: `pack_nr3d_v9_catalog_first`
- Raw output: `tmp/nr3d_tools_case600_v7_responses_passthrough_20260630/`
- Derived metrics:
  `docs/benchmark/nr3d/assets/v7_responses_passthrough_strat600_20260630_leaderboard.json`
