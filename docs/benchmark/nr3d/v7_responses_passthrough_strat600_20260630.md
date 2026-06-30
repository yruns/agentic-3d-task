# NR3D v7 - ModelHub responses passthrough adapter, strat600

Seventh **strat600** evaluation of the Codex Agent SDK runtime
(`src/codex_agent/`). This run validates the simplified in-repo ModelHub adapter:
Codex SDK localhost **Responses API** calls are proxied to ModelHub's native
**Responses API** endpoint with the request body left unchanged by default.

> **Status: COMPLETE - 600/600.** Launched and finished 2026-06-30
> (`exit=0`, finished Tue Jun 30 22:53:57 CST 2026). Raw artifacts:
> `tmp/nr3d_tools_case600_v7_responses_passthrough_20260630/`
> (`summary.json` + `per_sample/`). Durable derived metrics:
> `assets/v7_responses_passthrough_strat600_20260630_leaderboard.json`.
>
> **One-line takeaway:** v7 is **Acc@0.25 = 86.00% (516/600)** with
> **View-Dep = 82.46% (174/211)**, `cache_hit_rate = 0.9867`, and
> `mean_cache_ratio = 0.9297`. It clears the adapter-regression gate set for this
> migration and is statistically in the same band as v5 while slightly higher
> overall (+0.67 pp) and on View-Dep (+1.42 pp).

## What changed vs v5

v5 proved that the `/responses` path, not just the effort setting, was the major
accuracy lever. v7 keeps the same agent/runtime behavior and validates the new
adapter shape:

1. **Responses-first routing.** `auto` resolves to the ModelHub `/responses`
   upstream by default (`77a9ac2`).
2. **Request body passthrough.** `/v1/responses` forwards the SDK request body as
   received unless the explicit compatibility flag
   `AIDP_CODEX_PROXY_RESPONSES_BODY_MUTATION_ENABLED=true` is set (`77a9ac2`).
3. **Compact endpoint passthrough.** `/v1/responses/compact` proxies to upstream
   `/responses/compact` in responses mode instead of using the old local lossy
   compaction path (`33e4d0a`).
4. **Docs and launcher defaults.** Adapter docs and `restart_adapter.sh` now
   describe/set the responses path as the normal route (`9f0d994`).

The branch also contains the design and implementation plan commits
(`b5785f3`, `72f27e8`). No benchmark-time code drift was present.

## Pre-run

| Item | Value |
|---|---|
| Repo / branch | `agentic-3d-task` / `feat/codex-agent-upgrade` |
| Head commit at launch | `4b51027` (`4b510270554d5f57c7a4ceae805d8086a3c08977`) |
| Run-time code commit | `4b51027` (no worktree drift) |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json`, sha256 `3f3023595167fc7d744e5add205ea721a9eee01506079f02acc30df5c9b816b8` |
| Fold tiers | Easy 290 / Hard 310; View-Dep 211 / View-Indep 389 |
| Data root | `/Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet` |
| Pack | `pack_nr3d_v9_catalog_first` |
| Backend | `gpt-5.4-2026-03-05` via in-repo `codex_modelhub_adapter` on `127.0.0.1:8787` |
| Upstream API | **Responses API** (`AIDP_CODEX_PROXY_UPSTREAM_API=responses`; adapter health showed `responses_path=/responses`) |
| Request mutation | off by default (`responses_body_mutation_enabled=false`) |
| Sandbox | `workspace_write` + network access |
| Reasoning summary | `auto` |
| Turn budget | `--turn-timeout 900` |
| `model_context_window` | `900000` (`CODEX_AGENT_MODEL_CONTEXT_WINDOW`) |
| Concurrency | 40 workers, `sample_retries = 2` |
| Judge | none - deterministic oriented 3D IoU vs GT 9-DOF box |
| Date | 2026-06-30 |

### CLI

```bash
export AIDP_MODELHUB_UPSTREAMS_TOML=/Users/bytedance/project/agentic-3d-task/codex_modelhub_adapter/.modelhub_upstreams.toml
export AIDP_CODEX_PROXY_UPSTREAM_API=responses
export AIDP_CODEX_PROXY_UPSTREAM_ENV=office
export CODEX_AGENT_MODEL_CONTEXT_WINDOW=900000

PYTHONPATH=src python -m codex_agent.cli.run_nr3d \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --output-dir tmp/nr3d_tools_case600_v7_responses_passthrough_20260630 \
  --pack-name pack_nr3d_v9_catalog_first \
  --workers 40 --sample-retries 2 --tools \
  --reasoning-summary auto \
  --turn-timeout 900
```

## Results

Deterministic oriented-3D-IoU vs GT 9-DOF box. `Acc@0.25 == Acc@0.50` (pool is
`source = gt`, so IoU is bimodal and `mean IoU ~= Acc`). Metrics are over all 600
samples; failed samples count as wrong.

| Slice | n | v7 Acc | correct | v5 Acc | Delta vs v5 |
|---|---:|---:|---:|---:|---:|
| **Overall** | 600 | **86.00%** | 516 | 85.33% | +0.67 |
| Easy | 290 | 90.34% | 262 | 89.31% | +1.03 |
| Hard | 310 | 81.94% | 254 | 81.61% | +0.32 |
| **View-Dep** | 211 | **82.46%** | 174 | 81.04% | +1.42 |
| View-Indep | 389 | 87.92% | 342 | 87.66% | +0.26 |

The v7-v5 deltas are inside the strat600 comparison bands, so this should be read
as a regression check passing rather than a new accuracy claim.

### Acceptance gate

| Gate | Threshold | v7 | Result |
|---|---:|---:|:--:|
| Overall Acc@0.25 | >= 83.0% | 86.00% | pass |
| View-Dep Acc@0.25 | >= 76.5% | 82.46% | pass |
| `mean_cache_ratio` | >= 0.85 | 0.9297 | pass |
| `cache_hit_rate` | >= 0.98 | 0.9867 | pass |
| Hard failures | <= 10 | 2 | pass |
| Route fallback | no systematic chat/local compact fallback | 4,837 `POST /v1/responses`, 0 chat/crawl hits | pass |

### Run health

- **Completion:** 600/600 sample records. `status`: 598 completed, 2 failed.
- **Failures:** both failed rows were semantic "target absent as a standalone
  proposal" decisions, with no exception text. They count as wrong.
- **Reasoning summaries:** captured on **542/600 (90.33%)** samples, higher than
  v5's 491/600 (81.8%).
- **Latency:** median **165.3 s**, p90 **378.3 s**, max **830.6 s**.
- **Timeout backstop:** 2 turns emitted `exceeded turn_timeout_s=900` warnings and
  finalized from collected evidence; no sample exceeded the 900 s max duration in
  the final latency distribution.
- **Transport:** log grep found 4,837 `POST /v1/responses` entries, 0 chat/crawl
  route hits, 0 compact route hits during the benchmark, and 1 transient upstream
  503 that was absorbed by retry.
- **Adapter cleanup:** after the run, `127.0.0.1:8787` had no listening process.

### Failed samples

| Sample | Reason |
|---|---|
| `scannet/scene0353_00::5::27624` | Model saw the described orange shirt cue, but concluded no independent matching proposal existed in the pool. |
| `scannet/scene0578_00::18::13314` | Model saw the small black box cue on the whiteboard, but concluded it was not a standalone proposal. |

## Interpretation

This run answers the adapter question directly: the simplified adapter can be the
thin bridge

```text
Codex SDK localhost /v1/responses -> ModelHub /responses
```

with request-body mutation disabled by default. On the same strat600 fold, the
result is v5-level or slightly better, prefix caching stays healthy, and reasoning
summaries are still collected by the business runtime. That means the old adapter
logic that converted or compacted responses is no longer needed for this path,
except as explicit legacy compatibility.

The run does not prove a new model-quality improvement over v5 because the
v7-v5 differences are within the expected strat600 variance bands. It does prove
that the responses-passthrough adapter preserves the v5 behavior that mattered:
high View-Dep accuracy, high cache ratio, and cross-turn reasoning-summary
availability.

## Caveats

- **Single fold, single run.** Good enough for the adapter regression gate, but not
  a new leaderboard claim against the full NR3D set.
- **No SQLite ingestion for NR3D.** This repo still stores NR3D per-sample outputs
  as JSON under the run directory; no NR3D ingester exists yet.
- **Private upstream config omitted.** The run used the gitignored
  `codex_modelhub_adapter/.modelhub_upstreams.toml`; this doc records the path and
  route, not secrets.
