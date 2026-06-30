# NR3D v8 - encrypted reasoning state fail-closed adapter, strat600

Eighth **strat600** evaluation of the Codex Agent SDK runtime
(`src/codex_agent/`). This run validates the ModelHub Responses adapter after
removing the `invalid_encrypted_content` sanitizer/retry path. The adapter now
leaves encrypted reasoning state fully intact and lets the current Codex turn
fail if ModelHub rejects that state.

> **Status: COMPLETE - 600/600.** Launched 2026-07-01 00:17:10 CST and finished
> 2026-07-01 01:13:34 CST (`exit=0`). Raw artifacts:
> `tmp/nr3d_tools_case600_v8_failclosed_encrypted_state_20260701/`
> (`summary.json` + `per_sample/`). Durable derived metrics:
> `assets/v8_failclosed_encrypted_state_strat600_20260701_leaderboard.json`.
>
> **One-line takeaway:** v8 is **Acc@0.25 = 84.50% (507/600)** with
> **View-Dep = 77.25% (163/211)**, `cache_hit_rate = 0.9900`, and
> `mean_cache_ratio = 0.9301`. It still clears the adapter-regression gate, but
> unlike v7 it exposes one real `invalid_encrypted_content` turn failure because
> the adapter no longer silently strips reasoning state and retries.

## What changed vs v7

v7 validated a responses-passthrough adapter, but still had a compatibility
fallback that sanitized encrypted reasoning state when upstream returned
`invalid_encrypted_content`. That fallback was intentionally removed in
`8054f6b`:

1. **No encrypted-state sanitizer.** `adapter/encrypted_state.py` and its tests
   were deleted.
2. **No fallback flag.** `AIDP_CODEX_PROXY_ENCRYPTED_STATE_FALLBACK_ENABLED` was
   removed from config, health payload, and docs.
3. **Fail-closed upstream errors.** Non-429 upstream errors are buffered and
   returned to Codex unchanged. `429` failover is still retained.

This is the safer contract for prompt caching and cross-turn context: encrypted
reasoning state, `previous_response_id`, and reasoning items must correspond
exactly. A sanitized retry may create a false successful turn with missing
context.

## Pre-run

| Item | Value |
|---|---|
| Repo / branch | `agentic-3d-task` / `feat/codex-agent-upgrade` |
| Head commit at launch | `8054f6b` (`8054f6ba423c9cffd0116b6fc9e32db6d8ae333f`) |
| Run-time code commit | `8054f6b` |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json`, sha256 `3f3023595167fc7d744e5add205ea721a9eee01506079f02acc30df5c9b816b8` |
| Fold tiers | Easy 290 / Hard 310; View-Dep 211 / View-Indep 389 |
| Data root | `/Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet` |
| Pack | `pack_nr3d_v9_catalog_first` |
| Backend | `gpt-5.4-2026-03-05` via in-repo `codex_modelhub_adapter` on `127.0.0.1:8787` |
| Upstream API | **Responses API** (`AIDP_CODEX_PROXY_UPSTREAM_API=responses`; adapter health showed `responses_path=/responses`) |
| Encrypted state policy | **fail closed**; no sanitizer/retry fallback |
| Sandbox | `workspace_write` + network access |
| Reasoning summary | `auto` |
| Turn budget | `--turn-timeout 900` |
| `model_context_window` | `900000` (`CODEX_AGENT_MODEL_CONTEXT_WINDOW`) |
| Concurrency | 40 workers, `sample_retries = 2` |
| Judge | none - deterministic oriented 3D IoU vs GT 9-DOF box |
| Date | 2026-07-01 |

### CLI

The launch command is saved as:

```bash
docs/benchmark/nr3d/assets/run_v8_failclosed_encrypted_state_20260701.sh
```

Equivalent command:

```bash
export AIDP_MODELHUB_UPSTREAMS_TOML=/Users/bytedance/project/agentic-3d-task/codex_modelhub_adapter/.modelhub_upstreams.toml
export AIDP_CODEX_PROXY_UPSTREAM_API=responses
export AIDP_CODEX_PROXY_UPSTREAM_ENV=office
export CODEX_AGENT_MODEL_CONTEXT_WINDOW=900000

PYTHONPATH=src python -m codex_agent.cli.run_nr3d \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --output-dir tmp/nr3d_tools_case600_v8_failclosed_encrypted_state_20260701 \
  --pack-name pack_nr3d_v9_catalog_first \
  --workers 40 --sample-retries 2 --tools \
  --reasoning-summary auto \
  --turn-timeout 900
```

## Results

Deterministic oriented-3D-IoU vs GT 9-DOF box. `Acc@0.25 == Acc@0.50` (pool is
`source = gt`, so IoU is close to bimodal and `mean IoU ~= Acc`). Metrics are
over all 600 samples; failed samples count as wrong.

| Slice | n | v8 Acc | correct | v7 Acc | Delta vs v7 | v5 Acc | Delta vs v5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 600 | **84.50%** | 507 | 86.00% | -1.50 | 85.33% | -0.83 |
| Easy | 290 | 88.62% | 257 | 90.34% | -1.72 | 89.31% | -0.69 |
| Hard | 310 | 80.65% | 250 | 81.94% | -1.29 | 81.61% | -0.97 |
| **View-Dep** | 211 | **77.25%** | 163 | 82.46% | -5.21 | 81.04% | -3.79 |
| View-Indep | 389 | 88.43% | 344 | 87.92% | +0.51 | 87.66% | +0.77 |

### Acceptance gate

| Gate | Threshold | v8 | Result |
|---|---:|---:|:--:|
| Overall Acc@0.25 | >= 83.0% | 84.50% | pass |
| View-Dep Acc@0.25 | >= 76.5% | 77.25% | pass |
| `mean_cache_ratio` | >= 0.85 | 0.9301 | pass |
| `cache_hit_rate` | >= 0.98 | 0.9900 | pass |
| Hard failures | <= 10 | 4 | pass |
| Route fallback | no chat/crawl fallback | 0 chat/crawl hits | pass |
| Encrypted-state fallback | none | 1 fail-closed turn | pass |

### Run health

- **Completion:** 600/600 sample records. `status`: 596 completed, 4 failed.
- **Infrastructure failure:** 1 sample failed with `invalid_encrypted_content`;
  this is the expected fail-closed behavior after removing the sanitizer retry.
- **Semantic target-absent failures:** 3 failed rows were normal model outputs
  with summary/token metadata, where the model concluded that the referred target
  was absent from the proposal pool.
- **Reasoning summaries:** captured on **545/600 (90.83%)** samples.
- **Prompt cache:** `cache_hit_rate = 0.9900`, `mean_cache_ratio = 0.9301`.
- **Latency:** median **166.3 s**, p90 **359.5 s**, max **878.5 s**.
- **Timeout backstop:** 6 turns emitted `exceeded turn_timeout_s=900` warnings
  and finalized from collected evidence; no sample exceeded the 900 s cap in the
  final latency distribution.
- **Transport:** adapter log grep found 4,998 `POST /v1/responses` entries during
  the adapter session, 0 chat/crawl route hits, 4 HTTP 400 lines, and 1 transient
  upstream 503 absorbed by retry.

### Failed samples

| Sample | Type | Reason |
|---|---|---|
| `scannet/scene0081_00::0::10619` | infra | ModelHub returned `invalid_encrypted_content`; adapter propagated the error and the turn failed without a final response. |
| `scannet/scene0149_00::21::1743` | semantic absent | Model concluded the described under-counter cupboard was merged into a larger kitchen-cabinet proposal and not isolated as its own proposal. |
| `scannet/scene0169_00::26::28815` | semantic absent | Model concluded proposal #26 was a chair, while the smaller cabinet described in the query was not represented as its own cabinet proposal. |
| `scannet/scene0578_00::18::13314` | semantic absent | Model saw the small black box near the whiteboard, but it was not one of the listed/visible proposals. |

## Interpretation

The fail-closed adapter keeps the key v5/v7 properties: native Responses API,
high prompt-cache ratio, and reasoning-summary capture around 90%. The run still
passes the adapter-regression gate.

The important new signal is not the overall accuracy drop by itself; it is that
one encrypted-state mismatch now surfaces as a hard turn failure instead of being
masked by a sanitized retry. That matches the desired safety contract. However,
View-Dep accuracy fell to **77.25%**, only **0.75 pp** above the gate, so v8 is a
pass but a tighter one than v7. If repeated v8-style runs keep View-Dep near this
line, the next debugging target should be why ModelHub sometimes rejects encrypted
state, not reintroducing fallback.

## Caveats

- **Single fold, single run.** This is enough for the adapter regression gate, but
  not a new leaderboard claim against the full NR3D set.
- **One fail-closed encrypted-state failure.** This is expected behavior for the
  new contract, but it is still an upstream/state-stability issue worth tracking.
- **Private upstream config omitted.** The run used the gitignored
  `codex_modelhub_adapter/.modelhub_upstreams.toml`; this doc records the path and
  route, not secrets.
