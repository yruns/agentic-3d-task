# NR3D v6 — effort ablation: `medium` on the CHAT path (summary off), strat600

Ablation to **isolate the v5 +7 pp gain**: was it the reasoning-effort bump
(`""`→`medium`) or the `/responses` path that summaries forced? v5 changed both at
once. v6 holds effort at `medium` but turns **summaries off**, so the request stays
on the **chat** (`/v2/crawl`) path — the same path v4 used.

> **Status: COMPLETE — 600/600.** Launched + finished 2026-06-11 (`V6_EXIT_CODE=0`,
> ~64 min). Raw artifacts: `tmp/nr3d_tools_case600_v6_medium_chat/`
> (`summary.json` + `leaderboard_strat600.json` + `per_sample/`).
>
> **One-line takeaway — the ablation overturned the v5 hypothesis.** v6 (medium
> effort, **chat** path) = **Acc@0.25 = 80.33 % (482/600)**, only **+2.0 pp over v4**
> and **−5.0 pp under v5**. So the v5 +7 pp decomposes as **≈+2 pp from the effort
> bump (chat, near the noise band) + ≈+5 pp from the `/responses` path** (the real,
> dominant lever, holding effort fixed at medium). Routing chat→`/responses` —
> which preserves the model's encrypted reasoning state across the agent's
> multi-turn tool loop — is what actually moves NR3D grounding, especially View-Dep
> (the path alone is **+8.53 pp** there). Bumping `--reasoning-effort medium` on the
> cheap chat path buys only ~2 pp. Chat-path signature confirmed: `cache_hit_rate
> 0.965 / mean_cache_ratio 0.577` (≈ v4's 0.572; v5's /responses was 0.914).

## The 2×2 this completes

| | effort = `""` (model-default) | effort = `medium` |
|---|---|---|
| **chat path** | **v4** = 78.33% | **v6** = _this run_ |
| **/responses path** | _(not run)_ | **v5** = 85.33% |

- If **v6 ≈ v5 (~85 %)** → the lever is **effort alone**; the /responses path adds
  nothing, and medium effort can run on the cheaper chat path (summaries off).
- If **v6 ≈ v4 (~78 %)** → effort on the chat path does *not* help; the **/responses
  path** (native effort honouring + cross-turn encrypted reasoning carryover) is the
  real driver, and summaries-on is the price of the gain.
- In between → both contribute.

### Why this ablation is valid

The adapter's chat conversion **does forward reasoning effort** —
`adapter/mapping.py:119-123` maps `reasoning.effort` → chat `reasoning_effort` when
non-empty. So v6's `medium` genuinely reaches the chat endpoint (only
`reasoning.summary` is dropped on the chat path). v4 sent `effort=""` (falsy → not
forwarded → model default); v6 sends `medium`. Same path, only the effort param
differs → clean isolation.

## Pre-run

| Item | Value |
|---|---|
| Repo / branch | `agentic-3d-task` / `master` |
| Run-time code commit | `56e2d24` (code clean; only uncommitted tree changes are the v5 doc artifacts — no code drift) |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` — canonical strat600, sha256 `3f3023595167fc7d744e5add205ea721a9eee01506079f02acc30df5c9b816b8` (same 600 ids as v2–v5) |
| Data root | `/Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet` |
| Pack | `pack_nr3d_v9_catalog_first` |
| Backend | `gpt-5.4-2026-03-05` via vendored `codex_modelhub_adapter` (`127.0.0.1:8787`) |
| Upstream API | **chat `/v2/crawl`** (no summary requested → adapter's default model routing) |
| Reasoning effort | **`medium`** (forwarded to chat `reasoning_effort`) |
| Reasoning summary | **off** (the only diff vs v5) |
| Sandbox | `workspace_write` + `network_access=true` |
| Loop guard | `max_tool_calls=30`, `max_repeated_tool_calls=4` |
| Turn budget | `--turn-timeout 900` (matches v5; never fired in v5) |
| `model_context_window` | `900000` |
| Concurrency | 40 workers, `sample_retries = 2` |
| Judge | none — deterministic oriented 3D IoU vs GT 9-DOF box |
| Date | 2026-06-11 |

### CLI

```bash
export CODEX_AGENT_MODEL_CONTEXT_WINDOW=900000
PYTHONPATH=src python -m codex_agent.cli.run_nr3d \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --output-dir tmp/nr3d_tools_case600_v6_medium_chat \
  --pack-name pack_nr3d_v9_catalog_first \
  --workers 40 --sample-retries 2 --tools \
  --reasoning-effort medium --turn-timeout 900
```

## Results

`mean IoU = 0.8060`, `cache_hit_rate = 0.965`, `mean_cache_ratio = 0.577` — the
chat-path caching signature (≈ v4's 0.967 / 0.572), re-confirming v6 ran on
`/v2/crawl`, not `/responses`. Metrics over all 600 (failures = wrong).

| Slice | n | v6 (medium,chat) | v4 (default,chat) | v5 (medium,/resp) | **v6−v4** = effort@chat | **v5−v6** = /responses@medium |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | 600 | **80.33%** (482) | 78.33% | 85.33% | **+2.00** | **+5.00** |
| Easy | 290 | 84.48% | 82.76% | 89.31% | +1.72 | +4.83 |
| Hard | 310 | 76.45% | 74.19% | 81.61% | +2.26 | +5.16 |
| **View-Dep** | 211 | 72.51% | 68.25% | 81.04% | **+4.26** | **+8.53** |
| View-Indep | 389 | 84.58% | 83.80% | 87.66% | +0.78 | +3.08 |

### Decomposition of the v5 +7 pp (Overall)

```
v4  78.33%  (default effort, chat)
 │  +2.00 pp  ← reasoning effort  ""→medium   (chat path; ≤ Overall ±2.3 band → near-noise)
v6  80.33%  (medium effort, chat)
 │  +5.00 pp  ← upstream path  chat→/responses  (effort fixed = medium; > band → REAL)
v5  85.33%  (medium effort, /responses)
```

**The `/responses` path is the dominant lever (~5 pp), not reasoning effort (~2 pp).**
Both help View-Dep most (effort +4.26, path +8.53), but the path is ~2× the effect
everywhere and is the only piece that clears the variance bands on its own.

### Why the path matters more than the param

The chat (`/v2/crawl`) path is **stateless across turns**: each agentic turn
re-sends the transcript but the model's prior *reasoning* is gone (chat completions
returns no reasoning content; `mapping.py` drops empty reasoning items). The
`/responses` path can carry **encrypted reasoning state** across turns, so this
multi-turn evidence-seeking agent (inspect frames → rank proposals → decide) keeps
its chain-of-thought between tool calls. That compounding matters most on
**View-Dep** cases that need sustained spatial reasoning — exactly where the path
delta is largest (+8.53). The `reasoning_effort=medium` param *is* forwarded to
chat (`mapping.py:119-123`), and it does help a little (+2 pp), but it cannot
substitute for cross-turn reasoning continuity.

> **Residual confound (honest):** v6 = v4 + medium effort gives +2 pp on chat, but
> we cannot fully rule out that `/v2/crawl` only *partially* honours
> `reasoning_effort`. Even so, the conclusion holds directionally: at the same
> nominal medium effort, switching to `/responses` adds +5 pp — so the gain lives
> on the `/responses` path, however that path realises it (state carryover and/or
> more faithful effort honouring).

### Run health

- 600/600. `status`: 590 completed, **10 failed** (429-retry-exhausted under
  40-worker load; count as wrong). More than v5's 2 — the chat crawl upstream was
  somewhat more 429-prone in this window. 0 permanent infra errors.
- **Latency:** median 148.7 s, p90 353.7 s, max 755.6 s. **0** samples hit the
  900 s cap.
- No reasoning summaries (off by design — `with_summary = 0/600`, verified).

### Practical implication

To get the v5-level accuracy you must run on the **`/responses`** path (today that
means `--reasoning-summary auto`, which forces it). `--reasoning-effort medium` on
the cheaper chat path recovers only ~2 pp of the 7. A cleaner future lever would be
an explicit "use /responses" adapter/runtime toggle decoupled from summaries, so
the +5 pp path gain can be had without paying for summary generation.
