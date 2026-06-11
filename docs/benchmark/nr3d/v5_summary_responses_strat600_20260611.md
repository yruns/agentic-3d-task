# NR3D v5 — vendored adapter + reasoning summary (/responses path), strat600

Fifth **strat600** evaluation of the Codex Agent SDK runtime (`src/codex_agent/`).
This is the first full-fold run on the **in-repo (vendored) ModelHub adapter**
(`codex_modelhub_adapter/`) with **reasoning summaries enabled
(`--reasoning-summary auto`)**, which forces those requests onto the upstream
**`/responses`** API instead of the chat-completions (`/v2/crawl`) path that
[v4](v4_network_fix_strat600_20260610.md) used.

> **Status: COMPLETE — 600/600.** Launched + finished 2026-06-11 (`V5_EXIT_CODE=0`,
> ~71 min, no tail wedge, 0 samples hit the 900 s cap). Raw artifacts:
> `tmp/nr3d_tools_case600_v5_summary_responses/` (`summary.json` +
> `leaderboard_strat600.json` + `per_sample/`).
>
> **One-line takeaway:** v5 is **Acc@0.25 = 85.33 % (512/600), +7.00 pp over v4**
> — a uniform gain across *every* tier (Easy +6.55, Hard +7.42, **View-Dep +12.79**,
> View-Indep +3.86), all far outside the strat600 variance bands → **real, not
> nondeterminism** (v4↔v3 was Δ0.00). **The [v6 ablation](v6_effort_ablation_chat_strat600_20260611.md)
> isolated what drove it:** of the +7 pp, only **≈+2 pp is the reasoning-effort
> bump** (`""`→`medium`, measured on the chat path = v6) and **≈+5 pp is the
> `/responses` path itself** (forced by summaries), holding effort fixed. So the
> dominant lever is the **`/responses` path** (cross-turn encrypted reasoning
> carryover for this multi-turn agent), **not** the effort param as first
> hypothesised here. Reasoning summaries captured on **491/600 (81.8 %)**; only
> **2 hard failures** (vs v4's 11). _(This takeaway was corrected after v6; the
> original draft over-attributed the gain to effort — see Interpretation.)_

## What changed vs v4

Three deltas vs the v4 run: vendored adapter (byte-faithful, no behaviour change),
effort `""`→`medium`, and summaries-on (which forces `/responses`). Together they
moved Overall +7 pp. The [v6 ablation](v6_effort_ablation_chat_strat600_20260611.md)
later split that into **≈+2 pp from effort** and **≈+5 pp from the `/responses`
path** — see Interpretation.

1. **Vendored adapter.** v4 ran against the external
   `/Users/bytedance/aispace/codex_modelhub_adapter`. v5 runs against the copy now
   tracked in-repo at `codex_modelhub_adapter/` (commit `2df6ee0`), serving from
   its own `.venv` and local secret `codex_modelhub_adapter/.modelhub_upstreams.toml`
   (gitignored; same 3-AK weighted pool `gpt54_a:gpt54_b:gpt54_c = 5:1:5`).
2. **Reasoning summary on (`auto`).** The runtime now passes `summary` at the SDK
   call site (commit `88488f4`) and captures any `ReasoningThreadItem` summary into
   `CodexTurnMetadata.reasoning_summary` → `Nr3dSampleResult.reasoning_summary`.
   Empirically, summaries only stream from the **Responses** API, so the vendored
   adapter's `proxy.py` (commit `2df6ee0`) **forces `/responses`** for any request
   that explicitly asks for a non-empty, non-`none` `reasoning.summary`. Every other
   request still defaults to the chat path. Net effect: **all v5 turns route to
   `/responses`** (vs v4's chat path).
3. **Reasoning effort pinned to `medium`.** v4 used "model default (ON)". v5's code
   default is now `medium` (commit `88488f4` + `56e2d24`), passed explicitly at the
   call site rather than via TOML.

Everything else matches v4: inline playbook, loop guard
(`max_tool_calls=30`/`max_repeated_tool_calls=4`), image ≤768 px,
`model_context_window=900000`, workspace-write sandbox + network access, catalog
selectors (`pack_nr3d_v9_catalog_first`).

## Pre-run

| Item | Value |
|---|---|
| Repo / branch | `agentic-3d-task` / `master` |
| Head commit at launch | `56e2d24` (no worktree drift; head == run-time code) |
| Run-time code commit | `56e2d24` |
| Fold | `tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json` — canonical strat600, sha256 `3f3023595167fc7d744e5add205ea721a9eee01506079f02acc30df5c9b816b8` (byte-identical to `3DVLMReasoning/docs/benchmark/nr3d/assets/v9_3_strat600_sample_ids_20260517.json`; same 600 ids as v2/v3/v4) |
| Fold tiers | Easy 290 / Hard 310; View-Dep 211 / View-Indep 389 |
| Data root | `/Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet` |
| Pack | `pack_nr3d_v9_catalog_first` |
| Backend | `gpt-5.4-2026-03-05` via **vendored** Codex `codex_modelhub_adapter` (`127.0.0.1:8787`) |
| Upstream API | **`/responses`** (forced by `--reasoning-summary auto`; v4 used chat `/v2/crawl`) |
| AK pool | weighted `codex_modelhub_adapter/.modelhub_upstreams.toml` (`gpt54_a:gpt54_b:gpt54_c = 5:1:5`) |
| Sandbox | `workspace_write` + `network_access=true` |
| Loop guard | `max_tool_calls=30`, `max_repeated_tool_calls=4` (tools defaults) |
| Reasoning effort | **`medium`** (code default; v4 was model-default) |
| Reasoning summary | **`auto`** (NEW; captured into per-sample metadata) |
| Turn budget | **`--turn-timeout 900`** (safety vs the /responses stall seen at 10-case scale; v4 used 0/off) |
| `model_context_window` | `900000` (`CODEX_AGENT_MODEL_CONTEXT_WINDOW`) |
| Concurrency | **40 workers**, `sample_retries = 2` |
| Run home | not kept (default) — avoids run-home bloat |
| Judge | none — deterministic oriented 3D IoU vs GT 9-DOF box |
| Date | 2026-06-11 |

### CLI

```bash
export CODEX_AGENT_MODEL_CONTEXT_WINDOW=900000
PYTHONPATH=src python -m codex_agent.cli.run_nr3d \
  --sample-ids tmp/nr3d_artifacts/v9_3_strat600_sample_ids.json \
  --data-root /Users/bytedance/project/3DVLMReasoning/data/nr3d/scannet \
  --output-dir tmp/nr3d_tools_case600_v5_summary_responses \
  --pack-name pack_nr3d_v9_catalog_first \
  --workers 40 --sample-retries 2 --tools \
  --reasoning-summary auto --turn-timeout 900
```

## Results

Deterministic oriented-3D-IoU vs GT 9-DOF box. `Acc@0.25 == Acc@0.50` (pool is
`source = gt`, so IoU is bimodal and `mean IoU ≈ Acc`). `mean IoU = 0.8554`,
`cache_hit_rate = 0.9917`, `mean_cache_ratio = 0.9136`. Metrics computed over all
600 (failures count as wrong); derived from `per_sample/` joined with the fold's
tier flags → `leaderboard_strat600.json`.

| Slice | n | v5 Acc | correct | v4 Acc | Δ vs v4 | strat600 band | within band? |
|---|---:|---:|---:|---:|---:|---:|:--:|
| **Overall** | 600 | **85.33%** | 512 | 78.33% | **+7.00** | ±2.3 pp | ❌ (real) |
| Easy | 290 | 89.31% | 259 | 82.76% | +6.55 | ±2.7 pp | ❌ (real) |
| Hard | 310 | 81.61% | 253 | 74.19% | +7.42 | ±3.5 pp | ❌ (real) |
| **View-Dep** | 211 | **81.04%** | 171 | 68.25% | **+12.79** | ±4.5 pp | ❌ (real) |
| View-Indep | 389 | 87.66% | 341 | 83.80% | +3.86 | ±2.9 pp | ❌ (real) |

Every tier moved **up**, all well beyond the 90 % variance bands — so this is a
**systematic gain**, not the ±6-case shuffle seen between v3 and v4. The effect is
strongest exactly where more reasoning should help most: **View-Dep +12.79 pp**
(view-dependent spatial reasoning) and **Hard +7.42 pp**.

### Run health

- **Completion:** 600/600. `status`: 598 completed, **2 failed** (down from v4's
  11). Both failures are 429-retry-exhausted (`ResponseTooManyFailedAttempts`,
  `exceeded retry limit, last status: 429`) under 40-worker load on the /responses
  path; they count as wrong in the table above. 0 permanent infra errors.
- **Reasoning summaries:** captured on **491/600 (81.8 %)** samples (the remainder
  are best-effort empty — the model didn't emit summary fragments that turn; same
  behaviour as the 10-case smoke at 9/10). Stored per-sample in
  `reasoning_summary` for tracing/debugging.
- **Latency:** median **128.8 s** (v4 113.9 s), p90 378 s (v4 278 s), max 836 s
  (v4 788 s). ~13 % slower median — the cost of medium effort + summary generation
  + the /responses round-trip. **0 samples hit the 900 s turn cap**, so the
  `--turn-timeout 900` backstop never fired (no legit turn cut off, no indefinite
  stall — unlike the 10-case smoke which wedged under `turn_timeout=0`).
- **Caching:** `cache_hit_rate 0.9917`, `mean_cache_ratio 0.9136` — **higher** than
  v4's chat path (0.967 / 0.572). The /responses prefix cache held up well at
  40-worker concurrency despite a handful of transient 429s.

### Cross-version comparison (same strat600 fold)

| Version | Reasoning effort | Upstream API | Overall | View-Dep | Notes |
|---|---|---|---:|---:|---|
| v1 | model-default | chat | 63.83% | 51.66% | prompt-only floor (600/600) |
| v3 | model-default | chat | 78.33% | 71.09% | tools loop-fix; kf dead |
| v4 | **model-default (`""`)** | chat | 78.33% | 68.25% | tools; kf live; network fix |
| **v5** | **`medium`** | **/responses** | **85.33%** | **81.04%** | this run (600/600); summaries on |

### Interpretation — what drove +7 pp? (resolved by v6)

Three things changed vs v4 (vendored adapter, effort `""`→`medium`, summary→/responses).
The vendored adapter is a byte-faithful copy (`2df6ee0`), so it is not a behaviour
change. The remaining two — effort and API path — were confounded in this single
run, so v5 alone could not attribute the gain. The
**[v6 ablation](v6_effort_ablation_chat_strat600_20260611.md)** (medium effort on
the **chat** path, summaries off) settled it:

```
v4  78.33%  default effort, chat
 │  +2.00 pp   reasoning effort ""→medium      (v6, chat; ≤ ±2.3 band → near-noise)
v6  80.33%  medium effort, chat
 │  +5.00 pp   chat → /responses (effort fixed)  (this run, v5; > band → REAL)
v5  85.33%  medium effort, /responses
```

- **My first guess here (effort is the prime mover) was wrong.** Medium effort on
  the chat path buys only **+2 pp** (Overall, within/at the noise band).
- The **dominant lever is the `/responses` path itself: +5 pp** at fixed effort.
  Mechanism: `/responses` preserves the model's **encrypted reasoning state across
  the agent's multi-turn tool loop**, which the stateless chat path drops. For this
  inspect→rank→decide grounding agent, that cross-turn continuity is what moves the
  needle — most on **View-Dep**, where the path delta is **+8.53 pp** (vs effort's
  +4.26). The View-Dep / Hard concentration that *looked* like an "effort signature"
  in the first draft is actually the **path** signature.

**Practical upshot:** to get v5-level accuracy you need the `/responses` path (today
via `--reasoning-summary auto`); `--reasoning-effort medium` on the cheap chat path
recovers only ~2 of the 7 pp. See v6 for the full decomposition and the residual
"does chat fully honour effort?" confound.

## Caveats

- **Single fold, single run.** +7 pp is far outside the variance bands so it is
  unlikely to be noise. Attribution (effort vs /responses) **was isolated by
  [v6](v6_effort_ablation_chat_strat600_20260611.md)**: ≈+2 pp effort, ≈+5 pp the
  /responses path. A multi-seed run would still firm up the per-tier splits.
- **/responses caching differs from chat.** v5's `mean_cache_ratio 0.9136` is not
  directly comparable to v4's `0.572`; cost/latency figures are path-specific.
- **`--turn-timeout 900`** did not fire (max sample 836 s < 900 s), so it neither
  helped nor hurt this run; it remains a useful backstop for the /responses stall
  class seen at 10-case scale.
- **No SQLite ingestion / per-tier `keyframe_selector` usage** for NR3D in this repo
  yet (run home not kept) — same gap as v1–v4; tracked as a follow-up, not a
  blocker.

## Caveats (set before launch)

- **Not a controlled ablation.** v5 changes three things at once vs v4 (vendored
  adapter, /responses path, effort=medium, summary=on). A v5-vs-v4 accuracy delta
  cannot be cleanly attributed to any single factor; it is confounded with ordinary
  single-fold run-to-run nondeterminism (v4 itself showed a ±6-case shuffle vs v3 at
  Δ Overall 0.00). Treat v5 as: "does the new code path on the new adapter hold the
  ~78 % strat600 line while emitting summaries?" — not as a measurement of summary's
  accuracy effect.
- **/responses caching differs from chat.** v4's chat path hit `cache_hit_rate 0.967`,
  `mean_cache_ratio 0.572`. The /responses path has its own caching profile (the
  10-case probe showed `hit_rate 0.9`, `mean_cache_ratio 0.87`). Cost/latency
  numbers are therefore not directly comparable to v4.
- **`--turn-timeout 900`** can finalize a turn that legitimately runs long (v4's max
  *sample* latency was 788 s). Any sample that hits 900 s is finalized with its
  best-current proposal; the count of such samples is reported post-run.
