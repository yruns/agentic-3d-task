# Prompt caching for the Codex Agent runtime (NR3D)

Status: **configured + verified working** on 2026-06-08.

Reference: ModelHub Prompt Cache实战 doc
(`https://bytedance.sg.larkoffice.com/docx/QTGSd9IOAoAO5VxRRrDlAfZvgLp`).

## Mechanism (how a hit happens)

The Codex Agent path is `Codex SDK → modelhub_adapter (127.0.0.1:8787) → ModelHub
gpt-5.4`. ModelHub prompt cache requires **all** of:

1. AK has model permission (✓ our gpt-5.4 AKs).
2. Platform "account stickiness" switch enabled for the `business × model` (✓ —
   proven by the feasibility probe below).
3. Prompt prefix ≥ 1024 tokens (✓ — Codex base instructions + skill alone exceed
   this).
4. **Dynamic content at the prompt tail** (stable prefix first).
5. Request carries a **stable `extra.session_id`** (the gateway sticks the same
   session to the same backing instance, which holds the cache).

Our adapter already forwards a **stable `session_id`** to ModelHub while
selecting the AK by per-turn `chat_run_id` (load balancing). So each of the 3
gpt-5.4 AKs warms its own copy of the shared prefix; after a one-time cold write
per AK, subsequent same-prefix requests hit.

## Phase 1 — feasibility (direct ModelHub, isolates the platform switch)

`tmp/verify_prompt_cache.py` hits the crawl endpoint directly with our gpt-5.4 AK
(read from `configs/llm.toml`), a ~2400-token stable system prefix, a stable
`extra.session_id`, 4 rounds:

| Round | prompt_tokens | cached_tokens | ratio |
|---|---:|---:|---:|
| 1 | 3644 | 0 | first write |
| 2 | 3641 | 3456 | 94.9% |
| 3 | 3640 | 3456 | 94.9% |
| 4 | 3641 | 3456 | 94.9% |

→ Prompt cache is **feasible** for our setup (switch on, AK permitted, threshold met).

## Phase 2 — what was changed to make it work + observable end-to-end

### Adapter (`/Users/bytedance/aispace/codex_modelhub_adapter`)

The adapter previously **dropped** `cached_tokens` and returned no usage at all
for streamed chat-completions responses (so the Codex SDK saw `usage=None`). Fixed:

- `build_chat_completions_body`: add `stream_options={"include_usage": true}` on
  streamed upstream calls so usage (incl. `prompt_tokens_details.cached_tokens`)
  is returned.
- `iter_chat_sse_as_responses`: capture the upstream usage chunk and emit it in
  `response.completed.usage`.
- `_usage`: pass `cached_tokens` through as `input_tokens_details.cached_tokens`
  (so the Codex SDK populates `TokenUsageBreakdown.cached_input_tokens`).
- `log_cache_hit` (env-gated by `AIDP_LOG_PROMPT_CACHE=1`): one line per request:
  `[prompt-cache] alias=<gpt54_x> logid=<...> prompt_tokens=N cached_tokens=M ratio=…%`.

Backups: `adapter/mapping.py.bak.*`, `adapter/app.py.bak.*`. Adapter tests: 18 pass.
Restart via `restart_adapter.sh` (now exports `AIDP_LOG_PROMPT_CACHE=1`); running
in tmux session `codex-adapter`.

### Runtime (`src/codex_agent`)

- `CodexAgentConfig.prefix_cache_session_id` is the stable `session_id` (already
  sent via the `extra` header per turn); set it per run with
  `CODEX_AGENT_PREFIX_CACHE_SESSION_ID`.
- `CodexTurnMetadata` now carries `input_tokens`, `cached_input_tokens`, and a
  `cache_ratio` property (extracted from `result.usage.last`).
- `Nr3dSampleResult` records `input_tokens` / `cached_input_tokens`; the run
  `summary.json` now reports `cache_hit_rate` and `mean_cache_ratio`.

## Phase 3 — live verification on the NR3D path

8-case and 6-case batches (shared `session_id`, gpt-5.4, workers=2). Adapter log
+ Codex SDK usage agree:

- Shared prefix **`cached_tokens=7040`** hits consistently across different samples
  (the Codex base + skill + fixed preamble caches across samples).
- Within-turn finalization re-ask: **`cached_tokens=21504` (96.9%)** (full prior
  turn context cached).
- 6-case summary: `cache_hit_rate=0.5`, `mean_cache_ratio=0.29` — the misses are
  the one-time cold write per AK (3 AKs). At case600 scale the warmup is amortized,
  so steady-state hit rate is high.

The `cached_input_tokens` now appears in every per-sample checkpoint and the run
summary (previously `usage` was `None` for all 600 case600 samples).

## How to monitor a run

- Per-run: read `summary.json → cache_hit_rate / mean_cache_ratio`, or per-sample
  `cached_input_tokens / input_tokens`.
- Live: `grep "\[prompt-cache\]" /tmp/codex_modelhub_adapter_8787_restart.log`
  (requires `AIDP_LOG_PROMPT_CACHE=1` on the adapter).

## Caveats / follow-ups

- **Cold write per AK** is unavoidable (first time each account sees the prefix).
- **Ratio is bounded by the variable proposal list** (~13K tokens/sample, not
  cacheable). The cached prefix is ~7040 tokens. To raise the ratio, move all
  fixed text (rules + JSON schema) to the front and the per-sample query +
  proposals to the tail (doc principle #4). This changes prompt ordering and
  would need an NR3D A/B before adopting — deferred.
- Account stickiness pulls a shared `session_id` toward one account in pure
  clients; our adapter avoids that by rotating AKs on `chat_run_id` while still
  sending the stable `session_id` for cache.
