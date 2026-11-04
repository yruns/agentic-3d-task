# Adapter Protocol Diagnostics

This archive records direct protocol probes and adapter compatibility checks,
including failed requests. These synthetic checks have no dataset fold, judge,
or task accuracy leaderboard; they do not enter the NR3D/OpenEQA leaderboards.

| Date | Record | Scope | Result |
| --- | --- | --- | --- |
| 2026-11-03 | [v3 Codex–Gemini state/cache](v3_codex_gemini_cache_20261103.md) | Real SDK bridge plus direct controls, 11 runs | 65 requests, six tools, 34 completed SDK turns. Actual SDK cache hits; final reported reasoning/state cache lower bound 3,315 tokens. Failures and zero-cache runs retained. |
| 2026-11-03 | [v2 Gemini tool/signature probe](v2_gemini_tool_state_20261103.md) | Direct HTTP, `gemini-3.8-flash`, three subruns | 17 requests including six controls; seven tools. Signature replay passed; tool-signature corruption/removal failed. Initial fenced-JSON parser failure retained. |
| 2026-11-03 | [v1 ModelHub tool/state probe](v1_modelhub_tool_state_20261103.md) | Direct HTTP, `gpt-6-luna`, five subruns | 15 requests including two controls; seven local tool executions. Legacy Chat rejected; unified Chat works with effort `none`; unified Responses accepts unchanged encrypted-state replay and rejects corrupted state. |

Current routing policy: **GPT uses native Responses only; no Chat or automatic
downgrade.** Gemini's experimental signature codec passed SDK checks; production
integration and durable recovery remain unfinished. Historical Chat probes remain
comparison evidence, not route policy.

Durable sanitized requests, responses, summaries, and executed probe versions
live under [v1 assets](assets/v1_modelhub_tool_state_20261103/responses_replay/summary.json)
and [v2 assets](assets/v2_gemini_tool_state_20261103/crawl_replay/summary.json),
with [v3 analysis/assets](assets/v3_codex_gemini_cache_20261103/analysis.json).
`runs.sqlite` is a local derived archive, rebuilt using the version's
`assets/<version>/ingest.py`; the JSON records are the durable
source of truth.
