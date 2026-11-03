# v3: Codex–Gemini state replay and cache, 2026-11-03

## Result

**The real Codex SDK → experimental adapter → ModelHub Gemini chain completed
multi-turn tool use and produced cache hits while replaying native signed state.**
In the final 11-turn cache run, turn 11 reported 8,626 input tokens and 4,013
cached tokens. An otherwise matched request with signatures removed reported
698 input tokens. The 7,928-token difference equals the accumulated reported
reasoning tokens; under that usage-accounting interpretation, at least
**4,013 − 698 = 3,315 cached tokens** came from historical reasoning/state.
This is an inference from upstream usage and controlled requests, not inspection
of provider KV storage or plaintext hidden reasoning.

All **11 runs / 65 upstream model requests / six dynamic-tool executions** are
archived, including failures and negative cache results. HTTP counts were
**64 × 200, 1 × 400**; 34 SDK turns completed successfully. Health requests and
two initialization failures that made no model request are excluded from the
65-request count. This is a protocol/integration diagnostic, with no dataset,
fold, judge, or task-accuracy leaderboard.

## Actual chain and provenance

```text
openai-codex 0.163.0 / bundled Codex CLI 0.163.0
  → isolated http://127.0.0.1:8791/v1/responses (Responses SSE)
  → office ModelHub /api/modelhub/online/v2/crawl (non-streaming)
  → gemini-3.8-flash
```

The office hostname was `aidp-i18ntt-sg.tiktok-row.net`. Runs used a clean detached
worktree at HEAD `255b091849c0bafeef8f5e494d022a7750eefa91`, under
`tmp/gemini_codex_cache_worktree_20261103/`; summaries record `tracked_drift=false`.
The main checkout's earlier documentation changes were not execution code.
Untracked probe versions are separately identified by SHA256 in each summary;
all 36 recorded core source references match preserved versions in
[source_versions_verified.json](assets/v3_codex_gemini_cache_20261103/source_versions_verified.json).
The [manifest](assets/v3_codex_gemini_cache_20261103/manifest.json) inventories the archive.

Each SDK run used a fresh isolated HOME/CODEX_HOME/cwd, one thread and one fixed
session/header binding. The user-supplied AK was read into memory from the current
chat and held fixed; no other credential source or AK rotation was used. Raw AKs,
signatures and hidden reasoning were not written to the archived diagnostics.
Exact prompts, model catalogs, runtime configuration, SDK events/usage and native
request traces are preserved with the corresponding run.

The bridge stores complete native assistant messages in memory and issues an
HMAC-checked opaque handle through Responses `reasoning.encrypted_content`.
Codex replays that handle; the bridge restores the original signed native message.
Saved/restored message hashes and order match in all successful SDK runs.
This is a handle to server memory, not a portable encrypted native-state blob.
The bridge explicitly maps developer messages to system messages while preserving
text/order, and synthesizes downstream Responses SSE from non-streaming upstream
responses. Native tools, skills, multi-agent behavior and compaction were disabled;
the tool probe exposes only the SDK dynamic function `lookup_record`.

## Complete run inventory

Request counts below include accounting controls; a control is not an SDK turn.
Detailed timelines, native/SDK usage comparisons and replay checks are in
[analysis.json](assets/v3_codex_gemini_cache_20261103/analysis.json).

| Run | Model requests | Completed SDK turns / tools | Result |
| --- | ---: | ---: | --- |
| `cache_visible` | 3 | — / 0 | Direct HTTP; identical 8,796-token visible prefix; third request cached 4,078. |
| `cache_reasoning` | 6 | — / 0 | Direct HTTP; 68-token seed generated 5,156 reasoning tokens; three 5,309-token replays cached zero; two controls used 153 input tokens. |
| `cache_reasoning_repeat` | 11 | — / 0 | Direct HTTP; 68-token seed generated 7,863 reasoning tokens; four of eight identical-body replays cached 4,072; two controls used 165 input tokens. |
| `sdk_smoke` | 0 | 0 / 0 | Failed before model request: missing low-level SDK initialize handshake. |
| `sdk_smoke2` | 0 | 0 / 0 | Same failure reproduced and diagnosed as JSON-RPC `Not initialized`. |
| `sdk_smoke3` | 1 | 0 / 0 | Upstream HTTP 400. Original upstream error text was retained only as a hash/length and cannot be reconstructed. |
| `sdk_smoke4` | 5 | 2 / 3 | Passed after explicit developer→system mapping; `alpha, beta, alpha`, original marker/sum/receipts correct; cached zero. |
| `sdk_long` | 5 | 2 / 3 | Same tool chain passed with additional reasoning instructions; first response generated 592 reasoning tokens; cached zero. |
| `sdk_cache` | 9 + 1 control | 9 / 0 | One seed + eight acknowledgements; all completed, cached zero. Final input 5,549 − visible control 411 = 5,138 historical reasoning tokens. |
| `sdk_cache2` | 10 + 1 control | 10 / 0 | Two seeds + eight acknowledgements; cached zero. Final input 5,870 − visible control 553 = 5,317 historical reasoning tokens. |
| `sdk_cache3` | 11 + 2 controls | 11 / 0 | Three seeds + eight acknowledgements; four cache hits; exact matched accounting control establishes the final lower bound. |

The two tool runs verify actual SDK dispatch and results, not simulated tool
text. Cache runs score completion and `OK` acknowledgements, not alphametic
accuracy; all eight acknowledgements in each successful cache run were `OK`.
The successful compatibility change after `sdk_smoke3` is recorded without
claiming its unavailable error text proves a particular upstream root cause.

## Cache attribution and anomalies

For **real SDK `sdk_cache3`**, cached counts were 4,051 (turn 5), 4,032 (turn 8),
4,025 (turn 9), and 4,013 (turn 11). The three seed responses reported
3,433 + 1,600 + 2,810 reasoning tokens; acknowledgement turns added 29 + 28 + 28.
Their total is 7,928. Native raw usage and the SDK's corresponding usage fields
were checked together, as were every saved/restored native-message hash.

The primary [exact accounting control](assets/v3_codex_gemini_cache_20261103/sdk_cache3_exact_control/summary.json)
kept `max_tokens=8192`, matching turn 11, and changed only model-input signature
fields. Empty `reasoning_content` was reconstructed from its verified empty-string
hash when building the control from sanitized artifacts. It returned HTTP 200,
698 input tokens and zero cached tokens. Thus `8626 − 698 = 7928`, and even if
every visible token was cached, `4013 − 698 = 3315` cached tokens remain attributable
to the replayed historical state under the reported accounting. The earlier
128-output-token control also measured 698, but is auxiliary evidence because
its output limit differed. This control is not a production recovery strategy.

For **direct HTTP `cache_reasoning_repeat`**, all eight replay bodies had the same
SHA256. Repeats 4–7 each reported 4,072 cached tokens; the signature-removal and
visible-history controls both measured 165 input tokens. The corresponding
accounting lower bound is `4072 − 165 = 3907`. This is separate evidence from the
SDK experiment, not a substitute for it.

Two anomalies remain visible: direct replay 1 reported only 165 input tokens
and `finish_reason=length`, despite the same body hash as the later 8,028-token
requests. SDK turn 7 reported only 630 input tokens despite matching local state
restoration hashes; subsequent turns returned to over 8,000. Their upstream cause
is unresolved. Therefore state replay correctness in this bridge does not prove
that every upstream request applies or accounts for that state consistently.
The earlier zero-cache runs also show that replay does not guarantee a cache hit;
these measurements do not establish a universal length threshold or latency gain.

## Reproduction, archive and checks

The durable [assets](assets/v3_codex_gemini_cache_20261103/analysis.json) contain
all runs, four accounting controls, bridge traces, scripts and their historical
versions, prompts, gate logs, source verification and the SQLite importer.
`ingest.py` verified **11 runs / 65 requests / six tools** in the v3 portion of
`docs/benchmark/adapter/runs.sqlite`. Missing failure usage is not measured zero.

Reproduction must run in tmux with `agentic3d` and a clean execution worktree.
Copy the archived scripts to `<worktree>/tmp/gemini_bridge_probe/` first: drivers
derive their worktree from `Path(__file__).parents[2]`. Use fresh output names.
For historical runs, restore the versions matched in `source_versions_verified.json`.
The final bridge and cache driver use these commands from that directory, with
the service and client in separate tmux windows:

```bash
PROBE_TRACE_DIR="$PWD/repro_bridge_traces" python -m uvicorn bridge:app --host 127.0.0.1 --port 8791
timeout 480 python sdk_cache_driver.py \
  --credential-transcript '<local-current-chat-transcript>' \
  --output repro_sdk_cache3 --adapter-url http://127.0.0.1:8791/v1 \
  --second-seed --third-seed
```

The tool driver is `sdk_driver.py` with the same credential/output/adapter flags;
`sdk_long` additionally used `--prompt-file long_reasoning_prompt.txt`. Direct
`cache_probe.py` cases were `--case visible_prefix --repeats 3`,
`--case reasoning --repeats 3`, and `--case reasoning --repeats 8 --delay 2`;
see source-version mappings for their executed bytes. Accounting control scripts
and their parameters/results are archived alongside the relevant SDK sessions.

Final checks passed: **16 mock tests**, Ruff and Black on eight current probe
files, and mypy on the bridge. Earlier import/unused-import Ruff failures remain
in the historical logs. Source hashes and credential redaction were checked.
The isolated port 8791 service was stopped; the existing 8787 service was untouched.

## Limits and production policy

Production adapter source was not changed. The experimental bridge cannot resume
handles after restart and does not implement compact, image integration, durable
session recovery, native upstream streaming, or general cross-provider conversion.
No raw hidden reasoning was inspected; cache attribution relies on comparable,
accurate upstream token accounting. No task-accuracy improvement is claimed.
**GPT remains native Responses-only: no Chat conversion or automatic downgrade.**
These Gemini-specific experiments do not relax that persisted repository policy.
