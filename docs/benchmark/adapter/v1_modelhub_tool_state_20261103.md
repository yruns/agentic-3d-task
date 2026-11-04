# v1: ModelHub tool calls and opaque state, 2026-11-03

## Result and scope

**Unified Responses accepted unchanged encrypted reasoning state across a
five-request, three-tool chain. Changing one character of that state produced
HTTP 400 `invalid_encrypted_content`.** The successful final answer preserved the
original marker, sum, and both tool receipts.

This is a direct HTTP protocol diagnostic using `gpt-6-luna`, not a Codex SDK,
Codex CLI, or local adapter integration test. Five subruns made **15 HTTP
requests**: 13 main-chain requests plus two deliberate controls. The local probe
executed **seven** model-requested tools. All responses were non-streaming.
There is no dataset, frozen fold, LLM judge, or NR3D/OpenEQA accuracy metric.

| Subrun | Protocol / effort | HTTP requests | Local tools | Observed result |
| --- | --- | ---: | ---: | --- |
| [default](assets/v1_modelhub_tool_state_20261103/default/summary.json) | Legacy `/v2/crawl`; effort omitted | 1 | 0 | HTTP 403, code `-1002`: model no longer available on this endpoint; use unified protocol. |
| [unified_chat](assets/v1_modelhub_tool_state_20261103/unified_chat/summary.json) | Unified Chat; `medium` | 1 | 0 | HTTP 400: this model's Chat route rejects function tools with reasoning; suggests Responses or effort `none`. |
| [responses](assets/v1_modelhub_tool_state_20261103/responses/summary.json) | Unified Responses; `medium` | 3 | 2 | Three HTTP 200 responses; final JSON correct. Opaque state appeared only in the final response, so this subrun alone did not test its replay. |
| [unified_chat_none](assets/v1_modelhub_tool_state_20261103/unified_chat_none/summary.json) | Unified Chat; `none` | 3 | 2 | Three HTTP 200 responses; complete assistant messages replayed unchanged; final JSON correct. No opaque state observed. |
| [responses_replay](assets/v1_modelhub_tool_state_20261103/responses_replay/summary.json) | Unified Responses; `medium` | 5 + 2 controls | 3 | Five main-chain HTTP 200 responses; state replay hashes match. Corruption control: 400. Removal control: 200. |

The counts include the failed probes and controls; they are not a success-rate
benchmark. No streaming variant was executed.
Across all 15 requests, HTTP status counts were **200: 12, 400: 2, 403: 1**.
The 12 successful responses reported a combined **3,520 input tokens, 541 output
tokens, and zero cached input tokens**. Failed requests returned no usage;
missing usage is not recorded as a measured zero-token model execution.

## Provenance and fixed configuration

```text
Branch at launch:         feat/upgrade_codex_adapter
HEAD at launch:           255b091849c0bafeef8f5e494d022a7750eefa91
Tracked worktree drift:  false in all five subrun summaries
Requested model:         gpt-6-luna
Output token limit:      1600
Stream:                  false
Timezone:                Asia/Shanghai (+08:00)
```

The tracked tree was clean while requests ran. The probe scripts were untracked
diagnostic code, separately identified by SHA256 below; a clean HEAD does not
by itself identify that executed code. Documentation and archival changes were
made only after all model requests completed.

| Executed version | SHA256 | Subruns |
| --- | --- | --- |
| [probe_v1.py](assets/v1_modelhub_tool_state_20261103/probe_v1.py) | `5da798d0355415681edc7a9885f276dd465cfb56d7ec8e111eb49dc303dff337` | `default` |
| [probe_v2.py](assets/v1_modelhub_tool_state_20261103/probe_v2.py) | `eae440a8d099512398a50f7cc065cd03fb8af4e0707721d12a7faa782afd46ff` | `unified_chat`, `responses` |
| [probe.py](assets/v1_modelhub_tool_state_20261103/probe.py) | `40d64063b850c37fc0ae6d638c04bc0f71c0f146dd4074f634fc837e089e79f0` | `unified_chat_none`, `responses_replay` |

Each script version originally occupied
`tmp/modelhub_tool_state_probe_20261103/probe.py` when executed. The archived
versioned copies preserve those bytes.

| Subrun | Start time (+08:00) | Run/session ID |
| --- | --- | --- |
| `default` | 18:56:00 | `modelhub_tool_state_20261103_27322bc382` |
| `unified_chat` | 18:57:46 | `modelhub_tool_state_20261103_78eaf94fd2` |
| `responses` | 18:57:46 | `modelhub_tool_state_20261103_a11ecf51c7` |
| `unified_chat_none` | 18:59:02 | `modelhub_tool_state_20261103_97acff3eff` |
| `responses_replay` | 18:59:02 | `modelhub_tool_state_20261103_0359bc2079` |

The one AK explicitly supplied by the user in the current chat was loaded into
process memory from that chat's local transcript. No other credentials were
consulted. The AK was unchanged for all requests, including controls. Each
subrun used a distinct session; its `extra.session_id` and `extra.chat_run_id`
remained equal to its run ID. `X-TT-LOGID` distinguished individual requests.
Credentials, credential-bearing URLs, and raw opaque state are not archived.

All three endpoints used this office base URL:

```text
https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online
```

The exact path suffixes were `/v2/crawl`, `/unified/v1/chat/completions`, and
`/unified/v1/responses`. Successful Chat responses reported model
`gpt-6-luna-2026-09-22`; Responses reported
`deployment-gpt-6-luna-platform-global`. These are the returned identifiers;
their different forms do not establish identical or different underlying model
weights. The protocol and effort settings also differ between successful Chat
and Responses cases, so they are not a controlled model-performance comparison.

## Tool/history test and replay evidence

The prompt required sequential calls to `lookup_record(alpha)` and
`lookup_record(beta)`, then JSON containing the original random context marker,
the sum `137 + 211 = 348`, and both random tool receipts. The receipts were
returned by local Python tool execution, not supplied as expected answers in the
initial prompt. Responses were checked against exact expected JSON and tool
order. The successful Chat and initial Responses chains each executed two tools.

For Chat, the probe appended the complete returned assistant message, including
tool calls and any additional returned fields, before adding the matching tool
result. For Responses, it appended every returned `output` item unchanged,
followed by `function_call_output` items using the original call IDs. Responses
requests set `store=false` and `include=["reasoning.encrypted_content"]`; they
sent explicit accumulated `input`, without `previous_response_id`.

The five-request replay subrun added a user follow-up after the first correct
answer: call `alpha` once more, then reproduce the original final JSON. This
forced another model request after opaque state had appeared. The observed
sequence was:

| Request | Action returned | Reasoning tokens | Opaque state received | Opaque state sent in this request |
| --- | --- | ---: | --- | --- |
| 1 | `alpha` tool call | 0 | None | None |
| 2 | `beta` tool call | 0 | None | None |
| 3 | Correct initial JSON | 15 | 1,572-character `encrypted_content` | None |
| 4 | Follow-up `alpha` tool call | 117 | 3,020-character `encrypted_content` | Original 1,572-character value |
| 5 | Correct final JSON | 0 | None | Both original values |

The 1,572-character value has SHA256
`3dc6b317d77081b580e98a0a542d06ae80bc956b063134bd50e2775b93d49af5`.
The 3,020-character value has SHA256
`dab50fb98a0cd60bc96c7ec30111d23cda40d4b2f89843066d5774af5411f4c1`.
Those hashes match the corresponding next-request fields. For every successful
chain, the probe also checked that all prior output items were present in the
request and that their accumulated hashes matched (`replay_hashes_match=true`).
This verifies the probe's in-memory replay, not the behavior of another client.

`reasoning_tokens` is a usage count, not a state payload. The initial
three-request Responses probe returned 24 reasoning tokens and a 1,656-character
blob only at its final step; that establishes emission, not replay. Successful
Chat with effort `none` reported zero reasoning tokens and returned no opaque
field. Neither a zero token count nor an absent field on an error response is a
general statement about every model or endpoint.

## Deliberate controls

Both controls branched from the replay request containing the two original
reasoning items. They did not replace the successful main-chain requests and
were not automatic error recovery.

| Control | Modification | Result | Interpretation |
| --- | --- | --- | --- |
| [Corrupted](assets/v1_modelhub_tool_state_20261103/responses_replay/control_corrupted_response.json) | Change one character in the first blob; preserve its length and the other blob | HTTP 400, `invalid_encrypted_content`; no usage | This route verifies the supplied opaque state and rejects this corruption. |
| [Removed](assets/v1_modelhub_tool_state_20261103/responses_replay/control_removed_response.json) | Remove whole reasoning items containing encrypted state; retain visible history/tool results | HTTP 200; 17 reasoning tokens and a new 1,592-character blob | The route can accept this visible-history-only input. Acceptance does not prove equivalent reasoning, retained hidden context, or unchanged accuracy. |

The removal request is an intentionally lossy diagnostic. It is not evidence
supporting production deletion of encrypted state after an error. The probe did
not validate a general repair algorithm or compare task accuracy between these
controls.

## Commands and durable artifacts

The original outputs were under `tmp/modelhub_tool_state_probe_20261103/`.
Sanitized requests/responses, per-request field inventories, state lengths/hashes,
tool results, and summaries are preserved under
`docs/benchmark/adapter/assets/v1_modelhub_tool_state_20261103/`.
Raw blobs are replaced by redaction objects containing length and SHA256, so the
archived JSON is evidence rather than a directly replayable conversation.

The commands below preserve the executed options while replacing the local
credential transcript path with a placeholder. Run inside tmux in the existing
`agentic3d` environment from the repository root. The scripts infer the repository
root through `Path(__file__).parents[2]`; copy the desired archived version back
to the original two-level `tmp/` location before running. Running the archived
file directly under `docs/benchmark/.../assets/` would resolve the wrong root.
Use a fresh output directory for any reproduction and record its new run ID.
The script accepts `--credential-transcript`, not an AK command-line value or
an `env:MODELHUB_PROBE_AK` selector.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentic3d
mkdir -p tmp/modelhub_tool_state_probe_20261103

cp docs/benchmark/adapter/assets/v1_modelhub_tool_state_20261103/probe_v1.py tmp/modelhub_tool_state_probe_20261103/probe.py
python tmp/modelhub_tool_state_probe_20261103/probe.py \
  --credential-transcript '<local-current-chat-transcript>' \
  --output tmp/modelhub_tool_state_probe_20261103/default

cp docs/benchmark/adapter/assets/v1_modelhub_tool_state_20261103/probe_v2.py tmp/modelhub_tool_state_probe_20261103/probe.py
python tmp/modelhub_tool_state_probe_20261103/probe.py \
  --credential-transcript '<local-current-chat-transcript>' \
  --output tmp/modelhub_tool_state_probe_20261103/unified_chat \
  --protocol unified_chat --effort medium
python tmp/modelhub_tool_state_probe_20261103/probe.py \
  --credential-transcript '<local-current-chat-transcript>' \
  --output tmp/modelhub_tool_state_probe_20261103/responses \
  --protocol responses --effort medium

cp docs/benchmark/adapter/assets/v1_modelhub_tool_state_20261103/probe.py tmp/modelhub_tool_state_probe_20261103/probe.py
python tmp/modelhub_tool_state_probe_20261103/probe.py \
  --credential-transcript '<local-current-chat-transcript>' \
  --output tmp/modelhub_tool_state_probe_20261103/unified_chat_none \
  --protocol unified_chat --effort none
python tmp/modelhub_tool_state_probe_20261103/probe.py \
  --credential-transcript '<local-current-chat-transcript>' \
  --output tmp/modelhub_tool_state_probe_20261103/responses_replay \
  --protocol responses --effort medium --followup --controls
```

The durable source summaries include all negative results. The local SQLite
archive is `docs/benchmark/adapter/runs.sqlite`, rebuildable with
`assets/v1_modelhub_tool_state_20261103/ingest.py`. Ingestion was executed and
verified: **5 runs, 15 requests, 7 tool executions**. It is a protocol archive
and does not add rows to the NR3D accuracy leaderboard.

```bash
python docs/benchmark/adapter/assets/v1_modelhub_tool_state_20261103/ingest.py \
  --db docs/benchmark/adapter/runs.sqlite
```

## Artifact verification

- All three executed probe source hashes match their archived versions.
- Archived request/response checksums and SQLite row counts were verified.
- A scan using the actual user-provided AK found no occurrences in 332 generated
  artifact, log, database, documentation, and built-site files. The AK was never
  supplied as a shell command argument, script literal, or environment variable.
- Black checks passed for the four archived Python files; Ruff passed for the
  ingestion script. These are diagnostic artifacts, not production-code changes.
- `git diff --check` passed. `mkdocs build` completed successfully, with existing
  warnings about the missing SceneFunc leaderboard and older links/anchors.
- Business test suites were not run: `src/` and production adapter code were not
  changed. The actual HTTP/tool probes are the execution validation for this record.

Before the five HTTP subruns, one launcher stopped before any request because it
looked in the earlier segment of this chat's transcript and found no matching
credential. Selecting the segment containing the user's curl resolved that
local setup issue. It exposed only the exception type and made no network call.

## Limits and relation to prior records

- No Codex SDK/CLI, local adapter conversion, SSE streaming, images, compact,
  process restart/resume, AK rotation, or Claude/Gemini signature path was tested.
- The same AK was used throughout; the test does not establish whether encrypted
  state is portable across credentials, routes, deployments, or expiration.
- All reported `cached_tokens` and cache-write counts were zero. The synthetic
  inputs were short (successful input counts were at most 556 tokens), so this
  provides no positive prompt-cache evidence. Stable `extra.session_id` is not
  proof of saved conversation history; this probe explicitly replayed history.
- The failure with Chat `medium` is scoped to this model/route/request. Chat
  `none` succeeding without opaque state does not prove Chat can preserve
  reasoning state produced under another setting.
- The [historical reasoning-path analysis](../../codex_agent/reasoning_carryover_caching_ak_routing_20260611.md)
  describes an older mapper and `gpt-5.4` observations. This record adds current
  `gpt-6-luna` direct-protocol evidence; it does not reproduce those NR3D accuracy
  comparisons or attribute performance gains to encrypted-state replay.
- [NR3D v13](../nr3d/v13_office_responses_smoke10_20261103.md) exercised a different
  model and actual SDK/adapter chain. Its integration results cannot substitute
  for testing this probe's model and tool/state sequence through the adapter.
