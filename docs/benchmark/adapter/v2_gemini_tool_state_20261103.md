# v2: Gemini tool signatures, 2026-11-03

## Result and routing decision

**`gemini-3.8-flash` on the office ModelHub `/v2/crawl` endpoint completed a
five-request, three-tool chain with unchanged signatures. Corrupting the tool-call
signature or removing signatures produced HTTP 400.** This direct protocol test
does not validate a Codex SDK or local adapter integration.

The user confirmed the routing policy: **GPT must use native Responses; Chat and
automatic protocol downgrade are prohibited for GPT.** The current production
adapter is already Responses-only. A future Gemini integration needs its own
signature-preserving codec; this test does not enable a Chat fallback. The
[v1 Chat probes](v1_modelhub_tool_state_20261103.md) remain historical evidence.
This user decision is now persisted in `AGENTS.md` under runtime prerequisites;
`CLAUDE.md` remains a relative symlink to that file.

| Subrun | Main requests + controls | Executed tools | Recorded result |
| --- | ---: | ---: | --- |
| [crawl](assets/v2_gemini_tool_state_20261103/crawl/summary.json) | 3 + 2 | 2 | `answer_or_tool_order_mismatch`: strict JSON parser rejected a fenced JSON answer; enclosed values were correct and tool order passed. Original result preserved. |
| [crawl_replay](assets/v2_gemini_tool_state_20261103/crawl_replay/summary.json) | 5 + 4 | 3 | `passed`; original context marker, sum and tool receipts survived the follow-up; replay hashes match. |
| [crawl_image](assets/v2_gemini_tool_state_20261103/crawl_image/summary.json) | 3 + 0 | 2 | `passed` with an `image_url` input; transport acceptance and tool continuation tested, not visual understanding. |

Totals independently checked: **17 HTTP requests, seven local tool executions;
13 HTTP 200 and four HTTP 400**. There is no dataset fold, judge, or task-accuracy
leaderboard. No negative result was discarded or changed into a pass.
Successful responses reported aggregate `prompt_tokens=7307`,
`completion_tokens=517`, `total_tokens=8882`, and `cached_tokens=0`.
These are upstream usage fields; missing usage on failures is not measured zero.

## Provenance and configuration

Execution used a clean **detached** worktree at
`255b091849c0bafeef8f5e494d022a7750eefa91`; the branch field is empty and every
summary records `tracked_drift=false`. The main checkout contained v1 archival
edits, so it was not the execution checkout. The worktree was
`tmp/gemini_state_probe_worktree_20261103/`; probes lived under its
`tmp/gemini_tool_state_probe_20261103/` directory. Executed untracked scripts are
identified separately from HEAD:

| Archived script | SHA256 | Subruns |
| --- | --- | --- |
| [probe_v1.py](assets/v2_gemini_tool_state_20261103/probe_v1.py) | `a92d1db9d9ba396ccbc7522cec842951df238f544c239f45d3e20de231adac2d` | `crawl` |
| [probe.py](assets/v2_gemini_tool_state_20261103/probe.py) | `db454076251200a577b615f657fdde4e99e1aa2390ec991aa2113f63afd51e4d` | `crawl_replay`, `crawl_image` |

All requested and returned model IDs were `gemini-3.8-flash`; effort was omitted
(provider default), `max_tokens=4096`, `stream=false`, and redirects disabled.
The office endpoint was exactly:

```text
https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl
```

Start times were 20:07:12 (`crawl`) and 20:08:56 (`crawl_replay`, `crawl_image`),
Asia/Shanghai, 2026-11-03. Session IDs were respectively
`gemini_tool_state_20261103_30b3c01a3c`,
`gemini_tool_state_20261103_62c046f0d2`, and
`gemini_tool_state_20261103_73d41d533d`. Each subrun kept its own session ID fixed
and used the same user-supplied AK throughout, including controls. The AK was
read only from the current chat into memory; neither AK nor raw signatures were
written to logs or archived artifacts.

## State and control evidence

The probe copied whole assistant messages into history before appending matching
tool results. It requested sequential `alpha`, `beta` lookups (137 + 211 = 348),
then verified a random context marker and tool-generated receipts. The replay
variant requested another `alpha` lookup and the original answer. All three
subruns report unchanged output replay hashes. The second script accepts valid
JSON enclosed by one complete Markdown fence; it does not repair JSON content.

Signatures appeared at `choices[0].message.signature`,
`choices[0].message.tool_calls[].signature`, and
`choices[0].message.multimodal_contents[].signature`. The last form occurred on
text answers too; its name alone is not evidence of image reasoning. A converter
that preserves only `content` and standard tool-call fields loses these fields.
All 13 successful response messages had **empty `reasoning_content`**, while
11 reported nonzero `usage.reasoning_tokens` (sum 1,058); two omitted that usage
field. An empty readable reasoning string therefore does not mean no reasoning
or no opaque state. Missing usage fields are not measured zeroes.

The refined controls independently branched from **request 2** of `crawl_replay`,
which carried the first tool call and its result. The original signature was
804 characters, SHA256
`fe3654fd9ee7be57754a3e3e4c0c5663de6ba50f69074259a3107175900ed1b5`.

| Control | Result | Bounded interpretation |
| --- | --- | --- |
| Corrupt message-level copy only | 200 | The unchanged tool-call copy remained available; this does not establish that signatures are optional. |
| Corrupt tool-call copy only | 400, `Invalid thought signature`, code `-4003` | Message-level copy alone did not rescue the damaged tool-call state. |
| Corrupt all copies | 400, same error/code | Corrupted state was rejected. |
| Remove all copies | 400, code `-4003` | Gateway explicitly required the preceding response's signature fields to be returned unchanged. |

The initial `crawl` controls had changed only the first message-level copy
(200) and removed all copies (400). Those results motivated the more precise
controls; both versions remain archived. Successful control responses that
requested another tool were not executed, so they do not add to the seven-tool
count. These are deliberate diagnostic mutations, never production recovery.

## Reproduction and archive

Sanitized requests/responses, per-field lengths/hashes, tools and summaries live
under `assets/v2_gemini_tool_state_20261103/`. Raw signatures are replaced with
length/hash objects; these files are evidence, not replayable opaque state.
The protocol SQLite archive is `docs/benchmark/adapter/runs.sqlite`; the version's
`assets/v2_gemini_tool_state_20261103/ingest.py` rebuilds its rows from these assets.
Ingestion verified **3 Gemini runs / 17 requests / 7 tools**; the combined v1+v2
database contains **8 runs / 32 requests / 14 tools**. Archived script hashes
match the executed versions, and the credential scan passed.

The exact argument sets were:

| Original output suffix | Script version | Options after transcript/output arguments |
| --- | --- | --- |
| `crawl` | `probe_v1.py` copied as `probe.py` | `--protocol legacy_chat --followup --controls` |
| `crawl_replay` | `probe.py` | `--protocol legacy_chat --followup --controls` |
| `crawl_image` | `probe.py` | `--protocol legacy_chat --image` |

Run in tmux in `agentic3d` with a clean execution worktree. Copy the selected
archived script to `<worktree>/tmp/gemini_tool_state_probe_20261103/probe.py`
before execution: its `Path(__file__).parents[2]` must resolve to that worktree.
From that worktree, the replay invocation was:

```bash
python tmp/gemini_tool_state_probe_20261103/probe.py \
  --credential-transcript '<local-current-chat-transcript>' \
  --output tmp/gemini_tool_state_probe_20261103/crawl_replay \
  --protocol legacy_chat --followup --controls
```

Use fresh output names when reproducing; do not overwrite these historical
records. The transcript path is local and is not archived; no AK command-line
argument is supported.

## Limits

No Codex SDK/adapter bridge, SSE, compact, restart/resume, credential switching,
long-term cache, Claude, or Gemini native API was tested. Every reported cache
count was zero. Fixed session IDs do not replace explicit history replay.
The image run reported image input tokens, but its answer test did not depend on
image content. These observations support preserving all Gemini signature copies
and their message/tool association; they do not prove a particular Responses
envelope codec works or explain model-performance differences.
