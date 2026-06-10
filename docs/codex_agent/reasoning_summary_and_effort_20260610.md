# Reasoning summary + reasoning effort for the Codex agent

Status: **implemented + verified end-to-end** on 2026-06-10.

## Goal

Two related changes to how the Codex agent's *reasoning* is controlled and
surfaced:

1. Pass **reasoning effort** and **reasoning summary** as explicit per-turn SDK
   arguments (`thread.run/turn(effort=, summary=)`) instead of injecting a
   `model_reasoning_effort=...` string into `config.toml` overrides.
2. **Capture the reasoning summary** the model emits and surface it through the
   turn metadata into each NR3D sample's output, for tracing/debugging.
3. Make **`medium`** the default reasoning effort (reproducible), still
   overridable via CLI / env.

## Key empirical finding: summaries only exist on the Responses API

A reasoning *summary* is produced by a separate server-side summarizer over the
model's hidden chain-of-thought; it is a **Responses API** feature. The two
upstream paths the ModelHub adapter can use behave very differently. Verified
with direct upstream probes (`tmp/probe_upstream_reasoning.py`,
`tmp/probe_upstream_responses.py`; AK never printed):

| Upstream (gpt-5.4-2026-03-05) | Request | Reasoning returned |
| --- | --- | --- |
| chat `/v2/crawl` (the default route for `gpt-5.4*`) | `reasoning_effort=high` | **none** — delta keys are only `content` / `refusal` / `role` |
| `/responses` | `reasoning={effort, summary:"auto"}` | **full summary** — 82× `response.reasoning_summary_text.delta`, readable summary text |

Implication: editing the adapter's chat-path mapping to "forward
`reasoning.summary`" or "stop dropping reasoning items" would be **dead code** —
the chat upstream returns no reasoning content to map. The only working source of
summaries is the Responses API.

The SDK models corroborate this: a `ReasoningResponseItem` carries
`encrypted_content` (the opaque raw CoT) and `summary` (the human-readable
artifact) as **separate** fields — the summary is not the raw thoughts.

## Changes

### A. `agentic-3d-task` repo (git-tracked)

- `config.py`
  - new `reasoning_summary` field (`"" | auto | concise | detailed | none`),
    validated, read from `CODEX_AGENT_REASONING_SUMMARY`.
  - `DEFAULT_REASONING_EFFORT = "medium"`; `reasoning_effort` now defaults to it,
    and `from_env` falls back to it when the env var is unset. `""` still means
    "follow the model's own default".
- `runtime.py`
  - removed the `model_reasoning_effort=...` config-override injection.
  - `_effort()` / `_summary()` convert the config literals to the SDK
    `ReasoningEffort` / `ReasoningSummary` types and are passed to
    `thread.run` / `thread.turn`. `""` → `None` (no override).
  - `_reasoning_summary_from_items()` joins `ReasoningThreadItem.summary`
    fragments; captured on both the fast path (`TurnResult.items`) and the
    guarded streaming path (`_GuardedTurnResult.reasoning_summary`), then written
    to `CodexTurnMetadata.reasoning_summary`.
- `models.py` — `CodexTurnMetadata.reasoning_summary` (+ `as_dict`).
- `cli/run_nr3d.py` — `--reasoning-summary` flag; `--reasoning-effort` default is
  now `medium`; resolver simplified (CLI flag > env/config; dead
  `_DEFAULT_TOOLS_REASONING_EFFORT` removed).
- `evaluation/nr3d_runner.py` — `Nr3dSampleResult.reasoning_summary` persisted in
  each per-sample JSON.
- tests updated/added in `tests/test_runtime.py`, `tests/test_runtime_guard.py`,
  `tests/test_run_nr3d_cli.py`.

Override precedence for effort: `--reasoning-effort` > `CODEX_AGENT_REASONING_EFFORT`
> default `medium`.

### B. ModelHub adapter (`/Users/bytedance/aispace/codex_modelhub_adapter`, NOT git-tracked)

Because the chat upstream cannot produce summaries, the adapter now **routes a
turn to `/responses` whenever it requests a reasoning summary**, keeping every
other turn on the chat default (prefix cache + context trimming live there).

- `adapter/proxy.py` `resolve_upstream_api`: in `auto` mode, if the request body
  has `reasoning.summary ∈ {auto, concise, detailed}` → return `responses`.
  New helper `_requests_reasoning_summary`. `mapping.py` was intentionally **not**
  changed (the chat path has no reasoning to map).
- Backup: `adapter/proxy.py.bak.20260610_175952`. Started via new
  `start_adapter.sh` in tmux session `adapter` (logs `/tmp/adapter.log`).

## Verification

- **10-case** NR3D, tools mode, `--reasoning-summary auto`: 10/10 completed,
  Acc@0.25 = 0.80 (no regression vs prior smokes), **9/10** carried a non-empty
  `reasoning_summary` (466–4286 chars); the 10th was an empty summary on a
  completed/correct turn (summaries are best-effort, occasionally empty).
- **Prefix caching preserved** over the `/responses` route: `mean_cache_ratio`
  ≈ 0.97 on the tools run.
- **Regression**: a run *without* `--reasoning-summary` stays on the chat path
  and yields `reasoning_summary = null`.
- **Effort default**: rollout `turn_context` now shows
  `"reasoning_effort":"medium"` (was `null`).
- Gate: `ruff` / `black` / `mypy` clean; `pytest src/codex_agent/tests` → 167
  passed.

## How to use

```bash
# Capture reasoning summaries (routes these turns to /responses):
python -m codex_agent.cli.run_nr3d ... --tools --reasoning-summary auto
# Override effort (else medium):
python -m codex_agent.cli.run_nr3d ... --reasoning-effort high
# or: export CODEX_AGENT_REASONING_EFFORT=high / CODEX_AGENT_REASONING_SUMMARY=auto
```

The summary lands in `CodexTurnMetadata.reasoning_summary` and in each
`tmp/.../per_sample/*.json` under `reasoning_summary`. It is a lossy,
human-readable summary (good for trace viewers), **not** a verbatim
chain-of-thought.

## Caveats

- Summaries depend on the `/responses` upstream; if the adapter is reconfigured
  to force `upstream_api=chat_completions`, summaries silently disappear (no
  fallback — the field is simply `null`).
- The adapter repo is not under git; the only record of its change is this doc +
  the `*.bak` backup beside the edited file.
