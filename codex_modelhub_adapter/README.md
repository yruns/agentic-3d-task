# Codex ModelHub Adapter

Project-local Codex SDK adapter for the internal AIDP ModelHub endpoint.

This keeps Codex configuration inside this folder and does not touch the
user-level `~/.codex/config.toml`.

Codex ignores provider settings such as `model_provider` and `model_providers`
from project-scoped `.codex/config.toml` files. This project uses a local
`.codex-home/config.toml` and runs Codex with:

```bash
CODEX_HOME=$(pwd)/.codex-home
```

## What It Does

The adapter exposes local OpenAI-compatible endpoints:

```text
POST /v1/responses
POST /v1/responses/compact
GET  /health
```

The default route is a Responses API passthrough:

```text
Codex SDK Responses API
  -> http://127.0.0.1:8787/v1/responses
  -> AIDP ModelHub platform-selected endpoint /api/modelhub/online/responses
  -> Responses response/SSE
```

The default path keeps the request body unchanged and only handles gateway
concerns:

- office/online base URL switching
- Responses request body passthrough
- Responses response/SSE passthrough
- upstream `/responses/compact` proxying
- optional TOML upstream pool with `url`, `model_name`, `ak`, and `weight`
- optional sticky AK pool selected by `extra.session_id`
- 429 retry/failover
- invalid encrypted state retry after removing opaque state
- legacy Chat Completions conversion only when `upstream_api=chat_completions`

## Install

Use `uv` to create the local Python 3.12 environment:

```bash
cd codex_modelhub_adapter
uv sync --python 3.12
```

## Configure

```bash
cp .env.example .env
```

For a single upstream AK, set:

```bash
AIDP_GPT_AK=replace-with-modelhub-ak
```

For multiple weighted upstream targets, create a private TOML file:

```bash
cp .modelhub_upstreams.example.toml .modelhub_upstreams.toml
```

Then set real AK values in `.modelhub_upstreams.toml`:

```toml
[[upstreams]]
alias = "gpt54_a"
url = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online"
model_name = "gpt-5.4-2026-03-05"
ak = "replace-with-modelhub-ak-1"
weight = 5

[[upstreams]]
alias = "gpt54_b"
url = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online"
model_name = "gpt-5.4-2026-03-05"
ak = "replace-with-modelhub-ak-2"
weight = 1

[[upstreams]]
alias = "gpt54_c"
url = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online"
model_name = "gpt-5.4-2026-03-05"
ak = "replace-with-modelhub-ak-3"
weight = 5
```

Enable the TOML pool explicitly:

```bash
AIDP_MODELHUB_UPSTREAMS_TOML=.modelhub_upstreams.toml
```

If `AIDP_MODELHUB_UPSTREAMS_TOML` is not set, the adapter ignores the TOML pool
and falls back to legacy single-AK / JSON AK-pool environment variables.

When `AIDP_CODEX_PROXY_UPSTREAM_ENV` is unset, the adapter chooses the default
network from the host platform:

- Linux: `online`, resolving to
  `https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online`
- macOS: `office`, resolving to
  `https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online`

Set `AIDP_CODEX_PROXY_UPSTREAM_ENV=office` or
`AIDP_CODEX_PROXY_UPSTREAM_ENV=online` only when you need to override that
platform default.

The checked-in Codex config is:

```toml
model = "gpt-5.4-2026-03-05"
model_provider = "modelhub_adapter"

[model_providers.modelhub_adapter]
name = "ModelHub local adapter"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"
```

## Run

Start the adapter:

```bash
set -a
source .env
set +a
uv run uvicorn adapter.app:app --host 127.0.0.1 --port 8787
```

Health check:

```bash
curl http://127.0.0.1:8787/health
```

Smoke test the local Responses endpoint:

```bash
curl --request POST 'http://127.0.0.1:8787/v1/responses' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "gpt-5.4-2026-03-05",
    "input": "What is the result of 1+1?",
    "stream": false
  }'
```

Run Codex SDK in another terminal:

```bash
cd codex_modelhub_adapter
CODEX_HOME=$(pwd)/.codex-home uv run python examples/run_codex_sdk.py
```

Override the prompt:

```bash
CODEX_HOME=$(pwd)/.codex-home \
CODEX_PROMPT="Explain this repository in three bullets." \
uv run python examples/run_codex_sdk.py
```

## TOML Upstream Pool

`[[upstreams]]` entries use:

- `url`: ModelHub base URL, or a legacy full URL ending in `/v2/crawl` or `/responses`
- `model_name`: exact model name or a wildcard pattern such as `gpt-5.4*`
- `ak`: upstream ModelHub AK
- `weight`: positive integer traffic weight

The adapter first filters entries by request `model`, then selects one target by
`extra.session_id` with deterministic weighted hashing. The same session stays
sticky, while weights control distribution across sessions.

When a TOML file is configured and no entry matches the requested model, the
request fails closed instead of falling back to another AK.

`/health` reports target count, models, and URLs, but never returns AK values.

## Legacy AK Pool

To spread requests across multiple AKs while keeping a session sticky:

```bash
AIDP_MODELHUB_AK_POOL='[
  {"alias":"k1","ak":"replace-ak-1"},
  {"alias":"k2","ak":"replace-ak-2"}
]'
```

The adapter uses `extra.session_id` with rendezvous hashing. If no valid pool is
configured, it falls back to `AIDP_GPT_AK`.

## Compatibility Notes

The default path expects ModelHub's Responses API. The adapter can still make
Codex SDK talk to an AIDP Chat Completions-style model when explicitly launched
with `AIDP_CODEX_PROXY_UPSTREAM_API=chat_completions`, but that legacy mode
converts request/response shapes and does not preserve Responses reasoning state
with the same fidelity.

Legacy variables still work:

```bash
MODELHUB_AK=replace-with-modelhub-ak
MODELHUB_URL=https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/responses
```

## Test

```bash
uv run python -m unittest discover -s tests
```
