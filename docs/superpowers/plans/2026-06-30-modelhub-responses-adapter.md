# ModelHub Responses Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local ModelHub adapter a Responses API passthrough gateway by default, while keeping legacy Chat Completions behavior behind explicit configuration.

**Architecture:** `POST /v1/responses` builds a ModelHub `/responses` URL, injects AK/header routing metadata, and forwards the original JSON body. Chat mapping remains available only for `upstream_api=chat_completions`. `POST /v1/responses/compact` proxies to ModelHub `/responses/compact` in responses mode and uses local lossy compact only in explicit legacy chat mode.

**Tech Stack:** Python 3.10+, FastAPI, httpx, unittest, uv, black, ruff, mypy.

---

### Task 1: Lock Responses-First Routing And Body Passthrough Tests

**Files:**
- Modify: `codex_modelhub_adapter/tests/test_proxy.py`
- Modify: `codex_modelhub_adapter/tests/test_mapping.py`

- [ ] **Step 1: Update auto-route tests to expect responses-first**

In `codex_modelhub_adapter/tests/test_proxy.py`, replace the three old auto chat expectations with these tests:

```python
class ResolveUpstreamApiTest(unittest.TestCase):
    def test_chat_model_without_summary_uses_responses(self):
        body = {"model": "gpt-5.4-2026-03-05", "reasoning": {"effort": "medium"}}
        self.assertEqual(resolve_upstream_api(body, _auto_settings()), "responses")

    def test_chat_model_with_summary_uses_responses(self):
        body = {
            "model": "gpt-5.4-2026-03-05",
            "reasoning": {"effort": "medium", "summary": "auto"},
        }
        self.assertEqual(resolve_upstream_api(body, _auto_settings()), "responses")

    def test_summary_none_still_uses_responses(self):
        body = {"model": "gpt-5.4-2026-03-05", "reasoning": {"summary": "none"}}
        self.assertEqual(resolve_upstream_api(body, _auto_settings()), "responses")

    def test_non_chat_model_uses_responses_by_default(self):
        self.assertEqual(
            resolve_upstream_api({"model": "o4-mini"}, _auto_settings()), "responses"
        )

    def test_explicit_chat_config_wins_over_summary(self):
        settings = AdapterSettings(
            upstream_api="chat_completions",
            chat_completions_models=("gpt-5.4*",),
        )
        body = {"model": "gpt-5.4-2026-03-05", "reasoning": {"summary": "auto"}}
        self.assertEqual(resolve_upstream_api(body, settings), "chat_completions")
```

- [ ] **Step 2: Add a passthrough regression test**

In `codex_modelhub_adapter/tests/test_mapping.py`, replace
`test_build_upstream_request_routes_gpt54_to_chat_completions_by_default` with:

```python
    def test_build_upstream_request_routes_gpt54_to_responses_by_default(self):
        request = {
            "model": "gpt-5.4-2026-03-05",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "opaque-state",
                },
            ],
            "previous_response_id": "resp_previous",
            "max_output_tokens": 123,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                }
            },
        }

        upstream = build_upstream_request(
            request,
            settings=AdapterSettings(modelhub_ak="ak-1"),
        )

        self.assertEqual(
            upstream.url,
            "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/responses?ak=ak-1",
        )
        self.assertEqual(upstream.upstream_api, "responses")
        self.assertEqual(upstream.body, request)
```

- [ ] **Step 3: Keep explicit chat mapping covered**

In `codex_modelhub_adapter/tests/test_mapping.py`, update
`test_build_upstream_request_uses_office_chat_adapter_and_extra_header` so it still passes by explicitly setting `upstream_api="chat_completions"`.

- [ ] **Step 4: Update TOML upstream expectation**

In `codex_modelhub_adapter/tests/test_mapping.py`, change
`test_build_upstream_request_uses_toml_weighted_upstream_for_matching_model` to expect:

```python
self.assertEqual(
    upstream.url,
    "https://modelhub-a.example.test/api/modelhub/online/responses?ak=ak-from-toml",
)
self.assertEqual(upstream.upstream_api, "responses")
```

- [ ] **Step 5: Run focused tests and confirm they fail**

Run:

```bash
cd codex_modelhub_adapter
uv run python -m unittest tests.test_proxy tests.test_mapping
```

Expected before implementation: failures showing auto route still returns `chat_completions` and default body still contains normalized responses fields.

### Task 2: Implement Responses Passthrough In Proxy

**Files:**
- Modify: `codex_modelhub_adapter/adapter/proxy.py`
- Modify: `codex_modelhub_adapter/tests/test_proxy.py`
- Modify: `codex_modelhub_adapter/tests/test_mapping.py`

- [ ] **Step 1: Add explicit legacy responses mutation setting**

In `AdapterSettings`, add:

```python
responses_body_mutation_enabled: bool = False
```

In `AdapterSettings.from_env()`, populate it with:

```python
responses_body_mutation_enabled=_bool_env(
    "AIDP_CODEX_PROXY_RESPONSES_BODY_MUTATION_ENABLED",
    defaults.responses_body_mutation_enabled,
),
```

- [ ] **Step 2: Make responses body passthrough the default**

Replace the responses side of `build_upstream_request()` body construction with:

```python
body = (
    build_chat_completions_body(
        raw_body,
        max_output_tokens=resolved_settings.max_output_tokens,
        token_limit=chat_context_token_limit
        or resolved_settings.chat_context_token_limit,
        chars_per_token=resolved_settings.chat_context_chars_per_token,
    )
    if upstream_api == "chat_completions"
    else _build_responses_body(
        raw_body,
        max_output_tokens=resolved_settings.max_output_tokens,
        mutation_enabled=resolved_settings.responses_body_mutation_enabled,
    )
)
```

Add the helper near `build_upstream_request()`:

```python
def _build_responses_body(
    raw_body: Any,
    *,
    max_output_tokens: int,
    mutation_enabled: bool,
) -> JsonObject:
    if not isinstance(raw_body, dict):
        return {}
    if mutation_enabled:
        return normalize_responses_body(
            raw_body,
            max_output_tokens=max_output_tokens,
        )
    return raw_body
```

- [ ] **Step 3: Make `auto` responses-first**

Replace `resolve_upstream_api()` with:

```python
def resolve_upstream_api(raw_body: Any, settings: AdapterSettings) -> str:
    configured = settings.upstream_api
    if configured in {"responses", "chat_completions"}:
        return configured
    return "responses"
```

Leave `_requests_reasoning_summary()` in place because existing tests cover it as a request-shape helper, but route resolution no longer depends on it.

- [ ] **Step 4: Expose legacy mode in health**

Add this key to `health_payload()`:

```python
"responses_body_mutation_enabled": resolved.responses_body_mutation_enabled,
```

- [ ] **Step 5: Run focused tests and confirm pass**

Run:

```bash
cd codex_modelhub_adapter
uv run python -m unittest tests.test_proxy tests.test_mapping
```

Expected: all tests in those two files pass.

### Task 3: Proxy `/responses/compact` Upstream In Responses Mode

**Files:**
- Modify: `codex_modelhub_adapter/adapter/proxy.py`
- Modify: `codex_modelhub_adapter/adapter/app.py`
- Modify: `codex_modelhub_adapter/tests/test_app.py`

- [ ] **Step 1: Add compact request builder**

In `codex_modelhub_adapter/adapter/proxy.py`, add:

```python
def build_compact_upstream_request(
    raw_body: Any,
    *,
    settings: AdapterSettings | None = None,
    upstream_extra: dict[str, str] | None = None,
    logid: str | None = None,
    excluded_upstream_aliases: frozenset[str] = frozenset(),
) -> UpstreamRequest:
    resolved_settings = settings or AdapterSettings.from_env()
    upstream_api = resolve_upstream_api(raw_body, resolved_settings)
    if upstream_api == "chat_completions":
        raise RuntimeError("responses compact proxy requires upstream_api=responses")
    selected_upstream = resolve_modelhub_upstream(
        resolved_settings,
        raw_body,
        upstream_extra,
        excluded_upstream_aliases=excluded_upstream_aliases,
    )
    upstream_base_url = resolved_settings.upstream_base_url
    upstream_path_override = ""
    if selected_upstream is not None:
        key_alias = selected_upstream.alias
        ak = selected_upstream.ak
        key_selection = "toml_weighted_extra_hash"
        upstream_base_url, upstream_path_override = _split_modelhub_target_url(
            selected_upstream.url,
            upstream_api,
        )
    elif (
        selected_key := resolve_modelhub_key(resolved_settings, upstream_extra)
    ) is not None:
        key_alias, ak = selected_key
        key_selection = "extra_session_rendezvous_hash"
    else:
        ak = _valid_modelhub_ak(resolved_settings.modelhub_ak)
        key_alias = "single"
        key_selection = "single_key_fallback"
    if not ak:
        raise RuntimeError("AIDP_GPT_AK or AIDP_MODELHUB_AK is required")

    responses_path = upstream_path_override or _normalize_path(
        resolved_settings.responses_path
    )
    path = _responses_compact_path(responses_path)
    return UpstreamRequest(
        url=f"{upstream_base_url}{path}?ak={quote(ak, safe='')}",
        headers={
            "content-type": "application/json",
            "X-TT-LOGID": logid or _build_logid(),
            "extra": encode_upstream_extra(
                upstream_extra, fallback_session_id=resolved_settings.session_id
            ),
        },
        body=_build_responses_body(
            raw_body,
            max_output_tokens=resolved_settings.max_output_tokens,
            mutation_enabled=resolved_settings.responses_body_mutation_enabled,
        ),
        upstream_api=upstream_api,
        upstream_key_alias=key_alias,
        upstream_key_selection=key_selection,
    )
```

Add:

```python
def _responses_compact_path(responses_path: str) -> str:
    normalized = _normalize_path(responses_path)
    if normalized.endswith("/compact"):
        return normalized
    return f"{normalized}/compact"
```

- [ ] **Step 2: Change compact route**

In `codex_modelhub_adapter/adapter/app.py`, import `build_compact_upstream_request`.

Replace `compact_response()` with a route that:

1. Parses JSON.
2. Resolves settings and extra header.
3. If `settings.upstream_api == "chat_completions"`, returns existing local `build_compaction_response()`.
4. Otherwise builds `build_compact_upstream_request()`.
5. Opens upstream with `httpx.AsyncClient.stream()`.
6. Returns raw upstream bytes and status code.

Use the same response header copying helper already used by `/v1/responses`.

- [ ] **Step 3: Update app response route test**

In `test_responses_route_uses_office_chat_adapter`, rename it to
`test_responses_route_uses_office_responses_passthrough`, change fake upstream body to:

```python
body=json.dumps(
    {
        "id": "resp_test",
        "object": "response",
        "status": "completed",
        "model": "gpt-5.5-2026-04-24",
        "output_text": "2",
    }
).encode()
```

Expect the upstream URL to be `/responses?ak=ak-1` and `call["json"]` to equal the original request body.

- [ ] **Step 4: Update compact app test**

Replace `test_compact_route_returns_local_response_compaction` with a proxy test:

```python
    def test_compact_route_proxies_modelhub_responses_compact(self):
        _FakeAsyncClient.calls = []
        _FakeAsyncClient.response = _FakeUpstreamResponse(
            body=json.dumps(
                {
                    "id": "resp_compact_test",
                    "object": "response.compaction",
                    "status": "completed",
                    "output": [],
                }
            ).encode()
        )
        request_body = {
            "model": "gpt-5.5-2026-04-24",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            ],
        }

        with _env(
            {
                "AIDP_GPT_AK": "ak-1",
                "AIDP_CODEX_PROXY_UPSTREAM_ENV": "office",
            },
            remove=("MODELHUB_AK", "MODELHUB_URL"),
        ):
            with patch("adapter.app.httpx.AsyncClient", _FakeAsyncClient):
                response = TestClient(app).post(
                    "/v1/responses/compact",
                    json=request_body,
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "response.compaction")
        self.assertEqual(len(_FakeAsyncClient.calls), 1)
        call = _FakeAsyncClient.calls[0]
        self.assertEqual(
            call["url"],
            "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/responses/compact?ak=ak-1",
        )
        self.assertEqual(call["json"], request_body)
```

- [ ] **Step 5: Update 429 test fake upstream response**

In `test_responses_retries_429_with_different_toml_upstream`, change the second fake response to a Responses-shaped body:

```python
_FakeUpstreamResponse(
    body=json.dumps(
        {
            "id": "resp_test",
            "object": "response",
            "status": "completed",
            "model": "gpt-5.4-2026-03-05",
            "output_text": "ok",
        }
    ).encode()
)
```

- [ ] **Step 6: Run app tests**

Run:

```bash
cd codex_modelhub_adapter
uv run python -m unittest tests.test_app
```

Expected: all app tests pass.

### Task 4: Update Runtime Docs And Verification

**Files:**
- Modify: `codex_modelhub_adapter/README.md`
- Modify: `codex_modelhub_adapter/restart_adapter.sh`
- Modify: `codex_modelhub_adapter/docs/codex_agent_sdk_setup_guide.md`

- [ ] **Step 1: Update restart script**

In `codex_modelhub_adapter/restart_adapter.sh`, change:

```bash
export AIDP_CODEX_PROXY_UPSTREAM_API=auto
export AIDP_CODEX_PROXY_CHAT_COMPLETIONS_MODELS="gpt-5.4*,gpt-5.5*"
```

to:

```bash
export AIDP_CODEX_PROXY_UPSTREAM_API=responses
```

Remove the `AIDP_CODEX_PROXY_CHAT_COMPLETIONS_MODELS` export from the default startup path.

- [ ] **Step 2: Update README default route**

In `codex_modelhub_adapter/README.md`, replace the old default route section with:

```text
The default route is now a Responses API passthrough:

Codex SDK Responses API
  -> http://127.0.0.1:8787/v1/responses
  -> AIDP ModelHub /api/modelhub/online/responses
  -> Responses response/SSE
```

Also state that `chat_completions` is a legacy explicit fallback, not the default.

- [ ] **Step 3: Update setup guide targeted references**

In `codex_modelhub_adapter/docs/codex_agent_sdk_setup_guide.md`, update the sections that mention default `auto` routing to `/v2/crawl`, local compact as default, and restart env exports. Preserve historical notes when they describe the old implementation, but make the current default unambiguous.

- [ ] **Step 4: Run adapter unit tests**

Run:

```bash
cd codex_modelhub_adapter
uv run python -m unittest discover -s tests
```

Expected: all adapter tests pass.

- [ ] **Step 5: Run formatting and lint for adapter files**

Run:

```bash
black --check codex_modelhub_adapter
ruff check codex_modelhub_adapter
```

Expected: no formatting or lint errors.

- [ ] **Step 6: Run project quality gate**

Run from repository root:

```bash
source .venv/bin/activate
ruff check src/
black --check src/
mypy src/
PYTHONPATH=src pytest src/keyframe/tests -q
```

Expected: all commands pass, or any pre-existing environmental/tooling failure is reported with exact error text.

- [ ] **Step 7: Self-review against design**

Check:

- Default `/v1/responses` URL ends with `/responses?ak=<redacted>`.
- Default upstream body equals request body for dict JSON.
- `gpt-5.4*` no longer enters chat mapping in `auto`.
- `upstream_api=chat_completions` still explicitly enables legacy path.
- `/v1/responses/compact` proxies to `/responses/compact` unless explicit chat mode is configured.
- Health output does not expose AK.
- README and restart script no longer advertise `/v2/crawl` as the default path.
