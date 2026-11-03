# Codex multi-protocol adapter

The active implementation is `src/codex_adapter/`, packaged with the root
project. It replaces the old `codex_modelhub_adapter/` implementation; archived
benchmark scripts describing the old service remain historical evidence, not
current launch instructions.

## Responsibilities and supported protocols

```text
Task CLI -> CodexAgentRuntime -> openai_codex -> Codex app-server
                                                    |
                                             Responses HTTP/SSE
                                                    |
                                               codex_adapter
                                            /                 \
                                 native Responses         Gemini Chat codec
                                       |                        |
                                    ModelHub                 ModelHub
```

The adapter does not execute tools, read local image paths, judge OpenEQA
answers, or call SceneFunc3D sidecars. Codex still executes tools. Query parsing,
judging, and vision sidecars retain their independent configurations.

| Protocol | Request/response behavior | Streaming | Compact |
| --- | --- | --- | --- |
| `responses` | Native body and opaque reasoning are forwarded without Chat conversion | Native raw SSE | Native same-binding proxy |
| `gemini_chat` | Explicit Responses-to-Chat codec; complete native assistant messages persist | Upstream buffered; Responses SSE emitted after durable commit | Explicitly unsupported |

GPT-family models must use `responses`; there is no Chat fallback, encrypted
state deletion, automatic account failover, or wildcard model route. Future
providers need their own codec and capability contract; signature formats are
not interchangeable.

## Installation and configuration

Use the existing environment. On Linux activate `agentic3d`; do not recreate
virtual environments or change GPU services. On macOS install the root adapter
extra with uv. Run installations and services inside tmux.

```bash
uv sync --extra adapter --extra codex --extra dev
```

Copy `configs/codex_adapter.example.toml` to the ignored
`configs/codex_adapter.toml`. Each model contains `[[models.upstreams]]` entries,
each with its own `url`, direct `ak`, and `weight`. Real AKs go only in this
ignored private TOML; the tracked example contains placeholders. Protect the file
with owner-only permissions. There is no model-AK environment lookup, `ak_env`,
`secret_env`, implicit `.env` loading or legacy key fallback. Optional local
adapter authentication remains separate from upstream AK configuration.

The sample registers `gpt-5.4-2026-03-05` and `gemini-3.8-flash`. The model string sent by
Codex must exactly match a registered ID. It is not rewritten to a hidden alias.
A request for any other model returns `model_not_registered` before an upstream
call.

The state directory, database, journal, and backups store plaintext conversation
turns and responses locally (while upstream AKs are hashed and never written to
SQLite); protect the state directory with local owner-only permissions.

Validate configuration without contacting a model:

```bash
python -m codex_adapter check --config configs/codex_adapter.toml
```

Explicitly initialize a new state database once:

```bash
python -m codex_adapter init-state --config configs/codex_adapter.toml
```

Start the configured service in a dedicated tmux session:

```bash
python -m codex_adapter serve --config configs/codex_adapter.toml
```

The service does not automatically create missing state during normal startup.
Do not use multiple workers or `--reload`. A process-level database lock prevents
independent workers from silently maintaining divergent journals. Do not stop
unrelated services if the configured port is occupied.

Inspect safe registered capability descriptions:

```bash
python -m codex_adapter models --config configs/codex_adapter.toml
```

## Model registration and credential lifecycle

Each model specifies its exact ID, protocol, capabilities, and an upstream list;
there is no model-level URL. Each `[[models.upstreams]]` entry contains its own
stable `id`, `version`, full `url`, direct `ak`, nonnegative integer `weight`,
and `active`, `draining`, or `revoked` status. Two entries may share a URL but use
different AKs, or share an AK but use different URLs; they remain distinct routes.

- Only a model's own positive-weight active upstreams receive new sessions.
- Weights describe probabilities across new sessions, not exact short-run
  percentages, per-request balancing, token load, latency, or model selection.
- The first choice is committed atomically to SQLite. All subsequent model
  requests in that session reuse its AK, endpoint, model, and codec identity.
- Reordering a list, changing weights, or adding another AK does not move an
  existing session. Zero weight or `draining` stops new allocations but permits
  existing bindings to continue.
- Revocation, removal, changing a secret under its existing identity, or changing
  the bound endpoint/protocol/codec invalidates the old session explicitly. It
  never silently moves to another credential.
- Rotate upstream secrets by adding a new credential version. Retain the old
  version while its sessions drain, or revoke it knowingly and accept failures.
- Model switching and history branching require a new session; opaque state
  cannot be carried from one provider/model into another.

The service does not reload a private configuration on every request. Restart
with the same database and required state keys to apply configuration changes;
existing bindings are checked against the new registry.

### Migrating configuration schema 1

Schema 1 is rejected explicitly. Move each old credential into
`[[models.upstreams]]`, copy the old model endpoint to that entry's `url`, replace
its environment reference with the actual `ak`, and remove model-level
`endpoint`/`credentials`. Preserve every route's old ID, version, URL and AK to
continue existing sessions. This does not require rebuilding SQLite: its
`endpoint` and internal `credential_id/version/fingerprint` columns already
represent the selected upstream. Neither state claims nor cryptographic identity
are rewritten merely for this configuration change.

## Codex provider and session identity

The active provider is `codex_adapter`, with `wire_api="responses"`. The runtime
creates a unique `X-Codex-Session-ID` for each actual SDK thread and injects it
through the provider's environment-header mechanism. Tools, finalization,
retries and native GPT compact retain that same ID. Independent executions get
different IDs, including when prefix-cache tracing is disabled.

The old global `codex_agent` cache label is not an account-routing identity.
ModelHub receives an adapter-owned `extra` JSON header whose `session_id` comes
from the durable binding. Caller-provided AKs, endpoint overrides and mismatched
session metadata are rejected, not forwarded.

Initialize a fresh Codex home with the current bootstrap script:

```bash
scripts/codex_agent/bootstrap_codex_home.sh --model gpt-5.4-2026-03-05
```

The bootstrap must not overwrite an existing configuration silently. Back up
existing settings and migrate the provider deliberately; do not reuse an old
session while claiming its new binding is identical. An explicitly configured
external provider remains separate from this local adapter path.

When adapter authentication is enabled, `CODEX_ADAPTER_TOKEN` contains the local
adapter token. The runtime constructs the Bearer header; this token is not a
ModelHub AK. Non-loopback listeners require authentication. A bare session ID is
an affinity/capability identifier, not a tenant login: first-version deployments
are a trusted local service or one authenticated principal, not a multi-tenant
public gateway.

## Gemini capabilities and state preservation

The codec saves the complete native assistant message, including message-level,
function-call and multimodal-part signatures and unknown native extensions.
Codex sees a versioned authenticated state reference in a reasoning item's
`encrypted_content`; it is not an OpenAI-generated encrypted reasoning block.
Every model response, including tool continuations within one SDK turn, gets its
own state reference. On the next request, the adapter loads all saved turns in
sequence and restores every original native assistant message and its signatures,
not only the message named by the latest reference. It validates each projection
and the complete ordered prefix, skipping the visible assistant items so they
are not duplicated. The reference alone cannot recover native messages without
the matching SQLite state.

The protected prefix includes instructions, system/user history, tool definitions
and previous tool results, not only carrier references. Current-round generation
settings such as output budget, temperature, effort, and output format can change
without modifying the restored prefix; tool definitions remain fixed because they
define the reversible native function mapping. History mutation,
missing/reordered state, cross-session references and ambiguous tool IDs fail
explicitly. A single empty native tool-call ID has a reversible nonempty Codex
ID. Ambiguous repeated or multiple empty native IDs are not guessed.

The codec maps function tools, supported namespaces, tool choice, JSON Schema
formats, and supported text/image inputs explicitly. Unsupported fields, media,
custom grammar tools or reasoning options are rejected, never quietly dropped.
A strict JSON Schema request is not replaced by a prompt saying “return JSON”.

Image-capable configurations also preserve Codex `view_image` results as
multimodal content in the correlated Chat tool message; they are not dropped or
invented as user prose. Upstream acceptance of that tool-result representation
must be checked separately from ordinary user image input. The pinned SDK's
native tools require `parallel_tools=true`; declaring it false fails preflight
rather than silently changing the requested parallel semantics.

The supplied Gemini profile explicitly enables `structured_output=true` and
`images=true`, allowing the codec to translate strict-schema requests and image
inputs. Models without explicit capability declarations retain conservative
defaults. Declaring a capability is configuration, not evidence that the real
upstream accepts the complete strict-schema/task-image pipeline; local mock and
SDK tests do not replace that acceptance check. Current task runners always
request structured output, so a Gemini profile that disables it still fails
preflight. Reasoning effort must be a declared supported value or explicitly use
the provider default; the adapter does not silently swallow an unsupported
`medium` request.

Gemini automatic compact and implicit summaries must remain disabled in the
pinned Codex SDK/core integration. Compact requests return an unsupported error.
A finite context limit or merely a very large threshold is not proof that
compaction is disabled; context exhaustion is a failure, not permission to
throw away signatures. `/health` and mock tests do not certify a new SDK release
as compatible. Re-run the local SDK contract suite when upgrading.

## Persistence, retry and failure semantics

Configuration uses schema version 2; public health/capability documents still
use their existing version-1 contract. The persistent database retains internal
schema version 2 and carriers use
`codex_adapter.turn.v1.<binding_id>.<sequence>`. Pre-release prototype
databases/carriers are explicitly rejected, not automatically migrated or
replaced.

Original native messages and delivery data are stored as validated plaintext
UTF-8 JSON in SQLite (no runtime encryption/decryption overhead), while upstream
AKs are never persisted. Corrupting the database or removing required turn rows
fails closed rather than creating a new empty session silently.

For Gemini, the service reserves the request, validates history, obtains the
complete upstream result, and commits state and the stable Responses result
before delivering JSON/SSE. The same already-committed logical request can replay
that delivery without another model request. `Idempotency-Key` reuse with a
different body fails. A different concurrent request on the same session fails
rather than creating a fork.

Delivery replay is **not** an upstream prefix-cache hit. Its response header
`X-Codex-Adapter-Replay` and metrics distinguish it from generation. The adapter
does not cache arbitrary answers across sessions.

A timeout or crash after an upstream might have accepted a generation has an
uncertain outcome. The request is quarantined as indeterminate, not blindly
retried, moved to another AK, or reported as exactly-once. Native GPT streams are
not transformed into a locally replayable response cache. Disconnection may
leave a partially delivered stream; generation and tool side effects are not
guaranteed exactly-once.

Expired sessions return an explicit expiry error. Their identity tombstones
remain: the same ID must not silently acquire a new AK. `max_sessions` includes
these identities, while turn/payload quotas apply per session. Capacity failure
requires deliberate operator investigation/retention planning, not silent
history eviction. Back up and restore state with the matching keys; do not
manually delete a live database to clear an error. Authentication detects modified
records and missing pending journals while their authenticated evidence remains.
Without an external trusted monotonic anchor, one SQLite database cannot guarantee
detection of coordinated removal of every related record or rollback of the
entire database to a valid older snapshot; that is outside the first-version
threat model.

## Cache observability and evidence

The adapter preserves stable affinity and complete native history, but ModelHub
account affinity switches, TTL, backend capacity and cache eviction remain
outside its control. Request ordering and stable prefixes improve eligibility;
no client can force every round to hit.

Use actual native usage to distinguish input, cached input and reasoning counts.
Missing fields are unknown, not measured zeros. Gemini accounting follows the
versioned ModelHub contract observed in the probes, not an assumed formula for
all Chat APIs. The adapter neither logs hidden reasoning nor treats signatures
as visible prompt text.

For Gemini, Responses `metadata`, `client_metadata`, and `prompt_cache_key` are
validated adapter-local fields, not promised provider-side persistence or cache-key
controls. They are not copied into undocumented Chat fields or echoed as a
provider storage feature. ModelHub affinity is controlled by the bound
`extra.session_id`; a supplied Responses cache key does not override it.

A successful local test proves wire/state behavior, not real upstream caching or
strict-schema support. Paid upstream acceptance tests require separate
permission, a clean executed-code tree, recorded code/script hashes and a new
record under `docs/benchmark/adapter/`; retain negative and anomalous results.
The archived GPT/Gemini probes explain why state fidelity, server acceptance,
cache reads and task accuracy must be reported separately.

## Verification and security boundaries

Run adapter tests explicitly, including local HTTP/SQLite and pinned SDK tests:

```bash
python -m pytest tests/codex_adapter/ --cov=codex_adapter --cov-branch --cov-fail-under=80
```

Run the source regression suite as well:

```bash
python -m pytest src/ tests/codex_adapter/ -q
```

Use Black, Ruff and mypy; no production credentials or model access are required
by the local suites. The API validates external configuration and requests,
bounds body/response/concurrency/rate/state limits, and does not follow upstream
redirects. Logs omit prompts, tool content, state references, signatures, AKs and
secret-bearing URLs. Readiness checks local state and registry only; it is never
an upstream authentication or model-quality probe.
