# Reasoning carryover vs prompt caching vs AK routing — how the `/responses` +5 pp actually works

Status: **analysis / mechanism log**, 2026-06-11.

This is the mechanism writeup behind the NR3D
[v5](../benchmark/nr3d/v5_summary_responses_strat600_20260611.md) /
[v6](../benchmark/nr3d/v6_effort_ablation_chat_strat600_20260611.md) result. The
v6 ablation decomposed v5's **+7.00 pp** over v4 into **≈+2 pp from reasoning
effort** (`""`→`medium`, on the chat path) and **≈+5 pp from the `/responses`
path itself** (at fixed effort). This doc explains *why the path matters*, and
clears up three things that are easy to get wrong:

1. how reasoning is (not) carried across turns on each upstream path,
2. whether dropping reasoning breaks **prompt caching** (no — but it has a subtler
   cost), and
3. when the weighted **AK pool rotates** (only across chains, not within one).

It builds on three earlier notes and quantifies a mechanism the first already
hypothesised:

- [`skill_loop_and_reasoning_dropped_20260609.md`](skill_loop_and_reasoning_dropped_20260609.md)
  §3 first observed "the Chat Completions path must strip `encrypted_content`, so
  per-step reasoning has no continuity across tool turns."
- [`reasoning_summary_and_effort_20260610.md`](reasoning_summary_and_effort_20260610.md)
  — summaries (and reasoning content) only exist on the Responses API.
- [`prompt_caching_20260608.md`](prompt_caching_20260608.md) — the prompt-cache
  setup (stable `session_id`, AK by `chat_run_id`).

> **Terminology (important).** In the runtime, a **"turn"** = one `run_turn` call
> = one whole agent chain = one NR3D sample. A single turn contains *many* LLM
> HTTP round-trips (the inspect→rank→decide tool loop). When the older
> `prompt_caching` doc says "AK selected by **per-turn** `chat_run_id`", "turn"
> means `run_turn` (the chain), **not** per-LLM-call. This distinction is the
> whole answer to §3 below.

## 1. Two cross-turn reasoning mechanisms

The Codex SDK always builds the conversation in **Responses** shape: its `input`
array interleaves `message` / `function_call` / `function_call_output` items **and**
`type:"reasoning"` items (the model's prior chain-of-thought, carried as an opaque
`encrypted_content` blob). What happens to those reasoning items depends on which
upstream the adapter routes to.

### Chat path (`/v2/crawl`): reasoning is dropped — structurally

When the adapter converts the Responses `input` to chat `messages`, the per-item
loop only recognises three shapes — `function_call_output`, `function_call`, and
items with a `role` in `{user, assistant, system, developer}`:

```734:777:codex_modelhub_adapter/adapter/mapping.py
    for item in input_value:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call_output":
            ...
        if item_type == "function_call":
            ...
        role = item.get("role")
        if role in {"user", "assistant", "system", "developer"}:
            flush_pending_tools()
            messages.append(...)
    flush_pending_tools()
    return messages
```

A `reasoning` item is **neither** a `function_call*` **nor** role-bearing, so it
falls straight through the loop and is silently dropped. This is not a bug — the
Chat Completions protocol's `messages` has roles system/user/assistant/tool and
**no slot for reasoning**. The reverse direction matches: the chat upstream's
response returns only the final assistant message + tool calls and **no reasoning
content** (verified in `reasoning_summary_and_effort_20260610.md`). So on the chat
path, each step's private reasoning is produced, used once, and discarded — the
next step re-derives everything from the visible transcript.

### `/responses` path: reasoning is carried across turns

The Responses API treats reasoning as a first-class item with two carry channels:

1. **Encrypted reasoning replay.** The model emits a `reasoning` item containing
   an `encrypted_content` blob (server-encrypted CoT; the client can't read it but
   can replay it). The SDK puts those reasoning items back into the next turn's
   `input`, interleaved with the `function_call` / `function_call_output` items;
   the upstream decrypts them and the model **resumes its prior chain-of-thought**.
2. **`store` + `previous_response_id`.** The adapter defaults `/responses` to
   `store=true` (`normalize_responses_body(ensure_store=True)`; the chat path uses
   `ensure_store=False`), so the server can also restore prior reasoning by id.

Neither channel exists in the chat protocol. **This is the +5 pp lever**: for a
multi-turn evidence-seeking agent (inspect frames → rank proposals → decide),
keeping the chain-of-thought across tool calls is what moves accuracy — most on
View-Dep (the path delta there is **+8.53 pp** vs effort's +4.26).

## 2. The "drop empty reasoning item" is a *fallback*, not the chat path

A separate place also drops reasoning — `sanitize_encrypted_state` — but it is a
**defensive retry path**, fired only when the `/responses` upstream rejects an
encrypted blob:

```223:226:codex_modelhub_adapter/adapter/app.py
    elif settings.encrypted_state_fallback_enabled and is_invalid_encrypted_content(error_bytes):
        if isinstance(raw_body, dict):
            sanitized_body, stats = sanitize_encrypted_state(raw_body)
```

It strips `encrypted_content` + `previous_response_id`; a reasoning item stripped
of its blob is then an empty husk (only `id/type/status`), so it is dropped to
avoid sending something useless:

```1217:1227:codex_modelhub_adapter/adapter/mapping.py
def _is_empty_reasoning_item(item: JsonObject) -> bool:
    if item.get("type") != "reasoning":
        return False
    readable_keys = set(item) - {"id", "type", "status"}
    if not readable_keys:
        return True
    for key in readable_keys:
        value = item.get(key)
        if value not in (None, "", [], {}):
            return False
    return True
```

In the **current** setup this rarely fires, because the AK is pinned per chain
(§4) — the same AK that minted a blob also consumes it, so it stays decryptable.
(It would matter if `chat_run_id` were ever made per-LLM-call, or on server-side
state expiry.)

## 3. Does dropping reasoning break prompt caching? No — but there is a subtler cost

The common confusion: *"reasoning is generated by the model, so it's in the KV;
if you don't resend it, the context fractures and caching can't trigger."* This
conflates **two different caches**:

- **(a) the KV during a single generation** — yes, reasoning is in it and the
  output attends to it; but it is ephemeral, lives only for that one response.
- **(b) the cross-request prompt cache** (what `cache_hit_rate`/`cache_ratio`
  measure) — keys on the **literal token prefix of each request's INPUT**.

Reasoning is **output**, not input. Whether it becomes part of the *next* request's
input prefix is exactly the §1 difference. And because the chat path drops
reasoning **consistently every step**, consecutive chat requests are mutually
prefix-consistent:

```
chat req k:    [sys, user, (vis₁,fco₁), …, (vis_{k-1},fco_{k-1})]
chat req k+1:  [sys, user, (vis₁,fco₁), …, (vis_{k-1},fco_{k-1}), (vis_k,fco_k)]
```

`req k+1` is a superset of `req k` → the shared head hits. Empirically the chat
path (v4/v6) holds **`cache_hit_rate ≈ 0.965`** — caching clearly fires. A
"fracture" would require *mixing* (cache a sequence **with** reasoning, then send
one **without**); the chat path never has reasoning, so there is no mix.

**The real cost the intuition is sensing** shows up not in the hit *rate* but in
the cache *ratio* (chat `mean_cache_ratio 0.577` vs `/responses 0.914`). The
upstream is a reasoning model, so it generates reasoning **internally even on the
chat path**; the sequence it actually decodes (and caches) is
`[I_k | reasoning_k | vis_k]`. The next chat request is `[I_k | vis_k | fco_k]` —
after `I_k` the cached sequence has `reasoning_k` but the request has `vis_k`, so
they **diverge right there**: the next request can only reuse up to `I_k` and must
re-prefill its own prior visible output. On `/responses`, the next request replays
`reasoning_k` verbatim, so it matches the cached `[I_k | reasoning_k | fc_k]`
token-for-token and reuses everything up to the last tool result. Net: dropping
reasoning **reduces incremental reuse**, it does not stop caching.

> Confidence: (a)/(b) distinction, "consistent omission ≠ fracture", and the high
> chat hit-rate are **certain**. The exact 0.577 vs 0.914 attribution depends on
> the upstream caching decode tokens / `store=true` server-side behaviour, which is
> **plausible but not directly measured here**.

The bigger loss from dropping reasoning is **not caching at all** — it's that the
model can't *attend to / condition on* its prior reasoning (on chat, step k+1 can't
see `reasoning_{1..k}`; on `/responses` it can). That conditioning loss is the
+5 pp; cache efficiency is a separate axis.

## 4. When does the AK rotate?

The AK (weighted upstream) is a deterministic hash of a `selection_id`:

```563:573:codex_modelhub_adapter/adapter/proxy.py
def _modelhub_upstream_selection_id(
    settings: AdapterSettings,
    upstream_extra: dict[str, str] | None,
) -> str:
    extra = upstream_extra or {}
    return (
        _valid_upstream_extra_session_id(extra.get("chat_run_id"))
        or _valid_upstream_extra_session_id(extra.get("sandbox_session_id"))
        or _valid_upstream_extra_session_id(extra.get("session_id"))
        or settings.session_id
    )
```

```547:560:codex_modelhub_adapter/adapter/proxy.py
def _pick_modelhub_upstream_by_weighted_session_hash(
    session_id: str, model: str, upstreams: tuple[ModelHubUpstream, ...],
) -> ModelHubUpstream:
    total_weight = sum(upstream.weight for upstream in upstreams)
    digest = hashlib.sha256(f"{session_id}:{model}:modelhub-upstream".encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % total_weight
    ...
```

So `selection_id = chat_run_id`, and the whole question is *how often `chat_run_id`
changes*. The runtime sets it **once per `run_turn`** (= once per chain/sample):

```496:518:src/codex_agent/runtime.py
    def _build_turn_env(self, run_home: Path) -> dict[str, str]:
        ...
        chat_run_id = sanitize_session_id(
            f"{self.config.prefix_cache_session_id}_{uuid.uuid4().hex[:12]}"
        )
        extra = {
            "session_id": self.config.prefix_cache_session_id,  # stable across all samples
            "source": "codex_agent_sdk",
            "chat_run_id": chat_run_id,                          # fresh uuid per run_turn
        }
        env[MODELHUB_EXTRA_HEADER_ENV] = json.dumps(extra, ...)
        ...
```

```196:227:src/codex_agent/runtime.py
    def run_turn(self, request: CodexTurnRequest, ...):
        ...
        run_home = self._prepare_run_home()
        ...
                    env=self._build_turn_env(run_home),   # called ONCE; baked into the client
                )
            ) as client:
```

That env is baked into the client (`env_http_headers.extra=$MODELHUB_EXTRA_HEADER_ENV`)
and reused for **every** round-trip in the chain (including finalization re-asks).
So:

| Scope | `chat_run_id` | AK |
|---|---|---|
| **Within one chain** (all round-trips of a sample) | **stable** | **pinned** → stable per-AK cache + reasoning blobs decryptable |
| **Across samples** (40 concurrent workers) | fresh uuid each | re-hashed → spread over `gpt54_a:b:c = 5:1:5` |

**The AK rotates only *between* samples, never within a chain.** That is exactly
the property a multi-turn chain needs, and it's by design: the per-sample random
`chat_run_id` is the load-balancing knob, while the stable `session_id` + identical
instruction/tool prefix is what lets each AK warm and reuse the shared head.

Independent confirmation: `prompt_caching_20260608.md` Phase 3 measured a
**within-turn finalization re-ask reusing 96.9 % (cached_tokens=21504) of the full
prior-turn context** — only possible if the re-ask hit the **same** AK as the
original turn. "Rotation" cases: (1) a new sample → new `chat_run_id` → likely a
different AK (intended); (2) a sample **retry** → a fresh `run_turn` → fresh
`chat_run_id` → possibly a different AK, but also a fresh conversation (no replayed
blobs); (3) within a chain → none.

### Correction to a claim made while analysing this

An earlier verbal explanation implied the weighted pool routinely causes
`invalid_encrypted_content` mid-chain and erodes reasoning carryover. That is
**wrong for the current design** — the AK is pinned per chain, so blobs stay
decryptable. The v5/v6 failures (2 and 10 of 600) were **`429 Too Many Requests`**
(the pinned AK rate-limited, SDK retry exhausted), *not* encrypted-content/AK
mismatch. So the +5 pp `/responses` carryover is **not** being silently eroded by
rotation.

## 5. Practical implications

1. **To get v5-level accuracy you need the `/responses` path** (today forced via
   `--reasoning-summary auto`). `--reasoning-effort medium` on the cheap chat path
   recovers only ~2 of the 7 pp.
2. **A cleaner future lever:** an explicit runtime/adapter "use `/responses`" toggle
   **decoupled from summaries**, so the +5 pp reasoning-carryover gain can be had
   without paying for summary generation (summaries are useful for tracing, but are
   not the accuracy driver).
3. **Caching is healthy on both paths** within a chain; the lower chat `cache_ratio`
   is an efficiency difference, not a correctness problem.

### Residual confound (unchanged from v6)

v6 = v4 + medium effort gives +2 pp on the chat path, but we cannot fully rule out
that `/v2/crawl` only *partially* honours `reasoning_effort`. Either way the
direction holds: at the same nominal medium effort, switching to `/responses` adds
+5 pp — the gain lives on the `/responses` path, however that path realises it
(cross-turn reasoning state and/or more faithful effort honouring).
