# Mid-turn image viewing for the Codex agent (`view_image`)

Status: **root-caused + fixed + verified** on 2026-06-09.

## Goal

Let the Codex agent *see* an image **mid-turn** (e.g. an annotated first-person
NR3D frame produced by a tool), not just at the initial prompt. This is the
prerequisite for an agent-driven evidence loop where the agent calls a tool,
the tool renders/returns an image, and the agent inspects the pixels before
deciding.

## What works and what didn't (evidence)

All probes go through the real `CodexAgentRuntime` + the local ModelHub adapter
(`gpt-5.4-2026-03-05`, `wire_api="responses"`). Scripts live under `tmp/`
(`verify_vision_channel.py`, `verify_view_image.py`, `verify_view_image_items.py`,
`verify_mcp_image.py`, `mcp_image_probe_server.py`).

| Delivery | Result |
| --- | --- |
| `LocalImageInput` at the initial prompt | **works** — model describes a real kitchen frame correctly |
| Tool returns a **path** (text), agent told to "look at it" | **fails** — agent hallucinates (invents the CODE digits, calls a kitchen an "office") |
| Built-in `view_image` tool (agent calls it; emits an `imageView` item) | **failed before fix** — `imageView` item present, but model still read the wrong digits |
| MCP server tool returning an `image` content block | **not reachable** — Codex exposes MCP *resources* (`read_mcp_resource`), **not** MCP tools as callable functions; the agent never sees the tool |

So the official mid-turn mechanism is the built-in **`view_image`** tool. It was
silently broken by the adapter.

## Root cause

Dumping the Responses request the agent sent after `view_image` showed the image
is returned as a tool result:

```jsonc
{ "type": "function_call", "name": "view_image" }
{ "type": "function_call_output",
  "output": [ { "type": "input_image", "image_url": "data:image/png;base64,...", "detail": "..." } ] }
```

In `adapter/mapping.py`, `_responses_input_to_chat_messages` turned a
`function_call_output` into a Chat Completions `tool` message via
`_content_to_text(output)` — which keeps only text and **drops the
`input_image`**. Compounding this, Chat Completions `tool` role messages cannot
carry images at all; an image has to live on a `user`/`assistant` message.

## Fix (`adapter/mapping.py`, chat path only)

When a `function_call_output` carries image parts, keep the text on the `tool`
message and **surface the image parts as a following `user` message** so the
upstream Chat Completions model actually receives the pixels:

- added a `pending_tool_images` accumulator in `_responses_input_to_chat_messages`;
- `function_call_output` now splits text vs. image parts (`_extract_chat_image_parts`);
- `flush_pending_tools()` appends a `{"role": "user", "content": [image_url...]}`
  message right after the tool outputs.

Only the chat path (`_responses_input_to_chat_messages`, used by
`build_chat_completions_body`) was changed; the legacy
`_responses_input_to_modelhub_messages` path has the same gap but is not on the
active route. Adapter backups: `adapter/mapping.py.bak.*`.

## Verification (after fix)

| Probe | Before | After |
| --- | --- | --- |
| synthetic card, random 4-digit CODE | invented (e.g. 5233→5182) | **correct** (8743→8743, 2182→2182, 9487→9487) |
| real first-person frame (scene0011_00 frame 0) | "living room / wood floor / gray sofa" (all wrong) | **"dining area / dark-gray tile / brown chairs / dining table"** (all correct) |

## Implication

Unlocks an **agent-driven CLI + `view_image`** architecture for NR3D: inside one
`thread.run` (one runtime `execute`) the agent can shell out to NR3D CLI tools,
have them write annotated frames into the workspace, `view_image` those frames,
and iterate — first-person pixels included, with low turn overhead.

Guardrail to keep: a frame only counts as visual evidence once the agent has
actually `view_image`-d it; otherwise it is back to hallucinating from a path.
