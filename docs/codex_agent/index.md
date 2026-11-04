# Codex Agent

This chapter collects engineering notes for the Codex Agent runtime and its
tool-using NR3D/OpenEQA/SceneFunc3D workflows. These are design and investigation records,
not generated API documentation.

## Current Operations

- [ALFRED Incremental Execution](alfred.md): predicted depth and instance masks,
  online object/view/grid memory, simulator integration, and validation protocol.
- [Multi-Protocol Adapter](adapter.md): active `codex_adapter` setup, explicit
  registration, durable session/AK bindings, state preservation and capability
  limits. The investigation notes below preserve older experiments rather than
  replacing the current operational guide.

## Core Notes

- [Codex–Gemini State Replay and Cache](../benchmark/adapter/v3_codex_gemini_cache_20261103.md):
  real SDK dynamic tools and signature-preserving experimental bridge, actual
  cache hits, matched accounting controls, and retained failures/negative results.
- [Gemini Tool/Signature Protocol Probe](../benchmark/adapter/v2_gemini_tool_state_20261103.md):
  signature replay and targeted controls, image-input transport, and the confirmed
  GPT Responses-only routing policy; this earlier record tested direct HTTP only.
- [ModelHub Tool/State Protocol Probe](../benchmark/adapter/v1_modelhub_tool_state_20261103.md):
  2026-11-03 direct Chat/Responses tool loops, encrypted-state replay, and
  corruption/removal controls; separate from SDK/adapter integration validation.
- [Keyframe Selector Shared Foundation Review](keyframe_shared_foundation_review_20261003.md):
  deferred architecture gap between the thesis's shared evidence foundation and
  the three tasks' current retrieval paths, with code evidence and follow-up criteria.
- [NR3D Tools Plan](nr3d_tools_plan_20260609.md): design plan for agent-driven
  CLI tools, image viewing, and the phased NR3D evidence loop.
- [View Image Adapter Fix](view_image_adapter_fix_20260609.md): notes on making
  mid-turn image inspection work in the runtime.
- [Prompt Caching](prompt_caching_20260608.md): ModelHub prompt-cache mechanism,
  adapter changes, and live NR3D verification.
- [Reasoning Summary And Effort](reasoning_summary_and_effort_20260610.md):
  reasoning-summary and effort behavior in Codex/ModelHub runs.
- [Reasoning Carryover, Caching, And AK Routing](reasoning_carryover_caching_ak_routing_20260611.md):
  follow-up notes on reasoning state, cache behavior, and adapter routing.
- [Skill Loop And Reasoning Dropped](skill_loop_and_reasoning_dropped_20260609.md):
  investigation of tool-loop behavior and lost reasoning context.

## How To Use This Chapter

Start with the newest note that matches the subsystem you are touching, then
cross-check the current implementation under `src/codex_agent/` before changing
code. These files preserve historical decisions and measurements; current source
and tests remain the authority for runtime behavior.
