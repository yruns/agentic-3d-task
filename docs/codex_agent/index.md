# Codex Agent

This chapter collects engineering notes for the Codex Agent runtime and its
tool-using NR3D/OpenEQA workflows. These are design and investigation records,
not generated API documentation.

## Core Notes

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
