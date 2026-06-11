"""Agent-driven CLI tools for OpenEQA question answering.

Each tool is a small, deterministic function over a prepared OpenEQA clip
(:class:`~codex_agent.openeqa.tools.scene_context.OpenEqaToolScene`). ``list_objects``
prints compact JSON; ``view_frame`` / ``keyframe_selector`` / ``view_bev`` also
write an image into a writable scratch dir and return its path so the agent can
``view_image`` the pixels.

The single dispatcher entry point is::

    python -m codex_agent.openeqa.tools <tool> --scene-dir <clip_dir> --args '<json>'

so the agent can fetch first-person frames, retrieve language-grounded keyframes,
and view a top-down BEV inside one sandboxed turn.
"""

from __future__ import annotations

from .dispatch import TOOL_NAMES, run_tool
from .models import ToolInputError, ToolPayload
from .scene_context import OpenEqaToolScene, build_selector

__all__ = [
    "TOOL_NAMES",
    "run_tool",
    "ToolInputError",
    "ToolPayload",
    "OpenEqaToolScene",
    "build_selector",
]
