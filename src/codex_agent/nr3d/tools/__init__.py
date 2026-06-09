"""Agent-driven CLI tools for NR3D visual grounding.

Each tool is a small, deterministic function over a prepared
:class:`~codex_agent.nr3d.sample.Nr3dScene`. Text tools print compact JSON to
stdout; frame/BEV tools also write an annotated PNG into a writable scratch dir
and return its path so the agent can ``view_image`` the pixels.

The single dispatcher entry point is::

    python -m codex_agent.nr3d.tools <tool> --scene-dir <pack_dir> --args '<json>'

so the agent can run the whole evidence loop inside one sandboxed turn.
"""

from __future__ import annotations

from .dispatch import TOOL_NAMES, run_tool
from .models import ToolInputError, ToolPayload

__all__ = [
    "TOOL_NAMES",
    "run_tool",
    "ToolInputError",
    "ToolPayload",
]
