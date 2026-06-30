"""codex_agent: a task-agnostic Codex Agent SDK runtime.

The runtime drives single Codex Agent SDK turns and parses them into typed task
outcomes. Tasks plug in through the :class:`CodexTask` protocol, so the same
runtime serves NR3D visual grounding today and other task families later.

Example (NR3D visual grounding)::

    from codex_agent import CodexAgentConfig, CodexAgentRuntime
    from codex_agent.nr3d import Nr3dGroundingTask, Nr3dScene
    from codex_agent.nr3d.sample import load_sample, scene_dir_for

    runtime = CodexAgentRuntime(CodexAgentConfig.from_env())
    sample = load_sample(data_root, sample_id)
    scene = Nr3dScene.load(scene_dir_for(data_root, sample.scene_id))
    result = runtime.execute(Nr3dGroundingTask(sample=sample, scene=scene))
    print(result.outcome.proposal_id, result.outcome.confidence)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import CodexAgentConfig, SandboxMode
from .errors import (
    CodexAgentError,
    CodexConfigError,
    CodexResponseError,
    CodexTurnError,
    Nr3dDataError,
    OpenEqaDataError,
    OpenEqaJudgeError,
)
from .models import (
    CodexSkill,
    CodexTaskResult,
    CodexTurnMetadata,
    CodexTurnRequest,
    CodexTurnResult,
)
from .tasks import CodexTask

if TYPE_CHECKING:
    from .runtime import CodexAgentRuntime

__version__ = "0.1.0"

__all__ = [
    # Runtime + config
    "CodexAgentRuntime",
    "CodexAgentConfig",
    "SandboxMode",
    # Task seam
    "CodexTask",
    # Turn models
    "CodexSkill",
    "CodexTurnRequest",
    "CodexTurnResult",
    "CodexTurnMetadata",
    "CodexTaskResult",
    # Errors
    "CodexAgentError",
    "CodexConfigError",
    "CodexTurnError",
    "CodexResponseError",
    "Nr3dDataError",
    "OpenEqaDataError",
    "OpenEqaJudgeError",
]


def __getattr__(name: str) -> object:
    """Lazily expose optional Codex Agent SDK runtime symbols."""
    if name == "CodexAgentRuntime":
        from .runtime import CodexAgentRuntime

        return CodexAgentRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
