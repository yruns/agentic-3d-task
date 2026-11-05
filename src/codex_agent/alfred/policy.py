"""Tool-enabled ALFRED task and structured episode outcome for the Codex SDK.

In the redesigned architecture, one ALFRED episode runs as a **single Codex
Agent SDK turn**:
* Turn 0 provides the task instruction, the initial top-down Schematic BEV map,
  the initial annotated first-person RGB view, and the inlined 3D Object
  Catalog & Scene Region Layout (captured from an automatic initial 360-degree
  scan).
* Inside the turn, Codex autonomously invokes high-level CLI tools
  (``python -m codex_agent.alfred.tools <tool> --episode-dir <dir> --args ...``)
  and ``view_image``.
* Each macro action tool steps the simulator closed-loop via FMM / Auto-Framing,
  updates the 3D scene memory on every intermediate frame, and returns only a
  compact ``SceneDelta`` JSON.
* Once the goal is satisfied (or no further progress is possible), Codex calls
  ``finish_episode`` and emits the final :class:`AlfredEpisodeTurnOutcome` JSON.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from codex_agent.errors import CodexResponseError
from codex_agent.json_extraction import extract_json_object
from codex_agent.models import CodexTurnRequest

from .actor_runtime import AlfredPolicyRuntime as AlfredPolicyRuntime
from .dataset import AlfredInstruction
from .playbook import ALFRED_TOOLS_PLAYBOOK

NAVIGATION_ACTIONS = frozenset(
    {"MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"}
)
INTERACTION_ACTIONS = frozenset(
    {
        "PickupObject",
        "PutObject",
        "OpenObject",
        "CloseObject",
        "ToggleObjectOn",
        "ToggleObjectOff",
        "SliceObject",
    }
)


class AlfredEpisodeTurnOutcome(BaseModel):
    """Final structured response returned by Codex at the end of an episode turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed", "incomplete"] = Field(
        description="Whether all requested task goal conditions were completed."
    )
    summary: str = Field(
        min_length=1,
        max_length=2400,
        description="Concise summary of the high-level actions executed and final state.",
    )
    completed_subgoals: list[str] = Field(
        description="List of subgoals verified as completed during the episode."
    )


@dataclass(frozen=True)
class AlfredPolicyTask:
    """Single-turn tool-enabled ALFRED episode task for :class:`CodexAgentRuntime`."""

    instruction: AlfredInstruction
    episode_dir: Path
    initial_context: dict[str, object]
    image_paths: tuple[Path, ...]
    use_step_instructions: bool = False
    python_executable: str = sys.executable

    @property
    def task_name(self) -> str:
        return "alfred_interactive_episode"

    def build_turn_request(self) -> CodexTurnRequest:
        prompt = self._build_prompt()
        return CodexTurnRequest(
            prompt=prompt,
            output_schema=AlfredEpisodeTurnOutcome.model_json_schema(),
            image_paths=self.image_paths,
        )

    def _build_prompt(self) -> str:
        sections: list[str] = [
            "You are an embodied 3D indoor agent executing an interactive ALFRED household task.",
            f"Goal instruction: {self.instruction.goal}",
        ]
        if self.use_step_instructions and self.instruction.steps:
            steps_formatted = "\n".join(
                f"  {idx + 1}. {step}"
                for idx, step in enumerate(self.instruction.steps)
            )
            sections.append(f"Step-by-step hints:\n{steps_formatted}")

        ep_dir_str = str(self.episode_dir.resolve())
        sections.append(
            "Tool CLI invocation syntax (run via shell tool):\n"
            f"  {self.python_executable} -m codex_agent.alfred.tools <tool_name> \\\n"
            f'    --episode-dir "{ep_dir_str}" \\\n'
            "    --args '<json_object>'\n\n"
            "Examples:\n"
            f'  {self.python_executable} -m codex_agent.alfred.tools navigate_to --episode-dir "{ep_dir_str}" --args \'{{"object_id": 12}}\'\n'
            f'  {self.python_executable} -m codex_agent.alfred.tools interact_object --episode-dir "{ep_dir_str}" --args \'{{"action": "pickup", "object_id": 12}}\'\n'
            f'  {self.python_executable} -m codex_agent.alfred.tools explore_frontier --episode-dir "{ep_dir_str}" --args \'{{"stop_on_categories": ["Ladle", "SinkBasin"]}}\'\n'
            f'  {self.python_executable} -m codex_agent.alfred.tools view_crop --episode-dir "{ep_dir_str}" --args \'{{"object_id": 12}}\'\n'
            f'  {self.python_executable} -m codex_agent.alfred.tools finish_episode --episode-dir "{ep_dir_str}" --args \'{{"summary": "Completed task"}}\''
        )
        sections.append(ALFRED_TOOLS_PLAYBOOK)
        sections.append(
            "Initial Scene State (after automatic initial 360-degree scan; "
            "Image 1 = current annotated RGB, Image 2 = initial top-down BEV map):\n"
            + json.dumps(self.initial_context, ensure_ascii=False, indent=2)
        )
        sections.append(
            "Now execute the task using the high-level tools above. After calling "
            "`finish_episode`, respond ONLY with a JSON object matching the schema: "
            '{"status": "completed" | "incomplete", "summary": "<what you did>", '
            '"completed_subgoals": ["<subgoal 1>", ...]}.'
        )
        return "\n\n".join(sections)

    def parse_response(self, response_text: str) -> AlfredEpisodeTurnOutcome:
        try:
            return AlfredEpisodeTurnOutcome.model_validate(
                extract_json_object(response_text)
            )
        except (ValueError, ValidationError) as exc:
            raise CodexResponseError(
                f"invalid ALFRED episode turn outcome: {exc}"
            ) from exc

    def is_valid_response(self, response_text: str) -> bool:
        try:
            self.parse_response(response_text)
        except CodexResponseError:
            return False
        return True


__all__ = [
    "AlfredEpisodeTurnOutcome",
    "AlfredPolicyRuntime",
    "AlfredPolicyTask",
    "INTERACTION_ACTIONS",
    "NAVIGATION_ACTIONS",
]
