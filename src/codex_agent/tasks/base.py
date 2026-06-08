"""The :class:`CodexTask` protocol — the seam for adding new task families.

A *task* knows how to turn its own domain inputs into a generic
:class:`~codex_agent.models.CodexTurnRequest`, how to recognise a usable Codex
response, and how to parse that response into a typed outcome. The runtime stays
completely task-agnostic: it only consumes this protocol.

To add a new task family (QA, navigation, ...), implement this protocol; nothing
in :mod:`codex_agent.runtime` needs to change.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from ..models import CodexTaskResult, CodexTurnRequest

OutcomeT_co = TypeVar("OutcomeT_co", covariant=True)
ResultT = TypeVar("ResultT")


@runtime_checkable
class CodexTask(Protocol[OutcomeT_co]):
    """Bridge between a domain problem and one Codex turn."""

    @property
    def task_name(self) -> str:
        """Stable identifier for the task family (used in results/telemetry)."""
        ...

    def build_turn_request(self) -> CodexTurnRequest:
        """Construct the prompt, attachments, and output schema for the turn."""
        ...

    def is_valid_response(self, response_text: str) -> bool:
        """Return ``True`` if ``response_text`` is a parseable, valid answer.

        Used by the runtime to decide whether to issue a finalization re-ask.
        Implementations must not raise; return ``False`` on any parse failure.
        """
        ...

    def parse_response(self, response_text: str) -> OutcomeT_co:
        """Parse a (validated) response into a typed task outcome.

        Raises:
            codex_agent.errors.CodexResponseError: If the text cannot be parsed.
        """
        ...


@runtime_checkable
class CodexExecutor(Protocol):
    """Anything that can run a :class:`CodexTask` to a typed result.

    :class:`codex_agent.runtime.CodexAgentRuntime` is the production
    implementation; depending on this protocol (instead of the concrete class)
    keeps the evaluation harness decoupled and easy to fake in tests.
    """

    def execute(self, task: CodexTask[ResultT]) -> CodexTaskResult[ResultT]:
        """Build, run, and parse ``task`` into a :class:`CodexTaskResult`."""
        ...


__all__ = ["CodexTask", "CodexExecutor", "OutcomeT_co", "ResultT"]
