from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai_codex import Codex, Sandbox, SkillInput, TextInput
from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    CommandExecutionThreadItem,
    ItemCompletedNotification,
    McpToolCallThreadItem,
    ThreadTokenUsageUpdatedNotification,
    TurnCompletedNotification,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = PROJECT_ROOT / ".agents/skills/heart-task/SKILL.md"
os.environ.setdefault("CODEX_HOME", str(PROJECT_ROOT / ".codex-home"))


def _dump_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    return {}


def _item_type(item: Any) -> str:
    root = item.root if hasattr(item, "root") else item
    return getattr(root, "type", type(root).__name__)


def _print_item_summary(item: Any) -> None:
    root = item.root if hasattr(item, "root") else item
    if isinstance(root, McpToolCallThreadItem):
        print(
            "TRACE item=mcpToolCall "
            f"server={root.server} tool={root.tool} status={root.status.value}"
        )
        if root.result is not None:
            print("TRACE mcp_result=" + json.dumps(_dump_model(root.result), ensure_ascii=False))
        if root.error is not None:
            print("TRACE mcp_error=" + json.dumps(_dump_model(root.error), ensure_ascii=False))
        return

    if isinstance(root, CommandExecutionThreadItem):
        command = " ".join(root.command) if isinstance(root.command, list) else str(root.command)
        print(
            "TRACE item=commandExecution "
            f"status={root.status.value} exit_code={root.exit_code} command={command}"
        )
        return

    if isinstance(root, AgentMessageThreadItem):
        text = (root.text or "").replace("\n", "\\n")
        print(f"TRACE item=agentMessage phase={root.phase} text={text[:240]}")
        return

    print(f"TRACE item={_item_type(item)}")


def main() -> None:
    prompt = (
        "Use the heart-task skill. Use the MCP tool heart_template to get the exact heart. "
        "Create skill_mcp_heart.py in the current directory. The script must print exactly "
        "the heart from the MCP tool, then run `python3 skill_mcp_heart.py`. "
        "In the final answer, state whether you used the MCP tool and include exact stdout."
    )

    with Codex() as codex:
        thread = codex.thread_start(
            model="gpt-5.5-2026-04-24",
            sandbox=Sandbox.workspace_write,
            cwd=str(PROJECT_ROOT),
        )
        turn = thread.turn(
            [
                SkillInput(name="heart-task", path=str(SKILL_PATH)),
                TextInput(prompt),
            ],
            cwd=str(PROJECT_ROOT),
        )

        completed = None
        final_items = []
        usage = None
        for event in turn.stream():
            payload = event.payload
            print(f"TRACE event={type(payload).__name__}")
            if isinstance(payload, ItemCompletedNotification) and payload.turn_id == turn.id:
                final_items.append(payload.item)
                _print_item_summary(payload.item)
            elif isinstance(payload, ThreadTokenUsageUpdatedNotification) and payload.turn_id == turn.id:
                usage = payload.token_usage
                print("TRACE usage=" + json.dumps(_dump_model(usage), ensure_ascii=False))
            elif isinstance(payload, TurnCompletedNotification) and payload.turn.id == turn.id:
                completed = payload.turn
                print(
                    "TRACE turn_completed="
                    + json.dumps(_dump_model(completed), ensure_ascii=False)
                )

        print("TRACE final_item_types=" + json.dumps([_item_type(item) for item in final_items]))
        if usage is not None:
            print("TRACE final_usage=" + json.dumps(_dump_model(usage), ensure_ascii=False))
        if completed is not None:
            print(f"TRACE final_status={completed.status.value}")


if __name__ == "__main__":
    main()
