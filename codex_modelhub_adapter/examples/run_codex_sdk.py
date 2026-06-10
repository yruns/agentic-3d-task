from __future__ import annotations

import os
from pathlib import Path

from openai_codex import Codex, Sandbox


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CODEX_HOME", str(PROJECT_ROOT / ".codex-home"))


def main() -> None:
    prompt = os.environ.get(
        "CODEX_PROMPT",
        "Explain this repository in three bullets.",
    )

    with Codex() as codex:
        thread = codex.thread_start(
            model="gpt-5.5-2026-04-24",
            sandbox=Sandbox.workspace_write,
        )
        result = thread.run(prompt)
        print(result.final_response)


if __name__ == "__main__":
    main()
