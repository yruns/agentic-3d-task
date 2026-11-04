"""Check one native Responses turn after the recorded upstream outage."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from codex_agent.alfred.actor_runtime import AlfredPolicyRuntime
from codex_agent.alfred.policy import AlfredWireDecision
from codex_agent.alfred.provenance import code_provenance
from codex_agent.config import CodexAgentConfig
from codex_agent.models import CodexTurnRequest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("/home/ysh/codebase/agentic-3d-task")
    code = code_provenance(root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "actor").mkdir()
    fixture = output / "synthetic.png"
    Image.new("RGB", (300, 300), "gray").save(fixture)
    expected = {
        "operation": "Stop",
        "object_id": 0,
        "cell_x": 0,
        "cell_y": 0,
        "repeat": 1,
        "rationale": "Synthetic transport check",
        "memory_note": "No simulator task was run",
    }
    request = CodexTurnRequest(
        prompt="This is a synthetic connectivity check, not an embodied task. Return exactly: "
        + json.dumps(expected),
        output_schema=AlfredWireDecision.model_json_schema(),
        image_paths=(fixture,),
    )
    config = replace(
        CodexAgentConfig.from_env(),
        project_root=output / "actor",
        codex_home=root / ".codex-home",
        model="gpt-5.4-2026-03-05",
        reasoning_effort="medium",
        keep_run_home=True,
        sandbox="read_only",
        turn_timeout_s=90,
    )
    report: dict[str, object] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": code,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "turn_timeout_s": 90,
        "max_finalization_retries": 0,
        "scope": "Synthetic native Responses vision/schema transport probe; no simulator or task score",
        "expected": expected,
        "request": request.prompt,
    }
    started = time.monotonic()
    succeeded = False
    try:
        result = AlfredPolicyRuntime(config).run_turn(
            request, max_finalization_retries=0
        )
        actual = AlfredWireDecision.model_validate_json(result.final_response)
        succeeded = actual.operation == "Stop"
        report.update(
            actual=actual.model_dump(mode="json"), metadata=result.metadata.as_dict()
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report.update(succeeded=succeeded, wall_seconds=time.monotonic() - started)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(output / "report.json"),
                "succeeded": succeeded,
                "wall_seconds": report["wall_seconds"],
            }
        ),
        flush=True,
    )
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
