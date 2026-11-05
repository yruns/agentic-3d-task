"""Wait for the nine planned recoveries, then verify the exact 50-task cohort."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_pilot50_recovery_queue import read_object, require_equivalent_code

from codex_agent.alfred.provenance import code_provenance
from codex_agent.evaluation.alfred_runner import external_source_provenance

ROOT = Path("/home/ysh/codebase/agentic-3d-task")
WORK = ROOT / "tmp/alfred_setup"


def write_new(path: Path, payload: object) -> None:
    """Keep each verification result immutable once written."""
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> int:
    manifest = read_object(WORK / "pilot50_cohort_recovery3.json")
    result_path = WORK / "recovery3_final_audit_results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    previous = None
    while True:
        queues = [
            read_object(Path(queue["status_path"])) for queue in manifest["queues"]
        ]
        signature = tuple(
            (q["state"], sum(j["state"] == "completed" for j in q["jobs"]))
            for q in queues
        )
        if signature != previous:
            print(
                json.dumps(
                    {
                        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                        "queues": signature,
                    }
                ),
                flush=True,
            )
            previous = signature
        if any(q["state"] in {"failed", "interrupted"} for q in queues):
            write_new(
                result_path,
                {"status": "queue_failed", "queues": queues, "all_passed": False},
            )
            return 1
        if all(q["state"] == "completed" and q["exit_code"] == 0 for q in queues):
            break
        time.sleep(30)

    runs = [
        read_object(Path(row["run_json"]))
        for row in manifest["original_runs"] + manifest["recovery_runs"]
    ]
    current = code_provenance(ROOT)
    require_equivalent_code(current, manifest["code"])
    for run in runs:
        require_equivalent_code(run["code"], manifest["code"])
    external = {
        name: external_source_provenance(Path(info["path"]))
        for name, info in runs[0]["external_sources"].items()
    }
    if not all(run["external_sources"] == external for run in runs):
        raise ValueError("External source changed during original/recovery execution")
    write_new(
        WORK / "recovery3_end_source_audit.json",
        {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "current_code": current,
            "reference_code": manifest["code"],
            "source_identity_policy": manifest["source_identity_policy"],
            "all_19_runs_have_identical_executed_source": True,
            "actual_run_heads": {
                run["run_id"]: run["code"]["head_sha"] for run in runs
            },
            "external_sources": external,
        },
    )
    commands = [
        ["analyze_recovered_pilot50_v3.py"],
        ["rescore_recovered_pilot50_v3.py"],
        [
            "audit_recovery3_pilot_boundary.py",
            "--manifest",
            str(WORK / "pilot50_cohort_recovery3.json"),
            "--output",
            str(WORK / "pilot50_actor_boundary_recovery3_final.json"),
        ],
    ]
    checks = []
    for arguments in commands:
        command = [sys.executable, str(WORK / arguments[0]), *arguments[1:]]
        log = WORK / ("recovery3_" + arguments[0].removesuffix(".py") + "_final.txt")
        with log.open("x") as stream:
            result = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        record = {"command": command, "exit_code": result.returncode, "log": str(log)}
        checks.append(record)
        print(json.dumps(record), flush=True)
    passed = all(check["exit_code"] == 0 for check in checks)
    write_new(
        result_path,
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "verified" if passed else "audit_failed",
            "checks": checks,
            "all_passed": passed,
        },
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
