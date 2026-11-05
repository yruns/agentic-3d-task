"""Small fixture-only checks for pilot15 selection and diagnostic isolation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Callable
from pathlib import Path

import audit_pilot15 as audit
import ingest_diagnostics as diagnostics

CHECKS: dict[str, bool] = {}


def rejects(name: str, operation: Callable[[], object]) -> None:
    try:
        operation()
    except ValueError:
        CHECKS[name] = True
    else:
        CHECKS[name] = False


def completed(identity: str, *, success: bool) -> audit.JsonObject:
    return {
        "episode_id": identity,
        "status": "completed",
        "error": None,
        "score": {
            "success": success,
            "completed_goal_conditions": int(success),
            "total_goal_conditions": 1,
            "expert_actions": 10,
            "executed_actions": 5,
        },
    }


def failed(identity: str) -> audit.JsonObject:
    return {
        "episode_id": identity,
        "status": "error",
        "error": "synthetic transport failure",
        "score": None,
    }


selected, errors = audit.select_first_scores(
    [completed("a", success=True), failed("b")], {"a", "b"}, allow_partial=True
)
CHECKS["partial_keeps_one_score_and_one_null_error"] = len(selected) == len(errors) == 1
CHECKS["error_not_in_success_denominator"] = (
    audit.aggregate_scores(
        [audit.EpisodeScore.model_validate(row["score"]) for row in selected]
    ).success_rate
    == 1.0
)
rejects(
    "duplicate_terminal_failure_cannot_be_replaced_by_success",
    lambda: audit.select_first_scores(
        [completed("a", success=False), completed("a", success=True)],
        {"a"},
        allow_partial=False,
    ),
)
rejects(
    "fourteen_scores_cannot_pass_final_fifteen_gate",
    lambda: audit.select_first_scores(
        [completed(str(i), success=False) for i in range(14)],
        {str(i) for i in range(15)},
        allow_partial=False,
    ),
)
rejects(
    "infrastructure_error_cannot_have_an_official_score",
    lambda: audit.select_first_scores(
        [{**failed("a"), "score": completed("a", success=False)["score"]}],
        {"a"},
        allow_partial=True,
    ),
)
selected, errors = audit.select_first_scores(
    [failed("a"), completed("a", success=False)], {"a"}, allow_partial=False
)
CHECKS["unscored_error_then_first_score_retains_history"] = (
    len(selected) == len(errors) == 1
)
CHECKS["error_report_omits_original_message"] = (
    "synthetic transport failure"
    not in json.dumps(audit.safe_error("synthetic transport failure"))
)

with tempfile.TemporaryDirectory(prefix="pilot15_helper_checks_") as temporary:
    root = Path(temporary)
    for index in (1, 2):
        request = root / "episode/decisions" / f"{index:04d}" / "request.json"
        request.parent.mkdir(parents=True)
        request.write_text(json.dumps({"images": []}))
    response = root / "episode/decisions/0001/response.json"
    response.write_text(
        json.dumps(
            {
                "raw_response": json.dumps({"operation": "Stop"}),
                "metadata": {
                    "status": "completed",
                    "attempts": [{}],
                    "usage": {
                        "total": {
                            "input_tokens": 12,
                            "output_tokens": 3,
                            "total_tokens": 15,
                        }
                    },
                },
            }
        )
    )
    usage = audit.audit_usage(root / "episode", completed=False)
    CHECKS["unanswered_final_request_preserves_unknown_usage"] = usage[
        "exact_input_tokens"
    ] is None and usage["unanswered_decisions"] == [2]
    CHECKS["prefix_reports_only_observed_token_lower_bound"] = (
        usage["reported_usage_lower_bound"]["input_tokens"] == 12
    )
    rejects(
        "completed_score_requires_all_saved_requests_answered",
        lambda: audit.audit_usage(root / "episode", completed=True),
    )
    artifact = root / "capture_failure.txt"
    artifact.write_text("Synthetic conda activation failure; no Python or frames.")
    record = diagnostics.DiagnosticRecord(
        run_id="capture-fixture",
        kind="capture",
        status="failed",
        code=None,
        code_status="not_executed",
        artifact_path=str(artifact),
        summary={"python_started": False, "frames": 0},
    )
    absent = root / "check_only.sqlite"
    diagnostics.ingest_records(absent, [record], check_only=True)
    CHECKS["check_only_never_creates_database"] = not absent.exists()
    database = root / "fixture.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, metadata_json TEXT)"
        )
        connection.execute("CREATE TABLE attempts (run_id TEXT, score_json TEXT)")
        connection.execute("INSERT INTO runs VALUES ('task-fixture','{}')")
    diagnostics.ingest_records(database, [record], check_only=False)
    with sqlite3.connect(database) as connection:
        CHECKS["diagnostic_table_does_not_fabricate_task_attempts"] = (
            connection.execute("SELECT COUNT(*) FROM attempts").fetchone() == (0,)
        )
        CHECKS["task_run_table_unchanged"] = connection.execute(
            "SELECT COUNT(*) FROM runs"
        ).fetchone() == (1,)
        code, summary, metrics = connection.execute(
            "SELECT code_json,summary_json,metrics_json FROM diagnostic_runs"
        ).fetchone()
        CHECKS["shell_failure_has_no_fake_executed_code_or_metrics"] = (
            code is None
            and metrics is None
            and json.loads(summary)["code_status"] == "not_executed"
        )
    rejects(
        "diagnostic_history_cannot_be_overwritten",
        lambda: diagnostics.ingest_records(database, [record], check_only=False),
    )
    rejects(
        "check_only_also_rejects_duplicate_history",
        lambda: diagnostics.ingest_records(database, [record], check_only=True),
    )
    collision = record.model_copy(update={"run_id": "task-fixture"})
    rejects(
        "diagnostic_id_cannot_shadow_task_run",
        lambda: diagnostics.ingest_records(database, [collision], check_only=False),
    )
    rejects(
        "official_task_metrics_rejected_from_diagnostic_schema",
        lambda: diagnostics.DiagnosticRecord.model_validate(
            {**record.model_dump(), "metrics": {"success_rate": 1.0}}
        ),
    )

output = Path(__file__).with_name("helper_self_checks.json")
if output.exists():
    raise FileExistsError("Choose a new self-check evidence path")
report = {
    "scope": "Synthetic in-memory selections and a temporary SQLite fixture only; benchmark database/models/simulator unchanged",
    "passed": all(CHECKS.values()),
    "checks": CHECKS,
    "source_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(audit.__file__), Path(diagnostics.__file__), Path(__file__))
    },
}
with output.open("x") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
print(
    json.dumps(
        {"passed": sum(CHECKS.values()), "total": len(CHECKS), "output": str(output)},
        indent=2,
    )
)
if not all(CHECKS.values()):
    raise SystemExit(1)
