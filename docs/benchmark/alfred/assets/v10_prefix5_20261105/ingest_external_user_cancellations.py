"""Record externally cancelled task runs without modifying native artifacts.

Only original indices5/7/8 are inserted. Their preserved native checkpoints
remain running/pending; derived SQLite metadata explicitly records that gap.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import audit_pilot15 as audit

from codex_agent.alfred.results import EpisodeAttempt, ingest_run
from codex_agent.config import DEFAULT_PROJECT_ROOT

ROOT = DEFAULT_PROJECT_ROOT
WORK = ROOT / "tmp/alfred_v9"
DATABASE = ROOT / "docs/benchmark/alfred/runs.sqlite"
INDICES = (5, 7, 8)


class PreparedRecord(TypedDict):
    run_id: str
    metadata: audit.JsonObject
    attempt: audit.JsonObject


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_new(path: Path, payload: audit.JsonObject) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def task_rows(connection: sqlite3.Connection, run_ids: list[str]) -> audit.JsonObject:
    rows = {}
    for run_id in run_ids:
        stored = connection.execute(
            "SELECT metadata_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        attempts = connection.execute(
            "SELECT episode_id,status,artifact_dir,score_json,error FROM attempts WHERE run_id=? ORDER BY episode_id",
            (run_id,),
        ).fetchall()
        rows[run_id] = {
            "metadata_json": stored[0] if stored else None,
            "attempts": attempts,
        }
    return rows


def self_check() -> audit.JsonObject:
    checks = {}
    with tempfile.TemporaryDirectory(
        prefix="alfred_external_cancellation_"
    ) as temporary:
        database = Path(temporary) / "fixture.sqlite"
        native = {
            "status": "running",
            "fold_sha256": "synthetic",
            "command": ["fixture"],
        }
        original = copy.deepcopy(native)
        derived = {
            **native,
            "status": "interrupted",
            "record_origin": "derived_external_user_cancellation",
            "native_checkpoint": {"status": "running"},
            "finished_at_utc": None,
        }
        attempt = EpisodeAttempt(
            episode_id="synthetic-only",
            status="interrupted",
            artifact_dir=str(Path(temporary) / "episode"),
            score=None,
            error="External user cancellation; native terminal archive was not written",
        )
        ingest_run(
            database, run_id="synthetic-cancel", metadata=derived, attempts=[attempt]
        )
        with sqlite3.connect(database) as connection:
            row = connection.execute(
                "SELECT status,score_json FROM attempts"
            ).fetchone()
            metadata = json.loads(
                connection.execute("SELECT metadata_json FROM runs").fetchone()[0]
            )
        checks["null_score_and_interrupted_attempt"] = row == ("interrupted", None)
        checks["raw_running_and_derived_interrupted_both_explicit"] = (
            metadata["status"] == "interrupted"
            and metadata["native_checkpoint"]["status"] == "running"
        )
        checks["unknown_finish_time_stays_null"] = metadata["finished_at_utc"] is None
        checks["source_metadata_not_modified"] = native == original
        try:
            ingest_run(
                database,
                run_id="synthetic-cancel",
                metadata=derived,
                attempts=[attempt],
            )
        except ValueError:
            checks["existing_history_cannot_be_overwritten"] = True
        else:
            checks["existing_history_cannot_be_overwritten"] = False
    audit.require(all(checks.values()), "External cancellation fixture failed")
    return {
        "passed": True,
        "checks": checks,
        "scope": "Temporary fixture database only",
    }


def main() -> None:
    prepared_path = WORK / "external_cancellation_records_01.json"
    verification_path = WORK / "external_cancellation_sqlite_01.json"
    revised_path = WORK / "pilot5_excluded_user_cancellations_v2.json"
    for path in (prepared_path, verification_path, revised_path):
        audit.require(
            not path.exists(), "A new external cancellation record path is required"
        )
    test_result = self_check()
    cancelled_path = WORK / "pilot5_excluded_user_cancellations.json"
    cancelled = audit.read_json(cancelled_path)
    final_path = WORK / "pilot5_final_audit_02.json"
    final = audit.read_json(final_path)
    receipt_path = WORK / "queue_cleanup_final.json"
    receipt = audit.read_json(receipt_path)
    exit_path = WORK / "pilot5_v9_process_exit_check.json"
    exit_proof = audit.read_json(exit_path)
    audit.require(
        exit_proof["passed"]
        and exit_proof["matching_runner_or_worker_processes"] == [],
        "Process exit proof does not establish stopped workers",
    )
    receipt_ids = {entry["run_id"] for entry in receipt["extra_task_interrupts"]}
    expected_ids = {f"ALFRED_V9_20261105_pilot15_i{index:02d}" for index in INDICES}
    audit.require(receipt_ids == expected_ids, "External receipt identities differ")
    scored_ids = [entry["run_id"] for entry in final["episodes"]]
    prepared: list[PreparedRecord] = []
    immutable_files: dict[str, str] = {}
    for index in INDICES:
        run_id = f"ALFRED_V9_20261105_pilot15_i{index:02d}"
        source_row = next(
            row for row in cancelled["episodes"] if row["run_id"] == run_id
        )
        directory = ROOT / "outputs/alfred" / run_id
        raw_run_path, raw_summary_path = (
            directory / "run.json",
            directory / "summary.json",
        )
        native = audit.read_json(raw_run_path)
        summary = audit.read_json(raw_summary_path)
        audit.require(
            native["status"] == "running"
            and summary["pending"] == 1
            and summary["attempts"] == [],
            "Native checkpoint unexpectedly finalized",
        )
        audit.require(
            audit.configuration_signature(native) == final["configuration"],
            "Cancelled actor configuration differs from selected cohort",
        )
        for filename, expected_hash in source_row["source_sha256"].items():
            audit.require(
                audit.sha256(Path(filename)) == expected_hash,
                "Raw cancellation evidence changed",
            )
            immutable_files[filename] = expected_hash
        artifact = Path(source_row["artifact_dir"])
        audit.require(
            not (artifact / "attempt.json").exists()
            and not (artifact / "execution.json").exists()
            and not (artifact / "simulator/terminal_evaluation.json").exists(),
            "Native cancellation unexpectedly has terminal evidence",
        )
        usage = audit.audit_usage(artifact, completed=False)
        audit.require(usage == source_row["usage"], "Saved cancellation usage changed")
        description = "ExternalUserCancellation: original pilot15 scope was reduced to original-prefix5. This extra run was externally interrupted; the native runner did not write terminal metadata or an attempt record. Its original run.json remains running/pending. No official score was produced."
        attempt = EpisodeAttempt(
            episode_id=source_row["episode_id"],
            status="interrupted",
            artifact_dir=str(artifact),
            score=None,
            error=description,
        )
        metadata = copy.deepcopy(native)
        metadata.update(
            {
                "status": "interrupted",
                "error": description,
                "finished_at_utc": None,
                "record_origin": "derived_external_user_cancellation",
                "included_in_requested_prefix5": False,
                "included_in_infrastructure_error_count": False,
                "external_cancellation": {
                    "classification": "user_requested_scope_reduction_cancellation",
                    "registered_at_utc": utc_now(),
                    "receipt_at_utc": datetime.fromtimestamp(
                        receipt["at_unix"], timezone.utc
                    ).isoformat(),
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": audit.sha256(receipt_path),
                    "receipt_source": "parent queue cleanup after initial supervisor suspension did not persist",
                    "process_exit_verified_at_utc": exit_proof["checked_at_utc"],
                    "process_exit_proof_path": str(exit_path),
                    "process_exit_proof_sha256": audit.sha256(exit_path),
                    "native_terminal_archive_written": False,
                    "raw_run_json_path": str(raw_run_path),
                    "raw_run_json_sha256": audit.sha256(raw_run_path),
                    "raw_run_status": native["status"],
                    "raw_summary_json_path": str(raw_summary_path),
                    "raw_summary_json_sha256": audit.sha256(raw_summary_path),
                    "raw_summary_counts": source_row["raw_summary_counts"],
                    "native_attempt_json_present": False,
                    "exact_finish_at_utc": None,
                    "exact_finish_time_note": "Receipt and exit-verification timestamps bracket cancellation handling; exact process exit time was not recorded",
                    "partial_accounting": {
                        **source_row["counts"],
                        "reported_usage_lower_bound": usage[
                            "reported_usage_lower_bound"
                        ],
                        "exact_saved_request_input_tokens": usage["exact_input_tokens"],
                        "exact_saved_request_output_tokens": usage[
                            "exact_output_tokens"
                        ],
                        "unanswered_model_requests": usage["unanswered_decisions"],
                        "unknown_usage_decisions": usage["unknown_usage_decisions"],
                        "wall_seconds": None,
                        "policy_seconds": None,
                        "perception_seconds": None,
                    },
                },
            }
        )
        prepared.append(
            {
                "run_id": run_id,
                "metadata": metadata,
                "attempt": attempt.model_dump(mode="json"),
            }
        )
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as connection:
        before_scored = task_rows(connection, scored_ids)
        existing = task_rows(connection, sorted(expected_ids))
    audit.require(
        all(
            row["metadata_json"] is None and row["attempts"] == []
            for row in existing.values()
        ),
        "A target cancellation run already exists in SQLite",
    )
    prepared_payload = {
        "record_origin": "derived_external_user_cancellation",
        "created_at_utc": utc_now(),
        "source_cancellation_report_path": str(cancelled_path),
        "source_cancellation_report_sha256": audit.sha256(cancelled_path),
        "self_checks": test_result,
        "helper_sha256": audit.sha256(Path(__file__)),
        "accounting_helper_sha256": audit.sha256(Path(audit.__file__)),
        "records": prepared,
    }
    save_new(prepared_path, prepared_payload)
    for record in prepared:
        ingest_run(
            DATABASE,
            run_id=record["run_id"],
            metadata=record["metadata"],
            attempts=[EpisodeAttempt.model_validate(record["attempt"])],
        )
    verifications = []
    for record in prepared:
        check = audit.sqlite_check(
            DATABASE,
            record["run_id"],
            record["metadata"],
            [record["attempt"]],
            allow_missing=False,
        )
        verifications.append(
            {
                **check,
                "status": "interrupted",
                "score_is_null": True,
                "record_origin": "derived_external_user_cancellation",
                "native_artifact_status": "running",
            }
        )
    with sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True) as connection:
        after_scored = task_rows(connection, scored_ids)
    audit.require(before_scored == after_scored, "Official first5 SQLite rows changed")
    audit.require(
        all(
            audit.sha256(Path(name)) == expected_hash
            for name, expected_hash in immutable_files.items()
        ),
        "A native cancellation artifact changed",
    )
    verification = {
        "checked_at_utc": utc_now(),
        "passed": True,
        "records": verifications,
        "official_first5_sqlite_rows_unchanged": True,
        "native_artifact_hashes_unchanged": True,
        "prepared_records_path": str(prepared_path),
        "prepared_records_sha256": audit.sha256(prepared_path),
        "helper_sha256": audit.sha256(Path(__file__)),
        "database": str(DATABASE),
    }
    save_new(verification_path, verification)
    revised = copy.deepcopy(cancelled)
    revised["checked_at_utc"] = utc_now()
    revised["status"] = "verified_user_cancellations_with_external_sqlite_records"
    revised["previous_report_path"] = str(cancelled_path)
    revised["previous_report_sha256"] = audit.sha256(cancelled_path)
    revised["external_sqlite_ingestion"] = {
        "records": 3,
        "verification_path": str(verification_path),
        "verification_sha256": audit.sha256(verification_path),
        "method": "Existing ingest_run/EpisodeAttempt; interrupted with score NULL and explicit derived external receipt metadata",
    }
    by_run = {row["run_id"]: row for row in verifications}
    for row in revised["episodes"]:
        if row["run_id"] in by_run:
            row["sqlite"] = by_run[row["run_id"]]
            row["sqlite_native_terminal_archive_matches"] = False
            row["sqlite_derivation_note"] = (
                "Derived external cancellation record; original running/pending files remain unchanged"
            )
    revised["limits"][
        0
    ] = "Raw artifacts remain unchanged: three cancelled runs retain running/pending checkpoints. SQLite now separately records their external user cancellation with raw hashes, receipt/exit proof, null scores and unknown timing."
    save_new(revised_path, revised)
    print(
        json.dumps(
            {
                "inserted_external_cancellations": 3,
                "scores_inserted": 0,
                "first5_unchanged": True,
                "raw_artifacts_unchanged": True,
                "verification": str(verification_path),
                "updated_report": str(revised_path),
            }
        )
    )


if __name__ == "__main__":
    main()
