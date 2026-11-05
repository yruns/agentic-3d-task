"""Audit recovery3: 50 unique scores and nine separately retained error prefixes.

Only saved artifacts and SQLite are read. Historical HTTP errors are compared by
exact UTF-8 fingerprints and replaced with fixed safe labels in new reports.
Final output requires all nine prespecified recovery jobs to have complete scores.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyze_pilot50 import ROOT, read_json
from analyze_recovered_pilot50 import (
    RECOVERY_INDICES,
    CompletedCohort,
    SavedRun,
    SelectedEpisode,
    _database_check,
    sha256,
    write_new_report,
)
from analyze_recovered_pilot50 import build_report as build_artifact_report
from analyze_recovered_pilot50_v2 import PRIOR_FAILED_INDICES, read_singleton
from audit_original_pilot41 import collect_original41
from rescore_pilot50 import require

from codex_agent.alfred.dataset import AlfredEpisode, load_manifest

HISTORY_ROLES = ("original_direct", "recovery1_failed", "recovery2_failed")
SAFE_403 = "CodexTurnError: HTTP403; ROW code5004; DetectAffiliation dependency context deadline exceeded; system processing_error."
SAFE_502 = (
    "CodexTurnError: HTTP502; upstream unavailable; failed turn has no final response."
)


@dataclass(frozen=True)
class HistoricalCohort:
    """Validated immutable history, plus the selected future-job declarations."""

    manifest: dict[str, Any]
    references: tuple[AlfredEpisode, ...]
    originals: tuple[SelectedEpisode, ...]
    errors: tuple[SelectedEpisode, ...]
    runs: dict[str, SavedRun]
    lineage: tuple[dict[str, str], ...]


def safe_error(error: object) -> dict[str, Any]:
    """Return a fixed classification and hash without propagating upstream text."""
    require(isinstance(error, str) and bool(error), "missing historical error")
    assert isinstance(error, str)
    if "403" in error and "5004" in error and "DetectAffiliation" in error:
        classification = SAFE_403
    elif "502" in error:
        classification = SAFE_502
    else:
        classification = (
            "Upstream turn failed; raw error retained only in original artifacts."
        )
    return {
        "error": classification,
        "error_sha256": hashlib.sha256(error.encode("utf-8")).hexdigest(),
        "error_redacted": True,
        "raw_error_text_in_report": False,
    }


def read_lineage(path: Path) -> tuple[list[dict[str, Any]], tuple[dict[str, str], ...]]:
    """Read revision3 -> revision2 -> original, verifying every immutable link."""
    documents: list[dict[str, Any]] = []
    lineage: list[dict[str, str]] = []
    for expected_revision in (3, 2, 1):
        document = read_json(path)
        require(
            document["schema_version"] == 1
            and document.get("cohort_revision", 1) == expected_revision,
            "unexpected cohort revision chain",
        )
        documents.append(document)
        lineage.append({"path": str(path.resolve()), "sha256": sha256(path)})
        if expected_revision > 1:
            prior = document["previous_manifest"]
            path = Path(prior["path"])
            require(sha256(path) == prior["sha256"], "previous manifest changed")
    for previous in documents[1:]:
        for field in ("fold", "code", "common_configuration", "source_identity_policy"):
            require(documents[0][field] == previous[field], f"cohort changed {field}")
    return documents, tuple(lineage)


def historical_record_for_reader(record: dict[str, Any]) -> dict[str, Any]:
    """Adapt redacted manifest evidence to the preserved singleton validator.

    Raw errors exist only in this in-memory comparison. The source hashes and
    declared error fingerprint are verified before this adapter is constructed.
    """
    require(len(record["unscored_attempts"]) == 1, "expected one historical error")
    prefix = record["unscored_attempts"][0]
    if prefix.get("error_redacted") is not True:
        return record
    run_path, summary_path = Path(record["run_json"]), Path(record["summary_json"])
    require(
        sha256(run_path) == record["run_json_sha256"]
        and sha256(summary_path) == record["summary_json_sha256"],
        "redacted historical source hashes changed",
    )
    metadata, summary = read_json(run_path), read_json(summary_path)
    require(len(summary["attempts"]) == 1, "redacted source is not singleton")
    attempt = summary["attempts"][0]
    digest = safe_error(attempt["error"])["error_sha256"]
    require(
        prefix["error_sha256"] == digest
        and safe_error(metadata["error"])["error_sha256"] == digest
        and prefix["error"] == SAFE_403
        and prefix.get("score") is None,
        "redacted historical error fingerprint/classification differs",
    )
    expected = {
        "global_index": record["selected_global_indices"][0],
        **{key: attempt[key] for key in ("episode_id", "status", "artifact_dir")},
    }
    require(
        all(prefix[key] == value for key, value in expected.items()),
        "redacted historical attempt identity differs",
    )
    adapted = copy.deepcopy(record)
    adapted["unscored_attempts"] = [{**expected, "error": attempt["error"]}]
    return adapted


def load_history(path: Path, database: Path) -> HistoricalCohort:
    """Check all ten historical runs and all future declarations without final scores."""
    documents, lineage = read_lineage(path)
    manifest, previous, initial = documents
    originals, errors, runs, _ = collect_original41(initial, database)
    original_ids = [entry["run_id"] for entry in initial["original_runs"]]
    require(manifest["original_direct_run_ids"] == original_ids, "original IDs differ")
    count_fields = (
        "original_scored_count",
        "original_unscored_error_count",
        "original_unattempted_count",
        "recovery_planned_count",
        "historical_error_attempt_count",
        "historical_error_unique_identity_count",
        "currently_recorded_actual_runs",
        "currently_recorded_attempts",
        "future_actual_runs_if_complete",
        "future_actual_attempts_if_complete",
    )
    require(
        tuple(manifest[field] for field in count_fields)
        == (41, 3, 6, 9, 9, 5, 10, 50, 19, 59),
        "revision3 expected counts differ",
    )
    history = manifest["original_runs"]
    history_by_id = {entry["run_id"]: entry for entry in history}
    require(
        len(history) == len(history_by_id) == 10
        and Counter(entry["history_role"] for entry in history)
        == {"original_direct": 4, "recovery1_failed": 3, "recovery2_failed": 3},
        "historical membership/roles differ",
    )
    for old in previous["original_runs"]:
        require(
            history_by_id.get(old["run_id"]) == old, "previous historical lock changed"
        )
    for old in initial["original_runs"]:
        current = history_by_id[old["run_id"]]
        require(
            current["history_role"] == "original_direct"
            and {key: value for key, value in current.items() if key != "history_role"}
            == old,
            "original locked record changed",
        )
    fold = manifest["fold"]
    references = load_manifest(
        Path(fold["path"]), Path(fold["data_root"]), Path(fold["splits_path"])
    )
    failed = [entry for entry in history if entry["history_role"] != "original_direct"]
    require(
        len(manifest["failed_recovery_run_ids"]) == 6
        and set(manifest["failed_recovery_run_ids"])
        == {entry["run_id"] for entry in failed},
        "failed recovery IDs differ",
    )
    failed_by_role: dict[str, dict[int, str]] = {}
    for role, predecessor in (
        ("recovery1_failed", initial),
        ("recovery2_failed", previous),
    ):
        records = [entry for entry in failed if entry["history_role"] == role]
        jobs = {job["run_id"]: job for job in predecessor["recovery_runs"]}
        require(
            len(records) == 3
            and {record["selected_global_indices"][0] for record in records}
            == PRIOR_FAILED_INDICES,
            "failed recovery round membership differs",
        )
        failed_by_role[role] = {}
        for record in records:
            prior = jobs[record["run_id"]]
            index = record["selected_global_indices"][0]
            require(prior["global_index"] == index, "historical job index differs")
            require(
                all(
                    record[field] == prior[field]
                    for field in ("output_dir", "run_json", "summary_json")
                ),
                "historical job path differs from previous plan",
            )
            adapted = historical_record_for_reader(record)
            saved, item = read_singleton(
                adapted, manifest, references, historical_error=True
            )
            require(
                saved.metadata["command"] == prior["command"],
                "historical command changed",
            )
            require(item.run_id not in runs, "historical run duplicated")
            runs[item.run_id] = saved
            errors.append(item)
            failed_by_role[role][index] = item.run_id
    require(
        len(errors) == 9
        and len({item.reference.episode_id for item in errors}) == 5
        and len(runs) == 10
        and sum(len(run.summary["attempts"]) for run in runs.values()) == 50,
        "historical attempts were dropped or deduplicated",
    )
    jobs = manifest["recovery_runs"]
    require(
        len(jobs) == len({job["run_id"] for job in jobs}) == 9
        and {job["global_index"] for job in jobs} == RECOVERY_INDICES,
        "selected recovery membership differs",
    )
    previous_jobs = {job["global_index"]: job for job in previous["recovery_runs"]}
    for job in jobs:
        index = job["global_index"]
        require(job["run_id"] not in runs, "selected run replaces history")
        require(
            job["episode_id"] == references[index].episode_id,
            "planned identity differs",
        )
        origin = runs[job["origin_run_id"]]
        require(
            job["origin_run_id"] in original_ids
            and origin.metadata["shard_index"] == index % 4,
            "origin is not the original direct partition",
        )
        origin_attempt = next(
            (
                entry
                for entry in origin.summary["attempts"]
                if entry["episode_id"] == references[index].episode_id
            ),
            None,
        )
        if job["origin_status"] == "error":
            require(
                origin_attempt is not None
                and origin_attempt["status"] == "error"
                and origin_attempt["score"] is None
                and origin_attempt["error"] == job["origin_error"]
                and origin_attempt["artifact_dir"]
                == job["origin_attempt_artifact_dir"],
                "original error lineage differs",
            )
        else:
            require(
                job["origin_status"] == "pending" and origin_attempt is None,
                "original pending lineage differs",
            )
        if index in PRIOR_FAILED_INDICES:
            expected_ids = [failed_by_role[role][index] for role in HISTORY_ROLES[1:]]
            require(
                job["recovery_attempt_number"] == 3
                and job["previous_failed_run_id"] == expected_ids[-1]
                and job["prior_failed_run_ids"] == expected_ids,
                "third recovery lineage differs",
            )
        else:
            require(
                job["recovery_attempt_number"] == 1
                and job.get("previous_failed_run_id") is None
                and job.get("prior_failed_run_ids", []) == []
                and job["run_id"] == previous_jobs[index]["run_id"]
                and job["command"] == previous_jobs[index]["command"],
                "unused first recovery identity/command changed",
            )
    return HistoricalCohort(
        manifest, references, tuple(originals), tuple(errors), runs, lineage
    )


def load_completed_cohort(path: Path, database: Path) -> CompletedCohort:
    """Require all nine new scores and bind exactly one score per frozen identity."""
    history = load_history(path, database)
    manifest, runs = history.manifest, dict(history.runs)
    selected = {item.global_index: item for item in history.originals}
    jobs = {job["global_index"]: job for job in manifest["recovery_runs"]}
    for index, job in jobs.items():
        require(index not in selected, "retry would replace original score")
        saved, item = read_singleton(
            job, manifest, history.references, historical_error=False
        )
        runs[item.run_id] = saved
        selected[index] = item
    require(set(selected) == set(range(50)) and len(runs) == 19, "cohort incomplete")
    require(len(manifest["episodes"]) == 50, "canonical selection length differs")
    for index, entry in enumerate(manifest["episodes"]):
        item = selected[index]
        require(
            entry["global_index"] == index
            and entry["episode_id"] == item.reference.episode_id
            and entry["selected_source_run_id"] == item.run_id
            and Path(entry["expected_artifact_dir"]).resolve(strict=True)
            == item.directory
            and Path(entry["selected_source_output_dir"]).resolve()
            == runs[item.run_id].directory,
            "canonical selected source mismatch",
        )
        for field in (
            "split",
            "task",
            "repeat_idx",
            "task_type",
            "trajectory_sha256",
            "expert_actions",
        ):
            require(
                entry[field] == getattr(item.reference, field),
                f"canonical {field} differs",
            )
        if index in RECOVERY_INDICES:
            require(
                entry["selection_reason"]
                == "recovery_of_" + jobs[index]["origin_status"]
                and entry.get("previous_failed_run_id")
                == jobs[index].get("previous_failed_run_id"),
                "canonical recovery lineage differs",
            )
        else:
            require(
                entry["selection_reason"] == "original_completed",
                "original source changed",
            )
    require(
        sum(len(run.summary["attempts"]) for run in runs.values()) == 59,
        "expected 50 scores plus nine errors",
    )
    return CompletedCohort(
        path.resolve(),
        manifest,
        runs,
        tuple(selected[index] for index in range(50)),
        tuple(
            sorted(history.errors, key=lambda item: (item.global_index, item.run_id))
        ),
        _database_check(database, runs),
    )


def helper_hashes() -> dict[str, str]:
    return {
        name: sha256(Path(__file__).with_name(name))
        for name in (
            "analyze_recovered_pilot50_v3.py",
            "analyze_recovered_pilot50_v2.py",
            "analyze_recovered_pilot50.py",
            "audit_original_pilot41.py",
            "analyze_pilot50.py",
            "rescore_pilot50.py",
        )
    }


def build_report(cohort: CompletedCohort) -> dict[str, Any]:
    """Reuse detailed artifact accounting, redacting errors before any persistence."""
    report = build_artifact_report(cohort)
    historical = report.pop("unscored_original_attempts")
    roles = {
        entry["run_id"]: entry["history_role"]
        for entry in cohort.manifest["original_runs"]
    }
    for row in historical:
        row.update(safe_error(row.pop("error")), history_role=roles[row["run_id"]])
    report.update(
        cohort_revision=3,
        script_sha256=sha256(Path(__file__)),
        helper_script_sha256=helper_hashes(),
        manifest_lineage=list(read_lineage(cohort.manifest_path)[1]),
        previous_manifest=cohort.manifest["previous_manifest"],
        historical_error_attempt_count=9,
        historical_error_unique_identity_count=5,
        actual_run_count=19,
        actual_attempt_count=59,
        unscored_historical_attempts=historical,
        raw_upstream_errors_in_report=False,
        scope="Posthoc 50 unique scores from original41 + nine selected recoveries; retain nine unscored historical errors across19runs/59attempts. Unknown failed-turn usage remains unknown. No model or simulator invocation.",
    )
    report["unscored_prefix_totals"]["history_roles"] = dict(
        Counter(row["history_role"] for row in historical)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=ROOT / "tmp/alfred_setup/pilot50_cohort_recovery3.json",
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "docs/benchmark/alfred/runs.sqlite"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tmp/alfred_setup/pilot50_recovery3_analysis.json",
    )
    parser.add_argument(
        "--check-history-only",
        action="store_true",
        help="Validate immutable history and plan only; never emit final metrics.",
    )
    args = parser.parse_args()
    if args.check_history_only:
        history = load_history(args.cohort, args.database)
        print(
            json.dumps(
                {
                    "status": "history_validated_final_not_run",
                    "history_runs": len(history.runs),
                    "scored": len(history.originals),
                    "unscored_errors": len(history.errors),
                    "future_jobs": len(history.manifest["recovery_runs"]),
                    "manifest_lineage": list(history.lineage),
                },
                indent=2,
            )
        )
        return
    require(not args.output.exists(), f"output already exists: {args.output}")
    report = build_report(load_completed_cohort(args.cohort, args.database))
    write_new_report(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "selected",
                    "scored",
                    "metrics",
                    "by_split",
                    "by_task_type",
                    "scored_totals",
                    "unscored_prefix_totals",
                    "actual_run_count",
                    "actual_attempt_count",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
