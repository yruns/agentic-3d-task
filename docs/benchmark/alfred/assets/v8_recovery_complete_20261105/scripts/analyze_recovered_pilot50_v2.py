"""Audit revision-2 recovery: 50 scores plus all six historical error prefixes.

Reuse the prior artifact/usage checks, but retain three failed first recoveries
as additional locked history. Nothing is written until all nine selected
recovery jobs have complete scores. This script invokes no actor or simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_pilot50 import ROOT, read_json
from analyze_recovered_pilot50 import (
    RECOVERY_INDICES,
    CompletedCohort,
    SavedRun,
    SelectedEpisode,
    _database_check,
    _validate_summary,
    sha256,
    validate_run_configuration,
    write_new_report,
)
from analyze_recovered_pilot50 import (
    build_report as build_previous_report,
)
from audit_original_pilot41 import collect_original41
from rescore_pilot50 import require

from codex_agent.alfred.dataset import AlfredEpisode, load_manifest

PRIOR_FAILED_INDICES = {32, 36, 40}


def read_singleton(
    record: dict[str, Any],
    manifest: dict[str, Any],
    references: tuple[AlfredEpisode, ...],
    *,
    historical_error: bool,
) -> tuple[SavedRun, SelectedEpisode]:
    """Validate one immutable failed predecessor or one completed selected retry."""
    index = (
        record["selected_global_indices"][0]
        if historical_error
        else record["global_index"]
    )
    reference = references[index]
    run_path, summary_path = Path(record["run_json"]), Path(record["summary_json"])
    require(
        run_path.is_file() and summary_path.is_file(),
        f"selected recovery has not produced terminal artifacts: {record['run_id']}",
    )
    if historical_error:
        require(
            sha256(run_path) == record["run_json_sha256"]
            and sha256(summary_path) == record["summary_json_sha256"],
            "historical recovery artifacts changed",
        )
    run, summary = read_json(run_path), read_json(summary_path)
    directory = Path(record["output_dir"]).resolve(strict=True)
    require(
        run["run_id"] == record["run_id"]
        and Path(run["output_dir"]).resolve() == directory
        and run_path.resolve() == directory / "run.json"
        and summary_path.resolve() == directory / "summary.json",
        "singleton run identity/path mismatch",
    )
    require(
        run["status"] == ("error" if historical_error else "completed"),
        f"singleton is not eligible/terminal: {run['run_id']}: {run['status']}",
    )
    validate_run_configuration(run, manifest)
    require(
        run["num_shards"] == run["full_fold_size"] == 50
        and run["shard_index"] == index
        and run["limit"] is None
        and not run["diagnostic_prefix"],
        "singleton partition changed",
    )
    require(
        [entry["episode_id"] for entry in run["episodes"]] == [reference.episode_id],
        "singleton episode differs from frozen fold",
    )
    for field in ("split", "task_type", "trajectory_sha256", "expert_actions"):
        require(
            run["episodes"][0][field] == getattr(reference, field),
            f"singleton metadata mismatch: {field}",
        )
    saved = SavedRun(run, summary, directory)
    _validate_summary(saved, [reference.episode_id])
    require(
        summary["selected"] == len(summary["attempts"]) == 1
        and summary["pending"] == summary["interrupted"] == 0,
        "singleton must have exactly one terminal attempt",
    )
    attempt = summary["attempts"][0]
    if historical_error:
        expected = {
            "global_index": index,
            **{
                key: attempt[key]
                for key in ("episode_id", "status", "error", "artifact_dir")
            },
        }
        require(
            attempt["status"] == "error"
            and attempt["score"] is None
            and summary["errors"] == 1
            and summary["completed"] == 0
            and record["unscored_attempts"] == [expected]
            and record["completed_global_indices"]
            == record["pending_global_indices"]
            == [],
            "failed predecessor history differs",
        )
    else:
        require(
            run["command"] == record["command"]
            and record["episode_id"] == reference.episode_id,
            "selected recovery command/identity differs from plan",
        )
        require(
            attempt["status"] == "completed"
            and attempt["score"] is not None
            and summary["completed"] == 1
            and summary["errors"] == 0,
            "selected recovery has no complete score",
        )
    return saved, SelectedEpisode(
        reference,
        index,
        run["run_id"],
        Path(attempt["artifact_dir"]).resolve(strict=True),
        attempt,
    )


def load_completed_cohort(path: Path, database: Path) -> CompletedCohort:
    """Bind the original41 and nine selected scores while retaining six errors."""
    manifest = read_json(path)
    require(
        manifest["schema_version"] == 1 and manifest["cohort_revision"] == 2,
        "expected revision-2 cohort",
    )
    predecessor_path = Path(manifest["previous_manifest"]["path"])
    require(
        sha256(predecessor_path) == manifest["previous_manifest"]["sha256"],
        "previous manifest changed",
    )
    previous = read_json(predecessor_path)
    for field in ("fold", "code", "common_configuration", "source_identity_policy"):
        require(manifest[field] == previous[field], f"cohort changed {field}")
    originals, error_prefixes, runs, _ = collect_original41(previous, database)
    original_ids = [record["run_id"] for record in previous["original_runs"]]
    require(
        manifest["original_direct_run_ids"] == original_ids,
        "original direct run IDs differ",
    )
    counts = (
        manifest["original_scored_count"],
        manifest["original_unscored_error_count"],
        manifest["original_unattempted_count"],
        manifest["recovery_planned_count"],
        manifest["historical_error_attempt_count"],
        manifest["historical_error_unique_identity_count"],
        manifest["currently_recorded_actual_runs"],
        manifest["currently_recorded_attempts"],
        manifest["future_actual_runs_if_complete"],
        manifest["future_actual_attempts_if_complete"],
    )
    require(
        counts == (41, 3, 6, 9, 6, 5, 7, 47, 16, 56), "revision-2 history counts differ"
    )
    history = manifest["original_runs"]
    require(
        len(history) == 7 and len({entry["run_id"] for entry in history}) == 7,
        "expected seven unique historical runs",
    )
    history_by_id = {entry["run_id"]: entry for entry in history}
    for entry in previous["original_runs"]:
        current = history_by_id[entry["run_id"]]
        require(
            current["history_role"] == "original_direct"
            and {key: value for key, value in current.items() if key != "history_role"}
            == entry,
            "original locked record changed",
        )
    failed_ids = manifest["failed_recovery_run_ids"]
    failed_records = [
        entry for entry in history if entry["history_role"] == "recovery1_failed"
    ]
    require(
        len(failed_ids) == len(failed_records) == 3
        and set(failed_ids) == {entry["run_id"] for entry in failed_records}
        and {entry["selected_global_indices"][0] for entry in failed_records}
        == PRIOR_FAILED_INDICES,
        "first-recovery failure membership differs",
    )
    fold = manifest["fold"]
    references = load_manifest(
        Path(fold["path"]), Path(fold["data_root"]), Path(fold["splits_path"])
    )
    previous_jobs = {job["run_id"]: job for job in previous["recovery_runs"]}
    for record in failed_records:
        require(
            record["selected_global_indices"]
            == [previous_jobs[record["run_id"]]["global_index"]],
            "failed recovery was not in the previous plan",
        )
        for field in ("output_dir", "run_json", "summary_json"):
            require(
                record[field] == previous_jobs[record["run_id"]][field],
                "failed recovery path differs from previous plan",
            )
        saved, item = read_singleton(
            record, manifest, references, historical_error=True
        )
        runs[item.run_id] = saved
        error_prefixes.append(item)
    require(
        len(error_prefixes) == 6
        and len({item.reference.episode_id for item in error_prefixes}) == 5,
        "historical errors were dropped or deduplicated",
    )
    require(
        sum(len(run.summary["attempts"]) for run in runs.values()) == 47,
        "historical attempt count differs",
    )
    jobs = manifest["recovery_runs"]
    require(
        len(jobs) == 9
        and {job["global_index"] for job in jobs} == RECOVERY_INDICES
        and len({job["run_id"] for job in jobs}) == 9,
        "nine selected recovery jobs differ",
    )
    selected = {item.global_index: item for item in originals}
    jobs_by_index = {job["global_index"]: job for job in jobs}
    for job in jobs:
        index = job["global_index"]
        origin = runs[job["origin_run_id"]]
        require(
            job["origin_run_id"] in original_ids
            and origin.metadata["shard_index"] == index % 4,
            "recovery origin is not the original direct partition",
        )
        origin_attempt = next(
            (
                attempt
                for attempt in origin.summary["attempts"]
                if attempt["episode_id"] == references[index].episode_id
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
            prior_id = job["previous_failed_run_id"]
            require(
                job["recovery_attempt_number"] == 2
                and prior_id in failed_ids
                and runs[prior_id].metadata["shard_index"] == index,
                "failed-recovery lineage differs",
            )
        else:
            require(
                job["recovery_attempt_number"] == 1
                and job.get("previous_failed_run_id") is None,
                "unattempted singleton must retain its first-recovery identity",
            )
            require(
                job["run_id"] in previous_jobs
                and previous_jobs[job["run_id"]]["command"] == job["command"],
                "untouched recovery command/identity changed",
            )
        require(
            job["run_id"] not in runs and index not in selected,
            "selected retry would replace existing score/history",
        )
        saved, item = read_singleton(job, manifest, references, historical_error=False)
        runs[item.run_id] = saved
        selected[index] = item
    require(
        set(selected) == set(range(50)) and len(runs) == 16,
        "selected cohort is incomplete",
    )
    require(len(manifest["episodes"]) == 50, "canonical source list differs")
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
                f"canonical episode field changed: {field}",
            )
        if index in RECOVERY_INDICES:
            job = jobs_by_index[index]
            require(
                entry["selection_reason"] == "recovery_of_" + job["origin_status"]
                and entry.get("previous_failed_run_id")
                == job.get("previous_failed_run_id"),
                "canonical recovery lineage differs",
            )
        else:
            require(
                entry["selection_reason"] == "original_completed",
                "original selected-score provenance differs",
            )
    require(
        sum(len(run.summary["attempts"]) for run in runs.values()) == 56,
        "expected 50 scores and six separate error attempts",
    )
    return CompletedCohort(
        path.resolve(),
        manifest,
        runs,
        tuple(selected[index] for index in range(50)),
        tuple(
            sorted(error_prefixes, key=lambda item: (item.global_index, item.run_id))
        ),
        _database_check(database, runs),
    )


def build_report(cohort: CompletedCohort) -> dict[str, Any]:
    """Extend prior accounting without dropping or zero-filling failed recoveries."""
    report = build_previous_report(cohort)
    historical = report.pop("unscored_original_attempts")
    report.update(
        cohort_revision=2,
        script_sha256=sha256(Path(__file__)),
        artifact_audit_helper_sha256=sha256(
            Path(__file__).with_name("analyze_recovered_pilot50.py")
        ),
        original41_loader_sha256=sha256(
            Path(__file__).with_name("audit_original_pilot41.py")
        ),
        previous_manifest=cohort.manifest["previous_manifest"],
        historical_error_attempt_count=6,
        historical_error_unique_identity_count=5,
        actual_run_count=16,
        actual_attempt_count=56,
        unscored_historical_attempts=historical,
        scope="Posthoc 50 unique scores from original41 + nine selected recoveries; retain six unscored historical errors across16runs/56attempts. No model or simulator invocation.",
    )
    roles = {
        entry["run_id"]: entry["history_role"]
        for entry in cohort.manifest["original_runs"]
    }
    for row in historical:
        row["history_role"] = roles[row["run_id"]]
    report["unscored_prefix_totals"]["history_roles"] = {
        role: sum(row["history_role"] == role for row in historical)
        for role in ("original_direct", "recovery1_failed")
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=ROOT / "tmp/alfred_setup/pilot50_cohort_recovery2.json",
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "docs/benchmark/alfred/runs.sqlite"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tmp/alfred_setup/pilot50_recovery2_analysis.json",
    )
    args = parser.parse_args()
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
