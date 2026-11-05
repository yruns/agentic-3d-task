"""Audit exactly the frozen original 41 scored episodes, with explicit 41/50 coverage.

Only existing local artifacts are read. Official goals are independently
recomputed without constructors, model calls, simulator startup, or replay.
The separate full-50 analyzers retain their completion gate unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_pilot50 import ROOT, read_json
from analyze_recovered_pilot50 import (
    SIGNATURE_FIELDS,
    SavedRun,
    SelectedEpisode,
    _database_check,
    _validate_summary,
    audit_episode,
    command_settings,
    sha256,
    write_new_report,
)
from rescore_pilot50 import ALFRED_ROOT, prepare_goals, require, rescore_one

from codex_agent.alfred.dataset import load_manifest
from codex_agent.alfred.metrics import EpisodeScore, aggregate_scores


def collect_original41(manifest: dict[str, Any], database: Path) -> tuple[
    list[SelectedEpisode],
    list[SelectedEpisode],
    dict[str, SavedRun],
    tuple[dict[str, object], ...],
]:
    """Require the exact original scored set and retain all three error prefixes."""
    require(manifest["schema_version"] == 1, "unsupported cohort schema")
    require(
        (
            manifest["original_scored_count"],
            manifest["original_unscored_error_count"],
            manifest["original_unattempted_count"],
        )
        == (41, 3, 6),
        "original cohort counts differ",
    )
    fold = manifest["fold"]
    require(
        fold["episode_count"] == 50 and sha256(Path(fold["path"])) == fold["sha256"],
        "frozen fold changed",
    )
    require(
        sha256(Path(fold["splits_path"])) == fold["splits_sha256"],
        "official split file changed",
    )
    references = load_manifest(
        Path(fold["path"]), Path(fold["data_root"]), Path(fold["splits_path"])
    )
    by_id = {
        reference.episode_id: (index, reference)
        for index, reference in enumerate(references)
    }
    require(
        len(references) == len(by_id) == 50, "fold must contain 50 unique identities"
    )
    records = manifest["original_runs"]
    require(len(records) == 4, "expected four original runs")
    selected: list[SelectedEpisode] = []
    prefixes: list[SelectedEpisode] = []
    runs: dict[str, SavedRun] = {}
    for shard, source in enumerate(records):
        metadata_path, summary_path = Path(source["run_json"]), Path(
            source["summary_json"]
        )
        require(
            sha256(metadata_path) == source["run_json_sha256"]
            and sha256(summary_path) == source["summary_json_sha256"],
            "original artifact hashes differ",
        )
        metadata, summary = read_json(metadata_path), read_json(summary_path)
        run_id = source["run_id"]
        directory = Path(source["output_dir"]).resolve(strict=True)
        require(
            metadata["run_id"] == run_id and run_id not in runs,
            "original run identity mismatch",
        )
        require(
            metadata_path.resolve() == directory / "run.json"
            and summary_path.resolve() == directory / "summary.json"
            and Path(metadata["output_dir"]).resolve() == directory,
            "original run paths differ",
        )
        require(
            metadata["status"] == source["status"]
            and metadata["status"] in {"completed", "error"},
            "original run is not final",
        )
        require(
            metadata["num_shards"] == 4
            and metadata["shard_index"] == shard
            and metadata["limit"] is None
            and not metadata["diagnostic_prefix"],
            "original partition changed",
        )
        signature = {key: metadata[key] for key in SIGNATURE_FIELDS}
        signature["cli_arguments_excluding_run_and_shard"] = command_settings(
            metadata["command"]
        )
        require(
            signature == manifest["common_configuration"]
            and metadata["code"] == manifest["code"],
            "original configuration/code mismatch",
        )
        expected_ids = [reference.episode_id for reference in references[shard::4]]
        require(
            [entry["episode_id"] for entry in metadata["episodes"]] == expected_ids,
            "run differs from frozen partition",
        )
        saved = SavedRun(metadata, summary, directory)
        _validate_summary(saved, expected_ids)
        completed_indices: list[int] = []
        for attempt in summary["attempts"]:
            index, reference = by_id[attempt["episode_id"]]
            source_entry = next(
                entry
                for entry in metadata["episodes"]
                if entry["episode_id"] == reference.episode_id
            )
            for field in ("split", "task_type", "trajectory_sha256", "expert_actions"):
                require(
                    source_entry[field] == getattr(reference, field),
                    f"episode metadata mismatch: {field}",
                )
            item = SelectedEpisode(
                reference,
                index,
                run_id,
                Path(attempt["artifact_dir"]).resolve(strict=True),
                attempt,
            )
            if attempt["status"] == "completed":
                selected.append(item)
                completed_indices.append(index)
            else:
                require(
                    attempt["status"] == "error" and attempt["score"] is None,
                    "unscored error became a synthetic score",
                )
                prefixes.append(item)
        require(
            completed_indices == source["completed_global_indices"],
            "original completed-index set differs",
        )
        runs[run_id] = saved
    selected.sort(key=lambda item: item.global_index)
    prefixes.sort(key=lambda item: item.global_index)
    canonical = [
        entry
        for entry in manifest["episodes"]
        if entry["selection_reason"] == "original_completed"
    ]
    require(
        len(selected) == len(canonical) == 41 and len(prefixes) == 3,
        "original scored/error coverage mismatch",
    )
    require(
        len({item.reference.episode_id for item in selected}) == 41,
        "duplicate original scored identity",
    )
    for item, entry in zip(selected, canonical, strict=True):
        require(
            item.global_index == entry["global_index"]
            and item.reference.episode_id == entry["episode_id"]
            and item.run_id == entry["selected_source_run_id"]
            and item.directory
            == Path(entry["expected_artifact_dir"]).resolve(strict=True),
            "scored source differs from canonical original41 selection",
        )
    require(
        {item.global_index for item in prefixes} == {32, 43, 45},
        "original failed-prefix identities differ",
    )
    require(
        sum(run.summary["pending"] for run in runs.values()) == 6,
        "original pending count differs",
    )
    return selected, prefixes, runs, _database_check(database, runs)


def sum_counters(
    rows: list[dict[str, Any]], key: str, *, decisions: bool = False
) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        total.update(row["decision_audit"][key] if decisions else row[key])
    return dict(total)


def describe_scored(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report task metrics and exact persisted usage only for this scored subset."""
    result = {
        "scored": len(rows),
        "metrics": aggregate_scores(
            [EpisodeScore.model_validate(row["score"]) for row in rows]
        ).model_dump(),
        "execution_totals": {
            field: sum(row["execution"][field] for row in rows)
            for field in (
                "actions",
                "failures",
                "decisions",
                "input_tokens",
                "output_tokens",
                "perception_seconds",
                "policy_seconds",
            )
        },
        "observations": sum(row["observations"] for row in rows),
        "reported_usage": sum_counters(rows, "reported_usage", decisions=True),
        "stop_reasons": dict(Counter(row["execution"]["stop_reason"] for row in rows)),
        "operations": sum_counters(rows, "operations", decisions=True),
        "action_types": sum_counters(rows, "action_types"),
        "failed_action_types": sum_counters(rows, "failed_action_types"),
        "failure_messages": sum_counters(rows, "failure_messages"),
        "non_action_feedback": sum_counters(
            rows, "non_action_feedback", decisions=True
        ),
    }
    for field in (
        "requests",
        "responses",
        "submitted_images",
        "submitted_historical_images",
        "sdk_attempts_observed",
    ):
        result[field] = sum(row["decision_audit"][field] for row in rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", type=Path, default=ROOT / "tmp/alfred_setup/pilot50_cohort.json"
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "docs/benchmark/alfred/runs.sqlite"
    )
    parser.add_argument("--alfred-root", type=Path, default=ALFRED_ROOT)
    parser.add_argument(
        "--rescore-output",
        type=Path,
        default=ROOT / "tmp/alfred_setup/pilot41_terminal_rescore.json",
    )
    parser.add_argument(
        "--accounting-output",
        type=Path,
        default=ROOT / "tmp/alfred_setup/pilot41_accounting.json",
    )
    args = parser.parse_args()
    require(
        args.rescore_output.resolve() != args.accounting_output.resolve(),
        "output paths must differ",
    )
    for path in (args.rescore_output, args.accounting_output):
        require(not path.exists(), f"output already exists; refusing overwrite: {path}")
    manifest = read_json(args.cohort)
    selected, prefixes, runs, database_audit = collect_original41(
        manifest, args.database
    )
    rows = [audit_episode(item, scored=True) for item in selected]
    prefix_rows = [audit_episode(item, scored=False) for item in prefixes]
    goals, official_provenance = prepare_goals(args.alfred_root)
    recomputed = [
        rescore_one(
            item.reference,
            item.directory / "simulator/terminal_evaluation.json",
            item.attempt["score"],
            item.run_id,
            goals,
        )
        for item in selected
    ]
    mismatches = [row["episode_id"] for row in recomputed if not row["matches"]]
    by_split = {
        value: describe_scored([row for row in rows if row["split"] == value])
        for value in sorted({row["split"] for row in rows})
    }
    by_task = {
        value: describe_scored([row for row in rows if row["task_type"] == value])
        for value in sorted({row["task_type"] for row in rows})
    }
    for split, report in by_split.items():
        report["fold_requested"] = sum(
            entry["split"] == split for entry in manifest["episodes"]
        )
    lower_bound_usage = sum_counters(prefix_rows, "reported_usage", decisions=True)
    known_total: Counter[str] = Counter(
        sum_counters(rows, "reported_usage", decisions=True)
    )
    known_total.update(lower_bound_usage)
    shared = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Posthoc local audit of exactly the manifest's original41 completed episodes. No model/simulator invocation, replay, source changes, or feedback to actors.",
        "coverage": {
            "fold_requested": 50,
            "scored": 41,
            "unscored_error_prefixes": 3,
            "unattempted": 6,
            "missing_score_global_indices": [
                entry["global_index"]
                for entry in manifest["episodes"]
                if entry["selection_reason"] != "original_completed"
            ],
        },
        "cohort_path": str(args.cohort.resolve()),
        "cohort_sha256": sha256(args.cohort),
        "fold": manifest["fold"],
        "matching_configuration": manifest["common_configuration"],
        "script_sha256": sha256(Path(__file__)),
        "helper_script_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "analyze_recovered_pilot50.py",
                "analyze_pilot50.py",
                "rescore_pilot50.py",
            )
        },
        "database_audit": list(database_audit),
    }
    rescore_report = {
        **shared,
        "status": "partial_coverage_verified" if not mismatches else "score_mismatch",
        "official_provenance": official_provenance,
        "checked": len(recomputed),
        "matching": len(recomputed) - len(mismatches),
        "all_match": not mismatches,
        "mismatch_episode_ids": mismatches,
        "episodes": recomputed,
    }
    accounting = {
        **shared,
        "status": "partial_coverage_verified",
        "scored_totals": describe_scored(rows),
        "by_split": by_split,
        "by_task_type": by_task,
        "unscored_prefix_totals": {
            "attempts": 3,
            "score": None,
            "actions": sum(row["execution"]["actions"] for row in prefix_rows),
            "failures": sum(row["execution"]["failures"] for row in prefix_rows),
            "decisions_started": sum(
                row["execution"]["decisions"] for row in prefix_rows
            ),
            "observations": sum(row["observations"] for row in prefix_rows),
            "reported_usage_lower_bound": lower_bound_usage,
            "unknown_failed_request_usage": None,
            "exact_total_consumed_usage": None,
        },
        "all_original_attempts_reported_usage_lower_bound": dict(known_total),
        "usage_caveat": "Scored totals exactly sum persisted SDK totals for the41 scored tasks. Original error prefixes add only observed lower bounds; final502/timeout usage is unknown, never imputed as zero. These are not full50 results.",
        "episodes": rows,
        "unscored_original_attempts": prefix_rows,
        "official_rescore_all_match": not mismatches,
    }
    for run_id, run in runs.items():
        if run.metadata["status"] == "completed":
            require(
                run.metadata["perception_statistics"]["frames"]
                == sum(row["observations"] for row in rows if row["run_id"] == run_id),
                "completed run frame count mismatch",
            )
    write_new_report(args.rescore_output, rescore_report)
    write_new_report(args.accounting_output, accounting)
    print(
        json.dumps(
            {
                "coverage": shared["coverage"],
                "official_checked": len(recomputed),
                "official_matching": len(recomputed) - len(mismatches),
                "all_match": not mismatches,
                "metrics": accounting["scored_totals"]["metrics"],
                "scored_execution_totals": accounting["scored_totals"][
                    "execution_totals"
                ],
                "reported_usage": accounting["scored_totals"]["reported_usage"],
                "unscored_prefix_totals": accounting["unscored_prefix_totals"],
                "rescore_output": str(args.rescore_output),
                "accounting_output": str(args.accounting_output),
            },
            indent=2,
        )
    )
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
