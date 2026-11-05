"""Independent official terminal recheck for the revision-3 recovered cohort."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from analyze_recovered_pilot50 import compare_source_identity, sha256, write_new_report
from analyze_recovered_pilot50_v3 import (
    helper_hashes,
    load_completed_cohort,
    read_lineage,
    safe_error,
)
from rescore_pilot50 import ALFRED_ROOT, ROOT, prepare_goals, require, rescore_one


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=ROOT / "tmp/alfred_setup/pilot50_cohort_recovery3.json",
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "docs/benchmark/alfred/runs.sqlite"
    )
    parser.add_argument("--alfred-root", type=Path, default=ALFRED_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tmp/alfred_setup/pilot50_recovery3_terminal_rescore.json",
    )
    args = parser.parse_args()
    require(not args.output.exists(), f"output already exists: {args.output}")
    cohort = load_completed_cohort(args.cohort, args.database)
    goals, provenance = prepare_goals(args.alfred_root)
    results = [
        rescore_one(
            item.reference,
            item.directory / "simulator/terminal_evaluation.json",
            item.attempt["score"],
            item.run_id,
            goals,
        )
        for item in cohort.selected
    ]
    mismatches = [result["episode_id"] for result in results if not result["matches"]]
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Evaluator-only official terminal recheck of exactly50 unique scores. Nine historical errors remain unscored; unknown failed-turn usage remains unknown; no replay, model or simulator calls.",
        "cohort_revision": 3,
        "cohort_path": str(cohort.manifest_path),
        "cohort_sha256": sha256(cohort.manifest_path),
        "previous_manifest": cohort.manifest["previous_manifest"],
        "fold": cohort.manifest["fold"],
        "script_sha256": sha256(Path(__file__)),
        "helper_script_sha256": helper_hashes(),
        "manifest_lineage": list(read_lineage(cohort.manifest_path)[1]),
        "official_provenance": provenance,
        "expected": 50,
        "checked": len(results),
        "matching": len(results) - len(mismatches),
        "all_match": not mismatches,
        "mismatch_episode_ids": mismatches,
        "actual_run_count": 19,
        "actual_attempt_count": 59,
        "retained_unscored_historical_attempts": [
            {
                "episode_id": item.reference.episode_id,
                "run_id": item.run_id,
                "artifact_dir": str(item.directory),
                "score": None,
                **safe_error(item.attempt["error"]),
                "unknown_failed_turn_usage": None,
            }
            for item in cohort.error_prefixes
        ],
        "raw_upstream_errors_in_report": False,
        "database_audit": list(cohort.database_audit),
        "source_identity_policy": cohort.manifest["source_identity_policy"],
        "source_identity_audit": [
            {
                "run_id": run_id,
                **compare_source_identity(
                    run.metadata["code"],
                    cohort.manifest["code"],
                    cohort.manifest["source_identity_policy"],
                ),
            }
            for run_id, run in cohort.runs.items()
        ],
        "run_code_provenance": {
            run_id: run.metadata["code"] for run_id, run in cohort.runs.items()
        },
        "episodes": results,
    }
    write_new_report(args.output, report)
    print(
        json.dumps(
            {
                "checked": len(results),
                "matching": report["matching"],
                "all_match": not mismatches,
                "historical_error_attempts": len(cohort.error_prefixes),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
