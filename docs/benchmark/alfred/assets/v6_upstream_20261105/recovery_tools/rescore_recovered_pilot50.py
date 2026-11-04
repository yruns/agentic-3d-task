"""Independently rescore exactly 50 selected terminal snapshots after recovery.

Reuse the original hash-pinned official goal-method audit. No ThorEnv import,
task constructor, model loading, replay, or simulator startup occurs here.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from analyze_recovered_pilot50 import (
    compare_source_identity,
    load_completed_cohort,
    sha256,
    write_new_report,
)
from rescore_pilot50 import ALFRED_ROOT, ROOT, prepare_goals, require, rescore_one


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
        "--output",
        type=Path,
        default=ROOT / "tmp/alfred_setup/pilot50_recovered_terminal_rescore.json",
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
    mismatches = [row["episode_id"] for row in results if not row["matches"]]
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Evaluator-only independent terminal recheck of 41 original + 9 recovered unique frozen episodes; no replay or actor feedback.",
        "cohort_path": str(cohort.manifest_path),
        "cohort_sha256": sha256(cohort.manifest_path),
        "script_sha256": sha256(Path(__file__)),
        "cohort_loader_script_sha256": sha256(
            Path(__file__).with_name("analyze_recovered_pilot50.py")
        ),
        "official_scorer_script_sha256": sha256(
            Path(__file__).with_name("rescore_pilot50.py")
        ),
        "fold": cohort.manifest["fold"],
        "official_provenance": provenance,
        "expected": 50,
        "checked": len(results),
        "matching": len(results) - len(mismatches),
        "all_match": not mismatches,
        "mismatch_episode_ids": mismatches,
        "retained_unscored_original_attempts": [
            {
                "episode_id": item.reference.episode_id,
                "run_id": item.run_id,
                "score": None,
                "error": item.attempt["error"],
            }
            for item in cohort.error_prefixes
        ],
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
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
