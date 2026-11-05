"""Audit recovery3 without rewriting earlier scripts or exposing raw error text.

Use --history-only before completion to check historical provenance only.
The default final mode also runs the preserved generic 50-episode audit.
Neither mode calls a model, simulator, scorer, or service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit_pilot_boundary import BoundaryAudit, JsonObject, read_json, sha256
from audit_recovered_pilot_boundary import audit_recovered_cohort, validate_run


def error_digest(error: object) -> str | None:
    """Hash the unchanged UTF-8 error string, never its potentially sensitive text."""
    return (
        hashlib.sha256(error.encode("utf-8")).hexdigest()
        if isinstance(error, str)
        else None
    )


def validate_error_record(
    prefix: JsonObject,
    attempt: JsonObject,
    summary_attempt: JsonObject,
    metadata: JsonObject,
    audit: BoundaryAudit,
    path: Path,
) -> JsonObject:
    """Check the declared error fingerprint and return safe evidence only."""
    actual_digest = error_digest(attempt.get("error"))
    expected_digest = (
        prefix.get("error_sha256")
        if prefix.get("error_redacted") is True
        else error_digest(prefix.get("error"))
    )
    audit.check(
        actual_digest is not None
        and actual_digest == expected_digest
        and actual_digest == error_digest(summary_attempt.get("error"))
        and actual_digest == error_digest(metadata.get("error")),
        path,
        "historical_error_digest",
        "Attempt, summary, run and manifest must preserve the same original error fingerprint",
    )
    return {
        "error_sha256": actual_digest,
        "manifest_error_redacted": prefix.get("error_redacted") is True,
        "raw_error_text_in_report": False,
    }


def audit_history(manifest_path: Path, sdk_runs_root: Path) -> JsonObject:
    """Validate all ten historical runs, their nine errors, and preserved 41 scores."""
    manifest = read_json(manifest_path)
    audit = BoundaryAudit(sdk_runs_root)
    previous_path = Path(manifest["previous_manifest"]["path"])
    previous = read_json(previous_path)
    audit.check(
        manifest.get("cohort_revision") == 3
        and sha256(previous_path) == manifest["previous_manifest"]["sha256"],
        manifest_path,
        "revision_lineage",
        "Recovery3 must retain the hash-locked previous manifest",
    )
    audit.check(
        all(
            manifest[key] == previous[key]
            for key in (
                "fold",
                "code",
                "common_configuration",
                "source_identity_policy",
            )
        ),
        manifest_path,
        "frozen_configuration_lineage",
        "Recovery3 must retain the original fold, source and actor configuration",
    )
    histories = manifest["original_runs"]
    historical_ids = [item["run_id"] for item in histories]
    audit.check(
        len(histories) == len(set(historical_ids)) == 10
        and Counter(item["history_role"] for item in histories)
        == {"original_direct": 4, "recovery1_failed": 3, "recovery2_failed": 3},
        manifest_path,
        "historical_run_roles",
        "History must retain four original runs and both sets of three failed recoveries",
    )
    by_id = {item["run_id"]: item for item in histories}
    for old in previous["original_runs"]:
        current = by_id.get(old["run_id"], {})
        audit.check(
            all(
                current.get(key) == old[key]
                for key in ("run_json_sha256", "summary_json_sha256")
            ),
            manifest_path,
            "prior_history_locks",
            "All previously locked run and summary hashes must remain unchanged",
        )
    preserved = [
        item
        for item in previous["episodes"]
        if item["selection_reason"] == "original_completed"
    ]
    selected_by_index = {item["global_index"]: item for item in manifest["episodes"]}
    audit.check(
        len(preserved) == 41
        and all(
            selected_by_index.get(item["global_index"]) == item for item in preserved
        ),
        manifest_path,
        "preserved_41_selection",
        "All 41 previously scored episode selections must remain byte-equivalent JSON values",
    )
    prefix_reports: list[JsonObject] = []
    scored_count = 0
    for descriptor in histories:
        root, metadata, summary = validate_run(
            descriptor, manifest, audit, recovery=False
        )
        scored_count += summary["completed"]
        for prefix in descriptor["unscored_attempts"]:
            artifact = Path(prefix["artifact_dir"]).resolve(strict=True)
            attempt = read_json(artifact / "attempt.json")
            matching = [
                item
                for item in summary["attempts"]
                if item["episode_id"] == prefix["episode_id"]
            ]
            if not audit.check(
                len(matching) == 1
                and artifact.is_relative_to(root / "episodes")
                and attempt["episode_id"] == prefix["episode_id"]
                and attempt["status"] == prefix["status"] == "error"
                and attempt["score"] is None,
                artifact,
                "historical_error_identity",
                "Each historical error must retain its scoped episode identity and null score",
            ):
                continue
            prefix_reports.append(
                {
                    "run_id": descriptor["run_id"],
                    "history_role": descriptor["history_role"],
                    "episode_id": prefix["episode_id"],
                    "global_index": prefix["global_index"],
                    **validate_error_record(
                        prefix, attempt, matching[0], metadata, audit, artifact
                    ),
                }
            )
    audit.check(
        scored_count == 41
        and len(prefix_reports) == 9
        and len({(item["run_id"], item["episode_id"]) for item in prefix_reports}) == 9
        and len({item["episode_id"] for item in prefix_reports}) == 5,
        manifest_path,
        "historical_attempt_counts",
        "History must retain 41 scored episodes and nine distinct error attempts across five identities",
    )
    return {
        "scope": "Historical provenance only; actor boundary evidence is not audited by this preflight",
        "passed": not audit.violations,
        "historical_run_count": len(histories),
        "preserved_scored_count": scored_count,
        "historical_error_attempt_count": len(prefix_reports),
        "errors": prefix_reports,
        "violations": audit.violations,
    }


def main() -> int:
    setup = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=setup / "pilot50_cohort_recovery3.json"
    )
    parser.add_argument(
        "--sdk-runs-root", type=Path, default=setup.parents[1] / ".codex-home/runs"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite existing audit evidence")
    history = audit_history(args.manifest, args.sdk_runs_root)
    report: JsonObject = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_manifest_path": str(args.manifest.resolve()),
        "cohort_manifest_sha256": sha256(args.manifest),
        "history_only": args.history_only,
        "history_preflight": history,
        "passed": history["passed"],
    }
    if not args.history_only and history["passed"]:
        report.update(
            audit_recovered_cohort(args.manifest, sdk_runs_root=args.sdk_runs_root)
        )
    report["executed_script_hashes"] = {
        name: sha256(setup / name)
        for name in (
            "audit_recovery3_pilot_boundary.py",
            "audit_recovered_pilot_boundary.py",
            "audit_pilot_boundary.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "report": str(args.output.resolve()),
                "passed": report["passed"],
                "history_only": args.history_only,
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
