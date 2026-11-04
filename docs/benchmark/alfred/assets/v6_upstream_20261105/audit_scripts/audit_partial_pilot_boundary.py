"""Audit the original 41 scored episodes; explicitly retain incomplete 41/50 coverage.

Only a new report is written. This driver invokes no model, simulator, scorer,
service or evaluation, and reuses the preserved read-only boundary audit helpers.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_pilot_boundary import BoundaryAudit, JsonObject, read_json, sha256
from audit_recovered_pilot_boundary import (
    UnscoredPrefixAudit,
    canonical_episode_id,
    validate_run,
)


def main() -> int:
    setup = Path(__file__).resolve().parent
    project = setup.parents[1]
    output = setup / "pilot41_actor_boundary.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing snapshot: {output}")
    manifest_path = setup / "pilot50_cohort.json"
    cohort = read_json(manifest_path)
    frozen_fold_path = Path(cohort["fold"]["path"])
    frozen_fold = read_json(frozen_fold_path)
    selected = [
        entry
        for entry in cohort["episodes"]
        if entry["selection_reason"] == "original_completed"
    ]
    unscored = [
        entry
        for entry in cohort["episodes"]
        if entry["selection_reason"] != "original_completed"
    ]
    terminal = BoundaryAudit(project / ".codex-home/runs")
    prefixes = UnscoredPrefixAudit(project / ".codex-home/runs")
    prefixes.sdk_homes = terminal.sdk_homes
    terminal.check(
        sha256(frozen_fold_path) == cohort["fold"]["sha256"],
        frozen_fold_path,
        "fold_hash",
        "Original fold bytes must match the preserved cohort digest",
    )
    terminal.check(
        [entry["episode_id"] for entry in cohort["episodes"]]
        == [canonical_episode_id(sample) for sample in frozen_fold["samples"]],
        manifest_path,
        "frozen_identity",
        "Cohort identity order must match the original fold",
    )
    terminal.check(
        len(selected) == len({entry["episode_id"] for entry in selected}) == 41
        and len(unscored) == 9,
        manifest_path,
        "partial_cohort_size",
        "This audit specifically covers 41 scored of 50 original episodes, with nine unscored",
    )
    runs: dict[str, tuple[Path, JsonObject, JsonObject]] = {}
    run_reports: list[JsonObject] = []
    for descriptor in cohort["original_runs"]:
        root, metadata, summary = validate_run(
            descriptor, cohort, terminal, recovery=False
        )
        runs[descriptor["run_id"]] = root, metadata, summary
        run_reports.append(
            {
                "run_id": metadata["run_id"],
                "status": metadata["status"],
                "code": metadata["code"],
                "counts": {
                    key: summary[key]
                    for key in (
                        "selected",
                        "completed",
                        "errors",
                        "interrupted",
                        "pending",
                    )
                },
            }
        )
    scored_reports: list[JsonObject] = []
    for entry in selected:
        root, metadata, summary = runs[entry["selected_source_run_id"]]
        artifact = Path(entry["expected_artifact_dir"]).resolve(strict=True)
        if not terminal.check(
            artifact.is_relative_to(root / "episodes"),
            artifact,
            "artifact_scope",
            "Selected scored artifact must belong to the declared original run",
        ):
            continue
        attempt = read_json(artifact / "attempt.json")
        matching = [
            item
            for item in summary["attempts"]
            if item["episode_id"] == entry["episode_id"]
        ]
        terminal.check(
            attempt["status"] == "completed"
            and attempt["score"] is not None
            and attempt["episode_id"] == entry["episode_id"]
            and len(matching) == 1
            and matching[0]["score"] == attempt["score"],
            artifact,
            "scored_identity",
            "Selected artifact must match its scored run-summary identity",
        )
        before = terminal.counts.copy()
        terminal.episode(
            artifact,
            model=metadata["model"],
            step_instructions=metadata["execution"]["use_step_instructions"],
        )
        scored_reports.append(
            {
                "global_index": entry["global_index"],
                "episode_id": entry["episode_id"],
                "run_id": metadata["run_id"],
                "artifact_dir": str(artifact),
                "counts": dict(terminal.counts - before),
            }
        )
        print(
            json.dumps(
                {
                    "scored_episode_audited": entry["global_index"],
                    "progress": len(scored_reports),
                    "total_scored": 41,
                }
            ),
            flush=True,
        )
    prefix_reports: list[JsonObject] = []
    for descriptor in cohort["original_runs"]:
        root, metadata, summary = runs[descriptor["run_id"]]
        for entry in descriptor["unscored_attempts"]:
            artifact = Path(entry["artifact_dir"]).resolve(strict=True)
            if not prefixes.check(
                artifact.is_relative_to(root / "episodes"),
                artifact,
                "prefix_scope",
                "Error prefix must belong to its declared original run",
            ):
                continue
            attempt = read_json(artifact / "attempt.json")
            prefixes.check(
                attempt["status"] == "error"
                and attempt["score"] is None
                and attempt["episode_id"] == entry["episode_id"],
                artifact,
                "unscored_error",
                "Infrastructure errors remain unscored, not completed zero-score tasks",
            )
            before = prefixes.counts.copy()
            missing_start = len(prefixes.incomplete_requests)
            prefixes.episode(
                artifact,
                model=metadata["model"],
                step_instructions=metadata["execution"]["use_step_instructions"],
            )
            missing = prefixes.incomplete_requests[missing_start:]
            request_paths = sorted((artifact / "decisions").glob("*/request.json"))
            prefixes.check(
                len(missing) == 1
                and Path(missing[0]["request_path"]) == request_paths[-1],
                artifact,
                "unanswered_error_turn",
                "Each of these three upstream errors must retain its unanswered final request explicitly",
            )
            prefix_reports.append(
                {
                    "global_index": entry["global_index"],
                    "episode_id": entry["episode_id"],
                    "run_id": descriptor["run_id"],
                    "status": "error",
                    "score": None,
                    "included_in_scored_denominator": False,
                    "complete_execution_evidence": False,
                    "counts": dict(prefixes.counts - before),
                    "incomplete_decisions": missing,
                }
            )
            print(
                json.dumps(
                    {
                        "unscored_error_prefix_audited": entry["global_index"],
                        "unanswered_requests": len(missing),
                    }
                ),
                flush=True,
            )
    terminal.check(
        terminal.counts["episodes"] == 41
        and len(prefix_reports) == 3
        and len(prefixes.incomplete_requests) == 3,
        manifest_path,
        "audit_scope_counts",
        "Coverage must remain exactly 41 scored episodes plus three explicitly incomplete error prefixes",
    )
    artifacts = terminal.manifest + prefixes.manifest
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All original 41 scored episodes plus separately labelled three unscored infrastructure-error prefixes; incomplete 41/50 cohort coverage",
        "coverage_complete": False,
        "coverage": {
            "frozen_fold_size": 50,
            "scored_episodes_checked": 41,
            "unscored_error_prefixes_checked": 3,
            "pending_without_attempt": 6,
            "missing_scored_global_indices": [
                entry["global_index"] for entry in unscored
            ],
        },
        "available_boundary_checks_passed": not terminal.violations
        and not prefixes.violations,
        "cohort_manifest_sha256": sha256(manifest_path),
        "scripts": {
            name: sha256(setup / name)
            for name in (
                "audit_pilot_boundary.py",
                "audit_recovered_pilot_boundary.py",
                Path(__file__).name,
            )
        },
        "source_runs": run_reports,
        "scored_episode_audit": {
            "counts": dict(terminal.counts),
            "episodes": scored_reports,
            "violations": terminal.violations,
        },
        "unscored_error_prefix_audit": {
            "counts": dict(prefixes.counts),
            "prefixes": prefix_reports,
            "unanswered_request_count": len(prefixes.incomplete_requests),
            "available_boundary_violations": prefixes.violations,
        },
        "sdk_response_item_types": {
            "scored_episodes": dict(terminal.response_item_types),
            "unscored_prefixes": dict(prefixes.response_item_types),
        },
        "catalog_sha256_values": sorted(
            terminal.catalog_hashes | prefixes.catalog_hashes
        ),
        "artifact_manifest": artifacts,
        "artifact_manifest_sha256": hashlib.sha256(
            json.dumps(artifacts, sort_keys=True).encode()
        ).hexdigest(),
        "limits": [
            "41/50 coverage is explicitly incomplete; recovery jobs and the nine unscored identities are not represented as completed tasks.",
            "Three unanswered final SDK requests remain incomplete. Their failed SDK run-home catalog/trace cannot be joined without response metadata; they are not counted as completed decisions.",
            "Fresh-mask proof joins action source frame/memory identity to the corresponding prediction and saved memory NPZ; original simulator RPC mask arguments were not separately persisted.",
            "Catalogs check declared host-tool isolation and saved SDK traces check tool execution. Full on-wire tool lists were not captured for every pilot request.",
            "No model, simulator, scorer, runtime config, process environment, credentials or dataset trajectory contents are accessed by this audit.",
        ],
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "report": str(output),
                "coverage": report["coverage"],
                "available_boundary_checks_passed": report[
                    "available_boundary_checks_passed"
                ],
                "scored_counts": dict(terminal.counts),
                "prefix_counts": dict(prefixes.counts),
                "violations": len(terminal.violations) + len(prefixes.violations),
            }
        ),
        flush=True,
    )
    return 0 if report["available_boundary_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
