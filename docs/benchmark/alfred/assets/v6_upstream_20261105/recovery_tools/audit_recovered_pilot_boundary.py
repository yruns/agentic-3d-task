"""Audit the manifest-selected 50 scored episodes and separately identify error prefixes.

Run in tmux in agentic3d after all recovery jobs reach a terminal state::

    python tmp/alfred_setup/audit_recovered_pilot_boundary.py

This reuses the preserved read-only boundary checks in audit_pilot_boundary.py.
Only a new output report is written; no source, old snapshots, runtime config,
environment, credentials, simulator or model state is modified or queried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit_pilot_boundary import BoundaryAudit, JsonObject, read_json, sha256


class UnscoredPrefixAudit(BoundaryAudit):
    """Audit available inputs while explicitly recording unanswered decisions.

    A request without a response is never counted as a completed SDK decision.
    It is expected only as the final request of an infrastructure-error prefix,
    whose partial evidence is kept outside the 50 terminal scored episodes.
    """

    def __init__(self, sdk_runs_root: Path) -> None:
        super().__init__(sdk_runs_root)
        self.incomplete_requests: list[JsonObject] = []

    def check(self, condition: bool, path: Path, code: str, message: str) -> bool:
        if code == "missing_response" and not condition:
            self.incomplete_requests.append(
                {
                    "request_path": str(path),
                    "response_present": False,
                    "completed_decision": False,
                    "sdk_catalog_and_trace_status": "unverified: response metadata does not identify the failed SDK run home",
                }
            )
            return False
        return super().check(condition, path, code, message)


def canonical_episode_id(sample: JsonObject) -> str:
    return f"{sample['split']}/{sample['task']}#{sample['repeat_idx']}"


def normalized_cli_arguments(metadata: JsonObject) -> JsonObject:
    """Match the cohort's complete CLI signature without running the CLI main."""
    from codex_agent.cli.run_alfred import build_arg_parser

    arguments = vars(build_arg_parser().parse_args(metadata["command"][3:]))
    excluded = {"output_dir", "run_id", "num_shards", "shard_index"}
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in arguments.items()
        if key not in excluded
    }


def validate_source_identity(
    metadata: JsonObject, manifest: JsonObject, audit: BoundaryAudit, path: Path
) -> None:
    """Allow explicitly authorized HEAD-only drift while preserving executable identity."""
    policy = manifest["source_identity_policy"]
    required_true = (
        "require_same_branch",
        "require_clean_tracked_tree",
        "require_identical_executed_code_sha256",
        "actual_head_recorded_per_run",
    )
    valid_policy = (
        type(policy.get("version")) is int
        and policy["version"] == 1
        and type(policy.get("allow_head_sha_difference")) is bool
        and all(policy.get(field) is True for field in required_true)
    )
    audit.check(
        valid_policy,
        path,
        "source_identity_policy",
        "A versioned policy must preserve branch, clean tracked tree, executed digest and actual per-run HEAD",
    )
    actual, baseline = metadata["code"], manifest["code"]
    audit.check(
        actual["executed_code_sha256"] == baseline["executed_code_sha256"],
        path,
        "executed_code",
        "All attempts must execute the exact original source/config bytes",
    )
    audit.check(
        actual["branch"] == baseline["branch"],
        path,
        "branch_provenance",
        "All attempts must use the original branch",
    )
    audit.check(
        actual["tracked_worktree_drift"] == "",
        path,
        "worktree_provenance",
        "All attempts must start with a clean tracked working tree",
    )
    audit.check(
        isinstance(actual["head_sha"], str)
        and re.fullmatch(r"[0-9a-f]{40}", actual["head_sha"]) is not None,
        path,
        "actual_head",
        "Each attempt must retain its actual full Git HEAD SHA",
    )
    if not valid_policy or not policy["allow_head_sha_difference"]:
        audit.check(
            actual["head_sha"] == baseline["head_sha"],
            path,
            "head_provenance",
            "HEAD differences require explicit source-identity policy authorization",
        )


def validate_run(
    descriptor: JsonObject,
    manifest: JsonObject,
    audit: BoundaryAudit,
    *,
    recovery: bool,
) -> tuple[Path, JsonObject, JsonObject]:
    root = Path(descriptor["output_dir"]).resolve(strict=True)
    metadata = read_json(root / "run.json")
    summary = read_json(root / "summary.json")
    audit.check(
        metadata["run_id"] == descriptor["run_id"],
        root,
        "run_identity",
        "Run ID must match the cohort source descriptor",
    )
    audit.check(
        metadata["status"] in {"completed", "error", "interrupted"},
        root,
        "run_terminal_state",
        "All source and recovery runs must be terminal before the final audit",
    )
    if recovery:
        audit.check(
            metadata["status"] == "completed"
            and summary["selected"] == summary["completed"] == 1
            and summary["errors"] == summary["interrupted"] == summary["pending"] == 0,
            root,
            "recovery_complete",
            "Each selected recovery must finish exactly its one scored episode",
        )
    else:
        audit.check(
            sha256(root / "run.json") == descriptor["run_json_sha256"]
            and sha256(root / "summary.json") == descriptor["summary_json_sha256"],
            root,
            "original_artifact_hashes",
            "Original run and summary bytes are locked; their historical HEAD/status must never be rewritten",
        )
        audit.check(
            metadata["status"] == descriptor["status"],
            root,
            "original_terminal_state",
            "Original terminal status must match the preserved cohort record",
        )
    attempts = summary["attempts"]
    status_counts = Counter(attempt["status"] for attempt in attempts)
    audit.check(
        summary["selected"] == len(metadata["episodes"]),
        root,
        "selected_count",
        "Selected count must equal the original run episode list",
    )
    audit.check(
        summary["completed"] == status_counts["completed"]
        and summary["errors"] == status_counts["error"]
        and summary["interrupted"] == status_counts["interrupted"]
        and summary["pending"] == summary["selected"] - len(attempts),
        root,
        "summary_counts",
        "Summary counts must match actual completed/error/interrupted/pending attempts",
    )
    audit.check(
        len({attempt["episode_id"] for attempt in attempts}) == len(attempts),
        root,
        "duplicate_run_attempt",
        "A run must not contain duplicate episode attempts",
    )
    for attempt in attempts:
        completed = attempt["status"] == "completed"
        audit.check(
            completed == (attempt["score"] is not None),
            root,
            "attempt_score_boundary",
            "Only completed attempts may have scores; errors must retain null scores",
        )
    audit.check(
        (
            summary["metrics"] is None
            if summary["completed"] == 0
            else summary["metrics"]["num_evals"] == summary["completed"]
        ),
        root,
        "metric_denominator",
        "Metrics must use completed scored attempts only",
    )
    audit.check(
        metadata["fold_sha256"] == manifest["fold"]["sha256"]
        and metadata["splits_sha256"] == manifest["fold"]["splits_sha256"],
        root,
        "fold_provenance",
        "Every source must use the original frozen fold and split digests",
    )
    validate_source_identity(metadata, manifest, audit, root)
    common = manifest["common_configuration"]
    audit.check(
        common["code"] == manifest["code"],
        root,
        "baseline_code_consistency",
        "The preserved configuration baseline and source-policy baseline must agree",
    )
    for key, expected in common.items():
        if key in {"code", "cli_arguments_excluding_run_and_shard"}:
            continue
        audit.check(
            key in metadata and metadata[key] == expected,
            root,
            "matched_configuration",
            f"Configuration field {key} must match the cohort contract",
        )
    audit.check(
        normalized_cli_arguments(metadata)
        == common["cli_arguments_excluding_run_and_shard"],
        root,
        "matched_cli_configuration",
        "All CLI settings except output/run/shard identity must match the frozen configuration",
    )
    return root, metadata, summary


def audit_recovered_cohort(manifest_path: Path, *, sdk_runs_root: Path) -> JsonObject:
    manifest = read_json(manifest_path)
    terminal = BoundaryAudit(sdk_runs_root)
    prefixes = UnscoredPrefixAudit(sdk_runs_root)
    # Detect accidental SDK-home reuse even across selected terminal and old error attempts.
    prefixes.sdk_homes = terminal.sdk_homes
    fold_path = Path(manifest["fold"]["path"])
    fold = read_json(fold_path)
    samples = fold["samples"]
    expected_ids = [canonical_episode_id(sample) for sample in samples]
    terminal.check(
        len(expected_ids)
        == len(set(expected_ids))
        == manifest["fold"]["episode_count"]
        == 50,
        manifest_path,
        "frozen_cohort_size",
        "The original frozen fold must contain exactly 50 distinct episodes",
    )
    terminal.check(
        sha256(fold_path) == manifest["fold"]["sha256"],
        fold_path,
        "fold_file_hash",
        "Frozen fold bytes must match the cohort digest",
    )
    selected = manifest["episodes"]
    terminal.check(
        [entry["global_index"] for entry in selected] == list(range(50)),
        manifest_path,
        "global_index_order",
        "Canonical selected entries must preserve all original global indices in order",
    )
    terminal.check(
        [entry["episode_id"] for entry in selected] == expected_ids,
        manifest_path,
        "exact_fold_coverage",
        "Selected scoring identities must exactly cover the original frozen fold, with no substitutions",
    )
    for entry, sample in zip(selected, samples):
        terminal.check(
            entry["trajectory_sha256"] == sample["trajectory_sha256"],
            manifest_path,
            "trajectory_digest",
            "Selected episode provenance must match its frozen trajectory digest",
        )

    run_records: dict[str, tuple[Path, JsonObject, JsonObject]] = {}
    run_reports: list[JsonObject] = []
    all_scored_ids: list[str] = []
    unscored_actual: set[tuple[str, str]] = set()
    for recovery, group in (
        (False, manifest["original_runs"]),
        (True, manifest["recovery_runs"]),
    ):
        for descriptor in group:
            run_id = descriptor["run_id"]
            if not terminal.check(
                run_id not in run_records,
                manifest_path,
                "duplicate_source_run",
                "Source run IDs must be unique",
            ):
                continue
            root, metadata, summary = validate_run(
                descriptor, manifest, terminal, recovery=recovery
            )
            run_records[run_id] = root, metadata, summary
            all_scored_ids.extend(
                attempt["episode_id"]
                for attempt in summary["attempts"]
                if attempt["status"] == "completed"
            )
            unscored_actual.update(
                (run_id, attempt["episode_id"])
                for attempt in summary["attempts"]
                if attempt["status"] != "completed"
            )
            run_reports.append(
                {
                    "run_id": run_id,
                    "role": "recovery" if recovery else "original",
                    "status": metadata["status"],
                    "summary_counts": {
                        key: summary[key]
                        for key in (
                            "selected",
                            "completed",
                            "errors",
                            "interrupted",
                            "pending",
                        )
                    },
                    "code": metadata["code"],
                    "run_json_sha256": sha256(root / "run.json"),
                    "summary_json_sha256": sha256(root / "summary.json"),
                }
            )
    terminal.check(
        len(all_scored_ids) == len(set(all_scored_ids)) == 50
        and set(all_scored_ids) == set(expected_ids),
        manifest_path,
        "duplicate_or_missing_scored_identity",
        "Across all original and recovery sources there must be exactly one scored attempt for each original fold identity",
    )
    selected_artifacts: list[JsonObject] = []
    for entry in selected:
        run_id = entry["selected_source_run_id"]
        if not terminal.check(
            run_id in run_records,
            manifest_path,
            "unknown_selected_run",
            "Selected episode source must be a declared original or recovery run",
        ):
            continue
        root, metadata, summary = run_records[run_id]
        artifact = Path(entry["expected_artifact_dir"]).resolve(strict=True)
        if not terminal.check(
            artifact.is_relative_to(root / "episodes") and artifact.is_dir(),
            artifact,
            "selected_artifact_scope",
            "Selected episode artifacts must remain within their declared source run",
        ):
            continue
        terminal.check(
            Path(entry["selected_source_output_dir"]).resolve() == root,
            artifact,
            "selected_output_dir",
            "Selected source directory must match the declared run",
        )
        episode_metadata = [
            item
            for item in metadata["episodes"]
            if item["episode_id"] == entry["episode_id"]
        ]
        terminal.check(
            len(episode_metadata) == 1
            and episode_metadata[0]["trajectory_sha256"] == entry["trajectory_sha256"],
            artifact,
            "selected_run_trajectory",
            "The actually executed run episode must retain the frozen trajectory digest",
        )
        attempt = read_json(artifact / "attempt.json")
        terminal.check(
            attempt["episode_id"] == entry["episode_id"]
            and attempt["status"] == "completed"
            and attempt["score"] is not None,
            artifact,
            "selected_scored_attempt",
            "Each selected episode must be its declared completed scored attempt",
        )
        matching = [
            item
            for item in summary["attempts"]
            if item["episode_id"] == entry["episode_id"]
            and item["status"] == "completed"
        ]
        if not terminal.check(
            len(matching) == 1,
            artifact,
            "selected_summary_attempt",
            "Selected scored attempt must occur exactly once in its run summary",
        ):
            continue
        terminal.check(
            Path(matching[0]["artifact_dir"]).resolve() == artifact
            and matching[0]["score"] == attempt["score"],
            artifact,
            "selected_score_consistency",
            "Selected attempt artifact and score must match the run summary",
        )
        before = terminal.counts.copy()
        terminal.episode(
            artifact,
            model=metadata["model"],
            step_instructions=metadata["execution"]["use_step_instructions"],
        )
        selected_artifacts.append(
            {
                "global_index": entry["global_index"],
                "episode_id": entry["episode_id"],
                "run_id": run_id,
                "artifact_dir": str(artifact),
                "selection_reason": entry["selection_reason"],
                "counts": dict(terminal.counts - before),
            }
        )

    prefix_reports: list[JsonObject] = []
    unscored_declared: set[tuple[str, str]] = set()
    for original in manifest["original_runs"]:
        root, metadata, summary = run_records[original["run_id"]]
        for prefix in original["unscored_attempts"]:
            unscored_declared.add((original["run_id"], prefix["episode_id"]))
            artifact = Path(prefix["artifact_dir"]).resolve(strict=True)
            if not prefixes.check(
                artifact.is_relative_to(root / "episodes") and artifact.is_dir(),
                artifact,
                "prefix_scope",
                "Unscored prefixes must belong to their original run",
            ):
                continue
            attempt = read_json(artifact / "attempt.json")
            summary_prefixes = [
                item
                for item in summary["attempts"]
                if item["episode_id"] == prefix["episode_id"]
            ]
            prefixes.check(
                len(summary_prefixes) == 1
                and summary_prefixes[0]["status"] == prefix["status"]
                and summary_prefixes[0]["score"] is None
                and Path(summary_prefixes[0]["artifact_dir"]).resolve() == artifact,
                artifact,
                "prefix_summary_consistency",
                "Unscored prefix artifact, status and null score must match its run summary",
            )
            prefixes.check(
                attempt["status"] == prefix["status"]
                and attempt["status"] in {"error", "interrupted"}
                and attempt["score"] is None
                and attempt["episode_id"] == prefix["episode_id"],
                artifact,
                "unscored_prefix_status",
                "Old error prefixes remain explicitly unscored, never completed zeros",
            )
            before = prefixes.counts.copy()
            incomplete_before = len(prefixes.incomplete_requests)
            prefixes.episode(
                artifact,
                model=metadata["model"],
                step_instructions=metadata["execution"]["use_step_instructions"],
            )
            incomplete = prefixes.incomplete_requests[incomplete_before:]
            requests = sorted((artifact / "decisions").glob("*/request.json"))
            prefixes.check(
                len(incomplete) <= 1
                and (
                    not incomplete
                    or Path(incomplete[0]["request_path"]) == requests[-1]
                ),
                artifact,
                "incomplete_prefix_chronology",
                "Only the final request of an infrastructure-error prefix may lack a response",
            )
            prefix_reports.append(
                {
                    "run_id": original["run_id"],
                    "global_index": prefix["global_index"],
                    "episode_id": prefix["episode_id"],
                    "status": attempt["status"],
                    "score": None,
                    "included_in_50_terminal_episodes": False,
                    "artifact_dir": str(artifact),
                    "available_artifact_counts": dict(prefixes.counts - before),
                    "incomplete_decisions": incomplete,
                    "complete_execution_evidence": False,
                }
            )
    terminal.check(
        unscored_actual == unscored_declared,
        manifest_path,
        "unscored_prefix_coverage",
        "All original unscored attempts must be separately represented, with none entering terminal denominators",
    )
    terminal.check(
        terminal.counts["episodes"] == 50,
        manifest_path,
        "terminal_audit_count",
        "Exactly 50 selected terminal episode boundaries must have been audited",
    )
    combined_manifest = terminal.manifest + prefixes.manifest
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Manifest-selected exact frozen 50 scored episodes, plus separately labelled original unscored error prefixes",
        "cohort_manifest_path": str(manifest_path.resolve()),
        "cohort_manifest_sha256": sha256(manifest_path),
        "source_identity_policy": manifest["source_identity_policy"],
        "source_baseline": manifest["code"],
        "base_audit_script_sha256": sha256(
            Path(__file__).with_name("audit_pilot_boundary.py")
        ),
        "passed": not terminal.violations and not prefixes.violations,
        "source_runs": run_reports,
        "terminal_scored_cohort": {
            "episode_count": len(selected_artifacts),
            "counts": dict(terminal.counts),
            "episodes": selected_artifacts,
            "violations": terminal.violations,
        },
        "original_unscored_prefixes": {
            "prefix_count": len(prefix_reports),
            "included_in_50_terminal_denominator": False,
            "counts": dict(prefixes.counts),
            "unanswered_request_count": len(prefixes.incomplete_requests),
            "prefixes": prefix_reports,
            "available_boundary_violations": prefixes.violations,
        },
        "sdk_response_item_types": {
            "terminal": dict(terminal.response_item_types),
            "unscored_prefix": dict(prefixes.response_item_types),
        },
        "catalog_sha256_values": sorted(
            terminal.catalog_hashes | prefixes.catalog_hashes
        ),
        "artifact_manifest": combined_manifest,
        "artifact_manifest_sha256": hashlib.sha256(
            json.dumps(combined_manifest, sort_keys=True).encode()
        ).hexdigest(),
        "limits": [
            "The 50 terminal denominator contains only completed scored identities selected explicitly in the cohort manifest; errors and pending attempts are not zero-filled.",
            "Each unanswered original-prefix request is recorded separately as incomplete; without response metadata its failed SDK run home's catalog/trace cannot be verified by this join.",
            "Fresh-mask proof joins action source frame/memory identity to the corresponding prediction and saved memory NPZ; original simulator RPC mask arguments were not separately persisted.",
            "Catalogs verify declared host-tool isolation and preserved SDK traces verify no host-tool executions; full wire tool lists were not captured for every pilot request.",
            "No runtime config, process environment, credentials, simulator terminal metadata or dataset trajectory contents are read; image pixels are not re-rendered or semantically judged.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    setup_dir = Path(__file__).resolve().parent
    project = setup_dir.parents[1]
    parser.add_argument(
        "--manifest", type=Path, default=setup_dir / "pilot50_cohort.json"
    )
    parser.add_argument(
        "--sdk-runs-root", type=Path, default=project / ".codex-home/runs"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (
        args.output
        or setup_dir
        / f"actor_boundary_recovered_final_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing audit evidence: {output}"
        )
    report = audit_recovered_cohort(args.manifest, sdk_runs_root=args.sdk_runs_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "report": str(output.resolve()),
                "passed": report["passed"],
                "terminal_counts": report["terminal_scored_cohort"]["counts"],
                "original_unscored_prefix_count": report["original_unscored_prefixes"][
                    "prefix_count"
                ],
                "unanswered_prefix_requests": report["original_unscored_prefixes"][
                    "unanswered_request_count"
                ],
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
