"""Archive verified recovery3 results only after the complete 50-task audit.

This script reads saved artifacts and creates one new ignored staging directory.
It never invokes an audit, model, simulator, scorer, or service. Missing or failed
final evidence prevents staging creation. Original errors remain only in their
source artifacts; derived manifest errors use safe classifications and hashes.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import json
import re
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]  # Validated artifact IO boundary, never executed.
ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "tmp/alfred_setup"
ASSET_DIRECTORY = "assets/v8_recovery_complete_20261105"
EXPECTED_INDICES = {32, 36, 40, 43, 44, 45, 47, 48, 49}
URL_PATTERN = re.compile(r"(?:https?|socks5h?)://[^\s\"'<>]+", re.IGNORECASE)
SECRET_PATTERN = re.compile(
    r"Bearer\s+\S+|sk-[A-Za-z0-9]{12,}|"
    r"-----BEGIN (?:(?:RSA|DSA|EC|OPENSSH|ENCRYPTED) )?PRIVATE KEY-----",
    re.IGNORECASE,
)
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "access_token",
    "refresh_token",
}


def require(condition: bool, message: str) -> None:
    """Keep checks active under optimization without printing artifact contents."""
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()


def sanitized(value: object, changes: list[JsonObject], pointer: str = "") -> object:
    """Redact all error-bearing fields, preserving already declared error hashes."""
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            child = f"{pointer}/{key}"
            lower = key.lower()
            if (
                (lower == "error" or lower.endswith("_error"))
                and isinstance(item, str)
                and item
            ):
                if key == "error" and value.get("error_redacted") is True:
                    require(
                        re.fullmatch(r"[0-9a-f]{64}", str(value.get("error_sha256")))
                        is not None,
                        "Declared redacted error has invalid digest",
                    )
                    category = "Upstream request failed; original error text withheld"
                    result[key] = category
                    changes.append(
                        {
                            "json_pointer": child,
                            "transformation": "safe label normalized; existing original-error SHA256 retained",
                        }
                    )
                else:
                    category = (
                        "HTTP403 / ROW5004 / dependency timeout / processing_error"
                        if "403" in item and "5004" in item
                        else (
                            "HTTP502 / upstream unavailable"
                            if "502" in item
                            else "Error text withheld"
                        )
                    )
                    result[key] = {
                        "redacted": True,
                        "classification": category,
                        "error_sha256": digest(item.encode("utf-8")),
                    }
                    changes.append(
                        {
                            "json_pointer": child,
                            "transformation": "error text replaced by safe classification and exact UTF-8 SHA256",
                        }
                    )
            elif lower in SENSITIVE_KEYS and item:
                result[key] = "[REDACTED]"
                changes.append(
                    {
                        "json_pointer": child,
                        "transformation": "sensitive named field removed",
                    }
                )
            else:
                result[key] = sanitized(item, changes, child)
        return result
    if isinstance(value, list):
        return [
            sanitized(item, changes, f"{pointer}/{index}")
            for index, item in enumerate(value)
        ]
    if isinstance(value, str) and URL_PATTERN.search(value):
        changes.append(
            {"json_pointer": pointer, "transformation": "absolute URL removed"}
        )
        return URL_PATTERN.sub("[URL REDACTED]", value)
    return value


@dataclass
class Inputs:
    """Cache exact input bytes so the archived evidence is what was validated."""

    bytes_by_path: dict[Path, bytes] = field(default_factory=dict)

    def raw(self, path: Path) -> bytes:
        path = path.resolve(strict=True)
        if path not in self.bytes_by_path:
            self.bytes_by_path[path] = path.read_bytes()
        return self.bytes_by_path[path]

    def object(self, path: Path) -> JsonObject:
        value = json.loads(self.raw(path))
        require(isinstance(value, dict), f"Expected JSON object: {path.name}")
        return dict(value)

    def sha(self, path: Path) -> str:
        return digest(self.raw(path))


def source_matches(actual: JsonObject, baseline: JsonObject) -> bool:
    return (
        actual["branch"] == baseline["branch"]
        and actual["executed_code_sha256"] == baseline["executed_code_sha256"]
        and actual["tracked_worktree_drift"] == baseline["tracked_worktree_drift"] == ""
        and re.fullmatch(r"[0-9a-f]{40}", actual["head_sha"]) is not None
    )


def validate_inputs(args: argparse.Namespace, inputs: Inputs) -> JsonObject:
    """Require successful final reports, exact source locks and all terminal jobs."""
    final = inputs.object(args.final_checks)
    require(
        final.get("status") == "verified" and final.get("all_passed") is True,
        "Final audit watcher has not verified the complete cohort",
    )
    expected_scripts = {
        "analyze_recovered_pilot50_v3.py",
        "rescore_recovered_pilot50_v3.py",
        "audit_recovery3_pilot_boundary.py",
    }
    require(
        len(final["checks"]) == 3
        and {Path(row["command"][1]).name for row in final["checks"]}
        == expected_scripts
        and all(row["exit_code"] == 0 for row in final["checks"]),
        "Final watcher checks are incomplete or unsuccessful",
    )
    manifest = inputs.object(args.manifest)
    require(manifest.get("cohort_revision") == 3, "Expected revision3 manifest")
    baseline = manifest["code"]
    fold = manifest["fold"]
    require(
        inputs.sha(Path(fold["path"])) == fold["sha256"]
        and inputs.sha(Path(fold["splits_path"])) == fold["splits_sha256"],
        "Frozen fold or official splits changed",
    )
    canonical = manifest["episodes"]
    require(
        [row["global_index"] for row in canonical] == list(range(50))
        and len({row["episode_id"] for row in canonical}) == 50,
        "Canonical frozen selection is incomplete or duplicated",
    )
    recoveries = manifest["recovery_runs"]
    histories = manifest["original_runs"]
    require(
        len(histories) == 10
        and len(recoveries) == 9
        and {row["global_index"] for row in recoveries} == EXPECTED_INDICES,
        "Expected ten histories and nine selected recoveries",
    )
    reports = {
        "analysis": inputs.object(args.analysis),
        "rescore": inputs.object(args.rescore),
        "boundary": inputs.object(args.boundary),
        "source_end": inputs.object(args.source_end_audit),
    }
    analysis, rescore, boundary, source_end = (
        reports[name] for name in ("analysis", "rescore", "boundary", "source_end")
    )
    manifest_sha = inputs.sha(args.manifest)
    require(
        analysis["cohort_sha256"]
        == rescore["cohort_sha256"]
        == boundary["cohort_manifest_sha256"]
        == manifest_sha,
        "Final reports refer to another cohort snapshot",
    )
    require(
        analysis["status"] == "completed"
        and (
            analysis["selected"],
            analysis["scored"],
            analysis["original_scored"],
            analysis["recovered_scored"],
            analysis["actual_run_count"],
            analysis["actual_attempt_count"],
        )
        == (50, 50, 41, 9, 19, 59),
        "Analysis is not the complete selected cohort",
    )
    require(
        rescore["all_match"] is True
        and (
            rescore["expected"],
            rescore["checked"],
            rescore["matching"],
            rescore["actual_run_count"],
            rescore["actual_attempt_count"],
        )
        == (50, 50, 50, 19, 59)
        and not rescore["mismatch_episode_ids"],
        "Official terminal recheck did not match all50",
    )
    require(
        boundary["passed"] is True
        and boundary["history_only"] is False
        and boundary["history_preflight"]["passed"] is True
        and boundary["terminal_scored_cohort"]["episode_count"] == 50
        and boundary["original_unscored_prefixes"]["prefix_count"] == 9,
        "Actor boundary final audit is absent or failed",
    )
    require(
        source_end["all_19_runs_have_identical_executed_source"] is True
        and source_matches(source_end["current_code"], baseline)
        and source_end["reference_code"] == baseline,
        "End-of-run source audit failed",
    )
    require(
        source_end["external_sources"]
        == manifest["common_configuration"]["external_sources"],
        "External sources differ at completion",
    )
    selected_ids = [row["episode_id"] for row in canonical]
    for rows in (
        analysis["episodes"],
        rescore["episodes"],
        boundary["terminal_scored_cohort"]["episodes"],
    ):
        require(
            [row["episode_id"] for row in rows] == selected_ids,
            "Final report selected-identity order differs",
        )
    prefixes = analysis["unscored_historical_attempts"]
    require(
        len(prefixes) == len(rescore["retained_unscored_historical_attempts"]) == 9,
        "Historical error prefixes were dropped",
    )
    require(
        analysis["unscored_prefix_totals"]["exact_consumed_usage"] is None
        and analysis["unscored_prefix_totals"]["unknown_failed_turn_usage"] is None,
        "Unknown failed-request usage was converted into an exact total",
    )
    require(
        all(
            row["error_redacted"] is True
            and row["score"] is None
            and row["exact_consumed_usage"] is None
            for row in prefixes
        ),
        "Error prefix redaction/null-score/unknown-usage contract failed",
    )
    boundary_runs = {row["run_id"]: row for row in boundary["source_runs"]}
    require(len(boundary_runs) == 19, "Boundary report lacks exact19 source runs")
    observed_runs: dict[str, tuple[JsonObject, JsonObject]] = {}
    raw_errors = []
    all_scored = []
    for descriptor in histories + recoveries:
        run_path, summary_path = Path(descriptor["run_json"]), Path(
            descriptor["summary_json"]
        )
        run, summary = inputs.object(run_path), inputs.object(summary_path)
        run_id = descriptor["run_id"]
        require(
            run_id not in observed_runs
            and run["run_id"] == run_id
            and source_matches(run["code"], baseline),
            "Run identity or source changed",
        )
        require(
            source_end["actual_run_heads"][run_id] == run["code"]["head_sha"],
            "Actual run HEAD differs from end audit",
        )
        for key, expected in manifest["common_configuration"].items():
            if key not in {"code", "cli_arguments_excluding_run_and_shard"}:
                require(
                    run[key] == expected,
                    "Actor configuration differs from frozen manifest",
                )
        audit_row = boundary_runs[run_id]
        require(
            inputs.sha(run_path) == audit_row["run_json_sha256"]
            and inputs.sha(summary_path) == audit_row["summary_json_sha256"],
            "Run/summary changed after final boundary audit",
        )
        if descriptor in histories:
            require(
                inputs.sha(run_path) == descriptor["run_json_sha256"]
                and inputs.sha(summary_path) == descriptor["summary_json_sha256"],
                "Historical source lock changed",
            )
        else:
            require(
                run["status"] == "completed"
                and summary["selected"]
                == summary["completed"]
                == len(summary["attempts"])
                == 1
                and summary["errors"]
                == summary["interrupted"]
                == summary["pending"]
                == 0,
                "Selected recovery is not terminal-completed",
            )
            require(
                run["command"] == descriptor["command"],
                "Actual recovery command differs from plan",
            )
        for attempt in summary["attempts"]:
            require(
                inputs.object(Path(attempt["artifact_dir"]) / "attempt.json")
                == attempt,
                "Attempt changed after summary persistence",
            )
            if attempt["status"] == "completed":
                require(
                    attempt["score"] is not None, "Completed attempt has null score"
                )
                all_scored.append(attempt["episode_id"])
            else:
                require(
                    attempt["status"] == "error"
                    and attempt["score"] is None
                    and isinstance(attempt["error"], str),
                    "Historical error boundary changed",
                )
                raw_errors.append(attempt["error"])
        observed_runs[run_id] = run, summary
    require(
        len(all_scored) == len(set(all_scored)) == 50
        and set(all_scored) == set(selected_ids)
        and len(raw_errors) == 9,
        "Actual attempt coverage is not50scores+9errors",
    )
    for queue in manifest["queues"]:
        status = inputs.object(Path(queue["status_path"]))
        require(
            status["state"] == "completed"
            and status["exit_code"] == 0
            and status["manifest_sha256"] == manifest_sha
            and status["automatic_retries"] == 0,
            "Queue is not successfully terminal or refers to another plan",
        )
        require(
            [row["run_id"] for row in status["jobs"]] == queue["run_ids"]
            and all(
                row["state"] == "completed" and row["exit_code"] == 0
                for row in status["jobs"]
            ),
            "Queue jobs are incomplete",
        )
        require(
            inputs.sha(Path(queue["launcher_path"])) == queue["launcher_sha256"],
            "Queue launcher changed",
        )
    for job in recoveries:
        status = inputs.object(Path(job["status_path"]))
        require(
            status["state"] == "completed"
            and status["exit_code"] == 0
            and status["run_id"] == job["run_id"],
            "Job status is not completed",
        )
    return {
        "manifest": manifest,
        "reports": reports,
        "observed_runs": observed_runs,
        "raw_errors": raw_errors,
    }


def local_script_dependencies(seeds: set[Path], inputs: Inputs) -> set[Path]:
    """Copy the complete transitive set of local helper imports, without importing."""
    found: set[Path] = set()
    pending = list(seeds)
    while pending:
        path = pending.pop().resolve(strict=True)
        if path in found:
            continue
        found.add(path)
        tree = ast.parse(inputs.raw(path), filename=path.name)
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else (
                    [node.module]
                    if isinstance(node, ast.ImportFrom) and node.module
                    else []
                )
            )
            for name in names:
                sibling = path.parent / (name.split(".")[0] + ".py")
                if sibling.is_file():
                    pending.append(sibling)
    return found


def build_payload(
    args: argparse.Namespace, inputs: Inputs, validated: JsonObject
) -> dict[str, bytes]:
    """Prepare every byte and safety check before creating the output directory."""
    manifest, reports = validated["manifest"], validated["reports"]
    payload: dict[str, bytes] = {}
    records: list[JsonObject] = []

    def add(source: Path, name: str, *, transform_json: bool = False) -> None:
        data = inputs.raw(source)
        changes: list[JsonObject] = []
        if transform_json:
            value = sanitized(json.loads(data), changes)
            archived = encoded(value) if changes else data
        else:
            archived = data
        require(name not in payload, "Duplicate archive destination")
        payload[name] = archived
        records.append(
            {
                "source": str(source.resolve()),
                "source_sha256": digest(data),
                "archive": name,
                "archive_sha256": digest(archived),
                "byte_identical": data == archived,
                "transformations": changes,
            }
        )

    for job in manifest["recovery_runs"]:
        name = job["run_id"]
        run, summary = validated["observed_runs"][name]
        for source_field, filename in (
            ("run_json", "run.json"),
            ("summary_json", "summary.json"),
            ("status_path", "job_status.json"),
        ):
            add(Path(job[source_field]), f"runs/{name}/{filename}", transform_json=True)
        attempt_path = Path(summary["attempts"][0]["artifact_dir"]) / "attempt.json"
        add(attempt_path, f"runs/{name}/attempt.json", transform_json=True)
        command_name = f"runs/{name}/command.txt"
        command_bytes = (shlex.join(run["command"]) + "\n").encode()
        payload[command_name] = command_bytes
        records.append(
            {
                "source": str(Path(job["run_json"]).resolve()),
                "source_sha256": inputs.sha(Path(job["run_json"])),
                "archive": command_name,
                "archive_sha256": digest(command_bytes),
                "transformations": [
                    {
                        "transformation": "exact recorded argv rendered with shlex.join; no argument changes"
                    }
                ],
            }
        )
    for queue in manifest["queues"]:
        index = queue["queue_index"]
        add(Path(queue["status_path"]), f"queues/q{index}.json", transform_json=True)
        add(Path(queue["launcher_path"]), f"queues/q{index}.sh")
    for source, name in (
        (args.manifest, "cohort.redacted.json"),
        (args.source_end_audit, "source_end_audit.json"),
        (args.final_checks, "final_audit_results.json"),
        (args.analysis, "audits/pilot50_analysis.json"),
        (args.rescore, "audits/pilot50_terminal_rescore.json"),
    ):
        add(source, name, transform_json=True)

    case = inputs.object(args.interaction_case)
    case_job = next(
        job for job in manifest["recovery_runs"] if job["global_index"] == 45
    )
    _, case_summary = validated["observed_runs"][case_job["run_id"]]
    case_attempt = case_summary["attempts"][0]
    require(
        case["global_index"] == 45
        and case["run_id"] == case_job["run_id"]
        and case["episode_id"] == case_attempt["episode_id"]
        and case["official_saved_score"] == case_attempt["score"],
        "Interaction case differs from the selected completed i45 episode",
    )
    for path, expected_hash in case["raw_artifact_sha256"].items():
        require(
            inputs.sha(Path(path)) == expected_hash,
            "Interaction case source artifact changed",
        )
    case_script = SETUP / "archive_recovery3_i45_case.py"
    require(
        inputs.sha(case_script)
        == case["raw_artifact_sha256"][str(case_script.resolve())],
        "Interaction case generation script changed",
    )
    # This already-safe case retains observable simulator feedback verbatim.
    # The common final scan still rejects raw gateway errors or private URLs.
    add(args.interaction_case, "cases/recovery3_i45_interaction_case.json")

    readiness = inputs.object(args.launch_readiness)
    require(
        readiness["manifest_sha256"] == inputs.sha(args.manifest)
        and source_matches(readiness["code"], manifest["code"])
        and readiness["native_probe_succeeded"] is True
        and readiness["forward"]["tcp_connected"] is True
        and readiness["forward"]["socks_greeting_ready"] is True,
        "Launch readiness evidence does not match the recovered cohort",
    )
    readiness_fields = (
        "checked_at_utc",
        "code",
        "native_probe_sha256",
        "native_probe_succeeded",
        "manifest_sha256",
        "gpu_snapshot",
        "predictor_physical_gpu",
        "renderer_physical_gpu",
        "scored_original_episodes_not_repeated",
    )
    safe_readiness = {key: readiness[key] for key in readiness_fields}
    safe_readiness["forward"] = {
        key: readiness["forward"][key]
        for key in (
            "checked_at_utc",
            "endpoint",
            "tcp_connected",
            "socks_greeting_ready",
            "reply_hex",
            "wall_seconds",
        )
    }
    safe_readiness["scope"] = (
        "Allowlisted launch-time readiness snapshot; it does not establish "
        "continuous forwarding or upstream availability during execution."
    )
    readiness_name = "network/recovery3_launch_readiness.safe.json"
    readiness_bytes = encoded(safe_readiness)
    payload[readiness_name] = readiness_bytes
    records.append(
        {
            "source": str(args.launch_readiness.resolve()),
            "source_sha256": inputs.sha(args.launch_readiness),
            "archive": readiness_name,
            "archive_sha256": digest(readiness_bytes),
            "byte_identical": False,
            "transformations": [
                {
                    "transformation": "allowlisted code/forward/GPU/probe-hash/cohort-hash metadata; launch snapshot only"
                }
            ],
        }
    )

    boundary_bytes = inputs.raw(args.boundary)
    compressed = gzip.compress(boundary_bytes, mtime=0)
    payload["audits/pilot50_actor_boundary.full.json.gz"] = compressed
    readable = copy.deepcopy(reports["boundary"])
    entries = readable.pop("artifact_manifest")
    readable["full_report"] = {
        "source_path": str(args.boundary.resolve()),
        "source_sha256": digest(boundary_bytes),
        "compressed_archive": "pilot50_actor_boundary.full.json.gz",
        "compressed_sha256": digest(compressed),
        "lossless_compression": True,
        "omitted_from_readable_copy": "artifact_manifest",
        "omitted_entries": len(entries),
    }
    payload["audits/pilot50_actor_boundary.json"] = encoded(readable)
    records.append(
        {
            "source": str(args.boundary.resolve()),
            "source_sha256": digest(boundary_bytes),
            "archive": "audits/pilot50_actor_boundary.full.json.gz",
            "archive_sha256": digest(compressed),
            "transformation": "lossless gzip with mtime=0; decompressed SHA equals source SHA",
        }
    )

    pinned: dict[str, str] = {}
    for report in (reports["analysis"], reports["rescore"]):
        for name, value in report["helper_script_sha256"].items():
            require(
                name not in pinned or pinned[name] == value,
                "Audit reports disagree about helper execution digest",
            )
            pinned[name] = value
    for name, value in reports["boundary"]["executed_script_hashes"].items():
        require(
            name not in pinned or pinned[name] == value,
            "Boundary helper execution digest disagrees",
        )
        pinned[name] = value
    for name, value in (
        ("analyze_recovered_pilot50_v3.py", reports["analysis"]["script_sha256"]),
        ("rescore_recovered_pilot50_v3.py", reports["rescore"]["script_sha256"]),
        ("run_pilot50_recovery_queue.py", manifest["queue_driver"]["sha256"]),
        (
            case_script.name,
            case["raw_artifact_sha256"][str(case_script.resolve())],
        ),
    ):
        require(
            name not in pinned or pinned[name] == value,
            "Executed script digest disagreement",
        )
        pinned[name] = value
    for name, value in pinned.items():
        require(
            inputs.sha(SETUP / name) == value,
            "Executed helper changed after audit; archive exact executed version instead",
        )
    scripts = local_script_dependencies(
        {SETUP / name for name in pinned}
        | {Path(__file__), SETUP / "wait_recovery3_audits.py"},
        inputs,
    )
    for script in sorted(scripts):
        add(script, "scripts/" + script.name)
    payload["scripts/executed_hashes.json"] = encoded(
        {
            "recorded_executed_hashes": pinned,
            "all_local_script_hashes": {
                path.name: inputs.sha(path) for path in sorted(scripts)
            },
            "dependency_method": "AST traversal of local sibling imports; files copied without import/execution",
            "repository_dependencies": "codex_agent production imports remain in the original repository and are covered by recorded executed-source SHA; no source substitution",
        }
    )
    payload["manifest.json"] = encoded(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "New nine completed recovery runs and final50 audits only; original4 run bundle remains v6, six failed recoveries remain v7",
            "coverage": {
                "scored_unique": 50,
                "original_scored": 41,
                "new_scored": 9,
                "actual_runs": 19,
                "actual_attempts": 59,
                "unscored_historical_errors": 9,
            },
            "source_files": records,
            "redaction_policy": "error and *_error fields use safe labels and exact original UTF-8 error hashes; absolute URLs and sensitive named fields are removed. Redacted plan is derived evidence, not an executable replacement for the raw manifest.",
            "excluded": [
                "historical v6/v7 large bundles",
                "raw gateway error text",
                "raw console/SDK logs",
                "credentials, environment and provider/SDK configuration",
                "all task images and full simulator metadata",
            ],
        }
    )
    protected = validated["raw_errors"]
    for name, data in payload.items():
        text = gzip.decompress(data).decode() if name.endswith(".gz") else data.decode()
        require(
            not URL_PATTERN.search(text) and not SECRET_PATTERN.search(text),
            f"Archive safety scan rejected candidate: {name}",
        )
        require(
            all(value not in text for value in protected),
            "Raw historical error text survived archive redaction",
        )
    payload["sha256.json"] = encoded(
        {name: digest(data) for name, data in sorted(payload.items())}
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag, default in (
        ("manifest", "pilot50_cohort_recovery3.json"),
        ("final-checks", "recovery3_final_audit_results.json"),
        ("source-end-audit", "recovery3_end_source_audit.json"),
        ("analysis", "pilot50_recovery3_analysis.json"),
        ("rescore", "pilot50_recovery3_terminal_rescore.json"),
        ("boundary", "pilot50_actor_boundary_recovery3_final.json"),
        ("interaction-case", "recovery3_i45_interaction_case.json"),
        ("launch-readiness", "recovery3_launch_readiness_20261105.json"),
    ):
        parser.add_argument("--" + flag, type=Path, default=SETUP / default)
    parser.add_argument(
        "--staging", type=Path, default=SETUP / "v8_complete_20261105_staging"
    )
    args = parser.parse_args()
    require(
        not args.staging.exists(), "Refusing to overwrite an existing staging directory"
    )
    inputs = Inputs()
    validated = validate_inputs(args, inputs)
    payload = build_payload(args, inputs, validated)
    # All final gates and content scans have passed before the first filesystem write.
    args.staging.mkdir(parents=False, exist_ok=False)
    for relative, data in payload.items():
        path = args.staging / ASSET_DIRECTORY / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    inventory = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_cohort_sha256": inputs.sha(args.manifest),
        "asset_directory": ASSET_DIRECTORY,
        "asset_count": len(payload),
        "files": {name: digest(data) for name, data in sorted(payload.items())},
        "complete_before_staging_creation": True,
        "no_model_simulator_or_audit_invocation": True,
    }
    with (args.staging / "archive_inventory.json").open("xb") as stream:
        stream.write(encoded(inventory))
    print(
        json.dumps(
            {
                "status": "archived_verified_complete_cohort",
                "staging": str(args.staging.resolve()),
                "assets": len(payload),
                "scored": 50,
                "unscored_historical_errors": 9,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
