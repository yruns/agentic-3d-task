"""Audit the fixed 41+9 ALFRED cohort without actors, simulation, or source edits.

No report is written until all nine specified recoveries have terminal scores,
all 50 identities occur exactly once, and every source run agrees with SQLite.
Usage for the three abandoned error prefixes remains a separate lower bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_pilot50 import ROOT, read_json, read_lines
from rescore_pilot50 import require

from codex_agent.alfred.dataset import AlfredEpisode, load_manifest
from codex_agent.alfred.metrics import EpisodeScore, aggregate_scores
from codex_agent.alfred.results import EpisodeAttempt
from codex_agent.cli.run_alfred import build_arg_parser

RECOVERY_INDICES = {32, 36, 40, 43, 44, 45, 47, 48, 49}
SIGNATURE_FIELDS = (
    "fold_sha256",
    "splits_sha256",
    "model",
    "model_provider",
    "reasoning_effort",
    "execution",
    "geometry",
    "perception_profile",
    "perception",
    "simulator",
    "external_sources",
    "code",
)
SCORE_FIELDS = ("success", "completed_goal_conditions", "total_goal_conditions")
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_settings(command: list[str]) -> dict[str, object]:
    """Parse actual CLI defaults; ignore only output identity and shard selection."""
    require(command[1:3] == ["-m", "codex_agent.cli.run_alfred"], "unexpected CLI")
    arguments = vars(build_arg_parser().parse_args(command[3:]))
    for name in ("output_dir", "run_id", "num_shards", "shard_index"):
        arguments.pop(name)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in arguments.items()
    }


def compare_source_identity(
    actual: dict[str, Any], reference: dict[str, Any], policy: dict[str, Any]
) -> dict[str, object]:
    """Permit only a HEAD change while branch, clean state and executed bytes agree."""
    require(
        type(policy.get("version")) is int and policy["version"] == 1,
        "unsupported source identity policy",
    )
    flags = (
        "allow_head_sha_difference",
        "require_same_branch",
        "require_clean_tracked_tree",
        "require_identical_executed_code_sha256",
        "actual_head_recorded_per_run",
    )
    require(
        all(policy.get(field) is True for field in flags),
        "source identity policy must enforce all declared invariants",
    )
    require(
        isinstance(policy.get("reason"), str) and bool(policy["reason"].strip()),
        "source identity policy requires an explicit reason",
    )
    fields = {"branch", "head_sha", "executed_code_sha256", "tracked_worktree_drift"}
    require(
        set(actual) == set(reference) == fields, "unexpected code provenance fields"
    )
    for label, code in (("actual", actual), ("reference", reference)):
        require(
            isinstance(code["head_sha"], str)
            and re.fullmatch(r"[0-9a-f]{40}", code["head_sha"]) is not None,
            f"{label} HEAD must be a recorded full Git SHA",
        )
        require(
            isinstance(code["executed_code_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", code["executed_code_sha256"]) is not None,
            f"{label} executed code SHA must be a full SHA-256",
        )
    require(
        actual["branch"] == reference["branch"] and bool(reference["branch"]),
        "source branch changed",
    )
    require(
        actual["tracked_worktree_drift"] == reference["tracked_worktree_drift"] == "",
        "source tree is not clean",
    )
    require(
        actual["executed_code_sha256"] == reference["executed_code_sha256"],
        "executed source/config SHA changed",
    )
    return {
        "reference_head_sha": reference["head_sha"],
        "actual_head_sha": actual["head_sha"],
        "head_differs": actual["head_sha"] != reference["head_sha"],
        "branch": actual["branch"],
        "executed_code_sha256": actual["executed_code_sha256"],
        "tracked_worktree_drift": actual["tracked_worktree_drift"],
        "policy_version": policy["version"],
        "matches": True,
    }


def validate_run_configuration(
    metadata: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, object]:
    """Keep all model/input/budget comparisons exact; handle HEAD policy separately."""
    reference = manifest["common_configuration"]
    require(reference["code"] == manifest["code"], "cohort reference code differs")
    identity = compare_source_identity(
        metadata["code"], manifest["code"], manifest["source_identity_policy"]
    )
    actual = {field: metadata[field] for field in SIGNATURE_FIELDS if field != "code"}
    actual["cli_arguments_excluding_run_and_shard"] = command_settings(
        metadata["command"]
    )
    expected = {field: value for field, value in reference.items() if field != "code"}
    changed = sorted(
        field
        for field in set(actual) | set(expected)
        if actual.get(field) != expected.get(field)
    )
    require(not changed, f"actor/model/input/budget configuration changed: {changed}")
    return identity


@dataclass(frozen=True)
class SavedRun:
    """Validated terminal run artifacts; raw JSON is confined to this audit."""

    metadata: dict[str, Any]
    summary: dict[str, Any]
    directory: Path


@dataclass(frozen=True)
class SelectedEpisode:
    reference: AlfredEpisode
    global_index: int
    run_id: str
    directory: Path
    attempt: dict[str, Any]


@dataclass(frozen=True)
class CompletedCohort:
    manifest_path: Path
    manifest: dict[str, Any]
    runs: dict[str, SavedRun]
    selected: tuple[SelectedEpisode, ...]
    error_prefixes: tuple[SelectedEpisode, ...]
    database_audit: tuple[dict[str, object], ...]


def _validate_summary(run: SavedRun, expected_ids: list[str]) -> None:
    summary, metadata = run.summary, run.metadata
    attempts = summary["attempts"]
    require(summary["selected"] == len(expected_ids), "selected count mismatch")
    require(
        [item["episode_id"] for item in attempts] == expected_ids[: len(attempts)],
        f"attempt order/membership mismatch: {metadata['run_id']}",
    )
    for attempt in attempts:
        EpisodeAttempt.model_validate(attempt)
        directory = Path(attempt["artifact_dir"]).resolve(strict=True)
        require(directory.is_relative_to(run.directory), "attempt escapes source run")
        require(
            read_json(directory / "attempt.json") == attempt, "attempt artifact drift"
        )
    counts = Counter(attempt["status"] for attempt in attempts)
    for status, field in (
        ("completed", "completed"),
        ("error", "errors"),
        ("interrupted", "interrupted"),
    ):
        require(summary[field] == counts[status], f"summary {field} mismatch")
    require(summary["pending"] == len(expected_ids) - len(attempts), "pending mismatch")
    scores = [
        EpisodeScore.model_validate(item["score"])
        for item in attempts
        if item["status"] == "completed"
    ]
    metrics = aggregate_scores(scores).model_dump() if scores else None
    require(
        metrics == summary["metrics"], f"source metrics mismatch: {metadata['run_id']}"
    )
    for split in ("valid_seen", "valid_unseen"):
        split_scores = [
            EpisodeScore.model_validate(item["score"])
            for item in attempts
            if item["status"] == "completed"
            and item["episode_id"].startswith(split + "/")
        ]
        expected = aggregate_scores(split_scores).model_dump() if split_scores else None
        require(summary["by_split"][split] == expected, "source split metrics mismatch")


def _database_check(
    database: Path, runs: dict[str, SavedRun]
) -> tuple[dict[str, object], ...]:
    audits: list[dict[str, object]] = []
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        for run_id, saved in runs.items():
            stored = connection.execute(
                "SELECT metadata_json FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            require(
                stored is not None and json.loads(stored[0]) == saved.metadata,
                f"SQLite run missing/different: {run_id}",
            )
            records = connection.execute(
                "SELECT episode_id,status,artifact_dir,score_json,error FROM attempts WHERE run_id=?",
                (run_id,),
            ).fetchall()
            actual = {
                identity: {
                    "episode_id": identity,
                    "status": status,
                    "artifact_dir": directory,
                    "score": json.loads(score) if score is not None else None,
                    "error": error,
                }
                for identity, status, directory, score, error in records
            }
            expected = {item["episode_id"]: item for item in saved.summary["attempts"]}
            require(actual == expected, f"SQLite attempts differ: {run_id}")
            audits.append(
                {
                    "run_id": run_id,
                    "run_status": saved.metadata["status"],
                    "attempts": len(records),
                    "matches": True,
                }
            )
    return tuple(audits)


def load_completed_cohort(path: Path, database: Path) -> CompletedCohort:
    """Resolve exactly the prespecified scored source for each frozen fold item."""
    manifest = read_json(path)
    require(manifest["schema_version"] == 1, "unsupported cohort manifest")
    require(
        (
            manifest["original_scored_count"],
            manifest["original_unscored_error_count"],
            manifest["original_unattempted_count"],
            manifest["recovery_planned_count"],
        )
        == (41, 3, 6, 9),
        "cohort history counts changed",
    )
    fold = manifest["fold"]
    require(fold["episode_count"] == 50, "cohort is not pilot50")
    require(sha256(Path(fold["path"])) == fold["sha256"], "fold file changed")
    require(
        sha256(Path(fold["splits_path"])) == fold["splits_sha256"],
        "splits file changed",
    )
    references = load_manifest(
        Path(fold["path"]), Path(fold["data_root"]), Path(fold["splits_path"])
    )
    require(len(references) == 50, "frozen fold must contain exactly 50")
    by_id = {item.episode_id: (index, item) for index, item in enumerate(references)}
    originals, recoveries = manifest["original_runs"], manifest["recovery_runs"]
    require(
        len(originals) == 4 and len(recoveries) == 9,
        "expected four originals and nine recoveries",
    )
    require(
        {item["global_index"] for item in recoveries} == RECOVERY_INDICES,
        "recovery indices changed",
    )
    require(
        len({item["run_id"] for item in originals + recoveries}) == 13,
        "duplicate source run IDs",
    )
    runs: dict[str, SavedRun] = {}
    original_completed: set[str] = set()
    prefixes: list[SelectedEpisode] = []
    for record in originals + recoveries:
        run_id = record["run_id"]
        metadata_path, summary_path = Path(record["run_json"]), Path(
            record["summary_json"]
        )
        require(
            metadata_path.is_file() and summary_path.is_file(),
            f"planned source run has not produced terminal artifacts: {run_id}",
        )
        metadata, summary = read_json(metadata_path), read_json(summary_path)
        original = record in originals
        require(
            metadata["status"]
            in ({"completed", "error"} if original else {"completed"}),
            f"source run is not eligible/finished: {run_id}: {metadata['status']}",
        )
        if original:
            for file_path, hash_field in (
                (metadata_path, "run_json_sha256"),
                (summary_path, "summary_json_sha256"),
            ):
                require(
                    sha256(file_path) == record[hash_field],
                    f"original artifact changed: {file_path}",
                )
        directory = Path(record["output_dir"]).resolve(strict=True)
        require(
            metadata_path.resolve() == directory / "run.json"
            and summary_path.resolve() == directory / "summary.json",
            "run paths disagree",
        )
        require(
            metadata["run_id"] == run_id
            and Path(metadata["output_dir"]).resolve() == directory,
            "run identity mismatch",
        )
        validate_run_configuration(metadata, manifest)
        if original:
            require(
                metadata["code"] == manifest["code"],
                "original recorded code provenance changed",
            )
        require(
            metadata["full_fold_size"] == 50
            and metadata["limit"] is None
            and not metadata["diagnostic_prefix"],
            "noncanonical selection",
        )
        require(
            metadata["num_shards"] == (4 if original else 50), "unexpected shard count"
        )
        shard = metadata["shard_index"]
        if original:
            require(shard == originals.index(record), "original shard ordering changed")
            require(metadata["status"] == record["status"], "original status changed")
        else:
            require(
                shard == record["global_index"]
                and metadata["command"] == record["command"],
                "recovery command differs from plan",
            )
        selection = references[shard :: metadata["num_shards"]]
        expected_ids = [item.episode_id for item in selection]
        require(
            [item["episode_id"] for item in metadata["episodes"]] == expected_ids,
            "run differs from frozen partition",
        )
        for entry, reference in zip(metadata["episodes"], selection, strict=True):
            for field in ("split", "task_type", "trajectory_sha256", "expert_actions"):
                require(
                    entry[field] == getattr(reference, field),
                    f"episode metadata {field} mismatch",
                )
        saved = SavedRun(metadata, summary, directory)
        _validate_summary(saved, expected_ids)
        if original:
            selected_indices = list(range(shard, 50, 4))
            completed_indices = [
                by_id[item["episode_id"]][0]
                for item in summary["attempts"]
                if item["status"] == "completed"
            ]
            unscored = [
                {
                    "global_index": by_id[item["episode_id"]][0],
                    **{
                        key: item[key]
                        for key in ("episode_id", "status", "error", "artifact_dir")
                    },
                }
                for item in summary["attempts"]
                if item["status"] != "completed"
            ]
            require(
                record["selected_global_indices"] == selected_indices
                and record["completed_global_indices"] == completed_indices
                and record["pending_global_indices"]
                == selected_indices[len(summary["attempts"]) :]
                and record["unscored_attempts"] == unscored,
                "original selection/failure provenance differs from cohort",
            )
        else:
            require(
                summary["completed"] == summary["selected"] == 1
                and summary["errors"]
                == summary["interrupted"]
                == summary["pending"]
                == 0,
                f"recovery has no complete singleton score: {run_id}",
            )
        for attempt in summary["attempts"]:
            index, reference = by_id[attempt["episode_id"]]
            if original and attempt["status"] == "completed":
                require(
                    reference.episode_id not in original_completed,
                    "duplicate original score",
                )
                original_completed.add(reference.episode_id)
            elif original:
                require(
                    attempt["status"] == "error" and attempt["score"] is None,
                    "original failure must remain unscored",
                )
                prefixes.append(
                    SelectedEpisode(
                        reference, index, run_id, Path(attempt["artifact_dir"]), attempt
                    )
                )
        runs[run_id] = saved
    require(
        len(original_completed) == 41 and len(prefixes) == 3,
        "original 41+3 history changed",
    )
    missing = {
        index
        for index, item in enumerate(references)
        if item.episode_id not in original_completed
    }
    require(
        missing == RECOVERY_INDICES,
        "recovery is not exactly the original unscored remainder",
    )
    for recovery in recoveries:
        origin = runs[recovery["origin_run_id"]]
        index = recovery["global_index"]
        require(
            origin.metadata["num_shards"] == 4
            and origin.metadata["shard_index"] == index % 4,
            "recovery origin is not the original partition",
        )
        original_attempt = next(
            (
                attempt
                for attempt in origin.summary["attempts"]
                if attempt["episode_id"] == references[index].episode_id
            ),
            None,
        )
        require(
            recovery["episode_id"] == references[index].episode_id,
            "recovery planned identity differs from fold",
        )
        if recovery["origin_status"] == "error":
            require(
                original_attempt is not None
                and original_attempt["status"] == "error"
                and original_attempt["score"] is None
                and original_attempt["error"] == recovery["origin_error"]
                and original_attempt["artifact_dir"]
                == recovery["origin_attempt_artifact_dir"],
                "recovery error-origin provenance differs",
            )
        else:
            require(
                recovery["origin_status"] == "pending" and original_attempt is None,
                "recovery pending-origin provenance differs",
            )
    selected: list[SelectedEpisode] = []
    require(
        len(manifest["episodes"]) == 50,
        "canonical selected-source list must have 50 entries",
    )
    recovery_by_index = {item["global_index"]: item for item in recoveries}
    for index, (entry, reference) in enumerate(
        zip(manifest["episodes"], references, strict=True)
    ):
        require(
            entry["global_index"] == index
            and entry["episode_id"] == reference.episode_id,
            "canonical source order/identity mismatch",
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
                entry[field] == getattr(reference, field),
                f"cohort episode {field} mismatch",
            )
        run_id = entry["selected_source_run_id"]
        require(run_id in runs, "selected source is not an audited run")
        if index in RECOVERY_INDICES:
            require(
                run_id == recovery_by_index[index]["run_id"]
                and entry["selection_reason"]
                == "recovery_of_" + recovery_by_index[index]["origin_status"],
                "wrong recovery chosen",
            )
        else:
            require(
                run_id in {item["run_id"] for item in originals}
                and entry["selection_reason"] == "original_completed",
                "replaced an already-scored original",
            )
        run = runs[run_id]
        require(
            Path(entry["selected_source_output_dir"]).resolve() == run.directory,
            "source directory mismatch",
        )
        matches = [
            item
            for item in run.summary["attempts"]
            if item["episode_id"] == reference.episode_id
        ]
        require(
            len(matches) == 1
            and matches[0]["status"] == "completed"
            and matches[0]["score"] is not None,
            "selected episode lacks one terminal score",
        )
        directory = Path(entry["expected_artifact_dir"]).resolve(strict=True)
        require(
            directory == Path(matches[0]["artifact_dir"]).resolve()
            and directory.is_relative_to(run.directory),
            "selected artifact mismatch",
        )
        selected.append(
            SelectedEpisode(reference, index, run_id, directory, matches[0])
        )
    require(
        len({item.reference.episode_id for item in selected}) == 50,
        "duplicate final episode identity",
    )
    return CompletedCohort(
        path.resolve(),
        manifest,
        runs,
        tuple(selected),
        tuple(prefixes),
        _database_check(database, runs),
    )


def audit_decisions(directory: Path, *, require_complete: bool) -> dict[str, Any]:
    """Count persisted requests and reported SDK totals, keeping omissions explicit."""
    requests = sorted((directory / "decisions").glob("*/request.json"))
    responses = sorted((directory / "decisions").glob("*/response.json"))
    require(
        [int(path.parent.name) for path in requests]
        == list(range(1, len(requests) + 1)),
        "noncontiguous request trace",
    )
    require(
        {path.parent for path in responses}.issubset(
            {path.parent for path in requests}
        ),
        "response has no request",
    )
    if require_complete:
        require(
            len(responses) == len(requests),
            f"incomplete scored decision trace: {directory}",
        )
    tokens: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    feedback: Counter[str] = Counter()
    images = historical = sdk_attempts = 0
    absent_usage: list[int] = []
    for path in requests:
        request = read_json(path)
        for value in request["images"]:
            image = Path(value).resolve(strict=True)
            require(
                image.is_file() and image.is_relative_to(directory.resolve()),
                "input image escapes episode",
            )
        images += len(request["images"])
        historical += max(0, len(request["images"]) - 2)
        observations = json.loads(request["prompt"].splitlines()[-1])["observations"]
        message = observations["feedback"]
        if not message.startswith("{"):
            feedback[message] += 1
    for path in responses:
        response = read_json(path)
        operations[json.loads(response["raw_response"])["operation"]] += 1
        metadata = response["metadata"]
        require(metadata["status"] == "completed", "saved response is not completed")
        sdk_attempts += len(metadata["attempts"])
        usage = (metadata.get("usage") or {}).get("total")
        if usage is None:
            absent_usage.append(int(path.parent.name))
            continue
        require(
            all(
                type(usage.get(key)) is int and usage[key] >= 0 for key in TOKEN_FIELDS
            ),
            "invalid required usage counters",
        )
        require(
            usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"],
            "usage total differs from input+output",
        )
        for name, value in usage.items():
            require(type(value) is int and value >= 0, f"invalid usage counter: {name}")
            tokens[name] += value
    if require_complete:
        require(
            not absent_usage,
            f"scored decisions have unavailable usage: {directory}: {absent_usage}",
        )
    missing_responses = sorted(
        int(path.parent.name)
        for path in requests
        if not path.with_name("response.json").exists()
    )
    return {
        "requests": len(requests),
        "responses": len(responses),
        "reported_usage": dict(tokens),
        "missing_usage_decisions": absent_usage,
        "missing_response_decisions": missing_responses,
        "submitted_images": images,
        "submitted_historical_images": historical,
        "sdk_attempts_observed": sdk_attempts,
        "operations": dict(operations),
        "non_action_feedback": dict(feedback),
    }


def audit_episode(item: SelectedEpisode, *, scored: bool) -> dict[str, Any]:
    """Cross-check action, observation, terminal, execution and usage evidence."""
    directory = item.directory.resolve(strict=True)
    actions, observations = read_lines(directory / "actions.jsonl"), read_lines(
        directory / "observations.jsonl"
    )
    require(
        len(observations) == len(actions) + 1,
        f"observation/action mismatch: {directory}",
    )
    require(
        [row["observation"]["frame_id"] for row in observations]
        == list(range(len(observations))),
        "frame sequence mismatch",
    )
    for index, action in enumerate(actions, 1):
        require(
            action["index"] == index
            and action["source_frame"] == index - 1
            and action["result_frame"] == index,
            "action/frame linkage mismatch",
        )
    decision = audit_decisions(directory, require_complete=scored)
    execution = read_json(
        directory / ("execution.json" if scored else "partial_execution.json")
    )
    require(execution["actions"] == len(actions), "execution action count mismatch")
    require(
        execution["failures"] == sum(not row["success"] for row in actions),
        "execution failure count mismatch",
    )
    require(
        execution["decisions"] == decision["requests"],
        "decision/request count mismatch",
    )
    for token in ("input_tokens", "output_tokens"):
        field = token if scored else "observed_" + token + "_lower_bound"
        require(
            execution[field] == decision["reported_usage"].get(token, 0),
            f"reported usage/execution mismatch: {directory}: {field}",
        )
    for row in observations:
        predicted = row["prediction"]
        paths = [
            row["observation"]["rgb_path"],
            predicted["depth_path"],
            *(entry["mask_path"] for entry in predicted["detections"]),
        ]
        require(
            all(
                Path(value).is_file()
                and Path(value).resolve().is_relative_to(directory)
                for value in paths
            ),
            "observation artifact missing/foreign",
        )
    if scored:
        score = EpisodeScore.model_validate(item.attempt["score"])
        require(
            score.executed_actions == len(actions)
            and score.expert_actions == item.reference.expert_actions,
            "score path lengths mismatch",
        )
        terminal = read_json(directory / "simulator/terminal_evaluation.json")
        require(
            terminal["trajectory_sha256"] == item.reference.trajectory_sha256,
            "terminal trajectory hash mismatch",
        )
        require(
            Path(terminal["trajectory_path"]).resolve()
            == item.reference.trajectory_path.resolve(),
            "terminal trajectory path mismatch",
        )
        require(
            all(terminal[field] == getattr(score, field) for field in SCORE_FIELDS),
            "terminal/attempt score mismatch",
        )
        require(
            [(row["action"], row["success"]) for row in terminal["actions"]]
            == [(row["action"], row["success"]) for row in actions],
            "terminal action sequence differs",
        )
    else:
        require(
            item.attempt["score"] is None and item.attempt["status"] == "error",
            "failed prefix became a synthetic score",
        )
    row = {
        "global_index": item.global_index,
        "episode_id": item.reference.episode_id,
        "split": item.reference.split,
        "task_type": item.reference.task_type,
        "run_id": item.run_id,
        "artifact_dir": str(directory),
        "score": item.attempt["score"],
        "execution": execution,
        "observations": len(observations),
        "decision_audit": decision,
        "action_types": dict(Counter(action["action"] for action in actions)),
        "failed_action_types": dict(
            Counter(action["action"] for action in actions if not action["success"])
        ),
        "failure_messages": dict(
            Counter(action["error"] for action in actions if not action["success"])
        ),
    }
    if not scored:
        row.update(
            error=item.attempt["error"],
            exact_consumed_usage=None,
            failed_or_missing_decision_usage=None,
            usage_interpretation="Persisted successful-response totals are a lower bound; the final failed request/finalization may have consumed unreported tokens. Unknown usage is not zero.",
        )
    return row


def build_report(cohort: CompletedCohort) -> dict[str, Any]:
    rows = [audit_episode(item, scored=True) for item in cohort.selected]
    prefixes = [audit_episode(item, scored=False) for item in cohort.error_prefixes]
    scores = [EpisodeScore.model_validate(row["score"]) for row in rows]
    groups = {
        field: {
            value: aggregate_scores(
                [
                    EpisodeScore.model_validate(row["score"])
                    for row in rows
                    if row[field] == value
                ]
            ).model_dump()
            for value in sorted({row[field] for row in rows})
        }
        for field in ("split", "task_type")
    }
    usage: Counter[str] = Counter()
    prefix_usage: Counter[str] = Counter()
    for row in rows:
        usage.update(row["decision_audit"]["reported_usage"])
    for row in prefixes:
        prefix_usage.update(row["decision_audit"]["reported_usage"])
    total_fields = (
        "actions",
        "failures",
        "decisions",
        "input_tokens",
        "output_tokens",
        "perception_seconds",
        "policy_seconds",
    )
    totals = {
        field: sum(row["execution"][field] for row in rows) for field in total_fields
    }
    totals.update(
        reported_usage=dict(usage),
        observations=sum(row["observations"] for row in rows),
        stop_reasons=dict(Counter(row["execution"]["stop_reason"] for row in rows)),
    )
    for field in (
        "requests",
        "responses",
        "submitted_images",
        "submitted_historical_images",
        "sdk_attempts_observed",
    ):
        totals[field] = sum(row["decision_audit"][field] for row in rows)
    counts: dict[str, dict[str, int]] = {}
    for field in ("action_types", "failed_action_types", "failure_messages"):
        counter: Counter[str] = Counter()
        for row in rows:
            counter.update(row[field])
        counts[field] = dict(counter)
    for field in ("operations", "non_action_feedback"):
        counter = Counter()
        for row in rows:
            counter.update(row["decision_audit"][field])
        counts[field] = dict(counter)
    for run_id, run in cohort.runs.items():
        if run.metadata["status"] == "completed":
            expected_frames = sum(
                row["observations"] for row in rows if row["run_id"] == run_id
            )
            require(
                run.metadata["perception_statistics"]["frames"] == expected_frames,
                "perception frame count mismatch",
            )
    started = min(
        datetime.fromisoformat(run.metadata["started_at_utc"])
        for run in cohort.runs.values()
    )
    finished = max(
        datetime.fromisoformat(run.metadata["finished_at_utc"])
        for run in cohort.runs.values()
    )
    return {
        "status": "completed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only posthoc 41 original scored + 9 prespecified recovered scores; no model/simulator invocation.",
        "cohort_path": str(cohort.manifest_path),
        "cohort_sha256": sha256(cohort.manifest_path),
        "script_sha256": sha256(Path(__file__)),
        "io_helper_script_sha256": sha256(
            Path(__file__).with_name("analyze_pilot50.py")
        ),
        "validation_helper_script_sha256": sha256(
            Path(__file__).with_name("rescore_pilot50.py")
        ),
        "selected": 50,
        "scored": len(rows),
        "original_scored": 41,
        "recovered_scored": 9,
        "metrics": aggregate_scores(scores).model_dump(),
        "by_split": groups["split"],
        "by_task_type": groups["task_type"],
        "scored_totals": totals,
        "unscored_prefix_totals": {
            "attempts": len(prefixes),
            "score": None,
            "actions": sum(row["execution"]["actions"] for row in prefixes),
            "failures": sum(row["execution"]["failures"] for row in prefixes),
            "decisions_started": sum(row["execution"]["decisions"] for row in prefixes),
            "observations": sum(row["observations"] for row in prefixes),
            "persisted_completed_decisions": sum(
                row["decision_audit"]["responses"] for row in prefixes
            ),
            "missing_response_decisions": sum(
                len(row["decision_audit"]["missing_response_decisions"])
                for row in prefixes
            ),
            "sdk_attempts_observed_lower_bound": sum(
                row["decision_audit"]["sdk_attempts_observed"] for row in prefixes
            ),
            "observed_reported_usage_lower_bound": dict(prefix_usage),
            "exact_consumed_usage": None,
            "unknown_failed_turn_usage": None,
        },
        "counts": counts,
        "wall_seconds_including_outage_and_recovery": (
            finished - started
        ).total_seconds(),
        "matching_configuration": cohort.manifest["common_configuration"],
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
        "run_ids": list(cohort.runs),
        "database_audit": list(cohort.database_audit),
        "episodes": rows,
        "unscored_original_attempts": prefixes,
    }


def write_new_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", type=Path, default=ROOT / "tmp/alfred_setup/pilot50_cohort.json"
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "docs/benchmark/alfred/runs.sqlite"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tmp/alfred_setup/pilot50_recovered_analysis.json",
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
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
