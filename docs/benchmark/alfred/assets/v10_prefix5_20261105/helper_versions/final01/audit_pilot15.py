"""Audit saved pilot15 runs; no model, simulator, replay or implicit rescoring.

Run only when the parent supplies explicit run directories. Final mode requires
exactly one terminal score per frozen identity. --allow-partial writes a clearly
labelled snapshot; --rescore is an explicit final-only pure official-goal check.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sqlite3
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from codex_agent.alfred.dataset import AlfredEpisode, load_manifest
from codex_agent.alfred.metrics import EpisodeScore, aggregate_scores
from codex_agent.alfred.odometry import ActionOdometry, camera_intrinsics
from codex_agent.cli.run_alfred import build_arg_parser
from codex_agent.config import DEFAULT_PROJECT_ROOT

ROOT = DEFAULT_PROJECT_ROOT
ARCHIVED_HELPERS = (
    ROOT / "docs/benchmark/alfred/assets/v8_recovery_complete_20261105/scripts"
)
FOLD_SHA256 = "2fa0520b130ec9748762b31b7dc1c3aeeb072edbb5bb7e9393ea4c406da1f34a"
JsonObject = dict[str, Any]  # External artifact JSON boundary.


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> JsonObject:
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), f"Expected JSON object: {path.name}")
    return dict(value)


def read_lines(path: Path) -> list[JsonObject]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    require(
        all(isinstance(row, dict) for row in rows),
        f"Invalid JSONL objects: {path.name}",
    )
    return rows


def load_archived_helper(filename: str) -> Any:
    path = ARCHIVED_HELPERS / filename
    spec = importlib.util.spec_from_file_location("_pilot15_" + path.stem, path)
    require(
        spec is not None and spec.loader is not None, "Archived helper is unavailable"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_error(raw: str | None) -> JsonObject | None:
    if raw is None:
        return None
    category = (
        "http_502"
        if "502" in raw
        else "http_403" if "403" in raw else "infrastructure_error"
    )
    return {
        "classification": category,
        "error_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "raw_text_retained_only_in_source": True,
    }


def normalized_cli(command: list[str]) -> JsonObject:
    require(
        len(command) >= 3 and command[1:3] == ["-m", "codex_agent.cli.run_alfred"],
        "Unexpected recorded evaluation CLI",
    )
    parsed = vars(build_arg_parser().parse_args(command[3:]))
    selection_only = {"output_dir", "run_id", "num_shards", "shard_index", "limit"}
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in parsed.items()
        if key not in selection_only
    }


def configuration_signature(run: JsonObject) -> JsonObject:
    names = (
        "fold_sha256",
        "splits_sha256",
        "model",
        "model_provider",
        "reasoning_effort",
        "execution",
        "geometry",
        "perception_profile",
        "memory",
        "spatial_map",
        "simulator",
        "external_sources",
    )
    signature = {name: run[name] for name in names}
    signature["perception"] = {
        name: value
        for name, value in run["perception"].items()
        if name not in {"runtime_provenance", "libraries", "provenance_status"}
    }
    signature["cli"] = normalized_cli(run["command"])
    return signature


def select_first_scores(
    attempts: list[JsonObject], expected_ids: set[str], *, allow_partial: bool
) -> tuple[list[JsonObject], list[JsonObject]]:
    """Reject repeated official scores; historical infrastructure errors stay separate."""
    scored: dict[str, JsonObject] = {}
    errors: list[JsonObject] = []
    for attempt in attempts:
        identity = attempt["episode_id"]
        require(identity in expected_ids, "Attempt outside the frozen pilot15")
        if attempt["status"] == "completed":
            require(
                attempt["score"] is not None and attempt.get("error") is None,
                "Completed attempt has invalid score/error state",
            )
            require(
                identity not in scored,
                "Duplicate terminal score: scored task failures must not be retried",
            )
            EpisodeScore.model_validate(attempt["score"])
            scored[identity] = attempt
        else:
            require(
                attempt["status"] in {"error", "interrupted"}
                and attempt["score"] is None,
                "Infrastructure error must retain a null official score",
            )
            errors.append(attempt)
    require(
        allow_partial or set(scored) == expected_ids,
        f"Final audit requires all {len(expected_ids)} first terminal scores",
    )
    return list(scored.values()), errors


def sqlite_check(
    database: Path,
    run_id: str,
    metadata: JsonObject,
    attempts: list[JsonObject],
    *,
    allow_missing: bool,
) -> JsonObject:
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        stored = connection.execute(
            "SELECT metadata_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if stored is None and allow_missing:
            return {"run_id": run_id, "pending_ingestion": True}
        require(
            stored is not None and json.loads(stored[0]) == metadata,
            "SQLite run metadata differs from saved run",
        )
        rows = connection.execute(
            "SELECT episode_id,status,artifact_dir,score_json,error FROM attempts WHERE run_id=?",
            (run_id,),
        ).fetchall()
    actual = {
        identity: {
            "episode_id": identity,
            "status": status,
            "artifact_dir": directory,
            "score": json.loads(score) if score else None,
            "error": error,
        }
        for identity, status, directory, score, error in rows
    }
    require(
        actual == {entry["episode_id"]: entry for entry in attempts},
        "SQLite attempts differ from saved summary",
    )
    return {
        "run_id": run_id,
        "metadata_matches": True,
        "attempts_match": True,
        "attempt_count": len(rows),
    }


def audit_usage(directory: Path, *, completed: bool) -> JsonObject:
    requests = sorted((directory / "decisions").glob("*/request.json"))
    responses = sorted((directory / "decisions").glob("*/response.json"))
    require(
        [int(path.parent.name) for path in requests]
        == list(range(1, len(requests) + 1)),
        "Decision request IDs are not contiguous",
    )
    require(
        {path.parent for path in responses} <= {path.parent for path in requests},
        "Response without a saved request",
    )
    require(
        not completed or len(requests) == len(responses),
        "Completed task has an unanswered model request",
    )
    usage: Counter[str] = Counter()
    unknown_usage: list[int] = []
    operations: Counter[str] = Counter()
    sdk_attempts = 0
    for path in responses:
        response = read_json(path)
        metadata = response["metadata"]
        require(
            metadata["status"] == "completed", "Saved model response is not completed"
        )
        operations[json.loads(response["raw_response"])["operation"]] += 1
        require(
            isinstance(metadata.get("attempts"), list) and bool(metadata["attempts"]),
            "Completed response lacks SDK attempt evidence",
        )
        sdk_attempts += len(metadata["attempts"])
        total = (metadata.get("usage") or {}).get("total")
        if total is None:
            unknown_usage.append(int(path.parent.name))
            continue
        require(
            isinstance(total, dict)
            and all(
                type(total.get(key)) is int and total[key] >= 0
                for key in ("input_tokens", "output_tokens")
            ),
            "Invalid usage counters",
        )
        if "total_tokens" in total:
            require(
                total["total_tokens"] == total["input_tokens"] + total["output_tokens"],
                "Usage total mismatch",
            )
        for name, value in total.items():
            require(type(value) is int and value >= 0, "Invalid numeric usage field")
            usage[name] += value
    unanswered = [
        int(path.parent.name)
        for path in requests
        if not path.with_name("response.json").exists()
    ]
    require(
        not unanswered or unanswered == [len(requests)],
        "Only a prefix's final model request may be unanswered",
    )
    image_count = sum(len(read_json(path)["images"]) for path in requests)
    historical = sum(max(0, len(read_json(path)["images"]) - 2) for path in requests)
    known = not unknown_usage and not unanswered
    return {
        "requests": len(requests),
        "responses": len(responses),
        "sdk_attempts_observed": sdk_attempts,
        "reported_usage_lower_bound": dict(usage),
        "exact_input_tokens": usage["input_tokens"] if known else None,
        "exact_output_tokens": usage["output_tokens"] if known else None,
        "unknown_usage_decisions": unknown_usage,
        "unanswered_decisions": unanswered,
        "decision_image_attachments": image_count,
        "historical_image_attachments": historical,
        "operations": dict(operations),
    }


def audit_episode(
    attempt: JsonObject, reference: AlfredEpisode, run: JsonObject, boundary: Any
) -> JsonObject:
    directory = Path(attempt["artifact_dir"]).resolve(strict=True)
    completed = attempt["status"] == "completed"
    boundary_before = boundary.counts.copy()
    actions, observations = read_lines(directory / "actions.jsonl"), read_lines(
        directory / "observations.jsonl"
    )
    require(
        [row["observation"]["frame_id"] for row in observations]
        == list(range(len(observations))),
        "Observation frames are not contiguous",
    )
    require(
        (
            len(observations) == len(actions) + 1
            if completed
            else len(observations) <= len(actions) + 1
        ),
        "Action/observation counts differ",
    )
    by_frame = {row["observation"]["frame_id"]: row for row in observations}
    odometry = ActionOdometry()
    interactions = 0
    for index, row in enumerate(observations):
        observation = row["observation"]
        require(
            set(observation)
            == {
                "frame_id",
                "rgb_path",
                "yaw_degrees",
                "horizon_degrees",
                "last_action_success",
                "last_action_error",
            },
            "Unexpected simulator-facing observation field",
        )
        if index:
            action = actions[index - 1]
            previous = by_frame[index - 1]["observation"]
            odometry.update(
                action["action"],
                success=action["success"],
                yaw_degrees=previous["yaw_degrees"],
            )
            require(
                observation["last_action_success"] == action["success"],
                "Observation/action success differs",
            )
        require(
            np.allclose(row["agent_position"], odometry.agent_position),
            "Logged position differs from action odometry",
        )
        geometry = directory / "memory/observations" / f"frame_{index:06d}.npz"
        with np.load(geometry, allow_pickle=False) as saved:
            require(
                np.allclose(
                    saved["camera_to_world"],
                    odometry.camera_to_world(
                        yaw_degrees=observation["yaw_degrees"],
                        horizon_degrees=observation["horizon_degrees"],
                    ),
                )
                and np.allclose(saved["intrinsics"], camera_intrinsics()),
                "Saved geometry violates pose/intrinsic boundary",
            )
    for index, action in enumerate(actions, 1):
        require(
            action["index"] == index
            and action["source_frame"] == index - 1
            and action["result_frame"] == index,
            "Action/frame chronology differs",
        )
        require(index - 1 in by_frame, "Action has no preceding complete observation")
        interaction = action.get("interaction_mask")
        if interaction is None:
            continue
        interactions += 1
        path = Path(interaction["path"]).resolve(strict=True)
        require(
            path.is_relative_to(directory / "interaction_masks")
            and sha256(path) == interaction["sha256"],
            "Interaction mask path/hash differs",
        )
        source = by_frame[index - 1]
        detection_index = interaction["source_detection_index"]
        require(
            source["detection_objects"][str(detection_index)] == action["object_id"],
            "Interaction mask/instance association differs",
        )
        source_file = (
            directory / "memory/observations" / f"frame_{index - 1:06d}.mapping.npz"
        )
        with np.load(source_file, allow_pickle=False) as saved:
            indices = list(saved["detection_indices"])
            require(
                indices.count(detection_index) == 1,
                "Interaction mask missing from current effective mapping",
            )
            expected = saved["masks"][indices.index(detection_index)]
        supplied = np.load(path, allow_pickle=False)
        require(
            supplied.any() and np.array_equal(supplied, expected),
            "Interaction did not use the current effective mask",
        )
    delivered: set[int] = set()
    for request_path in sorted((directory / "decisions").glob("*/request.json")):
        payload = json.loads(read_json(request_path)["prompt"].rsplit("\n", 1)[-1])
        language: JsonObject = {"goal": reference.instruction.goal}
        if run["execution"]["use_step_instructions"]:
            language["step_instructions"] = list(reference.instruction.steps)
        require(
            payload["instruction"] == language,
            "Actor instruction differs from selected H/H+L annotation",
        )
        latest = boundary.request(
            request_path,
            by_frame,
            directory,
            delivered,
            model=run["model"],
            step_instructions=run["execution"]["use_step_instructions"],
        )
        require(latest >= 0, "Actor request cannot be bound to an observed frame")
        serialized = json.dumps(payload)
        require(
            not re.search(
                r"traj_data\.json|/json_2\.1\.0/|terminal_evaluation\.json", serialized
            ),
            "Evaluator artifact path leaked into actor request",
        )
    usage = audit_usage(directory, completed=completed)
    execution_path = directory / (
        "execution.json" if completed else "partial_execution.json"
    )
    execution = read_json(execution_path) if execution_path.exists() else None
    if execution is not None:
        require(
            execution["actions"] == len(actions)
            and execution["failures"]
            == sum(not action["success"] for action in actions),
            "Execution action/failure accounting differs",
        )
        require(
            (
                execution["decisions"] == usage["requests"]
                if completed
                else usage["requests"]
                <= execution["decisions"]
                <= usage["requests"] + 1
            ),
            "Execution decision accounting differs",
        )
        for kind in ("input", "output"):
            field = (
                f"{kind}_tokens" if completed else f"observed_{kind}_tokens_lower_bound"
            )
            reported = usage["reported_usage_lower_bound"].get(f"{kind}_tokens", 0)
            require(
                execution[field] is None or execution[field] == reported,
                "Execution/SDK usage differs",
            )
    if completed:
        require(execution is not None, "Completed task missing execution record")
        score = EpisodeScore.model_validate(attempt["score"])
        require(
            score.executed_actions == len(actions)
            and score.expert_actions == reference.expert_actions,
            "Official score path lengths differ",
        )
        terminal = read_json(directory / "simulator/terminal_evaluation.json")
        require(
            terminal["trajectory_sha256"] == reference.trajectory_sha256,
            "Terminal snapshot trajectory differs",
        )
        for field in ("success", "completed_goal_conditions", "total_goal_conditions"):
            require(
                terminal[field] == getattr(score, field),
                "Terminal snapshot score differs",
            )
        require(
            [(row["action"], row["success"]) for row in terminal["actions"]]
            == [(row["action"], row["success"]) for row in actions],
            "Terminal action trace differs",
        )
    boundary.counts.update(
        episodes=1, observations=len(observations), interactions=interactions
    )
    return {
        "run_id": run["run_id"],
        "episode_id": reference.episode_id,
        "split": reference.split,
        "task_type": reference.task_type,
        "artifact_dir": str(directory),
        "score": attempt["score"],
        "error": safe_error(attempt.get("error")),
        "execution": execution,
        "actions": len(actions),
        "failures": sum(not row["success"] for row in actions),
        "observations": len(observations),
        "interactions": interactions,
        "usage": usage,
        "boundary_counts": dict(boundary.counts - boundary_before),
    }


def summarize_rows(
    rows: list[JsonObject], *, include_metrics: bool = True
) -> JsonObject:
    scores = (
        [EpisodeScore.model_validate(row["score"]) for row in rows]
        if include_metrics
        else []
    )
    counts: Counter[str] = Counter()
    for row in rows:
        for field in ("actions", "failures", "observations", "interactions"):
            counts[field] += row[field]
        for field in (
            "requests",
            "responses",
            "sdk_attempts_observed",
            "decision_image_attachments",
            "historical_image_attachments",
        ):
            counts[field] += row["usage"][field]
    usage: Counter[str] = Counter()
    for row in rows:
        usage.update(row["usage"]["reported_usage_lower_bound"])
    exact = all(row["usage"]["exact_input_tokens"] is not None for row in rows)
    return {
        "episodes": len(rows),
        "metrics": aggregate_scores(scores).model_dump() if scores else None,
        "counts": dict(counts),
        "reported_usage_lower_bound": dict(usage),
        "exact_input_tokens": usage["input_tokens"] if exact else None,
        "exact_output_tokens": usage["output_tokens"] if exact else None,
        "policy_seconds": sum(
            row["execution"].get("policy_seconds", 0)
            for row in rows
            if row["execution"]
        ),
        "perception_seconds": sum(
            row["execution"].get("perception_seconds", 0)
            for row in rows
            if row["execution"]
        ),
    }


def historical_v8_reference(references: Sequence[AlfredEpisode]) -> JsonObject:
    """Match saved v8 outcomes to exactly these identities without new scoring."""
    source = ARCHIVED_HELPERS.parent / "audits/pilot50_analysis.json"
    archived = read_json(source)
    require(
        archived["status"] == "completed" and archived["scored"] == 50,
        "Historical reference is incomplete",
    )
    expected = {reference.episode_id: reference for reference in references}
    matched = [row for row in archived["episodes"] if row["episode_id"] in expected]
    require(
        len(matched) == len(expected)
        and len({row["episode_id"] for row in matched}) == len(expected),
        "Historical reference lacks unique matched identities",
    )
    evidence = []
    for reference in references:
        row = next(row for row in matched if row["episode_id"] == reference.episode_id)
        terminal_path = Path(row["artifact_dir"]) / "simulator/terminal_evaluation.json"
        terminal = read_json(terminal_path)
        score = EpisodeScore.model_validate(row["score"])
        require(
            terminal["trajectory_sha256"] == reference.trajectory_sha256,
            "Historical trajectory mismatch",
        )
        require(
            row["split"] == reference.split
            and row["task_type"] == reference.task_type
            and score.expert_actions == reference.expert_actions,
            "Historical annotation mismatch",
        )
        require(
            all(
                terminal[name] == getattr(score, name)
                for name in (
                    "success",
                    "completed_goal_conditions",
                    "total_goal_conditions",
                )
            ),
            "Historical snapshot/analysis score mismatch",
        )
        evidence.append(
            {
                "episode_id": reference.episode_id,
                "trajectory_sha256": reference.trajectory_sha256,
                "split": reference.split,
                "task_type": reference.task_type,
                "run_id": row["run_id"],
                "score": row["score"],
                "terminal_snapshot_path": str(terminal_path),
                "terminal_snapshot_sha256": sha256(terminal_path),
            }
        )
    scores = [EpisodeScore.model_validate(row["score"]) for row in evidence]
    prior_config = archived["matching_configuration"]
    return {
        "source_path": str(source),
        "source_sha256": sha256(source),
        "matched_count": len(evidence),
        "metrics": aggregate_scores(scores).model_dump(),
        "episodes": evidence,
        "model": prior_config["model"],
        "reasoning_effort": prior_config["reasoning_effort"],
        "execution": prior_config["execution"],
        "perception_profile": prior_config["perception_profile"],
        "limits": "Historical same-identity reference only. The policy, mapping, perception and model reasoning configuration differ; this is not a controlled memory ablation or a full-split performance comparison. No new historical run or rescore was performed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--expected-fold-sha", default=FOLD_SHA256)
    parser.add_argument(
        "--fold", type=Path, default=ROOT / "docs/benchmark/alfred/folds/pilot15.json"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ysh/Datasets/FILM/alfred/data/json_2.1.0"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("/home/ysh/Datasets/FILM/alfred_data_small/splits/oct21.json"),
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "docs/benchmark/alfred/runs.sqlite"
    )
    parser.add_argument("--sdk-runs-root", type=Path, default=ROOT / ".codex-home/runs")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument(
        "--prefix",
        type=int,
        choices=(5, 15),
        default=15,
        help="Use the user-requested first5 original identities, or the original full15 (default)",
    )
    parser.add_argument(
        "--alfred-root", type=Path, default=Path("/home/ysh/Datasets/FILM/alfred")
    )
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output exists; choose a fresh path")
    require(
        not (args.rescore and args.allow_partial),
        "Official final recheck is unavailable in partial snapshot mode",
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", args.expected_code_sha) is not None,
        "Expected code SHA must be a full SHA256",
    )
    require(sha256(args.fold) == args.expected_fold_sha, "Frozen pilot15 changed")
    full_references = load_manifest(args.fold, args.data_root, args.splits)
    require(len(full_references) == 15, "Expected the frozen15, not another fold")
    references = full_references[: args.prefix]
    by_id = {reference.episode_id: reference for reference in references}
    boundary_module = load_archived_helper("audit_pilot_boundary.py")

    BoundaryBase: Any = boundary_module.BoundaryAudit

    class PilotBoundary(BoundaryBase):
        def check(self, condition: bool, path: Path, code: str, message: str) -> bool:
            if code == "missing_response" and not condition:
                self.counts["unanswered_requests"] += 1
                return False
            return bool(super().check(condition, path, code, message))

    boundary = PilotBoundary(args.sdk_runs_root)
    signatures: list[JsonObject] = []
    worker_versions: list[JsonObject] = []
    run_records: list[JsonObject] = []
    all_attempts: list[JsonObject] = []
    runs: dict[str, JsonObject] = {}
    unavailable_runs: list[str] = []
    directories = [directory.absolute() for directory in args.run_dir]
    require(len(set(directories)) == len(directories), "Repeated run directory")
    for directory in directories:
        if (
            not (directory / "run.json").is_file()
            or not (directory / "summary.json").is_file()
        ):
            require(
                args.allow_partial, "Final audit requires all selected run artifacts"
            )
            unavailable_runs.append(str(directory))
            continue
        directory = directory.resolve(strict=True)
        run_bytes, summary_bytes = (directory / "run.json").read_bytes(), (
            directory / "summary.json"
        ).read_bytes()
        metadata, summary = json.loads(run_bytes), json.loads(summary_bytes)
        run_id, code = metadata["run_id"], metadata["code"]
        require(run_id not in runs, "Duplicate run ID")
        require(
            code["branch"] == "feat/alfred-benchmark"
            and code["tracked_worktree_drift"] == ""
            and code["executed_code_sha256"] == args.expected_code_sha,
            "Evaluation source differs from approved executable identity",
        )
        require(
            re.fullmatch(r"[0-9a-f]{40}", code["head_sha"]) is not None,
            "Run HEAD is not preserved",
        )
        require(
            Path(metadata["output_dir"]).resolve() == directory,
            "Run output path differs",
        )
        require(
            metadata["status"] in {"completed", "error", "interrupted", "running"},
            "Unexpected run status",
        )
        require(
            metadata["status"] != "running" or args.allow_partial,
            "Final audit refuses running sources",
        )
        require(
            metadata["fold_sha256"] == args.expected_fold_sha
            and metadata["splits_sha256"] == sha256(args.splits)
            and metadata["full_fold_size"] == 15,
            "Run fold differs",
        )
        selected = full_references[metadata["shard_index"] :: metadata["num_shards"]]
        if metadata["limit"] is not None:
            selected = selected[: metadata["limit"]]
        require(
            [row["episode_id"] for row in metadata["episodes"]]
            == [item.episode_id for item in selected],
            "Run selection differs from deterministic partition",
        )
        require(
            all(reference.episode_id in by_id for reference in selected),
            "Run selection falls outside the explicitly authorized prefix; cancellation records must stay separate",
        )
        for row, reference in zip(metadata["episodes"], selected, strict=True):
            require(
                all(
                    row[field] == getattr(reference, field)
                    for field in (
                        "split",
                        "task_type",
                        "trajectory_sha256",
                        "expert_actions",
                    )
                ),
                "Run episode metadata differs",
            )
        signatures.append(configuration_signature(metadata))
        runtime = metadata["perception"].get("runtime_provenance")
        if runtime is not None:
            worker_versions.append(runtime)
        attempts = summary["attempts"]
        require(
            [row["episode_id"] for row in attempts]
            == [item.episode_id for item in selected[: len(attempts)]],
            "Attempt order differs from selected episode prefix",
        )
        counts = Counter(row["status"] for row in attempts)
        require(
            summary["selected"] == len(selected)
            and summary["completed"] == counts["completed"]
            and summary["errors"] == counts["error"]
            and summary["interrupted"] == counts["interrupted"]
            and summary["pending"] == len(selected) - len(attempts),
            "Run summary counts differ",
        )
        source_scores = [
            EpisodeScore.model_validate(row["score"])
            for row in attempts
            if row["status"] == "completed"
        ]
        require(
            summary["metrics"]
            == (
                aggregate_scores(source_scores).model_dump() if source_scores else None
            ),
            "Source run aggregate differs from official saved scores",
        )
        db = (
            sqlite_check(
                args.database,
                run_id,
                metadata,
                attempts,
                allow_missing=args.allow_partial,
            )
            if metadata["status"] != "running"
            else {
                "run_id": run_id,
                "unverified": "running sources have not been ingested",
            }
        )
        for attempt in attempts:
            artifact = Path(attempt["artifact_dir"]).resolve(strict=True)
            require(
                artifact.is_relative_to(directory / "episodes"),
                "Episode artifact escapes its run",
            )
            require(
                read_json(artifact / "attempt.json") == attempt,
                "Attempt artifact differs from summary",
            )
            all_attempts.append({**attempt, "run_id": run_id})
        run_records.append(
            {
                "run_id": run_id,
                "output_dir": str(directory),
                "code": code,
                "status": metadata["status"],
                "run_json_sha256": hashlib.sha256(run_bytes).hexdigest(),
                "summary_json_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                "sqlite": db,
                "error": safe_error(metadata.get("error")),
                "worker_runtime_provenance": runtime,
                "worker_provenance_status": metadata["perception"].get(
                    "provenance_status"
                ),
                "started_at_utc": metadata["started_at_utc"],
                "finished_at_utc": metadata.get("finished_at_utc"),
                "reported_perception_statistics": metadata.get("perception_statistics"),
            }
        )
        runs[run_id] = metadata
    require(
        all(signature == signatures[0] for signature in signatures),
        "Run configurations are not matched",
    )
    require(
        all(version == worker_versions[0] for version in worker_versions),
        "Actual worker model environments differ",
    )
    selected_attempts, errors = select_first_scores(
        all_attempts, set(by_id), allow_partial=args.allow_partial
    )
    selected_attempts.sort(key=lambda row: list(by_id).index(row["episode_id"]))
    for attempt in selected_attempts:
        require(
            runs[attempt["run_id"]]["perception"].get("provenance_status")
            == "worker_reported",
            "Scored run lacks actual worker model provenance",
        )
    rows = [
        audit_episode(
            attempt, by_id[attempt["episode_id"]], runs[attempt["run_id"]], boundary
        )
        for attempt in selected_attempts
    ]
    prefixes = [
        audit_episode(
            attempt, by_id[attempt["episode_id"]], runs[attempt["run_id"]], boundary
        )
        for attempt in errors
    ]
    require(
        not boundary.violations,
        "Actor input/tool boundary violations found; no final report emitted",
    )
    closed_runs = [run for run in run_records if run["finished_at_utc"] is not None]
    elapsed_span = None
    if closed_runs:
        earliest = min(
            datetime.fromisoformat(run["started_at_utc"]) for run in closed_runs
        )
        latest = max(
            datetime.fromisoformat(run["finished_at_utc"]) for run in closed_runs
        )
        elapsed_span = {
            "start_utc": earliest.isoformat(),
            "end_utc": latest.isoformat(),
            "seconds": (latest - earliest).total_seconds(),
            "scope": "terminal runs only; includes inter-run gaps",
        }
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "partial_snapshot"
            if args.allow_partial
            else (
                "complete_verified_user_prefix5"
                if args.prefix == 5
                else "complete_verified_pilot15"
            )
        ),
        "selection": {
            "mode": (
                "user_requested_original_prefix"
                if args.prefix == 5
                else "original_frozen_fold"
            ),
            "full_fold_size": 15,
            "selected_prefix_length": args.prefix,
            "global_indices": list(range(args.prefix)),
            "episode_ids": list(by_id),
            "fold_unchanged": True,
            "scope_note": (
                "The user reduced the run count to the first5 original identities; no replacement/resampling of identities or scored retries"
                if args.prefix == 5
                else "Original frozen15 cohort"
            ),
        },
        "expected": args.prefix,
        "scored": len(rows),
        "coverage_complete": len(rows) == args.prefix,
        "unscored_error_attempts": len(prefixes),
        "unattempted_unique": sorted(
            set(by_id) - {entry["episode_id"] for entry in all_attempts}
        ),
        "missing_scores": [
            identity
            for identity in by_id
            if identity not in {row["episode_id"] for row in rows}
        ],
        "unavailable_run_directories": unavailable_runs,
        "fold_path": str(args.fold),
        "fold_sha256": args.expected_fold_sha,
        "executed_code_sha256": args.expected_code_sha,
        "script_sha256": sha256(Path(__file__)),
        "helper_sha256": {
            "audit_pilot_boundary.py": sha256(
                ARCHIVED_HELPERS / "audit_pilot_boundary.py"
            )
        },
        "configuration": signatures[0] if signatures else None,
        "runs": run_records,
        "terminal_run_elapsed_span": elapsed_span,
        "scored_summary": summarize_rows(rows),
        "error_prefix_summary": summarize_rows(prefixes, include_metrics=False),
        "by_split": {
            split: summarize_rows([row for row in rows if row["split"] == split])
            for split in ("valid_seen", "valid_unseen")
        },
        "by_task_type": {
            kind: summarize_rows([row for row in rows if row["task_type"] == kind])
            for kind in sorted({ref.task_type for ref in references})
        },
        "episodes": rows,
        "unscored_prefixes": prefixes,
        "actor_boundary": {
            "passed": True,
            "completed_episodes_checked": len(rows),
            "unscored_prefixes_checked": len(prefixes),
            "active_inflight_episodes_checked": 0,
            "counts_including_error_prefixes": dict(boundary.counts),
            "response_item_types": dict(boundary.response_item_types),
            "catalog_hashes": sorted(boundary.catalog_hashes),
            "artifact_manifest": boundary.manifest,
        },
        "limits": [
            "Error text is hashed/classified, never copied into this report.",
            "Usage is exact only when every saved response provides counters; unanswered/error requests remain unknown.",
            "Images count decision attachments, not all wire retransmissions.",
            "Catalogs/traces verify declared host-tool isolation and observed executions; full wire tool arrays are not captured for every turn.",
            "Effective current masks are verified against saved mapping arrays, not against simulator ground-truth instances.",
            "Summed policy/perception times are not elapsed runtime.",
            "Partial mode reports persisted attempts only; an active episode is not made into a scored or error attempt.",
        ],
    }
    if args.prefix == 5:
        report["historical_v8_same_identity_reference"] = historical_v8_reference(
            references
        )
    if args.rescore:
        official = load_archived_helper("rescore_pilot50.py")
        goals, source = official.prepare_goals(args.alfred_root)
        results = [
            official.rescore_one(
                by_id[row["episode_id"]],
                Path(row["artifact_dir"]) / "simulator/terminal_evaluation.json",
                row["score"],
                row["run_id"],
                goals,
            )
            for row in rows
        ]
        require(
            all(row["matches"] for row in results),
            "Independent official terminal score differs",
        )
        report["official_terminal_recheck"] = {
            "checked": len(results),
            "all_match": True,
            "official_source": source,
            "episodes": results,
        }
        report["helper_sha256"]["rescore_pilot50.py"] = sha256(
            ARCHIVED_HELPERS / "rescore_pilot50.py"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "scored": len(rows),
                "errors": len(prefixes),
                "metrics": report["scored_summary"]["metrics"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
