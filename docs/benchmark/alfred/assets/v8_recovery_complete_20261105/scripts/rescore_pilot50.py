"""Independently recheck all 50 terminal states with pinned official ALFRED goals.

Run only after all four direct shards finish. This script never imports ThorEnv,
AI2-THOR, model weights, navigation graphs, or task constructors. It compiles the
unchanged official goal methods and their one object-selector helper from
hash-verified source. Only saved terminal metadata, state-history sets, and the
matched trajectory's goal parameters are supplied to those methods.

Example (inside tmux, in agentic3d):
    python tmp/alfred_setup/rescore_pilot50.py
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from codex_agent.alfred.dataset import AlfredEpisode, load_manifest

ROOT = Path("/home/ysh/codebase/agentic-3d-task")
ALFRED_ROOT = Path("/home/ysh/Datasets/FILM/alfred")
OFFICIAL_COMMIT = "f91f4c0c96c7a29f33d0557f86b0a21035379b3b"
PINNED_SOURCES = {
    "env/tasks.py": "710a17085fa174f17f5cb06f11bcf9eafa8ba3be8d40defac8b006dc02694575",
    "gen/utils/game_util.py": "ec5e164425d3e94d47a0e17ff47a3b0131a1909befc9cf56adfa17fa5020920c",
}
TASK_CLASSES = {
    "pick_and_place_simple": "PickAndPlaceSimpleTask",
    "pick_two_obj_and_place": "PickTwoObjAndPlaceTask",
    "look_at_obj_in_light": "LookAtObjInLightTask",
    "pick_heat_then_place_in_recep": "PickHeatThenPlaceInRecepTask",
    "pick_cool_then_place_in_recep": "PickCoolThenPlaceInRecepTask",
    "pick_clean_then_place_in_recep": "PickCleanThenPlaceInRecepTask",
    "pick_and_place_with_movable_recep": "PickAndPlaceWithMovableRecepTask",
}
STATE_FIELDS = ("cleaned_objects", "cooled_objects", "heated_objects")
SCORE_FIELDS = ("success", "completed_goal_conditions", "total_goal_conditions")


def require(condition: bool, message: str) -> None:
    """Keep verification active even when Python optimization is enabled."""
    if not condition:
        raise ValueError(message)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    """Read one external JSON object; shape-specific validation follows below."""
    result = json.loads(path.read_bytes())
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return result


def prepare_goals(alfred_root: Path) -> tuple[dict[str, Any], dict[str, object]]:
    """Compile original goal bodies only, excluding imports and constructors."""
    import numpy as np

    source_trees: dict[str, ast.Module] = {}
    source_provenance: dict[str, object] = {}
    for relative, expected_hash in PINNED_SOURCES.items():
        path = alfred_root / relative
        content = path.read_bytes()
        actual_hash = digest(content)
        require(actual_hash == expected_hash, f"official source drift: {path}")
        source_trees[relative] = ast.parse(content.decode("utf-8"), filename=str(path))
        source_provenance[relative] = {"path": str(path), "sha256": actual_hash}

    namespace: dict[str, Any] = {"np": np, "__name__": "_alfred_saved_terminal_goals"}
    helpers = [
        node
        for node in source_trees["gen/utils/game_util.py"].body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_objects_with_name_and_prop"
    ]
    require(len(helpers) == 1, "expected exactly one pinned object-selector helper")
    exec(
        compile(
            ast.Module(body=list(helpers), type_ignores=[]),
            str(alfred_root / "gen/utils/game_util.py"),
            "exec",
        ),
        namespace,
    )

    compiled_names: set[str] = set()
    allowed_classes = {"BaseTask", *TASK_CLASSES.values()}
    for node in source_trees["env/tasks.py"].body:
        if not isinstance(node, ast.ClassDef) or node.name not in allowed_classes:
            continue
        method_names = (
            {"get_target", "get_targets"}
            if node.name == "BaseTask"
            else {"goal_satisfied", "goal_conditions_met"}
        )
        methods = [
            item
            for item in node.body
            if isinstance(item, ast.FunctionDef) and item.name in method_names
        ]
        require(
            {item.name for item in methods} == method_names,
            f"official goal methods missing: {node.name}",
        )
        isolated = copy.deepcopy(node)
        isolated.body = list(methods)
        module = ast.Module(body=[isolated], type_ignores=[])
        exec(compile(module, str(alfred_root / "env/tasks.py"), "exec"), namespace)
        compiled_names.add(node.name)
    require(
        compiled_names == allowed_classes, "official task-class coverage incomplete"
    )
    return namespace, {
        "official_commit": OFFICIAL_COMMIT,
        "sources": source_provenance,
        "numpy_version": np.__version__,
        "loaded_classes": sorted(compiled_names),
        "method": "original hash-pinned goal/helper AST bodies; object.__new__; no constructors/imported simulator",
    }


def completed_snapshots(
    project_root: Path, expected: tuple[AlfredEpisode, ...]
) -> list[tuple[AlfredEpisode, Path, dict[str, Any], str]]:
    """Refuse partial runs, then bind every saved score to exactly one fold entry."""
    rows: list[tuple[AlfredEpisode, Path, dict[str, Any], str]] = []
    observed: set[str] = set()
    for shard_index in range(4):
        run_id = f"REPRO_20261105_ALFRED_V3_20261105_pilot50_s{shard_index}_direct"
        directory = project_root / "outputs/alfred" / run_id
        run = read_object(directory / "run.json")
        summary = read_object(directory / "summary.json")
        require(
            run["status"] == "completed", f"run is not terminal-completed: {run_id}"
        )
        require(
            summary["completed"] == summary["selected"], f"incomplete scores: {run_id}"
        )
        require(
            summary["errors"] == summary["interrupted"] == summary["pending"] == 0,
            f"run has unscored attempts: {run_id}",
        )
        selection = expected[shard_index::4]
        require(len(selection) == summary["selected"], f"shard size mismatch: {run_id}")
        expected_by_id = {episode.episode_id: episode for episode in selection}
        require(
            [entry["episode_id"] for entry in run["episodes"]] == list(expected_by_id),
            f"frozen shard identity/order mismatch: {run_id}",
        )
        require(
            len(summary["attempts"]) == len(selection),
            f"attempt count mismatch: {run_id}",
        )
        for attempt in summary["attempts"]:
            identity = attempt["episode_id"]
            require(
                identity in expected_by_id and identity not in observed,
                f"unexpected/duplicate episode: {identity}",
            )
            require(
                attempt["status"] == "completed" and isinstance(attempt["score"], dict),
                f"episode lacks terminal score: {identity}",
            )
            artifact_dir = Path(attempt["artifact_dir"]).resolve(strict=True)
            require(
                artifact_dir.is_relative_to(directory.resolve()),
                f"artifact path escapes run: {identity}",
            )
            rows.append(
                (
                    expected_by_id[identity],
                    artifact_dir / "simulator/terminal_evaluation.json",
                    attempt["score"],
                    run_id,
                )
            )
            observed.add(identity)
    require(
        len(rows) == 50 and observed == {episode.episode_id for episode in expected},
        "expected exactly the full frozen 50 without missing or duplicate identities",
    )
    return rows


def rescore_one(
    episode: AlfredEpisode,
    snapshot_path: Path,
    saved_score: dict[str, Any],
    run_id: str,
    goals: dict[str, Any],
) -> dict[str, object]:
    """Evaluate an immutable terminal-state copy and compare both saved score views."""
    snapshot_bytes = snapshot_path.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    require(isinstance(snapshot, dict), f"invalid terminal snapshot: {snapshot_path}")
    require(
        snapshot["trajectory_sha256"] == episode.trajectory_sha256,
        f"snapshot trajectory hash mismatch: {episode.episode_id}",
    )
    require(
        Path(snapshot["trajectory_path"]).resolve(strict=True)
        == episode.trajectory_path.resolve(strict=True),
        f"snapshot trajectory path mismatch: {episode.episode_id}",
    )
    trajectory_bytes = episode.trajectory_path.read_bytes()
    require(
        digest(trajectory_bytes) == episode.trajectory_sha256,
        f"trajectory changed: {episode.episode_id}",
    )
    trajectory = json.loads(trajectory_bytes)
    require(
        trajectory["task_type"] == episode.task_type,
        f"trajectory task type mismatch: {episode.episode_id}",
    )
    require(
        isinstance(snapshot["metadata"], dict),
        f"terminal metadata missing: {episode.episode_id}",
    )
    histories: dict[str, frozenset[str]] = {}
    for name in STATE_FIELDS:
        values = snapshot[name]
        require(
            isinstance(values, list) and all(isinstance(item, str) for item in values),
            f"invalid {name}: {episode.episode_id}",
        )
        histories[name] = frozenset(values)
    for record in (snapshot, saved_score):
        require(
            type(record["success"]) is bool,
            f"invalid saved success: {episode.episode_id}",
        )
        require(
            all(type(record[key]) is int for key in SCORE_FIELDS[1:]),
            f"invalid saved goal counts: {episode.episode_id}",
        )

    task_class = goals[TASK_CLASSES[episode.task_type]]
    task = object.__new__(task_class)
    task.traj = {"pddl_params": copy.deepcopy(trajectory["pddl_params"])}
    task.env = SimpleNamespace(**histories)
    event = SimpleNamespace(metadata=copy.deepcopy(snapshot["metadata"]))
    completed, total = task.goal_conditions_met(event)
    success = bool(task.goal_satisfied(event))
    completed, total = int(completed), int(total)
    require(
        0 <= completed <= total and total > 0 and success == (completed == total),
        f"invalid official result: {episode.episode_id}",
    )
    actual = {
        "success": success,
        "completed_goal_conditions": completed,
        "total_goal_conditions": total,
    }
    snapshot_score = {key: snapshot[key] for key in SCORE_FIELDS}
    attempt_score = {key: saved_score[key] for key in SCORE_FIELDS}
    return {
        "episode_id": episode.episode_id,
        "run_id": run_id,
        "task_type": episode.task_type,
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": digest(snapshot_bytes),
        "trajectory_sha256": episode.trajectory_sha256,
        "official_class": TASK_CLASSES[episode.task_type],
        "state_history_counts": {key: len(value) for key, value in histories.items()},
        "saved_snapshot_score": snapshot_score,
        "saved_attempt_score": attempt_score,
        "recomputed_score": actual,
        "matches": actual == snapshot_score == attempt_score,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--alfred-root", type=Path, default=ALFRED_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tmp/alfred_setup/pilot50_terminal_rescore.json",
    )
    args = parser.parse_args()
    require(
        not args.output.exists(),
        f"output already exists; choose a new path: {args.output}",
    )
    fold_path = args.project_root / "docs/benchmark/alfred/folds/pilot50.json"
    expected = load_manifest(
        fold_path,
        args.alfred_root / "data/json_2.1.0",
        Path("/home/ysh/Datasets/FILM/alfred_data_small/splits/oct21.json"),
    )
    rows = completed_snapshots(args.project_root, expected)
    goals, provenance = prepare_goals(args.alfred_root)
    results = [rescore_one(*row, goals) for row in rows]
    mismatches = [result["episode_id"] for result in results if not result["matches"]]
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Evaluator-only independent terminal recheck; no replay, model calls, Unity startup, or actor modifications.",
        "fold_path": str(fold_path),
        "fold_sha256": digest(fold_path.read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "official_provenance": provenance,
        "expected": 50,
        "checked": len(results),
        "matching": len(results) - len(mismatches),
        "all_match": not mismatches,
        "mismatch_episode_ids": mismatches,
        "episodes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
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
