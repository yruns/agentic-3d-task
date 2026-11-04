"""Read-only final actor-boundary audit; run in tmux after the four pilot shards end.

Usage in the existing agentic3d environment::

    python tmp/alfred_setup/audit_pilot_boundary.py

Only the newly named output report is written. Existing snapshots, source,
episode artifacts, runtime configuration, environment and credentials are never
modified. No simulator, predictor, scorer or remote model is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORBIDDEN_KEYS = frozenset(
    {
        "cameraPosition",
        "cleaned_objects",
        "completed_goal_conditions",
        "cooled_objects",
        "episode_id",
        "expert_actions",
        "expert_plan",
        "global_pose",
        "global_position",
        "heated_objects",
        "high_pddl",
        "inventoryObjects",
        "low_actions",
        "objectId",
        "pddl_params",
        "receptacleObjectIds",
        "task_id",
        "task_type",
        "terminal_score",
        "total_goal_conditions",
        "trajectory_path",
        "trajectory_sha256",
    }
)
OBSERVATION_KEYS = frozenset(
    {
        "frame_id",
        "horizon_degrees",
        "last_action_error",
        "last_action_success",
        "rgb_path",
        "yaw_degrees",
    }
)
MOTION = frozenset({"MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"})
SIMULATOR_ID = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*(?:\|[+-]?\d+\.\d+){3}")
JsonObject = dict[str, Any]  # JSON artifact IO boundary; never an executable payload.


def read_json(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[JsonObject]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Expected JSON-object records: {path}")
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BoundaryAudit:
    """Collect explicit violations and non-secret evidence references."""

    def __init__(self, sdk_runs_root: Path) -> None:
        self.sdk_runs_root = sdk_runs_root.resolve()
        self.violations: list[JsonObject] = []
        self.counts: Counter[str] = Counter()
        self.response_item_types: Counter[str] = Counter()
        self.catalog_hashes: set[str] = set()
        self.manifest: list[JsonObject] = []
        self.sdk_homes: set[Path] = set()

    def check(self, condition: bool, path: Path, code: str, message: str) -> bool:
        if not condition:
            self.violations.append(
                {"path": str(path), "code": code, "message": message}
            )
        return condition

    def file_in_scope(self, path: Path, root: Path, *, role: str) -> bool:
        return self.check(
            path.is_absolute()
            and path.resolve().is_relative_to(root.resolve())
            and path.is_file(),
            path,
            "file_scope",
            f"{role} must be an existing file within {root}",
        )

    def forbidden_payload(self, value: Any, path: Path, trail: str = "payload") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                self.check(
                    key not in FORBIDDEN_KEYS,
                    path,
                    "forbidden_truth_key",
                    f"Forbidden key at {trail}.{key}",
                )
                self.forbidden_payload(item, path, f"{trail}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self.forbidden_payload(item, path, f"{trail}[{index}]")
        elif isinstance(value, str):
            self.check(
                not SIMULATOR_ID.search(value),
                path,
                "simulator_identity_leak",
                f"Simulator-style object ID found at {trail}; value omitted",
            )

    def sdk_response(
        self, response_path: Path, request_images: list[str], model: str
    ) -> None:
        response = read_json(response_path)
        self.counts["completed_responses"] += 1
        home_value = response["metadata"].get("run_home")
        if not self.check(
            isinstance(home_value, str),
            response_path,
            "missing_run_home",
            "Completed response must identify its preserved SDK run home",
        ):
            return
        home = Path(home_value)
        if not self.check(
            home.is_absolute()
            and home.resolve().is_relative_to(self.sdk_runs_root)
            and home.is_dir(),
            home,
            "sdk_home_scope",
            "SDK run home must exist within the explicit audit SDK-runs root",
        ):
            return
        if not self.check(
            home not in self.sdk_homes,
            response_path,
            "shared_sdk_home",
            "Independent decisions must have distinct SDK run homes",
        ):
            return
        self.sdk_homes.add(home)
        catalog_path = home / "alfred_actor_catalog.json"
        catalog = read_json(catalog_path)
        models = catalog.get("models", [])
        self.check(
            len(models) == 1,
            catalog_path,
            "catalog_models",
            "Actor catalog must declare one exact model",
        )
        if len(models) != 1:
            return
        expected = {
            "slug": model,
            "shell_type": "disabled",
            "apply_patch_tool_type": None,
            "experimental_supported_tools": [],
            "multi_agent_version": "disabled",
            "node_repl_disabled": True,
            "tool_mode": "direct",
            "include_skills_usage_instructions": False,
            "include_apps_usage_instructions": False,
            "include_plugin_usage_instructions": False,
        }
        for key, expected_value in expected.items():
            self.check(
                key in models[0] and models[0][key] == expected_value,
                catalog_path,
                "host_tool_catalog",
                f"Catalog field {key} must equal {expected_value!r}",
            )
        catalog_hash = sha256(catalog_path)
        self.catalog_hashes.add(catalog_hash)
        self.counts["catalogs"] += 1
        traces = sorted((home / "sessions").rglob("*.jsonl"))
        self.check(
            bool(traces), home, "missing_sdk_trace", "A preserved SDK trace is required"
        )
        attached_images: set[str] = set()
        for trace in traces:
            if not self.file_in_scope(trace, home, role="SDK trace"):
                continue
            for ordinal, record in enumerate(read_jsonl(trace), start=1):
                payload = record.get("payload", {})
                if record.get("type") == "response_item":
                    kind = str(payload.get("type"))
                    self.response_item_types[kind] += 1
                    self.check(
                        kind in {"message", "reasoning"},
                        trace,
                        "sdk_response_tool_item",
                        f"Unexpected response item {kind} at record {ordinal}; content omitted",
                    )
                if (
                    record.get("type") == "event_msg"
                    and payload.get("type") == "item_completed"
                ):
                    item = payload.get("item", {})
                    kind = str(item.get("type"))
                    self.check(
                        kind in {"UserMessage", "AgentMessage", "Reasoning"},
                        trace,
                        "sdk_event_tool_item",
                        f"Unexpected completed item {kind} at record {ordinal}; content omitted",
                    )
                    if kind == "UserMessage":
                        for content in item.get("content", []):
                            if content.get("type") == "local_image":
                                attached_images.add(content["path"])
            self.counts["sdk_trace_files"] += 1
            self.manifest.append(
                {"kind": "sdk_trace", "path": str(trace), "sha256": sha256(trace)}
            )
        self.check(
            attached_images == set(request_images),
            response_path,
            "sdk_image_set",
            "SDK local-image attachments must match the saved actor request",
        )
        self.manifest.append(
            {"kind": "catalog", "path": str(catalog_path), "sha256": catalog_hash}
        )

    def request(
        self,
        path: Path,
        observations: dict[int, JsonObject],
        episode: Path,
        delivered: set[int],
        *,
        model: str,
        step_instructions: bool,
    ) -> int:
        request = read_json(path)
        payload = json.loads(request["prompt"].rsplit("\n", 1)[-1])
        self.check(
            set(payload) == {"instruction", "observations"},
            path,
            "actor_payload_keys",
            "Actor payload must contain instruction and observations only",
        )
        self.forbidden_payload(payload, path)
        allowed_language = (
            {"goal", "step_instructions"} if step_instructions else {"goal"}
        )
        self.check(
            set(payload["instruction"]) == allowed_language,
            path,
            "language_profile",
            "Instruction fields must match the recorded H/H+L profile",
        )
        context = payload["observations"]
        memory = context["scene_memory"]
        latest = memory["latest_frame_id"]
        if not self.check(
            type(latest) is int and latest in observations,
            path,
            "current_frame",
            "Current frame must have a recorded observation",
        ):
            return -1
        observation = observations[latest]
        self.check(
            memory["agent_position"] == observation["agent_position"],
            path,
            "memory_odometry",
            "Actor memory position must equal logged action odometry",
        )
        self.check(
            memory["frame_count"] == sum(frame <= latest for frame in observations),
            path,
            "future_memory_frames",
            "Actor frame count must not include future observations",
        )
        self.check(
            all(
                item["result_frame"] <= latest
                for item in context.get("recent_actions", [])
            ),
            path,
            "future_action_feedback",
            "Recent action feedback cannot reference future frames",
        )
        for node in memory["objects"]:
            self.check(
                node["last_seen"] <= latest
                and all(frame <= latest for frame in node["recent_frames"]),
                path,
                "future_object_observation",
                "Object catalog references must have been observed",
            )
        image_paths = [Path(value) for value in request["images"]]
        historical = context["retrieved_frame_ids"]
        self.check(
            len(image_paths) == 2 + len(historical),
            path,
            "image_count",
            "Expected current RGB + BEV + the declared historical frame list",
        )
        self.check(
            len(image_paths) >= 2
            and image_paths[:2]
            == [path.parent / "current.png", path.parent / "bev.png"],
            path,
            "current_bev_paths",
            "Current/BEV images must come from this exact decision directory",
        )
        for image_path in image_paths:
            if self.file_in_scope(image_path, episode, role="Actor image"):
                self.check(
                    image_path.stat().st_mtime_ns <= path.stat().st_mtime_ns,
                    path,
                    "future_image_artifact",
                    "Image artifact must exist before the saved request",
                )
            self.counts["image_references"] += 1
        for offset, frame in enumerate(historical, start=2):
            if not self.check(
                type(frame) is int and frame in observations and frame < latest,
                path,
                "historical_frame",
                "Historical image must be an earlier observed frame, excluding current RGB",
            ):
                continue
            self.check(
                frame not in delivered,
                path,
                "duplicate_history_delivery",
                "Historical frames should not be delivered repeatedly within an episode",
            )
            delivered.add(frame)
            if offset < len(image_paths):
                self.check(
                    image_paths[offset]
                    == Path(observations[frame]["observation"]["rgb_path"]),
                    path,
                    "historical_image_path",
                    "Historical path must match that same episode's recorded raw RGB",
                )
        raw_rgb = Path(observation["observation"]["rgb_path"])
        if self.file_in_scope(raw_rgb, episode, role="Current raw RGB"):
            self.check(
                raw_rgb.stat().st_mtime_ns <= path.stat().st_mtime_ns,
                path,
                "future_observation",
                "Current raw RGB must precede the request",
            )
        self.counts["saved_requests"] += 1
        self.manifest.append(
            {"kind": "request", "path": str(path), "sha256": sha256(path)}
        )
        response = path.with_name("response.json")
        if self.check(
            response.is_file(),
            path,
            "missing_response",
            "A terminal completed episode cannot have an unanswered saved request",
        ):
            self.sdk_response(response, request["images"], model)
        return latest

    def episode(self, episode: Path, *, model: str, step_instructions: bool) -> None:
        import numpy as np

        records = read_jsonl(episode / "observations.jsonl")
        observations = {record["observation"]["frame_id"]: record for record in records}
        self.check(
            len(observations) == len(records),
            episode,
            "duplicate_observation",
            "Observation frame IDs must be unique",
        )
        for record in records:
            observation = record["observation"]
            self.check(
                set(observation) == OBSERVATION_KEYS,
                episode,
                "observation_boundary",
                "Only RGB/proprioception/action feedback fields may leave the simulator",
            )
            self.forbidden_payload(observation, episode / "observations.jsonl")
            self.file_in_scope(
                Path(observation["rgb_path"]), episode, role="Observed RGB"
            )
            self.check(
                record["prediction"]["frame_id"] == str(observation["frame_id"]),
                episode,
                "prediction_frame",
                "Prediction frame ID must match observed RGB",
            )
        self.counts["observations"] += len(records)
        action_path = episode / "actions.jsonl"
        actions = read_jsonl(action_path) if action_path.exists() else []
        x = y = 0.0
        self.check(
            observations[0]["agent_position"] == [0.0, 0.0, 0.0],
            episode,
            "initial_odometry",
            "Odometry must start at episode-relative zero",
        )
        previous_frame = 0
        for ordinal, action in enumerate(actions, start=1):
            source, result = action["source_frame"], action["result_frame"]
            if not self.check(
                source == previous_frame
                and result == source + 1
                and result in observations
                and action["index"] == ordinal,
                action_path,
                "action_frame_chain",
                "Each atomic action must use the latest frame and produce the next observation",
            ):
                continue
            previous_frame = result
            source_record = observations[source]
            if action["action"] == "MoveAhead" and action["success"]:
                yaw = math.radians(source_record["observation"]["yaw_degrees"])
                x, y = round(x + 0.25 * math.sin(yaw), 9), round(
                    y + 0.25 * math.cos(yaw), 9
                )
            self.check(
                observations[result]["agent_position"] == [x, y, 0.0],
                action_path,
                "action_odometry",
                "Logged odometry must follow successful moves, not simulator position metadata",
            )
            if action["action"] in MOTION:
                continue
            indices = [
                int(index)
                for index, identity in source_record["detection_objects"].items()
                if identity == action["object_id"]
            ]
            if not self.check(
                len(indices) == 1,
                action_path,
                "current_interaction_mask",
                f"Action {ordinal} must identify one mask in its exact source frame",
            ):
                continue
            index = indices[0]
            detection = source_record["prediction"]["detections"][index]
            mask_path = Path(detection["mask_path"])
            if not self.file_in_scope(
                mask_path, episode / "perception", role="Predicted interaction mask"
            ):
                continue
            self.check(
                mask_path.name.startswith(f"{source}_mask_"),
                mask_path,
                "mask_filename_frame",
                "Mask filename must identify the action's source frame",
            )
            mask = np.load(mask_path, allow_pickle=False)
            self.check(
                mask.shape == (300, 300)
                and np.isin(mask, [0, 1]).all()
                and bool(mask.any()),
                mask_path,
                "binary_mask",
                "Mask must be nonempty binary 300x300",
            )
            geometry_path = (
                episode / "memory" / "observations" / f"frame_{source:06d}.npz"
            )
            with np.load(geometry_path, allow_pickle=False) as geometry:
                self.check(
                    np.array_equal(mask.astype(bool), geometry["masks"][index]),
                    mask_path,
                    "memory_mask_match",
                    "Interaction mask must equal the source frame's preserved memory mask",
                )
            self.counts["interactions"] += 1
            self.manifest.append(
                {
                    "kind": "interaction_mask",
                    "path": str(mask_path),
                    "sha256": sha256(mask_path),
                    "source_frame": source,
                    "action_index": ordinal,
                }
            )
        delivered: set[int] = set()
        latest = -1
        requests = sorted((episode / "decisions").glob("*/request.json"))
        for request in requests:
            current = self.request(
                request,
                observations,
                episode,
                delivered,
                model=model,
                step_instructions=step_instructions,
            )
            self.check(
                current >= latest,
                request,
                "decision_chronology",
                "Decision frames cannot move backwards",
            )
            latest = current
        self.counts["episodes"] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        help="Repeat for each frozen-cohort shard; default is the four V3 direct runs.",
    )
    parser.add_argument(
        "--sdk-runs-root", type=Path, default=project / ".codex-home/runs"
    )
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON report path; existing files are never overwritten.",
    )
    args = parser.parse_args()
    if args.expected_episodes < 1:
        parser.error("--expected-episodes must be positive")
    run_dirs = args.run_dir or [
        project
        / "outputs/alfred"
        / f"REPRO_20261105_ALFRED_V3_20261105_pilot50_s{index}_direct"
        for index in range(4)
    ]
    timestamp = datetime.now(timezone.utc)
    output = (
        args.output
        or project
        / "tmp/alfred_setup"
        / f"actor_boundary_audit_final_{timestamp:%Y%m%dT%H%M%SZ}.json"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an audit snapshot: {output}")
    audit = BoundaryAudit(args.sdk_runs_root)
    run_reports: list[JsonObject] = []
    episode_ids: list[str] = []
    fold_hashes: set[str] = set()
    code_hashes: set[str] = set()
    for root_arg in run_dirs:
        root = root_arg.resolve()
        metadata = read_json(root / "run.json")
        summary = read_json(root / "summary.json")
        audit.check(
            metadata["status"] == "completed"
            and summary["completed"] == summary["selected"]
            and summary["errors"] == summary["interrupted"] == summary["pending"] == 0,
            root,
            "terminal_cohort",
            "Final audit requires every selected episode completed and no errors/pending entries",
        )
        fold_hashes.add(metadata["fold_sha256"])
        code_hashes.add(metadata["code"]["executed_code_sha256"])
        episode_ids.extend(attempt["episode_id"] for attempt in summary["attempts"])
        before = audit.counts.copy()
        for episode in sorted((root / "episodes").iterdir()):
            if episode.is_dir():
                audit.episode(
                    episode,
                    model=metadata["model"],
                    step_instructions=metadata["execution"]["use_step_instructions"],
                )
        run_reports.append(
            {
                "run_id": metadata["run_id"],
                "status": metadata["status"],
                "selected": summary["selected"],
                "completed": summary["completed"],
                "code": metadata["code"],
                "fold_sha256": metadata["fold_sha256"],
                "counts": dict(audit.counts - before),
            }
        )
    audit.check(
        len(episode_ids) == args.expected_episodes
        and len(set(episode_ids)) == args.expected_episodes,
        output,
        "cohort_size",
        f"Expected exactly {args.expected_episodes} distinct completed episode IDs",
    )
    audit.check(
        len(fold_hashes) == 1 and len(code_hashes) == 1,
        output,
        "matched_provenance",
        "All shards must share the same frozen fold and executed code digest",
    )
    report = {
        "checked_at_utc": timestamp.isoformat(),
        "scope": "Final read-only actor-boundary audit of all explicitly selected run artifacts",
        "passed": not audit.violations,
        "runs": run_reports,
        "counts": dict(audit.counts),
        "forbidden_truth_keys_checked": sorted(FORBIDDEN_KEYS),
        "sdk_response_item_types": dict(audit.response_item_types),
        "catalog_sha256_values": sorted(audit.catalog_hashes),
        "violations": audit.violations,
        "artifact_manifest": audit.manifest,
        "artifact_manifest_sha256": hashlib.sha256(
            json.dumps(audit.manifest, sort_keys=True).encode()
        ).hexdigest(),
        "limits": [
            "No runtime config, process environment, credentials, simulator terminal metadata or dataset trajectories are read.",
            "Fresh-mask proof joins action source frame/memory identity to the corresponding prediction and saved memory NPZ; original simulator RPC mask arguments were not separately persisted.",
            "Catalog checks verify declared host-tool isolation; preserved SDK traces verify no host-tool executions. Full on-wire tool lists were not captured for every pilot request; earlier synthetic wire audits remain separate evidence.",
            "Image path, existence, chronology and SDK-attachment consistency are checked; image pixels are not re-rendered or semantically judged.",
            "Missing or malformed required artifacts fail the audit instead of being treated as zero violations.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "report": str(output.resolve()),
                "passed": report["passed"],
                "counts": report["counts"],
                "violations": len(audit.violations),
            }
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
