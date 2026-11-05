"""Archive one completed episode's observable interaction failure evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tmp/alfred_setup/pilot50_cohort_recovery3.json"


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(MANIFEST)
job = next(job for job in manifest["recovery_runs"] if job["global_index"] == 45)
selected = manifest["episodes"][45]
directory = Path(job["output_dir"]) / "episodes/0001"
attempt = read(directory / "attempt.json")
if attempt["status"] != "completed" or attempt["score"] is None:
    raise ValueError("posthoc case requires completed scored episode")
actions = [
    json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines()
]
observations = [
    json.loads(line)
    for line in (directory / "observations.jsonl").read_text().splitlines()
]
frames = {row["observation"]["frame_id"]: row for row in observations}
execution = read(directory / "execution.json")
failed = [row for row in actions if not row["success"]]
if (
    len(actions) != 47
    or len(failed) != 10
    or any(row["action"] != "OpenObject" for row in failed)
):
    raise ValueError("unexpected case membership/counts")
evidence = []
for action in failed:
    row = frames[action["source_frame"]]
    indices = [
        int(index)
        for index, identity in row["detection_objects"].items()
        if identity == action["object_id"]
    ]
    if not indices:
        raise ValueError("failed interaction has no current detection association")
    index = indices[0]
    detection = row["prediction"]["detections"][index]
    mask_path = Path(detection["mask_path"])
    mask = np.load(mask_path, allow_pickle=False)
    if int(row["prediction"]["frame_id"]) != action["source_frame"]:
        raise ValueError("mask prediction frame differs from action source")
    result = frames[action["result_frame"]]["observation"]
    if (
        result["last_action_success"] != action["success"]
        or result["last_action_error"] != action["error"]
    ):
        raise ValueError("result observation differs from action record")
    evidence.append(
        {
            "action": action,
            "source_observation_frame_id": row["observation"]["frame_id"],
            "prediction_frame_id": row["prediction"]["frame_id"],
            "source_rgb_path": row["observation"]["rgb_path"],
            "source_rgb_sha256": digest(Path(row["observation"]["rgb_path"])),
            "predicted_detection_index": index,
            "all_detection_indices_for_memory_id": indices,
            "predicted_category": detection["category"],
            "confidence": detection["confidence"],
            "bbox_xyxy": detection["bbox_xyxy"],
            "mask_path": str(mask_path),
            "mask_sha256": digest(mask_path),
            "mask_shape": list(mask.shape),
            "mask_nonzero_pixels": int(np.count_nonzero(mask)),
            "same_frame_detection_to_memory_association": True,
            "selection_basis": "execution.py _step chooses the first current_detection_objects index matching the proposed memory object ID; persisted observations preserve that mapping and predicted mask path.",
        }
    )
first_request = directory / "decisions/0001/request.json"
instruction = json.loads(read(first_request)["prompt"].splitlines()[-1])["instruction"]
paths = [
    MANIFEST,
    Path(job["run_json"]),
    Path(job["summary_json"]),
    directory / "attempt.json",
    directory / "execution.json",
    directory / "actions.jsonl",
    directory / "observations.jsonl",
    directory / "simulator/terminal_evaluation.json",
    first_request,
    ROOT / "src/codex_agent/alfred/execution.py",
    ROOT / "src/codex_agent/alfred/simulator_worker.py",
    Path(__file__),
]
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Posthoc read-only evidence from a completed scored episode. No replay, model call, simulator call, or feedback to active actors. No ground-truth object metadata was inspected for this case.",
    "run_id": job["run_id"],
    "global_index": 45,
    "episode_id": attempt["episode_id"],
    "task": selected["task"],
    "task_type": selected["task_type"],
    "split": selected["split"],
    "repeat_idx": selected["repeat_idx"],
    "actor_instruction": instruction,
    "artifact_dir": str(directory),
    "execution": execution,
    "official_saved_score": attempt["score"],
    "action_count": len(actions),
    "successful_action_count": sum(row["success"] for row in actions),
    "failed_action_count": len(failed),
    "failed_open_object_count": 10,
    "action_types": dict(Counter(row["action"] for row in actions)),
    "failed_action_errors": dict(Counter(row["error"] for row in failed)),
    "proposed_memory_object_ids": dict(
        Counter(str(row["object_id"]) for row in failed)
    ),
    "observations": len(observations),
    "failed_interaction_evidence": evidence,
    "interpretation": "All ten attempted interactions used a predicted mask associated with the proposed memory ID in the immediately preceding source observation. Four failed during target lookup from the mask; six returned object-not-found. This establishes current-frame association, not correct simulator instance alignment or interaction readiness. The evidence does not identify a single causal failure without further controlled analysis.",
    "limits": [
        "Predicted category and memory ID are actor-side estimates, not ground truth.",
        "No per-step simulator instance masks or ground-truth object distances were read.",
        "No independent official rescore was run here; final cohort audit owns that check.",
    ],
    "raw_artifact_sha256": {str(path): digest(path) for path in paths},
}
output = ROOT / "tmp/alfred_setup/recovery3_i45_interaction_case.json"
with output.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write("\n")
print(
    json.dumps(
        {
            "output": str(output),
            "failed_open_object_count": 10,
            "failed_action_errors": report["failed_action_errors"],
            "mask_categories": dict(
                Counter(item["predicted_category"] for item in evidence)
            ),
            "same_frame_associations": sum(
                item["same_frame_detection_to_memory_association"] for item in evidence
            ),
        },
        indent=2,
    )
)
