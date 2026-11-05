"""Build the explicit five-record diagnostic manifest from completed evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from codex_agent.config import DEFAULT_PROJECT_ROOT

ROOT = DEFAULT_PROJECT_ROOT
OUTPUT = ROOT / "tmp/alfred_v9/diagnostic_manifest_01.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("Expected a diagnostic JSON object")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise ValueError("Manifest exists; do not overwrite diagnostic history")
    assets = ROOT / "docs/benchmark/alfred/v9_assets"
    capture_failure_path = assets / "capture_launch_attempt1.json"
    comparison_failure_path = assets / "depth_compare_launch_attempt1.json"
    capture_path = ROOT / "outputs/alfred_depth_calibration/train60_v1"
    comparison_path = ROOT / "outputs/alfred_depth_calibration/compare_train60_v2"
    probe_path = ROOT / "outputs/alfred/v9_native_probe_20261105"
    capture_failure = read_json(capture_failure_path)
    comparison_failure = read_json(comparison_failure_path)
    capture = read_json(capture_path / "capture_status.json")
    manifest = read_json(capture_path / "manifest.json")
    comparison = read_json(comparison_path / "comparison.json")
    protocol = read_json(comparison_path / "protocol.json")
    probe = read_json(probe_path / "report.json")
    if (
        capture_failure["status"] != "setup_error_before_python"
        or comparison_failure["status"] != "input_validation_error_before_model_load"
        or comparison_failure["inferred_frames"] != 0
        or capture["status"] != "completed"
        or comparison["status"] != "completed"
        or probe["succeeded"] is not True
        or probe["metadata"]["status"] != "completed"
    ):
        raise ValueError("Diagnostic state differs from the approved terminal records")
    if capture["frame_count"] != 60 or len(manifest["frames"]) != 60:
        raise ValueError("Capture did not preserve exactly 60 training frames")
    if manifest["provenance"]["code"] != manifest["provenance"]["end_code"]:
        raise ValueError("Capture source changed during execution")
    if comparison["manifest_sha256"] != digest(capture_path / "manifest.json"):
        raise ValueError("Depth comparison used another capture manifest")
    backends: dict[str, Any] = {}
    for name in ("da3metric-large", "unidepth-v2-vitl14"):
        record = read_json(comparison_path / f"{name}.json")
        if record["summary"]["frames"] != 60 or len(record["frames"]) != 60:
            raise ValueError("Depth comparison has incomplete backend coverage")
        backends[name] = {
            "summary": record["summary"],
            "inference_seconds": record["inference_seconds"],
            "gpu_wait_seconds": record["gpu_wait_seconds"],
            "load_seconds": record["load_seconds"],
            "peak_cuda_memory_bytes": record["peak_cuda_memory_bytes"],
        }
    records = [
        {
            "run_id": "ALFRED_V9_20261105_capture_launch1",
            "kind": "capture",
            "status": "failed",
            "code": None,
            "code_status": "not_executed",
            "artifact_path": str(capture_failure_path),
            "summary": {
                "source_status": capture_failure["status"],
                "failure_classification": "conda_activation_before_python",
                "error_sha256": hashlib.sha256(
                    capture_failure["error"].encode()
                ).hexdigest(),
                "launcher_sha256": capture_failure["launcher_sha256"],
                "log_sha256": capture_failure["log_sha256"],
                "frame_count": 0,
                "python_started": False,
            },
            "metrics": None,
        },
        {
            "run_id": "ALFRED_V9_20261105_train60_capture_v1",
            "kind": "capture",
            "status": "completed",
            "code": capture["code"],
            "code_status": "recorded",
            "artifact_path": str(capture_path),
            "summary": {
                "split": capture["split"],
                "frame_count": capture["frame_count"],
                "started_at_utc": capture["started_at_utc"],
                "finished_at_utc": capture["finished_at_utc"],
                "manifest_sha256": digest(capture_path / "manifest.json"),
                "plan_sha256": capture["plan_sha256"],
                "start_end_source_match": True,
                "ground_truth_scope": "Training-only depth comparison; never actor input or task scoring",
            },
            "metrics": None,
        },
        {
            "run_id": "ALFRED_V9_20261105_depth_compare_launch_v1",
            "kind": "depth_comparison",
            "status": "failed",
            "code": None,
            "code_status": "unavailable",
            "artifact_path": str(comparison_failure_path),
            "summary": {
                "source_status": comparison_failure["status"],
                "failure_classification": "signed_pitch_frame_id_validation",
                "error_sha256": hashlib.sha256(
                    comparison_failure["error"].encode()
                ).hexdigest(),
                "recorded_code_head_prefix": comparison_failure["code_head"],
                "recorded_executed_code_sha256": comparison_failure[
                    "executed_code_sha256"
                ],
                "code_evidence_limit": "Failure record preserves short HEAD/source SHA only; full contemporaneous branch/clean-state provenance is unavailable",
                "log_sha256": comparison_failure["log_sha256"],
                "inferred_frames": comparison_failure["inferred_frames"],
                "model_loaded": False,
            },
            "metrics": None,
        },
        {
            "run_id": "ALFRED_V9_20261105_depth_compare_train60_v2",
            "kind": "depth_comparison",
            "status": "completed",
            "code": protocol["code"],
            "code_status": "recorded",
            "artifact_path": str(comparison_path),
            "summary": {
                "started_at_utc": protocol["started_at_utc"],
                "finished_at_utc": comparison["finished_at_utc"],
                "manifest_sha256": comparison["manifest_sha256"],
                "frames_per_backend": comparison["frames_per_backend"],
                "selection": comparison["selection"],
                "scope": "Frozen training diagnostic; no ALFRED task rollout",
                "protocol_sha256": digest(comparison_path / "protocol.json"),
            },
            "metrics": backends,
        },
        {
            "run_id": "ALFRED_V9_20261105_native_probe",
            "kind": "transport_probe",
            "status": "completed",
            "code": probe["code"],
            "code_status": "recorded",
            "artifact_path": str(probe_path),
            "summary": {
                "started_at_utc": probe["started_at_utc"],
                "model": probe["model"],
                "reasoning_effort": probe["reasoning_effort"],
                "turn_timeout_s": probe["turn_timeout_s"],
                "max_finalization_retries": probe["max_finalization_retries"],
                "succeeded": probe["succeeded"],
                "scope": "Synthetic native Responses transport/structured output check; no task action executed",
            },
            "metrics": {
                "wall_seconds": probe["wall_seconds"],
                "usage": probe["metadata"]["usage"],
                "sdk_attempts": len(probe["metadata"]["attempts"]),
            },
        },
    ]
    with OUTPUT.open("x") as stream:
        json.dump(
            {"schema_version": 1, "diagnostics": records},
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    print(json.dumps({"output": str(OUTPUT), "records": len(records)}))


if __name__ == "__main__":
    main()
