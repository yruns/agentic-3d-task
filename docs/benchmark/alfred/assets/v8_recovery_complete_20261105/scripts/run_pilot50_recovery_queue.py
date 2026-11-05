"""Run one explicitly planned recovery queue, stopping on its first failure.

This operational wrapper does not change the actor, fold, or evaluation code.
It invokes the original CLI once per planned singleton and never retries. Use
--check-only to inspect source/fold/selection guards without launching anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codex_agent.alfred.provenance import code_provenance

ROOT = Path("/home/ysh/codebase/agentic-3d-task")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_status(path: Path, payload: dict[str, Any], *, initial: bool = False) -> None:
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if initial:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        temporary = path.with_suffix(path.suffix + ".pending")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


def require_equivalent_code(
    actual: Mapping[str, object], reference: Mapping[str, object]
) -> None:
    """Permit a documentation-only HEAD change, never executable drift.

    Callers obtain current identity through production code_provenance(), which
    hashes all tracked executable source/config bytes and rejects untracked
    executable files. Terminal metadata is checked against the same constraints.
    """
    if actual.get("branch") != reference["branch"]:
        raise ValueError("source branch differs from the frozen cohort")
    if actual.get("tracked_worktree_drift") != "":
        raise ValueError("source tracked working tree must remain clean")
    if actual.get("executed_code_sha256") != reference["executed_code_sha256"]:
        raise ValueError("executed code SHA-256 differs from the frozen cohort")
    head = actual.get("head_sha")
    if not isinstance(head, str) or re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ValueError("actual HEAD must be recorded as a full Git commit SHA")


def check_plan(
    manifest: dict[str, Any], queue_index: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if manifest["schema_version"] != 1 or manifest["status"] != "planned_not_launched":
        raise ValueError("expected the immutable planned recovery cohort manifest")
    if sha256(Path(__file__)) != manifest["queue_driver"]["sha256"]:
        raise ValueError("recovery queue driver changed after review")
    expected_policy = {
        "version": 1,
        "allow_head_sha_difference": True,
        "require_same_branch": True,
        "require_clean_tracked_tree": True,
        "require_identical_executed_code_sha256": True,
        "actual_head_recorded_per_run": True,
    }
    policy = manifest["source_identity_policy"]
    if any(policy.get(key) != value for key, value in expected_policy.items()):
        raise ValueError(
            "cohort source comparison policy is missing or weaker than required"
        )
    require_equivalent_code(code_provenance(ROOT), manifest["code"])
    if sha256(Path(manifest["fold"]["path"])) != manifest["fold"]["sha256"]:
        raise ValueError("frozen fold bytes differ from the cohort")
    if (
        sha256(Path(manifest["fold"]["splits_path"]))
        != manifest["fold"]["splits_sha256"]
    ):
        raise ValueError("official split bytes differ from the cohort")
    for original in manifest["original_runs"]:
        if sha256(Path(original["run_json"])) != original["run_json_sha256"]:
            raise ValueError(f'original run record changed: {original["run_id"]}')
        if sha256(Path(original["summary_json"])) != original["summary_json_sha256"]:
            raise ValueError(f'original summary changed: {original["run_id"]}')
    queue = next(
        item for item in manifest["queues"] if item["queue_index"] == queue_index
    )
    jobs = [
        item for item in manifest["recovery_runs"] if item["queue_index"] == queue_index
    ]
    if len(jobs) != 3 or [item["run_id"] for item in jobs] != queue["run_ids"]:
        raise ValueError("queue does not contain exactly its three planned runs")
    for job in jobs:
        command = job["command"]
        if command[1:3] != ["-m", "codex_agent.cli.run_alfred"]:
            raise ValueError("recovery must invoke the original ALFRED CLI")
        if command[command.index("--num-shards") + 1] != "50":
            raise ValueError("recovery requires singleton selection from original fold")
        if command[command.index("--shard-index") + 1] != str(job["global_index"]):
            raise ValueError("recovery global index differs from the plan")
        if "--limit" in command:
            raise ValueError("recovery must not select an additional prefix")
        for field in ("output_dir", "stdout_path", "status_path"):
            if Path(job[field]).exists():
                raise FileExistsError(
                    f"recovery never overwrites existing artifacts: {job[field]}"
                )
    if Path(queue["status_path"]).exists():
        raise FileExistsError(
            f'queue already started; automatic resume/retry is forbidden: {queue["status_path"]}'
        )
    return queue, jobs


def stop_owned_child(process: subprocess.Popen[bytes]) -> None:
    """Allow the single owned evaluation to archive interruption, then bound cleanup."""
    if process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError(f"owned evaluation process did not exit: {process.pid}")


def check_terminal(job: dict[str, Any], manifest: dict[str, Any]) -> None:
    run = read_object(Path(job["run_json"]))
    summary = read_object(Path(job["summary_json"]))
    if run["status"] != "completed" or run["run_id"] != job["run_id"]:
        raise ValueError("zero exit did not produce the expected completed run")
    require_equivalent_code(run["code"], manifest["code"])
    if run["num_shards"] != 50 or run["shard_index"] != job["global_index"]:
        raise ValueError("recovery singleton selection changed")
    for key, expected in manifest["common_configuration"].items():
        if key == "code":
            require_equivalent_code(run[key], expected)
        elif key != "cli_arguments_excluding_run_and_shard" and run[key] != expected:
            raise ValueError(f"recovery configuration changed: {key}")
    if not (
        summary["selected"] == summary["completed"] == 1
        and summary["errors"] == summary["interrupted"] == summary["pending"] == 0
    ):
        raise ValueError("recovery did not produce exactly one scored episode")
    attempts = summary["attempts"]
    if (
        len(attempts) != 1
        or attempts[0]["episode_id"] != job["episode_id"]
        or attempts[0]["status"] != "completed"
        or attempts[0]["score"] is None
    ):
        raise ValueError("recovery terminal episode identity or score missing")


def run_queue(manifest_path: Path, queue_index: int, *, check_only: bool) -> int:
    manifest = read_object(manifest_path)
    queue, jobs = check_plan(manifest, queue_index)
    if check_only:
        print(
            json.dumps(
                {
                    "status": "checked_not_launched",
                    "queue_index": queue_index,
                    "global_indices": queue["global_indices"],
                    "run_ids": queue["run_ids"],
                },
                indent=2,
            )
        )
        return 0
    status_path = Path(queue["status_path"])
    actual_code = code_provenance(ROOT)
    require_equivalent_code(actual_code, manifest["code"])
    status: dict[str, Any] = {
        "queue_index": queue_index,
        "state": "running",
        "started_at_utc": now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "code": actual_code,
        "reference_code": manifest["code"],
        "source_identity_policy": manifest["source_identity_policy"],
        "automatic_retries": 0,
        "jobs": [
            {
                "run_id": job["run_id"],
                "global_index": job["global_index"],
                "state": "pending",
            }
            for job in jobs
        ],
    }
    write_status(status_path, status, initial=True)
    current: dict[str, Any] | None = None
    current_job: dict[str, Any] | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        for job, job_status in zip(jobs, status["jobs"], strict=True):
            current, current_job = job_status, job
            actual_code = code_provenance(ROOT)
            require_equivalent_code(actual_code, manifest["code"])
            current.update(
                code=actual_code,
                state="running",
                started_at_utc=now(),
                command=job["command"],
                stdout_path=job["stdout_path"],
            )
            write_status(Path(job["status_path"]), current, initial=True)
            write_status(status_path, status)
            with Path(job["stdout_path"]).open("xb") as output:
                process = subprocess.Popen(
                    job["command"],
                    cwd=ROOT,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                current["pid"] = process.pid
                write_status(Path(job["status_path"]), current)
                write_status(status_path, status)
                exit_code = process.wait()
            current.update(exit_code=exit_code, finished_at_utc=now())
            if exit_code != 0:
                current["state"] = "failed"
                status.update(
                    state="failed", finished_at_utc=now(), exit_code=exit_code
                )
                write_status(Path(job["status_path"]), current)
                write_status(status_path, status)
                return exit_code if exit_code > 0 else 128 - exit_code
            check_terminal(job, manifest)
            current["evaluated_code"] = read_object(Path(job["run_json"]))["code"]
            current["state"] = "completed"
            write_status(Path(job["status_path"]), current)
            write_status(status_path, status)
            process = None
        status.update(state="completed", finished_at_utc=now(), exit_code=0)
        write_status(status_path, status)
        return 0
    except BaseException as exc:
        if process is not None:
            stop_owned_child(process)
        status.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            exit_code=130 if isinstance(exc, KeyboardInterrupt) else 1,
            error=f"{type(exc).__name__}: {exc}",
            finished_at_utc=now(),
        )
        if current is not None and current_job is not None:
            current.update(
                state=status["state"], error=status["error"], finished_at_utc=now()
            )
            if process is not None:
                current["exit_code"] = process.poll()
            write_status(Path(current_job["status_path"]), current)
        write_status(status_path, status)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "tmp/alfred_setup/pilot50_cohort.json"
    )
    parser.add_argument("--queue-index", type=int, choices=range(3), required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    return run_queue(
        args.manifest.resolve(strict=True), args.queue_index, check_only=args.check_only
    )


if __name__ == "__main__":
    raise SystemExit(main())
