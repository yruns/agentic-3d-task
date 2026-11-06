"""Sequential ALFRED evaluation with frozen folds and durable failure records."""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import re
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from codex_agent.alfred.dataset import VALIDATION_SPLITS, AlfredEpisode, load_manifest
from codex_agent.alfred.execution import EpisodeExecutor, ExecutionConfig, write_json
from codex_agent.alfred.memory import MemoryConfig
from codex_agent.alfred.metrics import AggregateMetrics, EpisodeScore, aggregate_scores
from codex_agent.alfred.odometry import ActionOdometry, camera_intrinsics
from codex_agent.alfred.perception import (
    PERCEPTION_PROFILE,
    ScenePerception,
    ScenePerceptionConfig,
)
from codex_agent.alfred.policy import AlfredPolicyRuntime
from codex_agent.alfred.provenance import (
    capture_runtime_provenance,
    code_provenance,
    external_source_provenance,
    file_sha256,
)
from codex_agent.alfred.provenance import (
    model_source_provenance as model_source_provenance,
)
from codex_agent.alfred.results import EpisodeAttempt, ingest_run
from codex_agent.alfred.simulator import SimulatorConfig, ThorSimulator
from codex_agent.alfred.spatial_map import SpatialMapConfig
from codex_agent.alfred.vocabulary import vocabulary_sha256
from codex_agent.cli.runtime_preflight import preflight_codex_runtime
from codex_agent.config import CodexAgentConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlfredRunConfig:
    """Explicit dataset, model, process, and archival inputs for one invocation."""

    source_root: Path
    data_root: Path
    splits_path: Path
    fold_path: Path
    output_dir: Path
    run_id: str
    database_path: Path
    simulator: SimulatorConfig
    perception: ScenePerceptionConfig
    runtime: CodexAgentConfig
    execution: ExecutionConfig
    command: tuple[str, ...]
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    spatial: SpatialMapConfig = field(default_factory=SpatialMapConfig)
    limit: int | None = None
    shard_index: int = 0
    num_shards: int = 1
    continue_on_error: bool = False
    resume: bool = False
    concurrency_state_path: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "source_root",
            "data_root",
            "splits_path",
            "fold_path",
            "output_dir",
            "database_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)).absolute())
        if self.concurrency_state_path is not None:
            object.__setattr__(
                self,
                "concurrency_state_path",
                Path(self.concurrency_state_path).absolute(),
            )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError(
                "run_id must start with an alphanumeric and contain only alphanumerics, '.', '_', '-'"
            )
        if self.limit is not None and (type(self.limit) is not int or self.limit < 1):
            raise ValueError("limit must be a positive integer or None")
        if type(self.num_shards) is not int or self.num_shards < 1:
            raise ValueError("num_shards must be a positive integer")
        if (
            type(self.shard_index) is not int
            or not 0 <= self.shard_index < self.num_shards
        ):
            raise ValueError(
                f"shard_index={self.shard_index!r} must be an integer in [0, {self.num_shards})"
            )
        if not self.command or not all(self.command):
            raise ValueError("command must record the executed argument vector")


class AlfredRunSummary(BaseModel):
    """Completed denominators stay separate from errors and unattempted entries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected: int
    completed: int
    errors: int
    interrupted: int
    pending: int
    metrics: AggregateMetrics | None
    by_split: dict[str, AggregateMetrics | None]
    attempts: tuple[EpisodeAttempt, ...]


def preflight_run(config: AlfredRunConfig) -> dict[str, object]:
    """Validate inputs/provider readiness without workers or benchmark execution."""
    manifest = load_manifest(config.fold_path, config.data_root, config.splits_path)
    episodes = _selected_episodes(config, manifest)
    if not (config.simulator.alfred_root / "env/thor_env.py").is_file():
        raise FileNotFoundError("official ALFRED env/thor_env.py is missing")
    preflight_codex_runtime(config.runtime)
    return {
        "status": "preflight_only",
        "selected": len(episodes),
        "full_fold_size": len(manifest),
        "shard_index": config.shard_index,
        "num_shards": config.num_shards,
        "by_split": {
            split: sum(episode.split == split for episode in episodes)
            for split in VALIDATION_SPLITS
        },
        "fold_sha256": file_sha256(config.fold_path),
        "perception": PERCEPTION_PROFILE,
        "simulator_started": False,
        "upstream_credentials_verified": False,
    }


def run_validation(config: AlfredRunConfig) -> AlfredRunSummary:
    """Run one immutable invocation; retain partial results and re-raise failures.

    Source must be committed and clean before any model or simulator is started.
    The predictor persists across episodes; each simulator owns a separate Unity
    lifecycle. Authoritative scoring occurs only after the actor has stopped.
    """
    manifest = load_manifest(config.fold_path, config.data_root, config.splits_path)
    episodes = _selected_episodes(config, manifest)
    code = code_provenance(config.source_root)
    if not config.resume:
        _require_unused_run_id(config.database_path, config.run_id)
        if config.output_dir.exists():
            raise FileExistsError(f"output_dir already exists: {config.output_dir}")
    metadata = _run_metadata(config, episodes, code, full_fold_size=len(manifest))
    preflight_codex_runtime(config.runtime)
    config.output_dir.mkdir(parents=True, exist_ok=config.resume)
    actor_dir = config.output_dir / "actor"
    actor_dir.mkdir(exist_ok=True)
    runtime = AlfredPolicyRuntime(
        replace(
            config.runtime,
            project_root=actor_dir,
            sandbox="workspace_write",
            sandbox_network_access=True,
            restrict_skills_to_project=True,
            keep_run_home=True,
            max_tool_calls=config.execution.max_decisions * 2,
            max_repeated_tool_calls=5,
        )
    )
    attempts: list[EpisodeAttempt] = []
    completed_indices: set[int] = set()
    if config.resume:
        completed_indices = _sync_disk_attempts(config, episodes, attempts)
    _checkpoint(config, metadata, episodes, attempts)
    perception: ScenePerception | None = None
    try:
        perception = ScenePerception(
            config.perception, log_path=config.output_dir / "perception.stderr.log"
        )
        runtime_provenance = perception.runtime_provenance
        perception_metadata = metadata["perception"]
        if not isinstance(perception_metadata, dict):
            raise TypeError("perception metadata must be a mapping")
        perception_metadata["runtime_provenance"] = runtime_provenance.model_dump(
            mode="json"
        )
        perception_metadata["libraries"] = {
            name: package.model_dump(mode="json")
            for name, package in runtime_provenance.packages.items()
        }
        perception_metadata["provenance_status"] = "worker_reported"
        _checkpoint(config, metadata, episodes, attempts)
        indexed_episodes = list(enumerate(episodes))
        if os.environ.get("ALFRED_REVERSE_EPISODES") == "1":
            indexed_episodes.reverse()
        for index, episode in indexed_episodes:
            if index in completed_indices:
                logger.info(
                    "ALFRED episode %s/%s: %s (already completed on resume, skipping)",
                    index + 1,
                    len(episodes),
                    episode.episode_id,
                )
                continue
            with _try_claim_episode(config, index) as claimed:
                if not claimed:
                    logger.info(
                        "ALFRED episode %s/%s: %s (completed or claimed by peer worker, skipping)",
                        index + 1,
                        len(episodes),
                        episode.episode_id,
                    )
                    continue
                logger.info(
                    "ALFRED episode %s/%s: %s",
                    index + 1,
                    len(episodes),
                    episode.episode_id,
                )
                with _adaptive_concurrency_slot(config) as slot_report:
                    perception, ep_status, had_retry = _run_episode_with_resilience(
                        config,
                        episode,
                        index=index,
                        perception=perception,
                        runtime=runtime,
                        attempts=attempts,
                    )
                    slot_report["status"] = ep_status
                    slot_report["had_retry"] = had_retry
                try:
                    metadata["perception_statistics"] = (
                        perception.statistics().model_dump(mode="json")
                    )
                except Exception:  # noqa: BLE001
                    pass
                _checkpoint(config, metadata, episodes, attempts)
        if config.resume:
            _sync_disk_attempts(config, episodes, attempts)
        metadata["status"] = (
            "completed"
            if all(a.status == "completed" for a in attempts)
            else "completed_with_errors"
        )
    except BaseException as exc:
        # Orchestration boundary: archive the failure and propagate it. No
        # synthetic score and no silent continuation after infrastructure loss.
        metadata["status"] = (
            "interrupted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "error"
        )
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        perception_metadata = metadata.get("perception")
        if (
            isinstance(perception_metadata, dict)
            and perception_metadata.get("runtime_provenance") is None
        ):
            perception_metadata["provenance_status"] = (
                "unavailable_initialization_failed"
            )
        raise
    finally:
        if perception is not None:
            try:
                perception.close()
            except Exception:  # noqa: BLE001
                pass
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _checkpoint(config, metadata, episodes, attempts)
        ingest_run(
            config.database_path,
            run_id=config.run_id,
            metadata=metadata,
            attempts=attempts,
            replace_existing=config.resume,
        )
    return summarize_attempts(episodes, attempts)


def _ensure_healthy_perception(
    config: AlfredRunConfig, perception: ScenePerception
) -> ScenePerception:
    try:
        perception.statistics()
        return perception
    except Exception as exc:  # noqa: BLE001
        logger.warning("Restarting unhealthy ScenePerception worker: %s", exc)
        try:
            perception.close()
        except Exception:  # noqa: BLE001
            pass
        return ScenePerception(
            config.perception, log_path=config.output_dir / "perception.stderr.log"
        )


def _run_episode_with_resilience(
    config: AlfredRunConfig,
    episode: AlfredEpisode,
    *,
    index: int,
    perception: ScenePerception,
    runtime: AlfredPolicyRuntime,
    attempts: list[EpisodeAttempt],
) -> tuple[ScenePerception, str, bool]:
    artifact_dir = config.output_dir / "episodes" / f"{index + 1:04d}"
    if artifact_dir.exists() and config.resume:
        backup = (
            config.output_dir / "episodes" / f"{index + 1:04d}_prev_{int(time.time())}"
        )
        artifact_dir.rename(backup)
    if not config.continue_on_error:
        _run_episode(
            config,
            episode,
            index=index,
            perception=perception,
            runtime=runtime,
            attempts=attempts,
        )
        return perception, "completed", False

    had_retry = False
    for try_idx in range(2):
        try:
            _run_episode(
                config,
                episode,
                index=index,
                perception=perception,
                runtime=runtime,
                attempts=attempts,
            )
            return perception, "completed", had_retry
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            perception = _ensure_healthy_perception(config, perception)
            if try_idx == 0:
                had_retry = True
                # Remove the recorded error attempt from attempts and archive the failed directory
                if attempts and attempts[-1].episode_id == episode.episode_id:
                    attempts.pop()
                if artifact_dir.exists():
                    retry_backup = (
                        config.output_dir
                        / "episodes"
                        / f"{index + 1:04d}_retry0_{int(time.time())}"
                    )
                    artifact_dir.rename(retry_backup)
                logger.warning(
                    "Episode %s failed on try 1 (%s: %s); retrying once after 6s...",
                    episode.episode_id,
                    type(exc).__name__,
                    exc,
                )
                time.sleep(6.0)
            else:
                logger.error(
                    "Episode %s failed after retry (%s: %s); continuing shard...",
                    episode.episode_id,
                    type(exc).__name__,
                    exc,
                )
                return perception, "error", True
    return perception, "error", True


@contextmanager
def _adaptive_concurrency_slot(
    config: AlfredRunConfig,
) -> Iterator[dict[str, object]]:
    """Coordinate active episodes across shards and step down concurrency on error spikes."""
    report: dict[str, object] = {"status": "error", "had_retry": False}
    state_path = config.concurrency_state_path
    if state_path is None:
        yield report
        return
    lock_path = state_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    pid_str = str(os.getpid())
    gpu_idx = config.perception.physical_gpu_index
    while True:
        with lock_path.open("a", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                state = _load_concurrency_state(state_path)
                active_pids = state["active_pids"]
                assert isinstance(active_pids, dict)
                _reap_dead_pids(active_pids)
                max_active = int(state.get("max_active", 12))
                num_gpus = int(state.get("num_gpus", 3))
                per_gpu_limit = max(1, math.ceil(max_active / max(1, num_gpus)))
                gpu_active = sum(
                    1
                    for info in active_pids.values()
                    if isinstance(info, dict) and info.get("gpu") == gpu_idx
                )
                gpu_ok = (
                    True
                    if config.perception.server_url is not None
                    else gpu_active < per_gpu_limit
                )
                if len(active_pids) < max_active and gpu_ok:
                    active_pids[pid_str] = {
                        "shard": config.shard_index,
                        "gpu": gpu_idx,
                        "started_at": time.time(),
                    }
                    _save_concurrency_state(state_path, state)
                    break
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        time.sleep(1.5)
    try:
        yield report
    finally:
        with lock_path.open("a", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                state = _load_concurrency_state(state_path)
                active_pids = state["active_pids"]
                assert isinstance(active_pids, dict)
                active_pids.pop(pid_str, None)
                _reap_dead_pids(active_pids)
                outcomes = state.setdefault("recent_outcomes", [])
                assert isinstance(outcomes, list)
                outcomes.append(
                    {
                        "ts": time.time(),
                        "shard": config.shard_index,
                        "status": str(report.get("status", "error")),
                        "had_retry": bool(report.get("had_retry", False)),
                    }
                )
                if len(outcomes) > 24:
                    del outcomes[:-24]
                _maybe_step_down_concurrency(state)
                _save_concurrency_state(state_path, state)
            finally:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def _load_concurrency_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict) and isinstance(data.get("active_pids"), dict):
                return data
        except Exception:  # noqa: BLE001
            pass
    return {
        "max_active": 12,
        "min_active": 3,
        "num_gpus": 3,
        "active_pids": {},
        "recent_outcomes": [],
        "last_throttle_ts": 0.0,
        "throttle_events": [],
    }


def _save_concurrency_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def _reap_dead_pids(active_pids: dict[str, Any]) -> None:
    dead: list[str] = []
    for pid_str in active_pids:
        try:
            os.kill(int(pid_str), 0)
        except OSError:
            dead.append(pid_str)
    for pid_str in dead:
        active_pids.pop(pid_str, None)


def _maybe_step_down_concurrency(state: dict[str, Any]) -> None:
    max_active = int(state.get("max_active", 12))
    min_active = int(state.get("min_active", 3))
    if max_active <= min_active:
        return
    last_throttle_ts = float(state.get("last_throttle_ts", 0.0))
    outcomes = state.get("recent_outcomes", [])
    if not isinstance(outcomes, list):
        return
    window = [
        o
        for o in outcomes
        if isinstance(o, dict) and float(o.get("ts", 0.0)) > last_throttle_ts
    ][-12:]
    if len(window) < 4:
        return
    errors = sum(1 for o in window if o.get("status") == "error")
    retries = sum(1 for o in window if o.get("had_retry"))
    if errors >= 3 or retries >= 5:
        new_max = max(min_active, max_active - 3)
        state["max_active"] = new_max
        state["last_throttle_ts"] = time.time()
        events = state.setdefault("throttle_events", [])
        if isinstance(events, list):
            events.append(
                {
                    "ts": time.time(),
                    "from": max_active,
                    "to": new_max,
                    "window_errors": errors,
                    "window_retries": retries,
                }
            )
        logger.warning(
            "Adaptive concurrency stepped down from %s to %s (errors=%s, retries=%s in window=%s)",
            max_active,
            new_max,
            errors,
            retries,
            len(window),
        )


def _run_episode(
    config: AlfredRunConfig,
    episode: AlfredEpisode,
    *,
    index: int,
    perception: ScenePerception,
    runtime: AlfredPolicyRuntime,
    attempts: list[EpisodeAttempt],
) -> None:
    artifact_dir = config.output_dir / "episodes" / f"{index + 1:04d}"
    artifact_dir.mkdir(parents=True)
    executor: EpisodeExecutor | None = None
    try:
        simulator_config = replace(
            config.simulator,
            output_dir=artifact_dir / "simulator",
            render_depth=(config.perception.depth_source == "simulator_gt"),
        )
        with ThorSimulator(simulator_config) as environment:
            initial = environment.reset(episode.trajectory_path)
            executor = EpisodeExecutor(
                environment=environment,
                perception=perception,
                runtime=runtime,
                output_dir=artifact_dir,
                config=config.execution,
                memory_config=config.memory,
                spatial_config=config.spatial,
            )
            outcome = executor.run(episode.instruction, initial)
            terminal = environment.score()
        score = EpisodeScore(
            success=terminal.success,
            completed_goal_conditions=terminal.completed_goal_conditions,
            total_goal_conditions=terminal.total_goal_conditions,
            expert_actions=episode.expert_actions,
            executed_actions=outcome.actions,
        )
        write_json(artifact_dir / "execution.json", asdict(outcome))
        attempt = EpisodeAttempt(
            episode_id=episode.episode_id,
            status="completed",
            artifact_dir=str(artifact_dir),
            score=score,
        )
    except BaseException as exc:
        attempt = EpisodeAttempt(
            episode_id=episode.episode_id,
            status=(
                "interrupted"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "error"
            ),
            artifact_dir=str(artifact_dir),
            error=f"{type(exc).__name__}: {exc}",
        )
        attempts.append(attempt)
        write_json(artifact_dir / "attempt.json", attempt.model_dump(mode="json"))
        if executor is not None:
            write_json(
                artifact_dir / "partial_execution.json",
                {
                    "actions": executor.actions,
                    "failures": executor.failures,
                    "decisions": executor.decisions,
                    "observed_input_tokens_lower_bound": executor.input_tokens,
                    "observed_output_tokens_lower_bound": executor.output_tokens,
                },
            )
        raise
    attempts.append(attempt)
    write_json(artifact_dir / "attempt.json", attempt.model_dump(mode="json"))


def summarize_attempts(
    episodes: Sequence[AlfredEpisode], attempts: Sequence[EpisodeAttempt]
) -> AlfredRunSummary:
    """Compute official aggregate metrics only from observed terminal scores."""
    scores = [attempt.score for attempt in attempts if attempt.score is not None]
    split_by_id = {episode.episode_id: episode.split for episode in episodes}
    by_split: dict[str, AggregateMetrics | None] = {}
    for split in VALIDATION_SPLITS:
        selected_scores = [
            attempt.score
            for attempt in attempts
            if attempt.score is not None and split_by_id[attempt.episode_id] == split
        ]
        by_split[split] = aggregate_scores(selected_scores) if selected_scores else None
    return AlfredRunSummary(
        selected=len(episodes),
        completed=sum(attempt.status == "completed" for attempt in attempts),
        errors=sum(attempt.status == "error" for attempt in attempts),
        interrupted=sum(attempt.status == "interrupted" for attempt in attempts),
        pending=len(episodes) - len(attempts),
        metrics=aggregate_scores(scores) if scores else None,
        by_split=by_split,
        attempts=tuple(attempts),
    )


def _selected_episodes(
    config: AlfredRunConfig, manifest: tuple[AlfredEpisode, ...]
) -> tuple[AlfredEpisode, ...]:
    """Take a stable strided shard, then apply an optional diagnostic prefix."""
    if len({episode.episode_id for episode in manifest}) != len(manifest):
        raise ValueError("frozen fold contains duplicate episode IDs")
    episodes = manifest[config.shard_index :: config.num_shards]
    if config.limit is not None:
        episodes = episodes[: config.limit]
    if not episodes:
        raise ValueError(
            f"frozen fold selected no ALFRED episodes for shard_index={config.shard_index}, "
            f"num_shards={config.num_shards}, full_fold_size={len(manifest)}"
        )
    return episodes


def perception_assets(config: ScenePerceptionConfig) -> dict[str, Path]:
    """List exact inference assets, excluding download caches and temporary files."""
    assets = {
        name: getattr(config, name)
        for name in (
            "dino_config",
            "dino_checkpoint",
            "sam_checkpoint",
            "clip_checkpoint",
        )
    }
    for name in ("dino_text_encoder_dir", "depth_model_dir"):
        directory = getattr(config, name)
        for path in sorted(directory.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                assets[f"{name}/{path.name}"] = path
    return assets


def _run_metadata(
    config: AlfredRunConfig,
    episodes: Sequence[AlfredEpisode],
    code: dict[str, object],
    *,
    full_fold_size: int,
) -> dict[str, object]:
    odometry = ActionOdometry()
    return {
        "run_id": config.run_id,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": code,
        "command": list(config.command),
        "output_dir": str(config.output_dir),
        "fold_path": str(config.fold_path),
        "fold_sha256": file_sha256(config.fold_path),
        "full_fold_size": full_fold_size,
        "shard_index": config.shard_index,
        "num_shards": config.num_shards,
        "splits_path": str(config.splits_path),
        "splits_sha256": file_sha256(config.splits_path),
        "limit": config.limit,
        "diagnostic_prefix": config.limit is not None,
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "split": episode.split,
                "task_type": episode.task_type,
                "trajectory_sha256": episode.trajectory_sha256,
                "expert_actions": episode.expert_actions,
            }
            for episode in episodes
        ],
        "model": config.runtime.model,
        "model_provider": config.runtime.model_provider,
        "reasoning_effort": config.runtime.reasoning_effort,
        "execution": asdict(config.execution),
        "geometry": {
            "runtime_provenance": capture_runtime_provenance(
                model_worker=False
            ).model_dump(mode="json"),
            "camera_height_m": odometry.camera_height_m,
            "step_size_m": odometry.step_size_m,
            "vertical_fov_degrees": 60.0,
            "intrinsics": camera_intrinsics().tolist(),
            "pose_source": "successful-action odometry and yaw/horizon proprioception",
            "world_axes": "episode-relative THOR X/Z/Y (Z-up)",
        },
        "perception_profile": PERCEPTION_PROFILE,
        "perception": {
            "configuration": config.perception.to_payload(),
            "vocabulary_sha256": vocabulary_sha256(),
            "checkpoints": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in perception_assets(config.perception).items()
            },
            "libraries": None,
            "runtime_provenance": None,
            "provenance_status": "not_started",
        },
        "memory": asdict(config.memory),
        "spatial_map": asdict(config.spatial),
        "simulator": {
            "python_executable": str(config.simulator.python_executable),
            "x_display": config.simulator.x_display,
            "ai2thor_version": "2.1.0",
            "build": "201909061227",
        },
        "external_sources": {
            "alfred": external_source_provenance(config.simulator.alfred_root),
        },
    }


def _sync_disk_attempts(
    config: AlfredRunConfig,
    episodes: Sequence[AlfredEpisode],
    attempts: list[EpisodeAttempt],
) -> set[int]:
    by_id: dict[str, EpisodeAttempt] = {a.episode_id: a for a in attempts}
    completed_indices: set[int] = set()
    for index, ep in enumerate(episodes):
        attempt_file = (
            config.output_dir / "episodes" / f"{index + 1:04d}" / "attempt.json"
        )
        if attempt_file.is_file():
            try:
                prev = EpisodeAttempt.model_validate_json(attempt_file.read_text())
                if prev.status == "completed":
                    by_id[ep.episode_id] = prev
                    completed_indices.add(index)
                elif ep.episode_id not in by_id:
                    by_id[ep.episode_id] = prev
            except Exception:  # noqa: BLE001
                pass
    attempts[:] = [by_id[ep.episode_id] for ep in episodes if ep.episode_id in by_id]
    return completed_indices


@contextmanager
def _try_claim_episode(config: AlfredRunConfig, index: int) -> Iterator[bool]:
    if not config.resume:
        yield True
        return
    ep_dir = config.output_dir / "episodes" / f"{index + 1:04d}"
    attempt_file = ep_dir / "attempt.json"
    if attempt_file.is_file():
        try:
            prev = EpisodeAttempt.model_validate_json(attempt_file.read_text())
            if prev.status == "completed":
                yield False
                return
        except Exception:  # noqa: BLE001
            pass
    lock_path = config.output_dir / "episodes" / f".{index + 1:04d}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            if attempt_file.is_file():
                try:
                    prev = EpisodeAttempt.model_validate_json(attempt_file.read_text())
                    if prev.status == "completed":
                        yield False
                        return
                except Exception:  # noqa: BLE001
                    pass
            elif ep_dir.exists():
                # Guard against an older worker running without .lock file
                try:
                    latest_mtime = max(
                        (p.stat().st_mtime for p in ep_dir.rglob("*")),
                        default=ep_dir.stat().st_mtime,
                    )
                    if time.time() - latest_mtime < 300.0:
                        yield False
                        return
                except OSError:
                    pass
            yield True
        finally:
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


def _checkpoint(
    config: AlfredRunConfig,
    metadata: dict[str, object],
    episodes: Sequence[AlfredEpisode],
    attempts: list[EpisodeAttempt],
) -> None:
    if config.resume:
        ckpt_lock = config.output_dir / ".checkpoint.lock"
        ckpt_lock.parent.mkdir(parents=True, exist_ok=True)
        with ckpt_lock.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                _sync_disk_attempts(config, episodes, attempts)
                write_json(config.output_dir / "run.json", metadata)
                write_json(
                    config.output_dir / "summary.json",
                    summarize_attempts(episodes, attempts).model_dump(mode="json"),
                )
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return
    write_json(config.output_dir / "run.json", metadata)
    write_json(
        config.output_dir / "summary.json",
        summarize_attempts(episodes, attempts).model_dump(mode="json"),
    )


def _require_unused_run_id(database_path: Path, run_id: str) -> None:
    if not database_path.exists():
        return
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'"
        ).fetchone()
        if (
            table
            and connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        ):
            raise ValueError(f"run_id={run_id!r} already exists; choose a new run ID")
