"""Sequential ALFRED evaluation with frozen folds and durable failure records."""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from codex_agent.alfred.dataset import VALIDATION_SPLITS, AlfredEpisode, load_manifest
from codex_agent.alfred.execution import EpisodeExecutor, ExecutionConfig, write_json
from codex_agent.alfred.metrics import AggregateMetrics, EpisodeScore, aggregate_scores
from codex_agent.alfred.odometry import ActionOdometry, camera_intrinsics
from codex_agent.alfred.perception import (
    FILM_CHECKPOINTS,
    FilmPerception,
    FilmPerceptionConfig,
)
from codex_agent.alfred.policy import AlfredPolicyRuntime
from codex_agent.alfred.provenance import code_provenance, file_sha256
from codex_agent.alfred.results import EpisodeAttempt, ingest_run
from codex_agent.alfred.simulator import SimulatorConfig, ThorSimulator
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
    perception: FilmPerceptionConfig
    runtime: CodexAgentConfig
    execution: ExecutionConfig
    command: tuple[str, ...]
    limit: int | None = None
    shard_index: int = 0
    num_shards: int = 1

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
        "perception": "film_predicted_rgb_v1",
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
    _require_unused_run_id(config.database_path, config.run_id)
    if config.output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {config.output_dir}")
    metadata = _run_metadata(config, episodes, code, full_fold_size=len(manifest))
    preflight_codex_runtime(config.runtime)
    config.output_dir.mkdir(parents=True, exist_ok=False)
    actor_dir = config.output_dir / "actor"
    actor_dir.mkdir()
    runtime = AlfredPolicyRuntime(
        replace(
            config.runtime,
            project_root=actor_dir,
            sandbox="read_only",
            sandbox_network_access=False,
            restrict_skills_to_project=True,
            keep_run_home=True,
        )
    )
    attempts: list[EpisodeAttempt] = []
    _checkpoint(config, metadata, episodes, attempts)
    try:
        with FilmPerception(
            config.perception, log_path=config.output_dir / "perception.stderr.log"
        ) as perception:
            for index, episode in enumerate(episodes):
                logger.info(
                    "ALFRED episode %s/%s: %s",
                    index + 1,
                    len(episodes),
                    episode.episode_id,
                )
                _run_episode(
                    config,
                    episode,
                    index=index,
                    perception=perception,
                    runtime=runtime,
                    attempts=attempts,
                )
                metadata["perception_statistics"] = perception.statistics().model_dump(
                    mode="json"
                )
                _checkpoint(config, metadata, episodes, attempts)
        metadata["status"] = "completed"
    except BaseException as exc:
        # Orchestration boundary: archive the failure and propagate it. No
        # synthetic score and no silent continuation after infrastructure loss.
        metadata["status"] = (
            "interrupted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "error"
        )
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _checkpoint(config, metadata, episodes, attempts)
        ingest_run(
            config.database_path,
            run_id=config.run_id,
            metadata=metadata,
            attempts=attempts,
        )
    return summarize_attempts(episodes, attempts)


def _run_episode(
    config: AlfredRunConfig,
    episode: AlfredEpisode,
    *,
    index: int,
    perception: FilmPerception,
    runtime: AlfredPolicyRuntime,
    attempts: list[EpisodeAttempt],
) -> None:
    artifact_dir = config.output_dir / "episodes" / f"{index + 1:04d}"
    artifact_dir.mkdir(parents=True)
    executor: EpisodeExecutor | None = None
    try:
        simulator_config = replace(
            config.simulator, output_dir=artifact_dir / "simulator"
        )
        with ThorSimulator(simulator_config) as environment:
            initial = environment.reset(episode.trajectory_path)
            executor = EpisodeExecutor(
                environment=environment,
                perception=perception,
                runtime=runtime,
                output_dir=artifact_dir,
                config=config.execution,
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


def external_source_provenance(path: Path) -> dict[str, object]:
    """Hash actual Python sources, including untracked imports, plus Git revision."""
    head = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(["git", "-C", str(path), "diff", "HEAD", "--"])
    source_paths = subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
        ]
    ).split(b"\0")
    digest = hashlib.sha256()
    paths = sorted({relative for relative in source_paths if relative})
    for relative in paths:
        digest.update(relative + b"\0")
        digest.update((path / relative.decode("utf-8")).read_bytes())
        digest.update(b"\0")
    return {
        "path": str(path),
        "head_sha": head,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_dirty": bool(diff),
        "python_source_sha256": digest.hexdigest(),
        "python_source_count": len(paths),
    }


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
            "camera_height_m": odometry.camera_height_m,
            "step_size_m": odometry.step_size_m,
            "vertical_fov_degrees": 60.0,
            "intrinsics": camera_intrinsics().tolist(),
            "pose_source": "successful-action odometry and yaw/horizon proprioception",
            "world_axes": "episode-relative THOR X/Z/Y (Z-up)",
        },
        "perception_profile": "film_predicted_rgb_v1",
        "perception": {
            "device": config.perception.device,
            "score_threshold": config.perception.score_threshold,
            "mask_threshold": config.perception.mask_threshold,
            "checkpoints": {
                name: file_sha256(config.perception.weights_root / name)
                for name in FILM_CHECKPOINTS
            },
        },
        "simulator": {
            "python_executable": str(config.simulator.python_executable),
            "x_display": config.simulator.x_display,
            "ai2thor_version": "2.1.0",
            "build": "201909061227",
        },
        "external_sources": {
            "alfred": external_source_provenance(config.simulator.alfred_root),
            "film": external_source_provenance(config.perception.film_root),
        },
    }


def _checkpoint(
    config: AlfredRunConfig,
    metadata: dict[str, object],
    episodes: Sequence[AlfredEpisode],
    attempts: Sequence[EpisodeAttempt],
) -> None:
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
