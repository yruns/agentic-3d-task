"""Typed, bounded subprocess boundary to the original ALFRED simulator stack.

The legacy Python interpreter and simulator checkout are explicit dependencies.
Only rendered observations leave the worker during acting; authoritative task
state is exposed separately through terminal scoring.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .worker import JsonWorker, WorkerError

NAVIGATION_ACTIONS = frozenset(
    {"MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"}
)
INTERACTION_ACTIONS = frozenset(
    {
        "PickupObject",
        "PutObject",
        "OpenObject",
        "CloseObject",
        "ToggleObjectOn",
        "ToggleObjectOff",
        "SliceObject",
    }
)


class SimulatorError(RuntimeError):
    """The external simulator failed or returned an invalid response."""


class SimulatorObservation(BaseModel):
    """RGB and action/proprioception feedback; no privileged scene information.

    Yaw and horizon are agent proprioception in degrees. Position, depth,
    instance labels, semantic state and inventory metadata remain worker-private.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    frame_id: int = Field(ge=0)
    rgb_path: Path
    yaw_degrees: float
    horizon_degrees: float
    last_action_success: bool
    last_action_error: str


class SimulatorScore(BaseModel):
    """Authoritative terminal evaluation, never used as policy feedback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    completed_goal_conditions: int = Field(ge=0)
    total_goal_conditions: int = Field(ge=1)
    snapshot_path: Path

    @model_validator(mode="after")
    def validate_count(self) -> SimulatorScore:
        if self.completed_goal_conditions > self.total_goal_conditions:
            raise ValueError("completed_goal_conditions exceeds total_goal_conditions")
        return self


@dataclass(frozen=True)
class SimulatorConfig:
    """Explicit legacy interpreter, checkout, display, and RPC deadline."""

    python_executable: Path
    alfred_root: Path
    output_dir: Path
    x_display: str = "5"
    timeout_seconds: float = 120.0
    render_depth: bool = False

    def __post_init__(self) -> None:
        for name in ("python_executable", "alfred_root", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).absolute())
        if not self.python_executable.is_file():
            raise FileNotFoundError(
                f"simulator Python missing: {self.python_executable}"
            )
        if not self.alfred_root.is_dir():
            raise FileNotFoundError(f"ALFRED checkout missing: {self.alfred_root}")
        if not re.fullmatch(r":?\d+(?:\.\d+)?", self.x_display):
            raise ValueError(
                f"x_display must be a numeric X display: {self.x_display!r}"
            )
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not isinstance(self.render_depth, bool):
            raise ValueError("render_depth must be bool")


class ThorSimulator:
    """Own one worker/Unity process group; intended for one sequential episode.

    The worker lives in its own session, so deadline cleanup affects only this
    simulator and its children. Callers must close it or use a context manager.
    """

    def __init__(
        self, config: SimulatorConfig, *, worker_path: Path | None = None
    ) -> None:
        self.config = config
        worker = worker_path or Path(__file__).with_name("simulator_worker.py")
        if (
            worker_path is None
            and not (config.alfred_root / "env/thor_env.py").is_file()
        ):
            raise FileNotFoundError(
                f"official ThorEnv missing under {config.alfred_root}"
            )
        if not worker.is_file():
            raise FileNotFoundError(f"simulator worker missing: {worker}")
        cmd = [
            str(config.python_executable),
            "-u",
            str(worker),
            "--alfred-root",
            str(config.alfred_root),
            "--output-dir",
            str(config.output_dir),
            "--x-display",
            config.x_display.lstrip(":"),
        ]
        if config.render_depth:
            cmd.append("--render-depth")
        self._worker = JsonWorker(
            command=cmd,
            cwd=config.alfred_root,
            env=os.environ,
            log_path=config.output_dir / "simulator.stderr.log",
            timeout_seconds=config.timeout_seconds,
        )

    @property
    def is_closed(self) -> bool:
        """Whether the owned worker has been shut down."""
        return self._worker.is_closed

    def __enter__(self) -> ThorSimulator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def reset(self, trajectory_path: Path) -> SimulatorObservation:
        """Restore the authoritative trajectory scene without exposing its plan."""
        trajectory_path = Path(trajectory_path).resolve(strict=True)
        return SimulatorObservation.model_validate(
            self._rpc({"command": "reset", "trajectory_path": str(trajectory_path)})
        )

    def step(
        self, action: str, *, mask_path: Path | None = None
    ) -> SimulatorObservation:
        """Execute one benchmark action, using a current-frame interaction mask."""
        if action not in NAVIGATION_ACTIONS | INTERACTION_ACTIONS:
            raise ValueError(f"unsupported ALFRED action: {action!r}")
        if action in INTERACTION_ACTIONS and mask_path is None:
            raise ValueError(f"mask_path is required for action {action}")
        if action in NAVIGATION_ACTIONS and mask_path is not None:
            raise ValueError(f"mask_path is not allowed for action {action}")
        path = Path(mask_path).resolve(strict=True) if mask_path is not None else None
        return SimulatorObservation.model_validate(
            self._rpc(
                {
                    "command": "step",
                    "action": action,
                    "mask_path": str(path) if path is not None else None,
                }
            )
        )

    def score(self) -> SimulatorScore:
        """Run the official task's terminal goal predicates and save their inputs."""
        return SimulatorScore.model_validate(self._rpc({"command": "score"}))

    def close(self) -> None:
        """Stop the worker and its Unity children, including after RPC timeouts."""
        self._worker.close()

    def _rpc(self, request: dict[str, object]) -> dict[str, object]:
        try:
            return self._worker.request(request)
        except WorkerError as exc:
            raise SimulatorError(str(exc)) from exc
