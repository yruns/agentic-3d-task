"""SceneFunc3D runner skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .playbook import SCENEFUNC3D_TOOLS_PLAYBOOK
from .sample import SceneFunc3dSample, load_sample


@dataclass(frozen=True)
class SceneFunc3dRunnerConfig:
    """Configuration for a SceneFunc3D single-sample run."""

    dataset_root: Path
    output_dir: Path


def build_prompt(sample: SceneFunc3dSample) -> str:
    """Build the prompt prefix for one SceneFunc3D sample."""
    return (
        f"{SCENEFUNC3D_TOOLS_PLAYBOOK}\n\n"
        "Task context:\n"
        f"{sample.agent_context}\n\n"
        "Generate a SceneFunc3D 3D mask artifact for this task."
    )


def load_runner_sample(
    config: SceneFunc3dRunnerConfig, sample_id: str
) -> SceneFunc3dSample:
    """Load the sample a future runtime turn will solve."""
    _ = config.output_dir
    return load_sample(config.dataset_root, sample_id)


__all__ = ["SceneFunc3dRunnerConfig", "build_prompt", "load_runner_sample"]
