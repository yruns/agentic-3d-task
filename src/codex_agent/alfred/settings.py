"""Validated ALFRED scene settings, separated from CLI and model initialization."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TypeVar

from pydantic import TypeAdapter

from .memory import MemoryConfig
from .object_mapping import MappingConfig
from .perception import ScenePerceptionConfig
from .spatial_map import SpatialMapConfig

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ConfigType = TypeVar(
    "ConfigType", ScenePerceptionConfig, MemoryConfig, MappingConfig, SpatialMapConfig
)


@dataclass(frozen=True)
class AlfredSceneSettings:
    """Typed perception, object mapping and navigation inputs for one run."""

    perception: ScenePerceptionConfig
    memory: MemoryConfig
    spatial: SpatialMapConfig


def _validated_section(
    payload: dict[str, object], name: str, model: type[ConfigType]
) -> ConfigType:
    section = payload.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    allowed = {field.name for field in fields(model)}
    unexpected = set(section) - allowed
    if unexpected:
        raise ValueError(f"[{name}] contains unknown fields: {sorted(unexpected)}")
    # JSON-mode strict validation accepts serialized paths but rejects bools as
    # numbers and strings as numeric parameters before dataclass invariants run.
    return TypeAdapter(model).validate_json(json.dumps(section), strict=True)


def load_scene_settings(path: Path) -> AlfredSceneSettings:
    """Load a complete local scene configuration without importing model packages.

    Paths must be absolute so launching from an actor work directory cannot
    silently resolve a different checkpoint. No environment interpolation occurs.
    """
    with path.open("rb") as stream:
        payload = tomllib.load(stream)
    unknown = set(payload) - {"perception", "memory", "spatial"}
    if unknown or "perception" not in payload:
        raise ValueError(
            f"{path}: require [perception], with optional [memory]/[spatial]; unknown={sorted(unknown)}"
        )
    perception = payload["perception"]
    if not isinstance(perception, dict):
        raise ValueError("[perception] must be a table")
    for field, value in perception.items():
        if (
            field.endswith(("_path", "_dir", "_checkpoint", "_config"))
            or field == "python_executable"
        ):
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"perception.{field} must be an absolute local path")
    memory = payload.get("memory", {})
    if not isinstance(memory, dict):
        raise ValueError("[memory] must be a table")
    if "mapping" in memory:
        _validated_section({"mapping": memory["mapping"]}, "mapping", MappingConfig)
    return AlfredSceneSettings(
        perception=_validated_section(payload, "perception", ScenePerceptionConfig),
        memory=_validated_section(payload, "memory", MemoryConfig),
        spatial=_validated_section(payload, "spatial", SpatialMapConfig),
    )
