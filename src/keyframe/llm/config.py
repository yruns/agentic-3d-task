"""Strongly-typed LLM configuration loaded from ``configs/llm.toml``.

The TOML file holds Azure-compatible endpoints, API keys, model names and
weighted-pool definitions. It is gitignored because it contains real API
keys; ``configs/llm.example.toml`` is the tracked placeholder template.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

#: Environment variable that overrides the default config location.
CONFIG_ENV_VAR = "KEYFRAME_LLM_CONFIG"


class LLMKeyConfig(BaseModel):
    """A single API key in a model's pool, with a round-robin weight."""

    model_config = ConfigDict(frozen=True)

    api_key: str
    weight: int = Field(default=1, ge=1)
    # Optional per-key endpoint override (a pool may span endpoints).
    endpoint: str | None = None


class LLMModelConfig(BaseModel):
    """Configuration for one named model and its key pool."""

    api_version: str
    endpoint: str = ""
    temperature: float = 0.0
    timeout: int = Field(default=120, gt=0)
    keys: list[LLMKeyConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_endpoint_per_key(self) -> LLMModelConfig:
        for key in self.keys:
            if not (key.endpoint or self.endpoint):
                raise ValueError(
                    "each key needs an endpoint (set key.endpoint or model.endpoint)"
                )
        return self

    def endpoint_for(self, key: LLMKeyConfig) -> str:
        """Resolve the endpoint for a key (key override, else model default)."""
        return key.endpoint or self.endpoint

    def weighted_keys(self) -> list[LLMKeyConfig]:
        """Keys expanded by weight, e.g. weights [1, 3] -> [k0, k1, k1, k1]."""
        expanded: list[LLMKeyConfig] = []
        for key in self.keys:
            expanded.extend([key] * key.weight)
        return expanded


class LLMConfig(BaseModel):
    """Top-level LLM configuration: a default model and a name -> config map."""

    default_model: str
    models: dict[str, LLMModelConfig]

    @model_validator(mode="after")
    def _require_known_default(self) -> LLMConfig:
        if self.default_model not in self.models:
            raise ValueError(
                f"default_model {self.default_model!r} not present in models"
            )
        return self

    def get_model(self, name: str | None = None) -> LLMModelConfig:
        """Look up a model config by name, falling back to the default model."""
        resolved = name or self.default_model
        if resolved not in self.models:
            known = ", ".join(sorted(self.models))
            raise KeyError(f"unknown model {resolved!r}; known models: {known}")
        return self.models[resolved]

    def resolve_name(self, name: str | None = None) -> str:
        """Return the concrete model name (default if ``name`` is None)."""
        return name or self.default_model

    @classmethod
    def from_toml(cls, path: Path) -> LLMConfig:
        """Load and validate configuration from a TOML file."""
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return cls.model_validate(data)


def default_config_path() -> Path:
    """Default config path: ``$KEYFRAME_LLM_CONFIG`` or ``<repo>/configs/llm.toml``."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "configs" / "llm.toml"
