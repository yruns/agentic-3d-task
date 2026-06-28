"""Strongly-typed LLM client with a weighted key pool and rate-limit rotation.

Wraps ``langchain_openai.AzureChatOpenAI``. Callers obtain a chat model via
``get_chat_model`` (which round-robins across the pool) and may attach
structured output themselves (e.g. ``.with_structured_output(schema)``), or
use ``invoke`` for a built-in rate-limit-aware retry across the pool.
"""

from __future__ import annotations

import threading
from typing import TypeAlias

from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_openai import AzureChatOpenAI
from loguru import logger
from pydantic import SecretStr

from keyframe.llm.config import (
    CONFIG_ENV_VAR,
    LLMConfig,
    LLMKeyConfig,
    LLMModelConfig,
    default_config_path,
)

#: Substrings that identify a rate-limit / quota error across providers.
_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "429",
    "quota exceeded",
    "quota_exceeded",
    "qps limit",
    "qps_limit",
    "resource exhausted",
)

#: Accepted prompt input types for ``LLMClient.invoke``.
PromptInput: TypeAlias = str | list[BaseMessage] | PromptValue


def is_rate_limit_error(error: Exception) -> bool:
    """Whether an exception looks like a rate-limit / quota error."""
    text = str(error).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


class _RoundRobin:
    """Thread-safe cyclic index over a fixed-size pool."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._index = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            current = self._index
            self._index = (self._index + 1) % self._size
            return current


class LLMClient:
    """Pooled, rate-limit-aware factory for Azure chat models."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._rotors: dict[str, _RoundRobin] = {}
        self._rotors_lock = threading.Lock()

    @classmethod
    def from_config(cls, config: LLMConfig) -> LLMClient:
        """Build a client from an already-loaded config."""
        return cls(config)

    @classmethod
    def from_toml(cls, path: str | None = None) -> LLMClient:
        """Load configuration from a TOML file (default: ``configs/llm.toml``)."""
        from pathlib import Path

        resolved = Path(path) if path is not None else default_config_path()
        if not resolved.exists():
            raise FileNotFoundError(
                f"LLM config not found at {resolved}. Copy configs/llm.example.toml "
                f"to configs/llm.toml or set ${CONFIG_ENV_VAR}."
            )
        return cls(LLMConfig.from_toml(resolved))

    @property
    def config(self) -> LLMConfig:
        """The underlying configuration."""
        return self._config

    def key_count(self, model: str | None = None) -> int:
        """Number of distinct API keys configured for a model."""
        return len(self._config.get_model(model).keys)

    def get_chat_model(
        self,
        model: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AzureChatOpenAI:
        """Return a chat model bound to the next key in the model's pool."""
        name = self._config.resolve_name(model)
        model_config = self._config.get_model(name)
        weighted = model_config.weighted_keys()
        rotor = self._get_rotor(name, len(weighted))
        key = weighted[rotor.next()]
        resolved_temperature = (
            model_config.temperature if temperature is None else temperature
        )
        return self._build_model(
            name, model_config, key, resolved_temperature, max_tokens
        )

    def invoke(
        self,
        prompt: PromptInput,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> BaseMessage:
        """Invoke a model, rotating across the pool on rate-limit errors."""
        name = self._config.resolve_name(model)
        model_config = self._config.get_model(name)
        attempts = len(model_config.weighted_keys())
        last_error: Exception | None = None
        for _ in range(attempts):
            chat_model = self.get_chat_model(
                name, temperature=temperature, max_tokens=max_tokens
            )
            try:
                return chat_model.invoke(prompt)
            except (
                Exception
            ) as error:  # noqa: BLE001 - re-raised below if not rate limit
                if is_rate_limit_error(error):
                    logger.warning(f"[LLMClient] rate limited on {name}, rotating key")
                    last_error = error
                    continue
                raise
        raise last_error or RuntimeError(f"all keys exhausted for model {name!r}")

    def _get_rotor(self, name: str, size: int) -> _RoundRobin:
        with self._rotors_lock:
            rotor = self._rotors.get(name)
            if rotor is None:
                rotor = _RoundRobin(size)
                self._rotors[name] = rotor
            return rotor

    def _build_model(
        self,
        name: str,
        model_config: LLMModelConfig,
        key: LLMKeyConfig,
        temperature: float,
        max_tokens: int | None,
    ) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_deployment=name,
            model=name,
            api_key=SecretStr(key.api_key),
            azure_endpoint=model_config.endpoint_for(key),
            api_version=model_config.api_version,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=model_config.timeout,
            max_retries=0,
        )
