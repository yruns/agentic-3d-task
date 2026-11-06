"""Strict, immutable public capability discovery for the Codex adapter."""

from __future__ import annotations

from typing import Annotated, Literal
from urllib.parse import quote, urlunparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    ValidationError,
    model_validator,
)

from .config import CodexAgentConfig
from .errors import CodexConfigError
from .provider_http import ProviderSettings, load_provider_settings, read_provider_json

_NonEmptyString = Annotated[
    str, StringConstraints(strict=True, min_length=1, strip_whitespace=False)
]


class AdapterCapabilities(BaseModel):
    """Only advertised capabilities are usable; omitted fields never default on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    structured_output: StrictBool
    images: StrictBool
    function_tools: StrictBool
    namespace_tools: StrictBool
    parallel_tools: StrictBool
    reasoning_summary: StrictBool
    reasoning_efforts: tuple[_NonEmptyString, ...]
    default_reasoning_effort: _NonEmptyString | None
    context_window: Annotated[StrictInt, Field(gt=0)] | None
    compact: StrictBool
    streaming: Literal["native", "buffered_sse"]

    @model_validator(mode="after")
    def validate_reasoning(self) -> AdapterCapabilities:
        if len(set(self.reasoning_efforts)) != len(self.reasoning_efforts):
            raise ValueError("reasoning_efforts must be unique")
        if (
            self.default_reasoning_effort is not None
            and self.default_reasoning_effort not in self.reasoning_efforts
        ):
            raise ValueError("default_reasoning_effort must be supported")
        return self


class AdapterModelProfile(BaseModel):
    """Credential-free wire document returned by application/catalog.py."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: _NonEmptyString
    protocol: Literal["responses", "gemini_chat"]
    codec_version: _NonEmptyString
    capabilities: AdapterCapabilities

    @model_validator(mode="after")
    def validate_protocol(self) -> AdapterModelProfile:
        if self.model.lower().startswith("gpt-") and self.protocol != "responses":
            raise ValueError("GPT models must use Responses end to end")
        if self.protocol == "gemini_chat" and self.capabilities.compact:
            raise ValueError("Gemini compact is unsupported")
        return self


class _AdapterHealth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ready"]
    service: Literal["codex-adapter"]
    version: Literal["1.0.0"]
    schema_version: StrictInt
    models: tuple[_NonEmptyString, ...]

    @model_validator(mode="after")
    def validate_schema_version(self) -> _AdapterHealth:
        if self.schema_version != 1:
            raise ValueError("Unsupported adapter health schema")
        return self


def discover_adapter_model(
    config: CodexAgentConfig,
    *,
    provider: ProviderSettings | None = None,
    timeout_seconds: float = 30.0,
) -> AdapterModelProfile:
    """Verify service identity and exact model before constructing an SDK client."""
    settings = provider or load_provider_settings(config)
    if not settings.health_url:
        raise CodexConfigError(
            "codex_adapter requires an HTTP(S) base_url for capability discovery"
        )
    health_body = read_provider_json(
        settings.health_url,
        timeout_seconds=timeout_seconds,
        adapter_token=config.adapter_token,
    )
    try:
        health = _AdapterHealth.model_validate(health_body)
    except ValidationError as exc:
        raise CodexConfigError(
            f"codex_adapter health identity is invalid or not ready; error_count={exc.error_count()}"
        ) from None
    if config.model not in health.models:
        raise CodexConfigError(
            f"codex_adapter model is not registered: model={config.model!r}"
        )
    path = f"{settings.parsed_url.path.rstrip('/')}/models/{quote(config.model, safe='')}/capabilities"
    capability_url = urlunparse(
        settings.parsed_url._replace(path=path, params="", query="", fragment="")
    )
    body = read_provider_json(
        capability_url,
        timeout_seconds=timeout_seconds,
        adapter_token=config.adapter_token,
    )
    try:
        profile = AdapterModelProfile.model_validate(body)
    except ValidationError as exc:
        raise CodexConfigError(
            f"codex_adapter model capabilities are invalid; error_count={exc.error_count()}"
        ) from None
    if profile.model != config.model:
        raise CodexConfigError(
            "codex_adapter capabilities do not match the exact requested model"
        )
    validate_model_settings(config, profile, settings)
    return profile


def validate_model_settings(
    config: CodexAgentConfig, profile: AdapterModelProfile, settings: ProviderSettings
) -> None:
    """Reject unsupported explicit settings rather than replacing their meaning."""
    capability = profile.capabilities
    if not capability.structured_output:
        raise CodexConfigError(
            f"model={profile.model!r} does not declare structured_output support"
        )
    if (
        config.reasoning_effort not in {"", "provider-default"}
        and config.reasoning_effort not in capability.reasoning_efforts
    ):
        raise CodexConfigError(
            f"model={profile.model!r} does not support reasoning_effort={config.reasoning_effort!r}; select --reasoning-effort provider-default or CODEX_AGENT_REASONING_EFFORT='' explicitly"
        )
    if (
        config.reasoning_summary not in {"", "none"}
        and not capability.reasoning_summary
    ):
        raise CodexConfigError(
            f"model={profile.model!r} does not support reasoning_summary={config.reasoning_summary!r}"
        )
    if not capability.parallel_tools:
        raise CodexConfigError(
            "parallel_tools must be enabled for pinned SDK/core 0.163.0 native function tools; no supported runtime override disables its wire flag"
        )
    if not capability.compact:
        _validate_no_implicit_compaction(config, settings)
    if profile.protocol == "gemini_chat" and settings.inherited_reasoning_effort:
        raise CodexConfigError(
            "Gemini cannot inherit model_reasoning_effort from Codex config; remove it and select provider-default explicitly"
        )


def _validate_no_implicit_compaction(
    config: CodexAgentConfig, settings: ProviderSettings
) -> None:
    if (
        config.model_context_window > 0
        or settings.inherited_context_window is not None
        or settings.inherited_auto_compact_limit is not None
    ):
        raise CodexConfigError(
            "Model does not support compaction: remove model_context_window/model_auto_compact_token_limit overrides; a positive context window silently enables SDK automatic compaction"
        )
