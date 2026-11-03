"""Shared preflight checks for Codex Agent CLI runtimes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from codex_agent.config import CodexAgentConfig
from codex_agent.errors import CodexConfigError
from codex_agent.provider_http import (
    ADAPTER_PROVIDER,
    load_provider_settings,
    read_provider_json,
    sanitize_provider_url,
)
from codex_agent.provider_profile import AdapterModelProfile, discover_adapter_model

CodexRuntimePreflightStatus = Literal["checked", "skipped_remote", "skipped_non_http"]


@dataclass(frozen=True)
class CodexRuntimePreflightResult:
    """Credential-free discovery outcome; remote providers are never probed."""

    model_provider: str
    base_url: str
    health_url: str
    status: CodexRuntimePreflightStatus
    model_profile: AdapterModelProfile | None = None


def preflight_codex_runtime(
    config: CodexAgentConfig,
    *,
    timeout_seconds: float = 1.0,
) -> CodexRuntimePreflightResult:
    """Check local adapters before a turn, preserving explicit legacy providers.

    Only ``codex_adapter`` must implement the ready identity and public model
    capabilities protocol. Remote providers retain the no-network preflight.
    Discovery describes configured capabilities, not upstream credential health.
    """
    if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
        raise CodexConfigError(
            f"timeout_seconds must be a finite positive number, got {timeout_seconds}"
        )
    provider = load_provider_settings(config)
    status: CodexRuntimePreflightStatus = "checked"
    profile: AdapterModelProfile | None = None
    if not provider.health_url:
        if config.model_provider == ADAPTER_PROVIDER:
            raise CodexConfigError(
                "codex_adapter requires an HTTP(S) base_url for capability discovery"
            )
        status = "skipped_non_http"
    elif not provider.is_local_http:
        status = "skipped_remote"
    elif config.model_provider == ADAPTER_PROVIDER:
        profile = discover_adapter_model(
            config, provider=provider, timeout_seconds=timeout_seconds
        )
    else:
        body = read_provider_json(provider.health_url, timeout_seconds=timeout_seconds)
        if not isinstance(body, (dict, list)):
            raise CodexConfigError(
                "Codex model provider health response is invalid: error_type=unexpected_json"
            )
    return CodexRuntimePreflightResult(
        model_provider=config.model_provider,
        base_url=sanitize_provider_url(provider.parsed_url),
        health_url=(
            sanitize_provider_url(provider.health_url) if provider.health_url else ""
        ),
        status=status,
        model_profile=profile,
    )


__all__ = [
    "CodexRuntimePreflightResult",
    "CodexRuntimePreflightStatus",
    "preflight_codex_runtime",
]
