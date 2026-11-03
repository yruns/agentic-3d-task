"""Per-thread routing environment and provider-specific Codex CLI overrides."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from .config import CodexAgentConfig
from .provider_http import ADAPTER_PROVIDER

ADAPTER_SESSION_ENV = "CODEX_ADAPTER_SESSION_ID"
ADAPTER_AUTHORIZATION_ENV = "CODEX_ADAPTER_AUTHORIZATION"
MODELHUB_EXTRA_HEADER_ENV = "CODEX_AGENT_MODELHUB_EXTRA_HEADER"
MODELHUB_LOGID_ENV = "CODEX_AGENT_MODELHUB_LOGID"
ISOLATED_HOME_DIRNAME = ".home"


def build_turn_environment(
    config: CodexAgentConfig, run_home: Path, *, session_id: str | None = None
) -> dict[str, str]:
    """Generate a full UUID route independently of cache policy and ambient env.

    The caller generates one identity per SDK thread and reuses it for all tool
    continuations/finalization. No upstream credential is read or injected.
    """
    identity = session_id or str(uuid.uuid4())
    env = {"CODEX_HOME": str(run_home)}
    if config.restrict_skills_to_project:
        env = {**env, "HOME": str(run_home / ISOLATED_HOME_DIRNAME)}
    if config.model_provider == ADAPTER_PROVIDER:
        return {
            **env,
            ADAPTER_SESSION_ENV: identity,
            ADAPTER_AUTHORIZATION_ENV: (
                f"Bearer {config.adapter_token}" if config.adapter_token else ""
            ),
        }
    if not config.enable_prefix_cache:
        return env
    # Legacy providers keep their wire format; a prefix-cache label is never the
    # routing identity, even on this explicitly selected compatibility path.
    extra = {
        "session_id": identity,
        "source": "codex_agent_sdk",
        "chat_run_id": identity,
        "prefix_cache_label": config.prefix_cache_session_id,
    }
    return {
        **env,
        MODELHUB_EXTRA_HEADER_ENV: json.dumps(
            extra, ensure_ascii=False, separators=(",", ":")
        ),
        MODELHUB_LOGID_ENV: f"codexsdk_{identity}_{int(time.time() * 1000)}",
    }


def build_provider_overrides(config: CodexAgentConfig) -> tuple[str, ...]:
    """Build non-secret overrides without loading environment or home files."""
    overrides: tuple[str, ...] = ("project_doc_max_bytes=0",)
    provider = f"model_providers.{toml_key_part(config.model_provider)}"
    if config.model_provider == ADAPTER_PROVIDER:
        overrides += (
            f'{provider}.wire_api="responses"',
            f"{provider}.requires_openai_auth=false",
            f"{provider}.supports_websockets=false",
            f"{provider}.request_max_retries=0",
            f"{provider}.stream_max_retries=0",
            f"{provider}.env_http_headers.X-Codex-Session-ID={toml_literal(ADAPTER_SESSION_ENV)}",
            f"{provider}.env_http_headers.Authorization={toml_literal(ADAPTER_AUTHORIZATION_ENV)}",
            "features.responses_websockets=false",
            "features.responses_websockets_v2=false",
        )
    elif config.enable_prefix_cache:
        overrides += (
            f"{provider}.env_http_headers.extra={toml_literal(MODELHUB_EXTRA_HEADER_ENV)}",
            f"{provider}.env_http_headers.X-TT-LOGID={toml_literal(MODELHUB_LOGID_ENV)}",
        )
    if config.sandbox == "workspace_write" and config.sandbox_network_access:
        overrides += ("sandbox_workspace_write.network_access=true",)
    if config.model_context_window > 0:
        overrides += (f"model_context_window={config.model_context_window}",)
    return overrides


def toml_literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def toml_key_part(value: str) -> str:
    return value if re.fullmatch(r"[A-Za-z0-9_-]+", value) else toml_literal(value)
