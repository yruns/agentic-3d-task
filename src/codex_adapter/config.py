"""Validated configuration references and centralized environment loading."""

from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.models import (
    Credential,
    ModelCapabilities,
    RegisteredModel,
    Registry,
)
from codex_adapter.domain.state import StateKey, StateSettings

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10.
    import tomli as tomllib

_DEFAULT_STATE_TTL = 86400


class _Document(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, hide_input_in_errors=True
    )


class _SecretReference(_Document):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    secret_env: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


class _CredentialDocument(_SecretReference):
    version: str = Field(min_length=1, max_length=64)
    weight: int = Field(default=1, ge=0, le=1000000)
    status: Literal["active", "draining", "revoked"] = "active"


class _CapabilitiesDocument(_Document):
    structured_output: bool = False
    images: bool = False
    function_tools: bool = True
    namespace_tools: bool = True
    parallel_tools: bool = True
    reasoning_summary: bool = False
    reasoning_efforts: list[str] = Field(default_factory=list)
    default_reasoning_effort: str | None = None
    context_window: int | None = Field(default=None, gt=0)
    default_max_output_tokens: int = Field(default=8192, gt=0, le=1000000)


class _ModelDocument(_Document):
    id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$"
    )
    protocol: Literal["responses", "gemini_chat"]
    endpoint: str
    credentials: list[_CredentialDocument] = Field(min_length=1)
    capabilities: _CapabilitiesDocument | None = None

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, endpoint: str) -> str:
        parsed = urlsplit(endpoint)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise ValueError("Endpoint requires HTTPS or loopback HTTP")
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Endpoint must not include credentials, queries or fragments"
            )
        _ = parsed.port
        return endpoint


class _StateDocument(_Document):
    database: str = Field(min_length=1)
    active_key_id: str = Field(min_length=1)
    keys: list[_SecretReference] = Field(min_length=1)
    session_ttl_seconds: int = Field(default=_DEFAULT_STATE_TTL, gt=0)
    max_sessions: int = Field(default=10000, gt=0)
    max_turns: int = Field(default=256, gt=0)
    max_state_bytes: int = Field(default=33554432, gt=0)


class ServerSettings(_Document):
    """Explicit server resource limits; no unbounded buffering or read deadline."""

    host: str = "127.0.0.1"
    port: int = Field(default=8787, ge=1, le=65535)
    auth_env: str | None = None
    max_request_bytes: int = Field(default=33554432, gt=0)
    max_response_bytes: int = Field(default=67108864, gt=0)
    max_concurrent_requests: int = Field(default=32, gt=0)
    requests_per_minute: int = Field(default=600, gt=0)
    upstream_timeout_seconds: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    connect_timeout_seconds: float = Field(default=15.0, gt=0, allow_inf_nan=False)


class _SettingsDocument(_Document):
    schema_version: Literal[1]
    state: _StateDocument
    server: ServerSettings = Field(default_factory=ServerSettings)
    models: list[_ModelDocument] = Field(min_length=1)


@dataclass(frozen=True)
class AdapterSettings:
    """Resolved immutable settings with secret-bearing fields excluded from repr."""

    registry: Registry
    state: StateSettings
    server: ServerSettings
    auth_token: str | None = field(default=None, repr=False)


def _read_secret(name: str, environ: Mapping[str, str]) -> str:
    value = environ.get(name, "")
    if not value.strip() or any(char in value for char in "\r\n\x00"):
        raise AdapterError(
            "invalid_config",
            f"Environment reference {name} is missing or invalid",
            status_code=503,
        )
    return value


def _load_state(
    document: _StateDocument, root: Path, environ: Mapping[str, str]
) -> StateSettings:
    try:
        keys = tuple(
            StateKey(
                reference.id,
                base64.b64decode(
                    _read_secret(reference.secret_env, environ),
                    altchars=b"-_",
                    validate=True,
                ),
            )
            for reference in document.keys
        )
    except (ValueError, binascii.Error):
        raise AdapterError(
            "invalid_config",
            "State keys must be base64-encoded 32-byte values",
            status_code=503,
        ) from None
    database = Path(document.database).expanduser()
    return StateSettings(
        database=database if database.is_absolute() else root / database,
        keys=keys,
        active_key_id=document.active_key_id,
        session_ttl_seconds=document.session_ttl_seconds,
        max_sessions=document.max_sessions,
        max_turns=document.max_turns,
        max_state_bytes=document.max_state_bytes,
    )


def _load_capabilities(document: _ModelDocument) -> ModelCapabilities:
    if document.capabilities is None:
        if document.protocol == "responses":
            return ModelCapabilities()
        return ModelCapabilities(
            structured_output=False,
            images=False,
            reasoning_summary=False,
            reasoning_efforts=(),
            default_reasoning_effort=None,
        )
    if document.protocol == "gemini_chat" and document.capabilities.reasoning_summary:
        raise AdapterError(
            "invalid_config",
            "Gemini reasoning summary mapping is unsupported",
            status_code=503,
        )
    values = document.capabilities.model_dump(exclude={"reasoning_efforts"})
    return ModelCapabilities(
        **values, reasoning_efforts=tuple(document.capabilities.reasoning_efforts)
    )


def _load_model(
    document: _ModelDocument, environ: Mapping[str, str]
) -> RegisteredModel:
    credentials = tuple(
        Credential(
            id=item.id,
            version=item.version,
            secret=_read_secret(item.secret_env, environ),
            weight=item.weight,
            status=item.status,
        )
        for item in document.credentials
    )
    return RegisteredModel(
        id=document.id,
        protocol=document.protocol,
        endpoint=document.endpoint,
        credentials=credentials,
        capabilities=_load_capabilities(document),
    )


def load_settings(
    path: Path, *, environ: Mapping[str, str] | None = None
) -> AdapterSettings:
    """Load explicit TOML once; reject unknown fields without echoing secrets."""
    environment = os.environ if environ is None else environ
    try:
        with path.open("rb") as source:
            document = _SettingsDocument.model_validate(tomllib.load(source))
    except (OSError, ValueError, ValidationError):
        raise AdapterError(
            "invalid_config",
            "Adapter configuration is missing or invalid; check schema and field constraints",
            status_code=503,
        ) from None
    auth = (
        _read_secret(document.server.auth_env, environment)
        if document.server.auth_env
        else None
    )
    if document.server.host not in {"127.0.0.1", "localhost", "::1"} and auth is None:
        raise AdapterError(
            "invalid_config",
            "Non-loopback listeners require adapter authentication",
            status_code=503,
        )
    return AdapterSettings(
        registry=Registry(
            tuple(_load_model(model, environment) for model in document.models)
        ),
        state=_load_state(document.state, path.resolve().parent, environment),
        server=document.server,
        auth_token=auth,
    )
