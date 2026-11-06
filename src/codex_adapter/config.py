"""Validated per-model upstream pools with direct, non-disclosing AK fields."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)

from codex_adapter.domain.errors import AdapterError
from codex_adapter.domain.models import (
    ModelCapabilities,
    RegisteredModel,
    Registry,
    Upstream,
    validate_upstream_url,
)
from codex_adapter.domain.state import StateSettings

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10.
    import tomli as tomllib

_DEFAULT_STATE_TTL = 86400
_CONFIG_SCHEMA_VERSION = 2


class _Document(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, hide_input_in_errors=True
    )


class _StateKeyReference(_Document):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    secret_env: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


class _UpstreamDocument(_Document):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    version: str = Field(min_length=1, max_length=64)
    url: str = Field(repr=False)
    ak: SecretStr = Field(repr=False)
    weight: int = Field(default=1, ge=0, le=1000000)
    status: Literal["active", "draining", "revoked"] = "active"

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: str) -> str:
        return validate_upstream_url(url)

    @field_validator("ak")
    @classmethod
    def validate_ak(cls, ak: SecretStr) -> SecretStr:
        value = ak.get_secret_value()
        if not value.strip() or any(char in value for char in "\r\n\x00"):
            raise ValueError("AK must be a nonempty single-line string")
        return ak


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
    upstreams: list[_UpstreamDocument] = Field(min_length=1)
    capabilities: _CapabilitiesDocument | None = None


class _StateDocument(_Document):
    database: str = Field(min_length=1)
    active_key_id: str | None = Field(default=None, min_length=1)
    keys: list[_StateKeyReference] = Field(default_factory=list)
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
    upstream_max_retries: int = Field(default=0, ge=0, le=5)
    upstream_retry_backoff_seconds: float = Field(
        default=1.0, ge=0.0, allow_inf_nan=False
    )


class _SettingsDocument(_Document):
    schema_version: Literal[2]
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
    """Resolve only local adapter auth references, never an upstream model AK."""
    value = environ.get(name, "")
    if not value.strip() or any(char in value for char in "\r\n\x00"):
        raise AdapterError(
            "invalid_config",
            f"Environment reference {name} is missing or invalid",
            status_code=503,
        )
    return value


def _load_state(document: _StateDocument, root: Path) -> StateSettings:
    database = Path(document.database).expanduser()
    return StateSettings(
        database=database if database.is_absolute() else root / database,
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


def _load_model(document: _ModelDocument) -> RegisteredModel:
    upstreams = tuple(
        Upstream(
            id=item.id,
            version=item.version,
            url=item.url,
            secret=item.ak.get_secret_value(),
            weight=item.weight,
            status=item.status,
        )
        for item in document.upstreams
    )
    return RegisteredModel(
        id=document.id,
        protocol=document.protocol,
        upstreams=upstreams,
        capabilities=_load_capabilities(document),
    )


def _read_document(path: Path) -> _SettingsDocument:
    try:
        with path.open("rb") as source:
            raw = tomllib.load(source)
    except (OSError, ValueError):
        raise AdapterError(
            "invalid_config",
            "Adapter configuration is missing or invalid",
            status_code=503,
        ) from None
    version = raw.get("schema_version")
    if type(version) is not int or version != _CONFIG_SCHEMA_VERSION:
        raise AdapterError(
            "invalid_config",
            "schema_version must be 2; migrate model endpoint/credentials to models.upstreams with url and direct ak",
            status_code=503,
        )
    try:
        return _SettingsDocument.model_validate(raw)
    except ValidationError:
        raise AdapterError(
            "invalid_config",
            "Adapter configuration is invalid; check upstream URL, direct AK and field constraints",
            status_code=503,
        ) from None


def load_settings(
    path: Path, *, environ: Mapping[str, str] | None = None
) -> AdapterSettings:
    """Load explicit TOML once; upstream AKs never read or fall back to environment."""
    environment = os.environ if environ is None else environ
    document = _read_document(path)
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
        registry=Registry(tuple(_load_model(model) for model in document.models)),
        state=_load_state(document.state, path.resolve().parent),
        server=document.server,
        auth_token=auth,
    )
