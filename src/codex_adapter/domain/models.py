"""Immutable model registry, capabilities, and secret-bearing upstreams."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

from .errors import AdapterError

ProtocolName = Literal["responses", "gemini_chat"]
UpstreamStatus = Literal["active", "draining", "revoked"]
# Retained for callers of the original constant; per-protocol versions live below.
CODEC_VERSION = "1"
_CODEC_VERSIONS: Mapping[ProtocolName, str] = MappingProxyType(
    {"responses": "1", "gemini_chat": "1"}
)


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_loopback(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_upstream_url(url: str) -> str:
    """Require a secret-free HTTPS URL or an HTTP URL to a loopback address."""
    message = "Upstream URL must be HTTPS or loopback HTTP without credentials, query or fragment"
    if (
        not isinstance(url, str)
        or not url
        or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 or char == "\\"
            for char in url
        )
    ):
        raise ValueError(message)
    try:
        url.encode("utf-8")
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _ = parsed.port
    except (ValueError, UnicodeError):
        pass
    else:
        if (
            hostname
            and (
                parsed.scheme == "https"
                or (parsed.scheme == "http" and _is_loopback(hostname))
            )
            and "@" not in parsed.netloc
            # Preserve old bindings with empty delimiters, but reject actual
            # query/fragment data and never normalize the stored URL identity.
            and not parsed.query
            and not parsed.fragment
        ):
            return url
    raise ValueError(message) from None


@dataclass(frozen=True)
class Upstream:
    """One server-owned URL and AK; neither is exposed by repr."""

    id: str
    version: str
    url: str = field(repr=False)
    secret: str = field(repr=False)
    weight: int = 1
    status: UpstreamStatus = "active"

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and bool(value.strip())
            for value in (self.id, self.version, self.secret)
        ):
            raise AdapterError(
                "invalid_config", "Upstream id, version and secret must be nonempty"
            )
        try:
            validate_upstream_url(self.url)
        except ValueError:
            invalid_url = True
        else:
            invalid_url = False
        if invalid_url:
            raise AdapterError("invalid_config", "Invalid upstream URL") from None
        if (
            not isinstance(self.weight, int)
            or isinstance(self.weight, bool)
            or self.weight < 0
        ):
            raise AdapterError(
                "invalid_config", "Upstream weight must be a nonnegative integer"
            )
        if self.status not in ("active", "draining", "revoked"):
            raise AdapterError("invalid_config", "Invalid upstream status")


@dataclass(frozen=True)
class ModelCapabilities:
    """Declared upstream capabilities, not inferred from a model name."""

    structured_output: bool = True
    images: bool = True
    function_tools: bool = True
    namespace_tools: bool = True
    parallel_tools: bool = True
    reasoning_summary: bool = True
    reasoning_efforts: tuple[str, ...] = ("minimal", "low", "medium", "high")
    default_reasoning_effort: str | None = "medium"
    context_window: int | None = None
    default_max_output_tokens: int = 8192

    def __post_init__(self) -> None:
        flags = (
            self.structured_output,
            self.images,
            self.function_tools,
            self.namespace_tools,
            self.parallel_tools,
            self.reasoning_summary,
        )
        if not all(isinstance(value, bool) for value in flags):
            raise AdapterError(
                "invalid_config", "Model capability flags must be booleans"
            )
        if not isinstance(self.reasoning_efforts, tuple) or not all(
            isinstance(value, str) and bool(value.strip())
            for value in self.reasoning_efforts
        ):
            raise AdapterError(
                "invalid_config",
                "Reasoning efforts must be an immutable tuple of names",
            )
        if len(set(self.reasoning_efforts)) != len(self.reasoning_efforts):
            raise AdapterError("invalid_config", "Reasoning efforts must be unique")
        if self.default_reasoning_effort is not None and (
            not isinstance(self.default_reasoning_effort, str)
            or self.default_reasoning_effort not in self.reasoning_efforts
        ):
            raise AdapterError(
                "invalid_config", "Default reasoning effort must be supported"
            )
        if self.context_window is not None and not _positive_integer(
            self.context_window
        ):
            raise AdapterError(
                "invalid_config", "Context window must be a positive integer"
            )
        if not _positive_integer(self.default_max_output_tokens):
            raise AdapterError(
                "invalid_config",
                "Default maximum output tokens must be a positive integer",
            )


@dataclass(frozen=True)
class RegisteredModel:
    """One exact public/native model ID with one immutable upstream protocol."""

    id: str
    protocol: ProtocolName
    upstreams: tuple[Upstream, ...]
    capabilities: ModelCapabilities

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not self.id.strip()
            or "*" in self.id
            or "?" in self.id
        ):
            raise AdapterError("invalid_config", "Model id must be exact and nonempty")
        if self.protocol not in ("responses", "gemini_chat"):
            raise AdapterError("invalid_config", "Unsupported provider protocol")
        if self.id.lower().startswith("gpt-") and self.protocol != "responses":
            raise AdapterError("invalid_config", "GPT models require native Responses")
        if (
            not isinstance(self.upstreams, tuple)
            or not self.upstreams
            or not all(isinstance(upstream, Upstream) for upstream in self.upstreams)
        ):
            raise AdapterError(
                "invalid_config", "A registered model requires immutable upstreams"
            )
        if not isinstance(self.capabilities, ModelCapabilities):
            raise AdapterError("invalid_config", "Model capabilities are required")
        identities = tuple((item.id, item.version) for item in self.upstreams)
        if len(set(identities)) != len(identities):
            raise AdapterError("invalid_config", "Duplicate upstream identity")

    @property
    def codec_version(self) -> str:
        return f"{self.protocol}:{_CODEC_VERSIONS[self.protocol]}"


@dataclass(frozen=True)
class Registry:
    """Exact model registration without wildcard or default-model fallback."""

    models: tuple[RegisteredModel, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.models, tuple)
            or not self.models
            or not all(isinstance(model, RegisteredModel) for model in self.models)
        ):
            raise AdapterError(
                "invalid_config", "Model registry requires an immutable tuple of models"
            )
        names = tuple(model.id for model in self.models)
        if len(set(names)) != len(names):
            raise AdapterError("invalid_config", "Duplicate or empty model registry")

    def get(self, model_id: str) -> RegisteredModel:
        """Resolve exactly, failing before credentials or upstream IO are used."""
        for model in self.models:
            if model.id == model_id:
                return model
        raise AdapterError(
            "model_not_registered",
            "Model is not registered",
            status_code=404,
            param="model",
        )
