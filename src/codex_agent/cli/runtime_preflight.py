"""Shared preflight checks for Codex Agent CLI runtimes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import ParseResult, urlparse, urlunparse
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib

from codex_agent.config import CodexAgentConfig
from codex_agent.errors import CodexConfigError

CodexRuntimePreflightStatus = Literal[
    "checked",
    "skipped_remote",
    "skipped_non_http",
]

_LOCAL_PROVIDER_HOSTS: frozenset[str] = frozenset(("127.0.0.1", "localhost"))


@dataclass(frozen=True)
class CodexRuntimePreflightResult:
    """Outcome of checking the configured Codex model provider."""

    model_provider: str
    base_url: str
    health_url: str
    status: CodexRuntimePreflightStatus


class _CodexProviderDocument(BaseModel):
    """Pydantic boundary model for one Codex model-provider section."""

    model_config = ConfigDict(extra="ignore")

    base_url: str = Field(min_length=1)


class _CodexConfigDocument(BaseModel):
    """Pydantic boundary model for ``$CODEX_HOME/config.toml``."""

    model_config = ConfigDict(extra="ignore")

    model_providers: Mapping[str, _CodexProviderDocument] = Field(default_factory=dict)


def preflight_codex_runtime(
    config: CodexAgentConfig,
    *,
    timeout_seconds: float = 1.0,
) -> CodexRuntimePreflightResult:
    """Fail fast when the configured local Codex model provider is unavailable.

    Remote HTTPS providers are intentionally not probed: the CLI preflight
    should catch missing local adapters without turning every benchmark launch
    into an external network dependency.

    Raises:
        CodexConfigError: If the Codex home config is missing, invalid, lacks the
            requested provider, or the local provider health endpoint is down.
    """
    if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
        raise CodexConfigError(
            f"timeout_seconds must be a finite positive number, got {timeout_seconds}"
        )

    provider = _load_provider_config(
        config.codex_home / "config.toml",
        model_provider=config.model_provider,
    )
    parsed_base_url = _parse_provider_base_url(provider.base_url)
    base_url = _sanitize_parsed_url(parsed_base_url)
    health_url = _derive_health_url(parsed_base_url)
    safe_health_url = _sanitize_url(health_url) if health_url else ""
    if not health_url:
        return CodexRuntimePreflightResult(
            model_provider=config.model_provider,
            base_url=base_url,
            health_url="",
            status="skipped_non_http",
        )

    if not _is_local_http_url(parsed_base_url):
        return CodexRuntimePreflightResult(
            model_provider=config.model_provider,
            base_url=base_url,
            health_url=safe_health_url,
            status="skipped_remote",
        )

    _check_health_url(health_url, timeout_seconds=timeout_seconds)
    return CodexRuntimePreflightResult(
        model_provider=config.model_provider,
        base_url=base_url,
        health_url=safe_health_url,
        status="checked",
    )


def _load_provider_config(
    config_path: Path,
    *,
    model_provider: str,
) -> _CodexProviderDocument:
    if not config_path.is_file():
        raise CodexConfigError(f"Codex config is missing: {config_path}")

    try:
        with config_path.open("rb") as handle:
            raw_document: object = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise CodexConfigError(
            f"Codex config has invalid TOML: {config_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise CodexConfigError(
            f"Codex config could not be read: {config_path}: {exc}"
        ) from exc

    try:
        document = _CodexConfigDocument.model_validate(raw_document)
    except ValidationError as exc:
        raise CodexConfigError(
            "Codex config validation failed: "
            f"path={config_path}; error_count={exc.error_count()}"
        ) from exc

    provider = document.model_providers.get(model_provider)
    if provider is None:
        raise CodexConfigError(
            "Codex model provider config is missing: "
            f"provider={model_provider!r}; path={config_path}"
        )
    return provider


def _parse_provider_base_url(base_url: str) -> ParseResult:
    try:
        parsed_url = urlparse(base_url)
        _ = parsed_url.port
    except ValueError as exc:
        raise CodexConfigError(
            "Codex model provider base_url is invalid: error_type=invalid_url"
        ) from exc

    if parsed_url.scheme in {"http", "https"} and not parsed_url.netloc:
        raise CodexConfigError(
            "Codex model provider base_url is invalid: error_type=missing_host"
        )
    return parsed_url


def _derive_health_url(parsed_url: ParseResult) -> str:
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return ""
    return urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            "/health",
            "",
            "",
            "",
        )
    )


def _sanitize_parsed_url(parsed_url: ParseResult) -> str:
    scheme = parsed_url.scheme or "<missing-scheme>"
    hostname = parsed_url.hostname or "<missing-host>"
    credentials = "<credentials>@" if parsed_url.username or parsed_url.password else ""
    port_text = f":{parsed_url.port}" if parsed_url.port is not None else ""
    path = parsed_url.path or "/"
    return f"{scheme}://{credentials}{hostname}{port_text}{path}"


def _is_local_http_url(parsed_url: ParseResult) -> bool:
    return parsed_url.scheme == "http" and parsed_url.hostname in _LOCAL_PROVIDER_HOSTS


def _check_health_url(health_url: str, *, timeout_seconds: float) -> None:
    request = Request(
        health_url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read()
    except HTTPError as exc:
        error_body = exc.read()
        raise CodexConfigError(
            "Codex model provider is unavailable: "
            f"url={_sanitize_url(health_url)}; error_type=http_error; "
            f"HTTP {exc.code}; response_bytes={len(error_body)}"
        ) from exc
    except TimeoutError as exc:
        raise CodexConfigError(
            "Codex model provider is unavailable: "
            f"url={_sanitize_url(health_url)}; error_type=request_timeout"
        ) from exc
    except URLError as exc:
        raise CodexConfigError(
            "Codex model provider is unavailable: "
            f"url={_sanitize_url(health_url)}; error_type=request_failed; "
            f"reason_type={exc.reason.__class__.__name__}"
        ) from exc
    except OSError as exc:
        raise CodexConfigError(
            "Codex model provider is unavailable: "
            f"url={_sanitize_url(health_url)}; error_type=request_failed; "
            f"reason_type={exc.__class__.__name__}"
        ) from exc

    _decode_health_response(response_body, health_url=health_url)


def _decode_health_response(response_body: bytes, *, health_url: str) -> None:
    try:
        response_text = response_body.decode("utf-8")
        decoded_body: object = json.loads(response_text)
    except UnicodeDecodeError as exc:
        raise CodexConfigError(
            "Codex model provider health response is invalid: "
            f"url={_sanitize_url(health_url)}; error_type=invalid_utf8; "
            f"response_bytes={len(response_body)}"
        ) from exc
    except JSONDecodeError as exc:
        raise CodexConfigError(
            "Codex model provider health response is invalid: "
            f"url={_sanitize_url(health_url)}; error_type=invalid_json; "
            f"response_bytes={len(response_body)}"
        ) from exc
    if not isinstance(decoded_body, (dict, list)):
        raise CodexConfigError(
            "Codex model provider health response is invalid: "
            f"url={_sanitize_url(health_url)}; error_type=unexpected_json"
        )


def _sanitize_url(url: str) -> str:
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme or "<missing-scheme>"
    hostname = parsed_url.hostname or "<missing-host>"
    credentials = "<credentials>@" if parsed_url.username or parsed_url.password else ""
    try:
        port = parsed_url.port
    except ValueError:
        port_text = ":<invalid-port>"
    else:
        port_text = f":{port}" if port is not None else ""
    path = parsed_url.path or "/"
    return f"{scheme}://{credentials}{hostname}{port_text}{path}"


__all__ = [
    "CodexRuntimePreflightResult",
    "CodexRuntimePreflightStatus",
    "preflight_codex_runtime",
]
