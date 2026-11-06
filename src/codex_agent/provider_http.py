"""Validated Codex-home configuration and provider discovery HTTP boundaries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import ParseResult, urlparse, urlunparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10.
    import tomli as tomllib

from .config import CodexAgentConfig
from .errors import CodexConfigError

ADAPTER_PROVIDER = "codex_adapter"
_MAX_DISCOVERY_BYTES = 1024 * 1024
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class _RuntimeDefaults(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    model_context_window: StrictInt | None = None
    model_auto_compact_token_limit: StrictInt | None = None
    model_reasoning_effort: StrictStr | None = None


class _ProviderDocument(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    base_url: StrictStr = Field(min_length=1)
    wire_api: StrictStr = "responses"
    env_key: StrictStr | None = None
    experimental_bearer_token: StrictStr | None = Field(default=None, repr=False)
    env_http_headers: Mapping[str, StrictStr] = Field(default_factory=dict)
    http_headers: Mapping[str, StrictStr] = Field(default_factory=dict)


class _HomeDocument(_RuntimeDefaults):
    model_providers: Mapping[str, _ProviderDocument] = Field(default_factory=dict)
    profile: StrictStr | None = None
    profiles: Mapping[str, _RuntimeDefaults] = Field(default_factory=dict)


@dataclass(frozen=True)
class ProviderSettings:
    """Non-secret immutable settings needed by runtime/preflight."""

    base_url: str
    parsed_url: ParseResult
    inherited_context_window: int | None
    inherited_auto_compact_limit: int | None
    inherited_reasoning_effort: str | None

    @property
    def health_url(self) -> str:
        if self.parsed_url.scheme not in {"http", "https"}:
            return ""
        return urlunparse(
            self.parsed_url._replace(path="/health", params="", query="", fragment="")
        )

    @property
    def is_local_http(self) -> bool:
        return (
            self.parsed_url.scheme == "http"
            and self.parsed_url.hostname in _LOCAL_HOSTS
        )


def load_provider_settings(config: CodexAgentConfig) -> ProviderSettings:
    """Read only the selected home, without credentials or endpoint substitution."""
    path = config.codex_home / "config.toml"
    document = _read_home_document(path)
    provider = document.model_providers.get(config.model_provider)
    if provider is None:
        raise CodexConfigError(
            f"Codex model provider config is missing: provider={config.model_provider!r}; path={path}"
        )
    if config.model_provider == ADAPTER_PROVIDER:
        _validate_adapter_provider(provider)
    parsed = parse_provider_url(provider.base_url)
    defaults = document.profiles.get(document.profile or "", document)
    return ProviderSettings(
        base_url=provider.base_url,
        parsed_url=parsed,
        inherited_context_window=_inherit(
            defaults.model_context_window, document.model_context_window
        ),
        inherited_auto_compact_limit=_inherit(
            defaults.model_auto_compact_token_limit,
            document.model_auto_compact_token_limit,
        ),
        inherited_reasoning_effort=_inherit(
            defaults.model_reasoning_effort, document.model_reasoning_effort
        ),
    )


_InheritedT = TypeVar("_InheritedT", int, str)


def _inherit(
    value: _InheritedT | None, fallback: _InheritedT | None
) -> _InheritedT | None:
    return value if value is not None else fallback


def _read_home_document(path: Path) -> _HomeDocument:
    if not path.is_file():
        raise CodexConfigError(f"Codex config is missing: {path}")
    try:
        with path.open("rb") as handle:
            raw: object = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise CodexConfigError(f"Codex config has invalid TOML: {path}") from exc
    except OSError as exc:
        raise CodexConfigError(f"Codex config could not be read: {path}") from exc
    try:
        return _HomeDocument.model_validate(raw)
    except ValidationError as exc:
        raise CodexConfigError(
            f"Codex config validation failed: path={path}; error_count={exc.error_count()}"
        ) from None


def _validate_adapter_provider(provider: _ProviderDocument) -> None:
    if provider.wire_api != "responses":
        raise CodexConfigError(
            "codex_adapter requires wire_api='responses'; Chat fallback is unsupported"
        )
    if (
        provider.env_key
        or provider.experimental_bearer_token
        or any(name.lower() == "authorization" for name in provider.http_headers)
    ):
        raise CodexConfigError(
            "codex_adapter must not configure upstream authentication in the client; use CODEX_ADAPTER_TOKEN only for local adapter auth"
        )
    header_names = (*provider.env_http_headers, *provider.http_headers)
    if any(name.lower().startswith("x-codex-upstream-") for name in header_names):
        raise CodexConfigError(
            "codex_adapter upstream credentials/endpoints must be registered in the adapter, not client headers"
        )
    if any(name.lower() == "extra" for name in header_names):
        raise CodexConfigError(
            "codex_adapter owns upstream extra/session headers; remove client extra headers"
        )


def parse_provider_url(base_url: str) -> ParseResult:
    try:
        parsed = urlparse(base_url)
        _ = parsed.port
    except ValueError:
        raise CodexConfigError(
            "Codex model provider base_url is invalid: error_type=invalid_url"
        ) from None
    if parsed.scheme in {"http", "https"} and not parsed.netloc:
        raise CodexConfigError(
            "Codex model provider base_url is invalid: error_type=missing_host"
        )
    return parsed


def sanitize_provider_url(url: str | ParseResult) -> str:
    parsed = urlparse(url) if isinstance(url, str) else url
    credentials = "<credentials>@" if parsed.username or parsed.password else ""
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        port = ":<invalid-port>"
    return f"{parsed.scheme or '<missing-scheme>'}://{credentials}{parsed.hostname or '<missing-host>'}{port}{parsed.path or '/'}"


class _NoDiscoveryRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        # Never forward an adapter token to a redirect target.
        return None


def read_provider_json(
    url: str, *, timeout_seconds: float, adapter_token: str = ""
) -> object:
    """Read bounded JSON without logging response bodies, tokens, or URL queries."""
    import time

    headers = {"Accept": "application/json"}
    if adapter_token:
        headers = {**headers, "Authorization": f"Bearer {adapter_token}"}
    request = Request(url, headers=headers, method="GET")
    parsed = urlparse(url)
    is_local = parsed.hostname in _LOCAL_HOSTS
    opener = (
        build_opener(_NoDiscoveryRedirect(), ProxyHandler({}))
        if is_local
        else build_opener(_NoDiscoveryRedirect())
    )
    max_attempts = 4 if (is_local and timeout_seconds >= 5.0) else 1
    body = b""
    for attempt in range(max_attempts):
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                body = response.read(_MAX_DISCOVERY_BYTES + 1)
            break
        except HTTPError as exc:
            raise _unavailable(url, f"http_error; HTTP {exc.code}") from None
        except TimeoutError:
            if attempt + 1 < max_attempts:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise _unavailable(url, "request_timeout") from None
        except (URLError, OSError) as exc:
            if attempt + 1 < max_attempts and isinstance(
                exc.reason if isinstance(exc, URLError) else exc, TimeoutError
            ):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise _unavailable(
                url, f"request_failed; reason_type={type(exc).__name__}"
            ) from None
    if len(body) > _MAX_DISCOVERY_BYTES:
        raise CodexConfigError(
            f"Codex model provider response is too large: url={sanitize_provider_url(url)}"
        )
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CodexConfigError(
            f"Codex model provider health/capabilities response is invalid: url={sanitize_provider_url(url)}; response_bytes={len(body)}"
        ) from None


def _unavailable(url: str, error_type: str) -> CodexConfigError:
    return CodexConfigError(
        f"Codex model provider is unavailable: url={sanitize_provider_url(url)}; error_type={error_type}"
    )
