from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any
from urllib.parse import quote, urlsplit

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python < 3.11.
    import tomli as tomllib


JsonObject = dict[str, Any]
_VALID_UPSTREAM_ENVS = frozenset({"office", "online"})
_PLACEHOLDER_SECRET_VALUES = frozenset(
    {
        "replace-with-modelhub-ak",
        "replace-with-modelhub-ak-1",
        "replace-with-modelhub-ak-2",
        "replace-with-modelhub-ak-3",
    }
)
_PLACEHOLDER_SECRET_PREFIXES = ("replace-", "replace_", "placeholder", "your-")
_RESPONSES_API_NAME = "responses"


@dataclass(frozen=True)
class ModelHubUpstream:
    url: str
    model_name: str
    ak: str
    weight: int = 1
    alias: str = ""


@dataclass(frozen=True)
class AdapterSettings:
    online_base_url: str = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online"
    office_base_url: str = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online"
    base_url: str = ""
    upstream_env: str = "office"
    responses_path: str = "/responses"
    encrypted_state_fallback_enabled: bool = True
    timeout_seconds: float = 300.0
    max_429_retries: int = 3
    session_id: str = "case-reviewer-codex"
    modelhub_ak: str = ""
    modelhub_ak_pool: tuple[tuple[str, str], ...] = ()
    modelhub_upstreams_toml: str = ""
    modelhub_upstreams: tuple[ModelHubUpstream, ...] = ()

    @classmethod
    def from_env(cls) -> AdapterSettings:
        defaults = cls()
        _validate_responses_only_upstream_api(
            os.getenv("AIDP_CODEX_PROXY_UPSTREAM_API")
        )
        legacy_url = (os.getenv("MODELHUB_URL") or "").strip()
        base_url = (os.getenv("AIDP_BASE_URL") or "").strip()
        responses_path = (
            os.getenv("AIDP_CODEX_PROXY_RESPONSES_PATH") or defaults.responses_path
        )
        modelhub_upstreams_toml = _resolve_modelhub_upstreams_toml_path(
            os.getenv("AIDP_MODELHUB_UPSTREAMS_TOML")
        )

        if legacy_url:
            parsed_base_url, parsed_path = _split_modelhub_url(legacy_url)
            if parsed_path == "/v2/crawl":
                raise RuntimeError(
                    "Responses API only: MODELHUB_URL must not use /v2/crawl"
                )
            if parsed_base_url:
                base_url = parsed_base_url
            if parsed_path:
                responses_path = parsed_path

        return cls(
            online_base_url=os.getenv("AIDP_CODEX_PROXY_ONLINE_BASE_URL")
            or defaults.online_base_url,
            office_base_url=os.getenv("AIDP_CODEX_PROXY_OFFICE_BASE_URL")
            or defaults.office_base_url,
            base_url=base_url,
            upstream_env=_upstream_env_from_env(
                os.getenv("AIDP_CODEX_PROXY_UPSTREAM_ENV")
            ),
            responses_path=responses_path,
            encrypted_state_fallback_enabled=_bool_env(
                "AIDP_CODEX_PROXY_ENCRYPTED_STATE_FALLBACK_ENABLED",
                defaults.encrypted_state_fallback_enabled,
            ),
            timeout_seconds=_float_env(
                "AIDP_CODEX_PROXY_TIMEOUT_SECONDS", defaults.timeout_seconds
            ),
            max_429_retries=_int_env(
                "AIDP_CODEX_PROXY_MAX_429_RETRIES", defaults.max_429_retries
            ),
            session_id=(
                os.getenv("AIDP_SESSION_ID")
                or os.getenv("AIDP_CODEX_PROXY_SESSION_ID")
                or defaults.session_id
            ).strip(),
            modelhub_ak=_valid_modelhub_ak(
                os.getenv("AIDP_GPT_AK")
                or os.getenv("AIDP_MODELHUB_AK")
                or os.getenv("MODELHUB_AK")
                or os.getenv("CASE_REVIEW_LLM_AK")
                or defaults.modelhub_ak
            ).strip(),
            modelhub_ak_pool=_parse_modelhub_key_pool(
                os.getenv("AIDP_MODELHUB_AK_POOL")
            ),
            modelhub_upstreams_toml=modelhub_upstreams_toml,
            modelhub_upstreams=_load_modelhub_upstreams_from_toml(
                modelhub_upstreams_toml
            ),
        )

    @property
    def upstream_base_url(self) -> str:
        if self.base_url:
            return _trim_trailing_slash(self.base_url)
        if self.upstream_env == "office":
            return _trim_trailing_slash(self.office_base_url)
        return _trim_trailing_slash(self.online_base_url)


@dataclass(frozen=True)
class UpstreamRequest:
    url: str
    headers: dict[str, str]
    body: JsonObject
    upstream_api: str = _RESPONSES_API_NAME
    upstream_key_alias: str = ""
    upstream_key_selection: str = ""


@dataclass(frozen=True)
class ResolvedExtra:
    value: dict[str, str]
    session_id: str
    source: str


@dataclass(frozen=True)
class _ResolvedModelHubTarget:
    base_url: str
    ak: str
    path_override: str
    key_alias: str
    key_selection: str


def build_upstream_request(
    raw_body: Any,
    *,
    settings: AdapterSettings | None = None,
    upstream_extra: dict[str, str] | None = None,
    logid: str | None = None,
    excluded_upstream_aliases: frozenset[str] = frozenset(),
) -> UpstreamRequest:
    resolved_settings = settings or AdapterSettings.from_env()
    target = _resolve_modelhub_target(
        resolved_settings,
        raw_body,
        upstream_extra,
        excluded_upstream_aliases=excluded_upstream_aliases,
    )
    path = target.path_override or _normalize_path(resolved_settings.responses_path)
    return UpstreamRequest(
        url=f"{target.base_url}{path}?ak={quote(target.ak, safe='')}",
        headers=_build_headers(
            upstream_extra=upstream_extra,
            fallback_session_id=resolved_settings.session_id,
            logid=logid,
        ),
        body=_responses_body(raw_body),
        upstream_key_alias=target.key_alias,
        upstream_key_selection=target.key_selection,
    )


def build_compact_upstream_request(
    raw_body: Any,
    *,
    settings: AdapterSettings | None = None,
    upstream_extra: dict[str, str] | None = None,
    logid: str | None = None,
    excluded_upstream_aliases: frozenset[str] = frozenset(),
) -> UpstreamRequest:
    resolved_settings = settings or AdapterSettings.from_env()
    target = _resolve_modelhub_target(
        resolved_settings,
        raw_body,
        upstream_extra,
        excluded_upstream_aliases=excluded_upstream_aliases,
    )
    responses_path = target.path_override or _normalize_path(
        resolved_settings.responses_path
    )
    path = _responses_compact_path(responses_path)
    return UpstreamRequest(
        url=f"{target.base_url}{path}?ak={quote(target.ak, safe='')}",
        headers=_build_headers(
            upstream_extra=upstream_extra,
            fallback_session_id=resolved_settings.session_id,
            logid=logid,
        ),
        body=_responses_body(raw_body),
        upstream_key_alias=target.key_alias,
        upstream_key_selection=target.key_selection,
    )


def _resolve_modelhub_target(
    settings: AdapterSettings,
    raw_body: Any,
    upstream_extra: dict[str, str] | None,
    *,
    excluded_upstream_aliases: frozenset[str],
) -> _ResolvedModelHubTarget:
    selected_upstream = resolve_modelhub_upstream(
        settings,
        raw_body,
        upstream_extra,
        excluded_upstream_aliases=excluded_upstream_aliases,
    )
    if selected_upstream is not None:
        base_url, path_override = _split_modelhub_target_url(selected_upstream.url)
        return _ResolvedModelHubTarget(
            base_url=base_url,
            ak=selected_upstream.ak,
            path_override=path_override,
            key_alias=selected_upstream.alias,
            key_selection="toml_weighted_extra_hash",
        )

    selected_key = resolve_modelhub_key(settings, upstream_extra)
    if selected_key is not None:
        key_alias, ak = selected_key
        return _ResolvedModelHubTarget(
            base_url=settings.upstream_base_url,
            ak=ak,
            path_override="",
            key_alias=key_alias,
            key_selection="extra_session_rendezvous_hash",
        )

    ak = _valid_modelhub_ak(settings.modelhub_ak)
    if not ak:
        raise RuntimeError("AIDP_GPT_AK or AIDP_MODELHUB_AK is required")
    return _ResolvedModelHubTarget(
        base_url=settings.upstream_base_url,
        ak=ak,
        path_override="",
        key_alias="single",
        key_selection="single_key_fallback",
    )


def resolve_modelhub_key(
    settings: AdapterSettings,
    upstream_extra: dict[str, str] | None = None,
) -> tuple[str, str] | None:
    keys = _normalize_key_pool(settings.modelhub_ak_pool)
    if not keys:
        return None
    session_id = (
        _valid_upstream_extra_session_id((upstream_extra or {}).get("session_id"))
        or settings.session_id
    )
    return _pick_modelhub_key_by_rendezvous_hash(session_id, keys)


def resolve_modelhub_upstream(
    settings: AdapterSettings,
    raw_body: Any,
    upstream_extra: dict[str, str] | None = None,
    *,
    excluded_upstream_aliases: frozenset[str] = frozenset(),
) -> ModelHubUpstream | None:
    upstreams = _normalize_modelhub_upstreams(settings.modelhub_upstreams)
    if not upstreams:
        return None
    model = _request_model_name(raw_body)
    matches = tuple(
        upstream
        for upstream in upstreams
        if _modelhub_upstream_matches_model(upstream, model)
    )
    if not matches:
        raise RuntimeError(
            f"No ModelHub TOML upstream matches model '{model or '<empty>'}'"
        )
    available_matches = tuple(
        upstream
        for upstream in matches
        if upstream.alias not in excluded_upstream_aliases
    )
    if not available_matches:
        available_matches = matches
    selection_id = _modelhub_upstream_selection_id(settings, upstream_extra)
    return _pick_modelhub_upstream_by_weighted_session_hash(
        selection_id, model, available_matches
    )


def resolve_upstream_extra(
    client_extra_header: str | None,
    *,
    fallback_session_id: str,
) -> ResolvedExtra:
    default_session_id = _safe_default_upstream_extra_session_id(fallback_session_id)
    if client_extra_header:
        try:
            parsed = json.loads(client_extra_header)
        except Exception:
            parsed = None
        session_id = (
            _valid_upstream_extra_session_id(parsed.get("session_id"))
            if isinstance(parsed, dict)
            else ""
        )
        request_source = (
            _valid_upstream_extra_source(parsed.get("source"))
            if isinstance(parsed, dict)
            else ""
        )
        payload = _build_upstream_extra_payload(
            session_id=session_id or default_session_id,
            request_source=request_source or None,
            chat_run_id=(
                str(parsed.get("chat_run_id"))
                if isinstance(parsed, dict) and parsed.get("chat_run_id") is not None
                else None
            ),
            sandbox_session_id=(
                str(parsed.get("sandbox_session_id"))
                if isinstance(parsed, dict)
                and parsed.get("sandbox_session_id") is not None
                else None
            ),
        )
        return ResolvedExtra(
            value=payload,
            session_id=session_id or default_session_id,
            source="client_header" if session_id else "default_invalid_client_header",
        )
    payload = _build_upstream_extra_payload(session_id=default_session_id)
    return ResolvedExtra(value=payload, session_id=default_session_id, source="default")


def encode_upstream_extra(
    upstream_extra: dict[str, str] | None,
    *,
    fallback_session_id: str,
) -> str:
    payload = (
        _build_upstream_extra_payload(
            session_id=(
                _valid_upstream_extra_session_id(upstream_extra.get("session_id"))
                if isinstance(upstream_extra, dict)
                else ""
            )
            or fallback_session_id,
            request_source=(
                str(upstream_extra.get("source"))
                if isinstance(upstream_extra, dict)
                and upstream_extra.get("source") is not None
                else None
            ),
            chat_run_id=(
                str(upstream_extra.get("chat_run_id"))
                if isinstance(upstream_extra, dict)
                and upstream_extra.get("chat_run_id") is not None
                else None
            ),
            sandbox_session_id=(
                str(upstream_extra.get("sandbox_session_id"))
                if isinstance(upstream_extra, dict)
                and upstream_extra.get("sandbox_session_id") is not None
                else None
            ),
        )
        if isinstance(upstream_extra, dict)
        else _build_upstream_extra_payload(session_id=fallback_session_id)
    )
    return json.dumps(payload, separators=(",", ":"))


def health_payload(settings: AdapterSettings | None = None) -> JsonObject:
    resolved = settings or AdapterSettings.from_env()
    keys = _normalize_key_pool(resolved.modelhub_ak_pool)
    toml_upstreams = _normalize_modelhub_upstreams(resolved.modelhub_upstreams)
    return {
        "upstream_base_url": resolved.upstream_base_url,
        "upstream_api": _RESPONSES_API_NAME,
        "responses_path": _normalize_path(resolved.responses_path),
        "encrypted_state_fallback_enabled": resolved.encrypted_state_fallback_enabled,
        "session_id": resolved.session_id,
        "modelhub_key_pool_enabled": bool(keys),
        "modelhub_key_pool_size": len(keys),
        "modelhub_key_aliases": [alias for alias, _ in keys],
        "modelhub_toml_path": resolved.modelhub_upstreams_toml,
        "modelhub_toml_upstream_count": len(toml_upstreams),
        "modelhub_toml_models": sorted(
            {upstream.model_name for upstream in toml_upstreams}
        ),
        "modelhub_toml_urls": sorted({upstream.url for upstream in toml_upstreams}),
        "has_upstream_ak": bool(
            _valid_modelhub_ak(resolved.modelhub_ak) or keys or toml_upstreams
        ),
    }


def _build_headers(
    *,
    upstream_extra: dict[str, str] | None,
    fallback_session_id: str,
    logid: str | None,
) -> dict[str, str]:
    return {
        "content-type": "application/json",
        "X-TT-LOGID": logid or _build_logid(),
        "extra": encode_upstream_extra(
            upstream_extra, fallback_session_id=fallback_session_id
        ),
    }


def _responses_body(raw_body: Any) -> JsonObject:
    if not isinstance(raw_body, dict):
        return {}
    return raw_body


def _responses_compact_path(responses_path: str) -> str:
    normalized = _normalize_path(responses_path)
    if normalized.endswith("/compact"):
        return normalized
    return f"{normalized}/compact"


def _validate_responses_only_upstream_api(value: str | None) -> None:
    if value is None or not value.strip():
        return
    if value.strip().lower() == _RESPONSES_API_NAME:
        return
    raise RuntimeError("Responses API only: chat completions upstream is unsupported")


def _default_upstream_env() -> str:
    if platform.system() == "Linux":
        return "online"
    return "office"


def _upstream_env_from_env(value: str | None) -> str:
    if value is None or not value.strip():
        return _default_upstream_env()
    upstream_env = value.strip().lower()
    if upstream_env not in _VALID_UPSTREAM_ENVS:
        raise RuntimeError(
            "AIDP_CODEX_PROXY_UPSTREAM_ENV must be one of "
            f"{sorted(_VALID_UPSTREAM_ENVS)}, got {value!r}"
        )
    return upstream_env


def _split_modelhub_url(url: str) -> tuple[str, str]:
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        return "", ""
    path = parsed.path.rstrip("/")
    for suffix in ("/v2/crawl", "/responses"):
        if path.endswith(suffix):
            base_path = path[: -len(suffix)]
            return f"{parsed.scheme}://{parsed.netloc}{base_path}", suffix
    return f"{parsed.scheme}://{parsed.netloc}{path}", ""


def _parse_modelhub_key_pool(value: str | None) -> tuple[tuple[str, str], ...]:
    raw = (value or "").strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    pairs: list[tuple[str, str]] = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            alias = _valid_modelhub_key_alias(item.get("alias"))
            ak = str(item.get("ak") or "").strip()
            if alias and ak:
                pairs.append((alias, ak))
    else:
        for index, item in enumerate(raw.replace("\n", ",").split(",")):
            if not item.strip():
                continue
            alias, separator, ak = item.partition("=")
            if not separator:
                alias = f"k{index + 1}"
                ak = item
            safe_alias = _valid_modelhub_key_alias(alias)
            ak = ak.strip()
            if safe_alias and ak:
                pairs.append((safe_alias, ak))
    return tuple(dict(pairs).items())


def _resolve_modelhub_upstreams_toml_path(value: str | None) -> str:
    configured = str(value or "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    return ""


def _load_modelhub_upstreams_from_toml(path: str) -> tuple[ModelHubUpstream, ...]:
    if not path:
        return ()
    if not os.path.exists(path):
        raise RuntimeError(f"ModelHub TOML upstream config not found: {path}")
    try:
        with open(path, "rb") as handle:
            parsed = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(
            f"ModelHub TOML upstream config is invalid: {path}: {exc}"
        ) from exc

    items = _modelhub_toml_upstream_items(parsed)
    if not items:
        raise RuntimeError(
            "ModelHub TOML upstream config must define at least one [[upstreams]] entry"
        )

    upstreams: list[ModelHubUpstream] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"ModelHub TOML upstream entry #{index + 1} must be a table"
            )
        upstreams.append(_parse_modelhub_upstream_item(item, index))
    return tuple(upstreams)


def _modelhub_toml_upstream_items(parsed: JsonObject) -> list[Any]:
    upstreams = parsed.get("upstreams")
    if isinstance(upstreams, list):
        return upstreams
    modelhub = parsed.get("modelhub")
    if isinstance(modelhub, dict) and isinstance(modelhub.get("upstreams"), list):
        return modelhub["upstreams"]
    legacy = parsed.get("modelhub_upstreams")
    if isinstance(legacy, list):
        return legacy
    return []


def _parse_modelhub_upstream_item(item: dict[str, Any], index: int) -> ModelHubUpstream:
    url = str(item.get("url") or "").strip()
    model_name = str(item.get("model_name") or item.get("model") or "").strip()
    ak = str(item.get("ak") or "").strip()
    alias = _valid_modelhub_key_alias(item.get("alias")) or f"u{index + 1}"

    try:
        weight = int(item.get("weight", 1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"ModelHub TOML upstream entry '{alias}' has invalid weight"
        ) from exc

    if not _is_valid_http_url(url):
        raise RuntimeError(
            f"ModelHub TOML upstream entry '{alias}' must provide a valid http(s) url"
        )
    _, path = _split_modelhub_url(url)
    if path == "/v2/crawl":
        raise RuntimeError(
            f"Responses API only: ModelHub TOML upstream '{alias}' must not use /v2/crawl"
        )
    if not model_name:
        raise RuntimeError(
            f"ModelHub TOML upstream entry '{alias}' must provide model_name"
        )
    if not ak:
        raise RuntimeError(f"ModelHub TOML upstream entry '{alias}' must provide ak")
    if weight <= 0:
        raise RuntimeError(f"ModelHub TOML upstream entry '{alias}' weight must be > 0")

    return ModelHubUpstream(
        url=_trim_trailing_slash(url),
        model_name=model_name,
        ak=ak,
        weight=weight,
        alias=alias,
    )


def _normalize_modelhub_upstreams(
    value: tuple[ModelHubUpstream, ...],
) -> tuple[ModelHubUpstream, ...]:
    upstreams: list[ModelHubUpstream] = []
    seen_aliases: set[str] = set()
    for index, upstream in enumerate(value):
        alias = _valid_modelhub_key_alias(upstream.alias) or f"u{index + 1}"
        if alias in seen_aliases:
            continue
        seen_aliases.add(alias)
        try:
            weight = int(upstream.weight)
        except (TypeError, ValueError):
            continue
        url = _trim_trailing_slash(str(upstream.url or "").strip())
        model_name = str(upstream.model_name or "").strip()
        ak = _valid_modelhub_ak(upstream.ak)
        _, path = _split_modelhub_url(url)
        if (
            not _is_valid_http_url(url)
            or path == "/v2/crawl"
            or not model_name
            or not ak
            or weight <= 0
        ):
            continue
        upstreams.append(
            ModelHubUpstream(
                url=url,
                model_name=model_name,
                ak=ak,
                weight=weight,
                alias=alias,
            )
        )
    return tuple(upstreams)


def _modelhub_upstream_matches_model(upstream: ModelHubUpstream, model: str) -> bool:
    pattern = upstream.model_name.strip()
    if pattern == "*":
        return True
    return bool(model) and fnmatchcase(model, pattern)


def _pick_modelhub_upstream_by_weighted_session_hash(
    session_id: str,
    model: str,
    upstreams: tuple[ModelHubUpstream, ...],
) -> ModelHubUpstream:
    total_weight = sum(upstream.weight for upstream in upstreams)
    digest = hashlib.sha256(
        f"{session_id}:{model}:modelhub-upstream".encode()
    ).hexdigest()
    bucket = int(digest, 16) % total_weight
    cumulative = 0
    for upstream in upstreams:
        cumulative += upstream.weight
        if bucket < cumulative:
            return upstream
    return upstreams[-1]


def _modelhub_upstream_selection_id(
    settings: AdapterSettings,
    upstream_extra: dict[str, str] | None,
) -> str:
    extra = upstream_extra or {}
    return (
        _valid_upstream_extra_session_id(extra.get("chat_run_id"))
        or _valid_upstream_extra_session_id(extra.get("sandbox_session_id"))
        or _valid_upstream_extra_session_id(extra.get("session_id"))
        or settings.session_id
    )


def _split_modelhub_target_url(url: str) -> tuple[str, str]:
    base_url, path = _split_modelhub_url(url)
    if not path:
        return _trim_trailing_slash(base_url or url), ""
    if path == "/responses":
        return _trim_trailing_slash(base_url), _normalize_path(path)
    raise RuntimeError(f"Responses API only: unsupported ModelHub URL path '{path}'")


def _normalize_key_pool(
    value: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for alias, ak in value:
        safe_alias = _valid_modelhub_key_alias(alias)
        safe_ak = _valid_modelhub_ak(ak)
        if not safe_alias or not safe_ak or safe_alias in seen:
            continue
        seen.add(safe_alias)
        pairs.append((safe_alias, safe_ak))
    return tuple(pairs)


def _pick_modelhub_key_by_rendezvous_hash(
    session_id: str,
    keys: tuple[tuple[str, str], ...],
) -> tuple[str, str]:
    best_key = keys[0]
    best_score = -1
    for key in keys:
        alias, _ = key
        digest = hashlib.sha256(f"{session_id}:{alias}".encode()).hexdigest()
        score = int(digest, 16)
        if score > best_score:
            best_score = score
            best_key = key
    return best_key


def _request_model_name(raw_body: Any) -> str:
    if not isinstance(raw_body, dict):
        return ""
    return str(raw_body.get("model") or "").strip()


def _build_upstream_extra_payload(
    *,
    session_id: str,
    request_source: str | None = None,
    chat_run_id: str | None = None,
    sandbox_session_id: str | None = None,
) -> dict[str, str]:
    payload = {"session_id": _safe_default_upstream_extra_session_id(session_id)}
    source = _valid_upstream_extra_source(request_source)
    if source:
        payload["source"] = source
    normalized_chat_run_id = _valid_upstream_extra_session_id(chat_run_id)
    if normalized_chat_run_id:
        payload["chat_run_id"] = normalized_chat_run_id
    normalized_sandbox_session_id = _valid_upstream_extra_session_id(sandbox_session_id)
    if normalized_sandbox_session_id:
        payload["sandbox_session_id"] = normalized_sandbox_session_id
    return payload


_UPSTREAM_EXTRA_SESSION_ID_MAX_LENGTH = 128
_UPSTREAM_EXTRA_SOURCE_MAX_LENGTH = 64
_MODELHUB_KEY_ALIAS_MAX_LENGTH = 64
_ALLOWED_EXTRA_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
)


def _valid_upstream_extra_session_id(value: Any) -> str:
    session_id = str(value or "").strip()
    if not session_id or len(session_id) > _UPSTREAM_EXTRA_SESSION_ID_MAX_LENGTH:
        return ""
    if any(ch not in _ALLOWED_EXTRA_CHARS for ch in session_id):
        return ""
    return session_id


def _valid_upstream_extra_source(value: Any) -> str:
    source = str(value or "").strip()
    if not source or len(source) > _UPSTREAM_EXTRA_SOURCE_MAX_LENGTH:
        return ""
    if any(ch not in _ALLOWED_EXTRA_CHARS for ch in source):
        return ""
    return source


def _valid_modelhub_key_alias(value: Any) -> str:
    alias = str(value or "").strip()
    if not alias or len(alias) > _MODELHUB_KEY_ALIAS_MAX_LENGTH:
        return ""
    if any(ch not in _ALLOWED_EXTRA_CHARS for ch in alias):
        return ""
    return alias


def _valid_modelhub_ak(value: Any) -> str:
    secret = str(value or "").strip()
    if not secret:
        return ""
    normalized = secret.lower()
    if normalized in _PLACEHOLDER_SECRET_VALUES:
        return ""
    if any(normalized.startswith(prefix) for prefix in _PLACEHOLDER_SECRET_PREFIXES):
        return ""
    return secret


def _safe_default_upstream_extra_session_id(value: str) -> str:
    valid = _valid_upstream_extra_session_id(value)
    if valid:
        return valid
    return "case-reviewer-codex"


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise RuntimeError(f"{name} must be >= 0, got {value}")
    return value


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0, got {value}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_logid() -> str:
    return f"codex-adapter-{uuid.uuid4().hex}"


def _normalize_path(path: str) -> str:
    stripped = (path or "").strip()
    if not stripped:
        return "/responses"
    return "/" + stripped.strip("/")


def _trim_trailing_slash(url: str) -> str:
    return url.rstrip("/")


def _is_valid_http_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
