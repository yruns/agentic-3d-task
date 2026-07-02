from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from adapter.proxy import (
    AdapterSettings,
    UpstreamRequest,
    build_compact_upstream_request,
    build_upstream_request,
    health_payload,
    resolve_upstream_extra,
)

app = FastAPI(title="Codex ModelHub Adapter")


@app.get("/health")
async def health() -> dict[str, Any]:
    settings = AdapterSettings.from_env()
    config = health_payload(settings)
    return {
        "status": "healthy" if config["has_upstream_ak"] else "degraded",
        "service": "codex-modelhub-adapter",
        "config": config,
    }


@app.post("/v1/responses")
async def create_response(
    request: Request,
    x_tt_logid: str | None = Header(default=None, alias="X-TT-LOGID"),
    extra_header: str | None = Header(default=None, alias="extra"),
) -> Any:
    try:
        raw_body: Any = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Request body must be valid JSON"
        ) from exc

    settings = AdapterSettings.from_env()
    resolved_extra = resolve_upstream_extra(
        extra_header,
        fallback_session_id=settings.session_id,
    )
    try:
        upstream_request = build_upstream_request(
            raw_body,
            settings=settings,
            upstream_extra=resolved_extra.value,
            logid=x_tt_logid,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    timeout = httpx.Timeout(
        settings.timeout_seconds,
        connect=min(settings.timeout_seconds, 20.0),
        read=None,
    )
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)

    try:
        stream_context, upstream, upstream_request = await _open_upstream_with_retries(
            client,
            upstream_request,
            raw_body=raw_body,
            settings=settings,
            upstream_extra=resolved_extra.value,
            logid=x_tt_logid,
        )
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=502, detail=f"Failed to reach AIDP upstream: {exc}"
        ) from exc

    response_headers = _copy_response_headers(upstream)
    if upstream_request.headers.get("X-TT-LOGID"):
        response_headers["X-TT-LOGID"] = upstream_request.headers["X-TT-LOGID"]

    if upstream.status_code >= 400:
        body = await upstream.aread()
        await stream_context.__aexit__(None, None, None)
        await client.aclose()
        return Response(
            content=body,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type") or "application/json",
            headers=response_headers,
        )

    if bool(upstream_request.body.get("stream")):

        async def iter_upstream_body():
            try:
                async for chunk in upstream.aiter_raw():
                    if chunk:
                        yield chunk
            finally:
                await stream_context.__aexit__(None, None, None)
                await client.aclose()

        return StreamingResponse(
            iter_upstream_body(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type") or "text/event-stream",
            headers=response_headers,
        )

    upstream_bytes = await upstream.aread()
    await stream_context.__aexit__(None, None, None)
    await client.aclose()
    return Response(
        content=upstream_bytes,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type") or "application/json",
        headers=response_headers,
    )


@app.post("/v1/responses/compact")
async def compact_response(
    request: Request,
    x_tt_logid: str | None = Header(default=None, alias="X-TT-LOGID"),
    extra_header: str | None = Header(default=None, alias="extra"),
) -> Response:
    try:
        raw_body: Any = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Request body must be valid JSON"
        ) from exc
    settings = AdapterSettings.from_env()
    resolved_extra = resolve_upstream_extra(
        extra_header,
        fallback_session_id=settings.session_id,
    )
    try:
        upstream_request = build_compact_upstream_request(
            raw_body,
            settings=settings,
            upstream_extra=resolved_extra.value,
            logid=x_tt_logid,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    timeout = httpx.Timeout(
        settings.timeout_seconds,
        connect=min(settings.timeout_seconds, 20.0),
        read=None,
    )
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
    try:
        stream_context, upstream = await _open_upstream(client, upstream_request)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=502, detail=f"Failed to reach AIDP upstream: {exc}"
        ) from exc
    body = await upstream.aread()
    await stream_context.__aexit__(None, None, None)
    await client.aclose()

    response_headers = _copy_response_headers(upstream)
    if upstream_request.headers.get("X-TT-LOGID"):
        response_headers["X-TT-LOGID"] = upstream_request.headers["X-TT-LOGID"]
    return Response(
        content=body,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type") or "application/json",
        headers=response_headers,
    )


async def _open_upstream_with_retries(
    client: httpx.AsyncClient,
    upstream_request: UpstreamRequest,
    *,
    raw_body: Any,
    settings: AdapterSettings,
    upstream_extra: dict[str, str],
    logid: str | None,
) -> tuple[Any, httpx.Response, UpstreamRequest]:
    stream_context = None
    upstream = None
    excluded_upstream_aliases: frozenset[str] = frozenset()
    for attempt in range(settings.max_429_retries + 1):
        stream_context, upstream = await _open_upstream(client, upstream_request)
        if upstream.status_code == 429 and attempt < settings.max_429_retries:
            excluded_upstream_aliases = _updated_excluded_upstream_aliases(
                upstream_request,
                excluded_upstream_aliases,
            )
            upstream_request = build_upstream_request(
                raw_body,
                settings=settings,
                upstream_extra=upstream_extra,
                logid=logid,
                excluded_upstream_aliases=excluded_upstream_aliases,
            )
            await stream_context.__aexit__(None, None, None)
            await asyncio.sleep(1.0 * (2**attempt))
            continue
        break

    if stream_context is None or upstream is None:
        raise RuntimeError("failed to open upstream request")

    if upstream.status_code < 400:
        return stream_context, upstream, upstream_request

    error_bytes = await upstream.aread()
    await stream_context.__aexit__(None, None, None)
    return (
        _BufferedResponseContext(),
        _BufferedResponse(upstream, error_bytes),
        upstream_request,
    )


def _updated_excluded_upstream_aliases(
    upstream_request: Any,
    excluded_upstream_aliases: frozenset[str],
) -> frozenset[str]:
    if upstream_request.upstream_key_selection != "toml_weighted_extra_hash":
        return excluded_upstream_aliases
    if not upstream_request.upstream_key_alias:
        return excluded_upstream_aliases
    return excluded_upstream_aliases | frozenset((upstream_request.upstream_key_alias,))


async def _open_upstream(
    client: httpx.AsyncClient, upstream_request: Any
) -> tuple[Any, httpx.Response]:
    stream_context = client.stream(
        "POST",
        upstream_request.url,
        headers=upstream_request.headers,
        json=upstream_request.body,
    )
    upstream = await stream_context.__aenter__()
    return stream_context, upstream


class _BufferedResponseContext:
    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _BufferedResponse:
    def __init__(self, original: httpx.Response, body: bytes) -> None:
        self.status_code = original.status_code
        self.headers = original.headers
        self._body = body

    async def aread(self) -> bytes:
        return self._body

    async def aiter_raw(self):
        yield self._body

    async def aiter_text(self):
        yield self._body.decode("utf-8", errors="replace")


def _copy_response_headers(upstream: httpx.Response) -> dict[str, str]:
    headers: dict[str, str] = {
        "Cache-Control": upstream.headers.get("cache-control") or "no-cache",
        "X-Accel-Buffering": "no",
    }
    content_type = upstream.headers.get("content-type")
    if content_type:
        headers["Content-Type"] = content_type
    return headers
