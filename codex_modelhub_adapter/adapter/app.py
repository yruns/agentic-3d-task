from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from adapter.mapping import (
    build_compaction_response,
    chat_completion_to_response,
    is_context_length_exceeded,
    is_invalid_encrypted_content,
    iter_chat_sse_as_responses,
    log_cache_hit,
    sanitize_encrypted_state,
)
from adapter.proxy import (
    AdapterSettings,
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
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc

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
        raise HTTPException(status_code=502, detail=f"Failed to reach AIDP upstream: {exc}") from exc

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

    if upstream_request.upstream_api == "chat_completions":
        if bool(upstream_request.body.get("stream")):
            async def iter_chat_as_responses():
                try:
                    async for chunk in iter_chat_sse_as_responses(
                        upstream.aiter_text(),
                        logid=upstream_request.headers.get("X-TT-LOGID", ""),
                        key_alias=upstream_request.upstream_key_alias,
                    ):
                        yield chunk
                finally:
                    await stream_context.__aexit__(None, None, None)
                    await client.aclose()

            return StreamingResponse(
                iter_chat_as_responses(),
                status_code=upstream.status_code,
                media_type="text/event-stream",
                headers={**response_headers, "Content-Type": "text/event-stream"},
            )

        upstream_bytes = await upstream.aread()
        await stream_context.__aexit__(None, None, None)
        await client.aclose()
        try:
            upstream_payload = json.loads(upstream_bytes.decode("utf-8")) if upstream_bytes else {}
        except Exception:
            upstream_payload = {}
        if isinstance(upstream_payload, dict):
            log_cache_hit(
                upstream_payload.get("usage"),
                logid=upstream_request.headers.get("X-TT-LOGID", ""),
                key_alias=upstream_request.upstream_key_alias,
            )
        return JSONResponse(
            status_code=upstream.status_code,
            content=chat_completion_to_response(
                upstream_payload if isinstance(upstream_payload, dict) else {}
            ),
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
async def compact_response(request: Request) -> JSONResponse:
    try:
        raw_body: Any = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
    settings = AdapterSettings.from_env()
    return JSONResponse(
        content=build_compaction_response(
            raw_body,
            max_chars=settings.compact_summary_max_chars,
        )
    )


async def _open_upstream_with_retries(
    client: httpx.AsyncClient,
    upstream_request: Any,
    *,
    raw_body: Any,
    settings: AdapterSettings,
    upstream_extra: dict[str, str],
    logid: str | None,
) -> tuple[Any, httpx.Response, Any]:
    stream_context = None
    upstream = None
    for attempt in range(settings.max_429_retries + 1):
        stream_context, upstream = await _open_upstream(client, upstream_request)
        if upstream.status_code == 429 and attempt < settings.max_429_retries:
            await stream_context.__aexit__(None, None, None)
            await asyncio.sleep(1.0 * (2 ** attempt))
            continue
        break

    if stream_context is None or upstream is None:
        raise RuntimeError("failed to open upstream request")

    if upstream.status_code < 400:
        return stream_context, upstream, upstream_request

    error_bytes = await upstream.aread()
    retry_request = None
    if upstream_request.upstream_api == "chat_completions" and is_context_length_exceeded(error_bytes):
        retry_request = build_upstream_request(
            raw_body,
            settings=settings,
            chat_context_token_limit=settings.chat_context_retry_token_limit,
            upstream_extra=upstream_extra,
            logid=logid,
        )
    elif settings.encrypted_state_fallback_enabled and is_invalid_encrypted_content(error_bytes):
        if isinstance(raw_body, dict):
            sanitized_body, stats = sanitize_encrypted_state(raw_body)
            if (
                stats.removed_encrypted_content_count
                or stats.dropped_empty_reasoning_count
                or stats.removed_previous_response_id
            ):
                retry_request = build_upstream_request(
                    sanitized_body,
                    settings=settings,
                    upstream_extra=upstream_extra,
                    logid=logid,
                )

    if retry_request is None:
        return _BufferedResponseContext(), _BufferedResponse(upstream, error_bytes), upstream_request

    await stream_context.__aexit__(None, None, None)
    retry_stream_context, retry_upstream = await _open_upstream(client, retry_request)
    return retry_stream_context, retry_upstream, retry_request


async def _open_upstream(client: httpx.AsyncClient, upstream_request: Any) -> tuple[Any, httpx.Response]:
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
