"""Single-variable signature-removal control with the original output limit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from probe_support import credential, save

ENDPOINT = "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl"
EMPTY_HASH = hashlib.sha256(b'""').hexdigest()


def visible_body(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        if value.get("redacted") is True:
            if (
                key == "reasoning_content"
                and value.get("bytes") == 2
                and value.get("sha256") == EMPTY_HASH
            ):
                return ""
            raise ValueError("Cannot reconstruct a nonempty redacted field")
        return {k: visible_body(v, k) for k, v in value.items() if k != "signature"}
    if isinstance(value, list):
        return [visible_body(v, key) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-transcript", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ak = credential(args.credential_transcript)
    sid = json.loads(args.summary.read_text())["session_id"]
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    source = [
        r
        for r in rows
        if r.get("session_id") == sid and r.get("kind") != "visible_token_control"
    ][-1]
    original = source["upstream_request"]
    body = visible_body(original)
    assert body["max_tokens"] == original["max_tokens"] == 8192
    args.output.mkdir(parents=True, exist_ok=True)
    headers = {
        "extra": json.dumps({"session_id": sid}),
        "X-TT-LOGID": "exact_visible_" + uuid.uuid4().hex,
    }
    save(args.output / "request.json", {"headers": headers, "body": body}, ak)
    started = time.monotonic()
    with httpx.Client(timeout=120, follow_redirects=False) as client:
        response = client.post(ENDPOINT, params={"ak": ak}, headers=headers, json=body)
    payload = response.json()
    save(args.output / "response.json", payload, ak)
    record = {
        "session_id": sid,
        "source_request_id": source["request_id"],
        "http_status": response.status_code,
        "max_tokens": 8192,
        "only_model_input_change": "remove all signature fields; empty redacted reasoning_content restored using verified hash",
        "usage": payload.get("usage"),
        "error": payload.get("error"),
        "elapsed_s": time.monotonic() - started,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save(args.output / "summary.json", record, ak)
    print(json.dumps({"http_status": response.status_code, "usage": record["usage"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps({"error_type": type(exc).__name__, "details_suppressed": True})
        )
        raise SystemExit(1) from None
