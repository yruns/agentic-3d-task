"""Run the isolated bridge's explicit visible-token accounting control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx

from probe_support import credential, save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential-transcript", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ak = credential(args.credential_transcript)
    run = json.loads(args.summary.read_text())
    if run["result"] != "passed":
        raise ValueError("Run must finish before accounting control")
    with httpx.Client(timeout=120, follow_redirects=False) as client:
        response = client.post(
            "http://127.0.0.1:8791/probe/visible-token-control",
            headers={
                "X-Codex-Session-ID": run["session_id"],
                "X-Codex-Upstream-AK": ak,
            },
            json={},
        )
    payload = response.json()
    record = {
        "session_id": run["session_id"],
        "http_status": response.status_code,
        "control_response": payload,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save(args.output, record, ak)
    print(
        json.dumps(
            {"http_status": response.status_code, "usage": payload.get("raw_usage")}
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps({"error_type": type(exc).__name__, "details_suppressed": True})
        )
        raise SystemExit(1) from None
