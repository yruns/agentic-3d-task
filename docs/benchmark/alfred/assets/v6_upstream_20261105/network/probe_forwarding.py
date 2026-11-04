"""Bounded, credential-free SOCKS greeting checks of the existing local forward."""

from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path


def check_forward() -> dict[str, object]:
    """Check the same no-authentication greeting used by the loaded network shim."""
    result: dict[str, object] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "127.0.0.1:10800",
        "tcp_connected": False,
        "socks_greeting_ready": False,
    }
    started = time.monotonic()
    try:
        with socket.create_connection(("127.0.0.1", 10800), timeout=3) as stream:
            result["tcp_connected"] = True
            stream.sendall(bytes.fromhex("050100"))
            received = b""
            while len(received) < 2:
                part = stream.recv(2 - len(received))
                if not part:
                    break
                received += part
        result["reply_hex"] = received.hex()
        result["socks_greeting_ready"] = received == bytes.fromhex("0500")
    except OSError as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
    result["wall_seconds"] = time.monotonic() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for index in range(10):
            record = check_forward()
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(json.dumps(record), flush=True)
            if record["socks_greeting_ready"]:
                return 0
            if index < 9:
                time.sleep(30)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
