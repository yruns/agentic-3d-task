"""CLI entry point for high-level ALFRED agent tools.

Usage (invoked by Codex inside the episode turn)::

    python -m codex_agent.alfred.tools <tool> \\
        --episode-dir <episode_artifact_dir> --args '<json>'
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..episode_server import SESSION_ENDPOINT_FILENAME
from ..playbook import ALFRED_TOOL_NAMES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex_agent.alfred.tools", description=__doc__
    )
    parser.add_argument(
        "tool",
        choices=list(ALFRED_TOOL_NAMES),
        help="High-level ALFRED tool to execute.",
    )
    parser.add_argument(
        "--episode-dir",
        required=True,
        type=Path,
        help="Active episode artifact directory containing session_endpoint.json.",
    )
    parser.add_argument(
        "--args",
        default="{}",
        help="Tool arguments as a single JSON object (default: '{}').",
    )
    return parser


def _parse_args_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--args must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--args must decode to a JSON object.")
    return parsed


def call_episode_server(
    episode_dir: Path, tool_name: str, args: dict[str, Any], *, timeout: float = 600.0
) -> dict[str, Any]:
    """Send one tool request to the active EpisodeToolServer over loopback."""
    endpoint_file = episode_dir.resolve() / SESSION_ENDPOINT_FILENAME
    if not endpoint_file.is_file():
        return {
            "error": f"Active episode endpoint not found at {endpoint_file}. Is the episode running?"
        }
    info = json.loads(endpoint_file.read_text(encoding="utf-8"))
    url = f"{info['url'].rstrip('/')}/tool"
    body = json.dumps({"tool": tool_name, "args": args}, ensure_ascii=False).encode(
        "utf-8"
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    # Explicitly bypass any environment HTTP/SOCKS proxies for 127.0.0.1 RPC.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            raw_resp = response.read().decode("utf-8")
            result = json.loads(raw_resp)
            return result if isinstance(result, dict) else {"result": result}
    except urllib.error.HTTPError as exc:
        err_text = exc.read().decode("utf-8", errors="replace")
        return {"error": f"Episode server HTTP {exc.code}: {err_text}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Episode server RPC failed: {type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    ns = parser.parse_args(argv)
    try:
        parsed_args = _parse_args_json(ns.args)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 0

    payload = call_episode_server(ns.episode_dir, ns.tool, parsed_args)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
