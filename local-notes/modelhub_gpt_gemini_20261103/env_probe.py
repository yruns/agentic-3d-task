"""Run an archived ModelHub diagnostic with an explicit environment credential.

Only the credential loader and provenance label are adapted. The archived
request construction, state replay, controls and assertions remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPTS = {
    "gpt_probe.py": "MODELHUB_GPT_AK",
    "gemini_probe.py": "MODELHUB_GEMINI_AK",
    "sdk_driver.py": "MODELHUB_GEMINI_AK",
    "sdk_cache_driver.py": "MODELHUB_GEMINI_AK",
    "cache_probe.py": "MODELHUB_GEMINI_AK",
    "exact_accounting_control.py": "MODELHUB_GEMINI_AK",
    "control_client.py": "MODELHUB_GEMINI_AK",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", choices=SCRIPTS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if any(arg.startswith("--credential-transcript") for arg in arguments):
        raise ValueError("This launcher supplies the credential loader")
    if args.script == "gpt_probe.py":
        # Historical scripts have Chat modes; the current GPT policy forbids them.
        # Match the historical parser's abbreviation rules so a later --prot
        # cannot override the validated protocol in the target parser.
        options = argparse.ArgumentParser(add_help=False)
        options.add_argument("--protocol")
        parsed, _ = options.parse_known_args(arguments)
        if parsed.protocol != "responses":
            raise ValueError("GPT reproduction requires --protocol responses")
    output_options = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    output_options.add_argument("--output", required=True, type=Path)
    parsed_output, _ = output_options.parse_known_args(arguments)
    output = parsed_output.output.resolve()
    if output.exists():
        raise ValueError("Choose a new reproduction output path")
    env_name = SCRIPTS[args.script]
    ak = os.environ.pop(env_name)
    for credential_env in set(SCRIPTS.values()):
        os.environ.pop(credential_env, None)
    if not ak or any(char.isspace() for char in ak):
        raise ValueError("Invalid environment credential")
    script = Path(__file__).resolve().with_name(args.script)
    source = "Explicit environment variable " + env_name
    record = output.with_name(output.name + ".launch.json")
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "credential_source": source,
                "script": args.script,
                "arguments": arguments,
                "source_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(script.parent.glob("*.py"))
                },
                "changes_to_archived_logic": [
                    "Credential loader reads the selected environment variable",
                    "Summary credential_source reflects that environment source",
                ],
            },
            handle,
            indent=2,
        )
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("reproduction_target", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load archived script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.credential = lambda _path: ak
    original_save = module.save

    def save_with_source(path: Path, value: Any, credential: str) -> None:
        if isinstance(value, dict) and path.name == "summary.json":
            value = {**value, "credential_source": source}
        original_save(path, value, credential)

    module.save = save_with_source
    # The original CLI still requires a transcript argument; it is never read.
    sys.argv = [str(script), "--credential-transcript", "/dev/null", *arguments]
    result = module.main()
    return int(result or 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps({"error_type": type(exc).__name__, "details_suppressed": True})
        )
        raise SystemExit(1) from None
