"""Exercise tunnel reconnection and cancellation using local fake commands."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/scope/run_tunnel.sh"
_PRIVATE_MARKER = "fake-private-credential-must-not-reach-tunnel-logs"


@dataclass(frozen=True)
class TunnelFixture:
    """Fake command boundaries and files owned solely by this integration test."""

    config_path: Path
    attempts_path: Path
    sleep_delays_path: Path
    ssh_ready_path: Path
    sleep_ready_path: Path
    ssh_stopped_path: Path
    sleep_stopped_path: Path
    environment: dict[str, str]


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _make_fixture(
    directory: Path, exit_codes: str, *, block_backoff: bool = False
) -> TunnelFixture:
    """Build fake ssh and sleep without inheriting credentials or using a network."""
    fakebin = directory / "fakebin"
    fakebin.mkdir()
    config_path = directory / "private-ssh-config"
    config_path.write_text(_PRIVATE_MARKER + "\n", encoding="utf-8")
    _write_executable(
        fakebin / "ssh",
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "directory=$FAKE_TUNNEL_TEST_DIR\n"
        'count_file="$directory/attempts"\n'
        "count=0\n"
        '[[ ! -f "$count_file" ]] || read -r count < "$count_file"\n'
        "count=$((count + 1))\n"
        'printf "%s\\n" "$count" > "$count_file"\n'
        f"printf '%s\\n' '{_PRIVATE_MARKER}' >&2\n"
        'IFS=" " read -r -a outcomes <<< "$FAKE_SSH_EXIT_CODES"\n'
        "outcome=${outcomes[$((count - 1))]:-block}\n"
        'if [[ "$outcome" != block ]]; then exit "$outcome"; fi\n'
        'trap \'printf "stopped\\n" >> "$directory/ssh_stopped"; exit 0\' TERM\n'
        'printf "%s\\n" "$count" > "$directory/ssh_ready"\n'
        "while true; do /bin/sleep 0.02; done\n",
    )
    _write_executable(
        fakebin / "sleep",
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "directory=$FAKE_TUNNEL_TEST_DIR\n"
        'printf "%s\\n" "$1" >> "$directory/sleep_delays"\n'
        'if [[ "$FAKE_BLOCK_BACKOFF" == 1 ]]; then\n'
        '  trap \'printf "stopped\\n" >> "$directory/sleep_stopped"; exit 0\' TERM\n'
        '  printf "ready\\n" > "$directory/sleep_ready"\n'
        "  while true; do /bin/sleep 0.02; done\n"
        "fi\n"
        "/bin/sleep 0.02\n",
    )
    environment = {
        "PATH": str(fakebin) + os.pathsep + os.defpath,
        "LC_ALL": "C",
        "FAKE_TUNNEL_TEST_DIR": str(directory),
        "FAKE_SSH_EXIT_CODES": exit_codes,
        "FAKE_BLOCK_BACKOFF": "1" if block_backoff else "0",
    }
    return TunnelFixture(
        config_path,
        directory / "attempts",
        directory / "sleep_delays",
        directory / "ssh_ready",
        directory / "sleep_ready",
        directory / "ssh_stopped",
        directory / "sleep_stopped",
        environment,
    )


@contextmanager
def _running_tunnel(fixture: TunnelFixture) -> Iterator[subprocess.Popen[str]]:
    """Always clean up the local test helper, including when assertions fail."""
    process = subprocess.Popen(
        [
            "bash",
            str(_SCRIPT),
            "--ssh-config",
            str(fixture.config_path),
            "--host",
            "bupt",
        ],
        env=fixture.environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=3)


def _wait_for_attempts(
    process: subprocess.Popen[str], fixture: TunnelFixture, count: int
) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if fixture.attempts_path.is_file():
            text = fixture.attempts_path.read_text(encoding="utf-8").strip()
            ready = (
                fixture.ssh_ready_path.read_text(encoding="utf-8").strip()
                if fixture.ssh_ready_path.is_file()
                else ""
            )
            if text and ready and int(text) >= count and int(ready) >= count:
                return
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            pytest.fail(f"tunnel terminated early: {stdout}\n{stderr}")
        time.sleep(0.01)
    pytest.fail(f"tunnel did not reach {count} fake connection attempts")


def _wait_for_backoff(process: subprocess.Popen[str], fixture: TunnelFixture) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if fixture.sleep_ready_path.is_file():
            return
        if process.poll() is not None:
            pytest.fail("tunnel terminated before entering backoff")
        time.sleep(0.01)
    pytest.fail("tunnel did not enter backoff")


@pytest.mark.parametrize("stop_signal", (signal.SIGTERM, signal.SIGINT))
def test_reconnects_two_failures_and_stops_owned_ssh(
    tmp_path: Path, stop_signal: signal.Signals
) -> None:
    fixture = _make_fixture(tmp_path, "255 255 block")
    with _running_tunnel(fixture) as process:
        _wait_for_attempts(process, fixture, 3)
        process.send_signal(stop_signal)
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 0
        assert fixture.ssh_stopped_path.read_text(encoding="utf-8") == "stopped\n"
        assert fixture.sleep_delays_path.read_text(encoding="utf-8").splitlines() == [
            "2",
            "4",
        ]
        time.sleep(0.05)
        assert fixture.attempts_path.read_text(encoding="utf-8").strip() == "3"
        assert "attempt=1" in stdout and "exit_code=255" in stdout
        assert "state=stopped" in stdout
        assert _PRIVATE_MARKER not in stdout + stderr
        assert str(fixture.config_path) not in stdout + stderr


def test_reconnects_successful_disconnect_and_caps_backoff(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, "0 255 255 255 255 255 255 block")
    with _running_tunnel(fixture) as process:
        _wait_for_attempts(process, fixture, 8)
        process.terminate()
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 0
        assert "attempt=1 exit_code=0" in stdout
        assert fixture.sleep_delays_path.read_text(encoding="utf-8").splitlines() == [
            "2",
            "2",
            "4",
            "8",
            "16",
            "30",
            "30",
        ]
        assert _PRIVATE_MARKER not in stdout + stderr


def test_sigterm_during_backoff_stops_without_reconnecting(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, "255 block", block_backoff=True)
    with _running_tunnel(fixture) as process:
        _wait_for_backoff(process, fixture)
        process.terminate()
        stdout, stderr = process.communicate(timeout=3)
        assert process.returncode == 0
        assert fixture.sleep_stopped_path.read_text(encoding="utf-8") == "stopped\n"
        assert fixture.attempts_path.read_text(encoding="utf-8").strip() == "1"
        assert "state=stopped" in stdout
        assert _PRIVATE_MARKER not in stdout + stderr


def test_rejects_missing_config_without_connecting(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path, "block")
    fixture.config_path.unlink()
    result = subprocess.run(
        ["bash", str(_SCRIPT), "--ssh-config", str(fixture.config_path)],
        env=fixture.environment,
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert result.returncode == 2
    assert not fixture.attempts_path.exists()
    assert str(fixture.config_path) not in result.stdout + result.stderr
