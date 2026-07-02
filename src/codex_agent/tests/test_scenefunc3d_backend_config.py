"""Tests for SceneFunc3D sidecar backend configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from codex_agent.scenefunc3d.backends.config import (
    SceneFunc3dBackendSettings,
    load_backend_settings,
)
from codex_agent.scenefunc3d.tools.models import ToolInputError


def _write_valid_backend_config(
    config_path: Path,
    *,
    image_root: Path,
    output_root: Path,
) -> None:
    config_path.write_text(
        f"""
molmo_url = "http://127.0.0.1:8711"
sam_url = "http://localhost:8712"
request_timeout_seconds = 3.5
allowed_image_roots = ["{image_root}"]
allowed_output_roots = ["{output_root}"]
""",
        encoding="utf-8",
    )


def test_load_backend_settings_from_toml(tmp_path: Path) -> None:
    allowed_root = tmp_path / "runs"
    allowed_root.mkdir()
    config_path = tmp_path / "scenefunc3d_backends.toml"
    _write_valid_backend_config(
        config_path,
        image_root=tmp_path,
        output_root=allowed_root,
    )

    settings = load_backend_settings(config_path)

    assert settings.molmo_url == "http://127.0.0.1:8711"
    assert settings.sam_url == "http://localhost:8712"
    assert settings.request_timeout_seconds == 3.5
    assert settings.allowed_image_roots == (tmp_path,)
    assert settings.allowed_output_roots == (allowed_root,)


def test_load_backend_settings_derives_urls_from_sidecar_base_url(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "scenefunc3d_backends.toml"
    config_path.write_text(
        f"""
sidecar_base_url = "https://workspace-proxy-candy-maliva-tce.tiktok-row.org/pws-token"
request_timeout_seconds = 3.5
allowed_image_roots = ["{tmp_path}"]
allowed_output_roots = ["{tmp_path}"]
""",
        encoding="utf-8",
    )

    settings = load_backend_settings(config_path)

    assert (
        settings.molmo_url
        == "https://workspace-proxy-candy-maliva-tce.tiktok-row.org/pws-token/molmo"
    )
    assert (
        settings.sam_url
        == "https://workspace-proxy-candy-maliva-tce.tiktok-row.org/pws-token/sam"
    )


def test_load_backend_settings_reads_request_headers_file(tmp_path: Path) -> None:
    headers_path = tmp_path / "sidecar_headers.toml"
    headers_path.write_text(
        """
[headers]
Cookie = "arnold_proxy_sc=token"
X-Test-Header = "expected-value"
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "scenefunc3d_backends.toml"
    config_path.write_text(
        f"""
sidecar_base_url = "https://workspace-proxy-candy-maliva-tce.tiktok-row.org/pws-token"
request_headers_path = "{headers_path.name}"
request_timeout_seconds = 3.5
allowed_image_roots = ["{tmp_path}"]
allowed_output_roots = ["{tmp_path}"]
""",
        encoding="utf-8",
    )

    settings = load_backend_settings(config_path)

    assert {header.name: header.value for header in settings.request_headers} == {
        "Cookie": "arnold_proxy_sc=token",
        "X-Test-Header": "expected-value",
    }


def test_load_backend_settings_rejects_unsafe_request_header_value(
    tmp_path: Path,
) -> None:
    headers_path = tmp_path / "sidecar_headers.toml"
    headers_path.write_text(
        """
[headers]
Cookie = "valid\\nInjected: bad"
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "scenefunc3d_backends.toml"
    config_path.write_text(
        f"""
sidecar_base_url = "https://workspace-proxy-candy-maliva-tce.tiktok-row.org/pws-token"
request_headers_path = "{headers_path}"
request_timeout_seconds = 3.5
allowed_image_roots = ["{tmp_path}"]
allowed_output_roots = ["{tmp_path}"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ToolInputError, match="unsafe request header"):
        load_backend_settings(config_path)


def test_load_backend_settings_missing_file_is_recoverable(tmp_path: Path) -> None:
    with pytest.raises(ToolInputError, match="backend config is missing"):
        load_backend_settings(tmp_path / "missing.toml")


def test_load_backend_settings_invalid_toml_is_recoverable(tmp_path: Path) -> None:
    config_path = tmp_path / "scenefunc3d_backends.toml"
    config_path.write_text("molmo_url = [not-valid", encoding="utf-8")

    with pytest.raises(ToolInputError, match="invalid TOML"):
        load_backend_settings(config_path)


def test_load_backend_settings_rejects_empty_allowed_roots(tmp_path: Path) -> None:
    config_path = tmp_path / "scenefunc3d_backends.toml"
    config_path.write_text(
        f"""
molmo_url = "http://127.0.0.1:8711"
sam_url = "http://127.0.0.1:8712"
request_timeout_seconds = 3.5
allowed_image_roots = []
allowed_output_roots = ["{tmp_path}"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ToolInputError, match="allowed_image_roots"):
        load_backend_settings(config_path)


def test_load_backend_settings_rejects_infinite_timeout(tmp_path: Path) -> None:
    config_path = tmp_path / "scenefunc3d_backends.toml"
    config_path.write_text(
        f"""
molmo_url = "http://127.0.0.1:8711"
sam_url = "http://127.0.0.1:8712"
request_timeout_seconds = inf
allowed_image_roots = ["{tmp_path}"]
allowed_output_roots = ["{tmp_path}"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ToolInputError, match="finite positive"):
        load_backend_settings(config_path)


def test_load_backend_settings_redacts_extra_secret_values(tmp_path: Path) -> None:
    config_path = tmp_path / "scenefunc3d_backends.toml"
    config_path.write_text(
        f"""
molmo_url = "http://127.0.0.1:8711"
sam_url = "http://127.0.0.1:8712"
request_timeout_seconds = 3.5
allowed_image_roots = ["{tmp_path}"]
allowed_output_roots = ["{tmp_path}"]
api_key = "secret-token-value"
""",
        encoding="utf-8",
    )

    with pytest.raises(ToolInputError) as exc_info:
        load_backend_settings(config_path)

    message = str(exc_info.value)
    assert "api_key" in message
    assert "secret-token-value" not in message


def test_backend_settings_rejects_non_local_url(tmp_path: Path) -> None:
    with pytest.raises(ToolInputError, match="127.0.0.1"):
        SceneFunc3dBackendSettings(
            molmo_url="http://10.1.2.3:8711",
            sam_url="http://127.0.0.1:8712",
            request_timeout_seconds=3.0,
            allowed_image_roots=(tmp_path,),
            allowed_output_roots=(tmp_path,),
        )


def test_backend_settings_redacts_url_credentials(tmp_path: Path) -> None:
    with pytest.raises(ToolInputError) as exc_info:
        SceneFunc3dBackendSettings(
            molmo_url="http://user:secret-password@127.0.0.1:8711",
            sam_url="http://127.0.0.1:8712",
            request_timeout_seconds=3.0,
            allowed_image_roots=(tmp_path,),
            allowed_output_roots=(tmp_path,),
        )

    message = str(exc_info.value)
    assert "<credentials>" in message
    assert "secret-password" not in message


def test_backend_settings_redacts_url_query_tokens(tmp_path: Path) -> None:
    with pytest.raises(ToolInputError) as exc_info:
        SceneFunc3dBackendSettings(
            molmo_url="http://127.0.0.1:8711?token=secret-token-value",
            sam_url="http://127.0.0.1:8712",
            request_timeout_seconds=3.0,
            allowed_image_roots=(tmp_path,),
            allowed_output_roots=(tmp_path,),
        )

    message = str(exc_info.value)
    assert "secret-token-value" not in message


def test_backend_settings_wraps_path_resolution_errors(tmp_path: Path) -> None:
    loop_path = tmp_path / "loop"
    loop_path.symlink_to(loop_path)

    with pytest.raises(ToolInputError, match="could not be resolved"):
        SceneFunc3dBackendSettings(
            molmo_url="http://127.0.0.1:8711",
            sam_url="http://127.0.0.1:8712",
            request_timeout_seconds=3.0,
            allowed_image_roots=(loop_path,),
            allowed_output_roots=(tmp_path,),
        )


def test_backend_settings_rejects_url_without_port(tmp_path: Path) -> None:
    with pytest.raises(ToolInputError, match="explicit port"):
        SceneFunc3dBackendSettings(
            molmo_url="http://127.0.0.1",
            sam_url="http://127.0.0.1:8712",
            request_timeout_seconds=3.0,
            allowed_image_roots=(tmp_path,),
            allowed_output_roots=(tmp_path,),
        )


def test_backend_settings_accepts_local_base_paths(tmp_path: Path) -> None:
    settings = SceneFunc3dBackendSettings(
        molmo_url="http://127.0.0.1:9001/s/export-token/molmo",
        sam_url="http://localhost:9001/s/export-token/sam",
        request_timeout_seconds=3.0,
        allowed_image_roots=(tmp_path,),
        allowed_output_roots=(tmp_path,),
    )

    assert settings.molmo_url == "http://127.0.0.1:9001/s/export-token/molmo"
    assert settings.sam_url == "http://localhost:9001/s/export-token/sam"


def test_backend_settings_rejects_non_positive_timeout(tmp_path: Path) -> None:
    with pytest.raises(ToolInputError, match="request_timeout_seconds"):
        SceneFunc3dBackendSettings(
            molmo_url="http://127.0.0.1:8711",
            sam_url="http://127.0.0.1:8712",
            request_timeout_seconds=0.0,
            allowed_image_roots=(tmp_path,),
            allowed_output_roots=(tmp_path,),
        )


def test_backend_settings_rejects_infinite_timeout(tmp_path: Path) -> None:
    with pytest.raises(ToolInputError, match="finite positive"):
        SceneFunc3dBackendSettings(
            molmo_url="http://127.0.0.1:8711",
            sam_url="http://127.0.0.1:8712",
            request_timeout_seconds=float("inf"),
            allowed_image_roots=(tmp_path,),
            allowed_output_roots=(tmp_path,),
        )
