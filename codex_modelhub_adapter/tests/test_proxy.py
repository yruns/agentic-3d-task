from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapter.proxy import (
    AdapterSettings,
    build_upstream_request,
    health_payload,
)


class ResponsesOnlyProxyTest(unittest.TestCase):
    def test_chat_upstream_env_fails_closed(self) -> None:
        with patch.dict(
            "os.environ",
            {"AIDP_CODEX_PROXY_UPSTREAM_API": "chat_completions"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "Responses API only"):
                AdapterSettings.from_env()

    def test_legacy_chat_modelhub_url_fails_closed(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MODELHUB_URL": "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "Responses API only"):
                AdapterSettings.from_env()

    def test_legacy_mutation_env_does_not_change_request_body(self) -> None:
        request = {
            "model": "gpt-5.4-2026-03-05",
            "input": "hello",
            "max_output_tokens": 123,
            "store": False,
        }
        with patch.dict(
            "os.environ",
            {
                "AIDP_GPT_AK": "ak-1",
                "AIDP_CODEX_PROXY_RESPONSES_BODY_MUTATION_ENABLED": "true",
            },
            clear=True,
        ):
            settings = AdapterSettings.from_env()

        upstream = build_upstream_request(request, settings=settings)

        self.assertEqual(upstream.upstream_api, "responses")
        self.assertEqual(upstream.body, request)

    def test_chat_completion_toml_url_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            toml_path = Path(tmp_dir) / "upstreams.toml"
            toml_path.write_text(
                "\n".join(
                    (
                        "[[upstreams]]",
                        'alias = "legacy_chat"',
                        'url = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl"',
                        'model_name = "gpt-5.4-2026-03-05"',
                        'ak = "ak-1"',
                        "weight = 1",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {"AIDP_MODELHUB_UPSTREAMS_TOML": str(toml_path)},
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "Responses API"):
                    AdapterSettings.from_env()

    def test_health_reports_only_responses_route_fields(self) -> None:
        settings = AdapterSettings(modelhub_ak="ak-1")

        payload = health_payload(settings)

        self.assertEqual(payload["upstream_api"], "responses")
        self.assertEqual(payload["responses_path"], "/responses")
        self.assertNotIn("chat_completions_path", payload)
        self.assertNotIn("chat_completions_models", payload)
        self.assertNotIn("responses_body_mutation_enabled", payload)


class DefaultGatewayTest(unittest.TestCase):
    def test_linux_defaults_to_online_byteintl_gateway(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("adapter.proxy.platform.system", return_value="Linux"),
        ):
            settings = AdapterSettings.from_env()

        self.assertEqual(settings.upstream_env, "online")
        self.assertEqual(
            settings.upstream_base_url,
            "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online",
        )

    def test_macos_defaults_to_office_gateway(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("adapter.proxy.platform.system", return_value="Darwin"),
        ):
            settings = AdapterSettings.from_env()

        self.assertEqual(settings.upstream_env, "office")
        self.assertEqual(
            settings.upstream_base_url,
            "https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online",
        )

    def test_explicit_gateway_env_overrides_platform_default(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {"AIDP_CODEX_PROXY_UPSTREAM_ENV": "office"},
                clear=True,
            ),
            patch("adapter.proxy.platform.system", return_value="Linux"),
        ):
            settings = AdapterSettings.from_env()

        self.assertEqual(settings.upstream_env, "office")

    def test_invalid_gateway_env_fails_closed(self) -> None:
        with patch.dict(
            "os.environ",
            {"AIDP_CODEX_PROXY_UPSTREAM_ENV": "ofice"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "AIDP_CODEX_PROXY_UPSTREAM_ENV",
            ):
                AdapterSettings.from_env()


class PlaceholderCredentialTest(unittest.TestCase):
    def test_placeholder_single_ak_does_not_count_as_configured(self) -> None:
        with (
            patch.dict(
                "os.environ", {"AIDP_GPT_AK": "replace-with-modelhub-ak"}, clear=True
            ),
            patch("adapter.proxy.platform.system", return_value="Linux"),
        ):
            settings = AdapterSettings.from_env()

        self.assertFalse(health_payload(settings)["has_upstream_ak"])
        with self.assertRaisesRegex(RuntimeError, "AIDP_GPT_AK"):
            build_upstream_request({"model": "gpt-5.4-2026-03-05"}, settings=settings)

    def test_placeholder_toml_upstream_does_not_count_as_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            toml_path = Path(tmp_dir) / "upstreams.toml"
            toml_path.write_text(
                "\n".join(
                    (
                        "[[upstreams]]",
                        'alias = "placeholder"',
                        'url = "https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online"',
                        'model_name = "gpt-5.4-2026-03-05"',
                        'ak = "replace-with-modelhub-ak-1"',
                        "weight = 1",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            with (
                patch.dict(
                    "os.environ",
                    {"AIDP_MODELHUB_UPSTREAMS_TOML": str(toml_path)},
                    clear=True,
                ),
                patch("adapter.proxy.platform.system", return_value="Linux"),
            ):
                settings = AdapterSettings.from_env()

        self.assertFalse(health_payload(settings)["has_upstream_ak"])
        with self.assertRaisesRegex(RuntimeError, "AIDP_GPT_AK"):
            build_upstream_request({"model": "gpt-5.4-2026-03-05"}, settings=settings)


if __name__ == "__main__":
    unittest.main()
