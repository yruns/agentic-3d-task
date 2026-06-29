from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adapter.proxy import (
    AdapterSettings,
    build_upstream_request,
    health_payload,
    _requests_reasoning_summary,
    resolve_upstream_api,
)


def _auto_settings() -> AdapterSettings:
    return AdapterSettings(
        upstream_api="auto",
        chat_completions_models=("gpt-5.4*", "gpt-5.5*"),
    )


class RequestsReasoningSummaryTest(unittest.TestCase):
    def test_true_for_auto_concise_detailed(self):
        for value in ("auto", "concise", "detailed", "AUTO"):
            self.assertTrue(
                _requests_reasoning_summary({"reasoning": {"summary": value}}), value
            )

    def test_false_for_none_or_empty(self):
        for value in ("none", "", "  ", "NONE"):
            self.assertFalse(
                _requests_reasoning_summary({"reasoning": {"summary": value}}), value
            )

    def test_false_when_no_summary_field(self):
        self.assertFalse(_requests_reasoning_summary({}))
        self.assertFalse(_requests_reasoning_summary({"reasoning": {"effort": "high"}}))

    def test_false_when_reasoning_not_dict_or_body_not_dict(self):
        self.assertFalse(_requests_reasoning_summary({"reasoning": "high"}))
        self.assertFalse(_requests_reasoning_summary("not a dict"))
        self.assertFalse(_requests_reasoning_summary({"reasoning": {"summary": 1}}))


class ResolveUpstreamApiTest(unittest.TestCase):
    def test_chat_model_without_summary_uses_chat(self):
        body = {"model": "gpt-5.4-2026-03-05", "reasoning": {"effort": "medium"}}
        self.assertEqual(
            resolve_upstream_api(body, _auto_settings()), "chat_completions"
        )

    def test_chat_model_with_summary_routes_to_responses(self):
        # The chat upstream returns no reasoning content, so a summary request
        # must go to /responses.
        body = {
            "model": "gpt-5.4-2026-03-05",
            "reasoning": {"effort": "medium", "summary": "auto"},
        }
        self.assertEqual(resolve_upstream_api(body, _auto_settings()), "responses")

    def test_summary_none_stays_on_chat(self):
        body = {"model": "gpt-5.4-2026-03-05", "reasoning": {"summary": "none"}}
        self.assertEqual(
            resolve_upstream_api(body, _auto_settings()), "chat_completions"
        )

    def test_non_chat_model_uses_responses_by_default(self):
        self.assertEqual(
            resolve_upstream_api({"model": "o4-mini"}, _auto_settings()), "responses"
        )

    def test_explicit_chat_config_wins_over_summary(self):
        # An operator who pins upstream_api=chat_completions keeps that route even
        # for a summary request (documented limitation: summaries then disappear).
        settings = AdapterSettings(
            upstream_api="chat_completions",
            chat_completions_models=("gpt-5.4*",),
        )
        body = {"model": "gpt-5.4-2026-03-05", "reasoning": {"summary": "auto"}}
        self.assertEqual(resolve_upstream_api(body, settings), "chat_completions")


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
