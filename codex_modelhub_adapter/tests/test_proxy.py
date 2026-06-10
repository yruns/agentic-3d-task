from __future__ import annotations

import unittest

from adapter.proxy import (
    AdapterSettings,
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
        self.assertEqual(resolve_upstream_api(body, _auto_settings()), "chat_completions")

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
        self.assertEqual(resolve_upstream_api(body, _auto_settings()), "chat_completions")

    def test_non_chat_model_uses_responses_by_default(self):
        self.assertEqual(resolve_upstream_api({"model": "o4-mini"}, _auto_settings()), "responses")

    def test_explicit_chat_config_wins_over_summary(self):
        # An operator who pins upstream_api=chat_completions keeps that route even
        # for a summary request (documented limitation: summaries then disappear).
        settings = AdapterSettings(
            upstream_api="chat_completions",
            chat_completions_models=("gpt-5.4*",),
        )
        body = {"model": "gpt-5.4-2026-03-05", "reasoning": {"summary": "auto"}}
        self.assertEqual(resolve_upstream_api(body, settings), "chat_completions")


if __name__ == "__main__":
    unittest.main()
