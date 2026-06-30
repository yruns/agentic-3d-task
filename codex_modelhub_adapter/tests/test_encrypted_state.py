from __future__ import annotations

import json
import unittest

from adapter.encrypted_state import (
    is_invalid_encrypted_content,
    sanitize_encrypted_state,
)


class EncryptedStateTest(unittest.TestCase):
    def test_sanitize_encrypted_state_removes_opaque_codex_state(self) -> None:
        request = {
            "model": "gpt-5.4-2026-03-05",
            "previous_response_id": "resp_old",
            "input": [
                {"type": "message", "role": "user", "content": "continue"},
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "gAAA",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "visible"}],
                    "encrypted_content": "gBBB",
                },
            ],
        }

        sanitized, stats = sanitize_encrypted_state(request)

        self.assertNotIn("previous_response_id", sanitized)
        self.assertNotIn("encrypted_content", json.dumps(sanitized, ensure_ascii=False))
        self.assertEqual(stats.removed_encrypted_content_count, 2)
        self.assertEqual(stats.dropped_empty_reasoning_count, 1)
        self.assertTrue(stats.removed_previous_response_id)
        self.assertEqual(len(sanitized["input"]), 2)

    def test_invalid_encrypted_content_detector_reads_error_code(self) -> None:
        body = json.dumps(
            {
                "error": {
                    "code": "invalid_encrypted_content",
                    "message": "Encrypted content is invalid",
                }
            }
        ).encode("utf-8")

        self.assertTrue(is_invalid_encrypted_content(body))


if __name__ == "__main__":
    unittest.main()
