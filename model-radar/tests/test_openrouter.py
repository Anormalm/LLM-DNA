from __future__ import annotations

import unittest

from model_radar.openrouter import HTTPResult, OpenRouterClient, parse_completion


class OpenRouterTests(unittest.TestCase):
    def test_model_pagination(self) -> None:
        calls = []

        def fake(method, url, headers, body, timeout):
            calls.append(url)
            if len(calls) == 1:
                return HTTPResult(200, {}, {"data": [{"id": "a"}], "links": {"next": "https://next"}})
            return HTTPResult(200, {}, {"data": [{"id": "b"}], "links": {"next": None}})

        client = OpenRouterClient(request_fn=fake)
        self.assertEqual([item["id"] for item in client.list_models()], ["a", "b"])

    def test_completion_parser_preserves_operational_metadata(self) -> None:
        parsed = parse_completion(
            {
                "id": "gen-1",
                "model": "stealth/ox-alpha",
                "provider": "Stealth",
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 12},
            }
        )
        self.assertEqual(parsed["response"], "answer")
        self.assertEqual(parsed["generation_id"], "gen-1")
        self.assertEqual(parsed["provider"], "Stealth")


if __name__ == "__main__":
    unittest.main()

