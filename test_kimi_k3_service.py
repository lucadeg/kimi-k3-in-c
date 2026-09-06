from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import kimi_k3_service as service


TOOL_CALLS = [{
    "id": "call_1",
    "type": "function",
    "function": {"name": "read", "arguments": "{\"path\":\"package.json\"}"},
}]


class TestKimiHydraBridge(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(service.app)

    @patch("kimi_k3_service.route_heavy_work_to_hydra")
    def test_non_stream_response_preserves_tool_calls(self, route):
        route.return_value = {
            "content": "",
            "message": {"role": "assistant", "content": "", "tool_calls": TOOL_CALLS},
            "finish_reason": "tool_calls",
            "upstream": {"model": "upstream", "router_metadata": {"backend": "gemini"}},
        }
        response = self.client.post("/v1/chat/completions", json={
            "model": "kimi-k3-moe",
            "messages": [{"role": "user", "content": "read the file"}],
            "tools": [{"type": "function", "function": {"name": "read"}}],
            "stream": False,
        })
        self.assertEqual(response.status_code, 200)
        choice = response.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"], TOOL_CALLS)
        self.assertEqual(route.call_args.args[3][0]["function"]["name"], "read")

    @patch("kimi_k3_service.route_heavy_work_to_hydra")
    def test_stream_response_emits_tool_call_delta(self, route):
        route.return_value = {
            "content": "",
            "message": {"role": "assistant", "content": "", "tool_calls": TOOL_CALLS},
            "finish_reason": "tool_calls",
            "upstream": {"model": "upstream", "router_metadata": {"backend": "gemini"}},
        }
        response = self.client.post("/v1/chat/completions", json={
            "model": "kimi-k3-moe",
            "messages": [{"role": "user", "content": "read the file"}],
            "tools": [{"type": "function", "function": {"name": "read"}}],
            "stream": True,
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('"tool_calls"', response.text)
        self.assertIn('"finish_reason": "tool_calls"', response.text)
        self.assertTrue(response.text.rstrip().endswith("data: [DONE]"))


if __name__ == "__main__":
    unittest.main()
