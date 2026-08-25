import unittest
from unittest.mock import patch

from ergix_hormozi.ollama import ModelClient, ModelClientError


class FakeResponse:
    def __init__(self, body):
        self.body = body
        self.text = str(body)

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class OllamaTests(unittest.TestCase):
    @patch("ergix_hormozi.ollama.requests.post")
    def test_disables_thinking_for_application_answers(self, post):
        post.return_value = FakeResponse({"message": {"content": "A direct answer."}})
        client = ModelClient("http://localhost:11434", "qwen", "embed")
        self.assertEqual(client.chat([{"role": "user", "content": "Question"}]), "A direct answer.")
        self.assertIs(post.call_args.kwargs["json"]["think"], False)

    @patch("ergix_hormozi.ollama.requests.post")
    def test_rejects_empty_model_answer(self, post):
        post.return_value = FakeResponse({"message": {"content": "", "thinking": "unfinished"}})
        client = ModelClient("http://localhost:11434", "qwen", "embed")
        with self.assertRaises(ModelClientError):
            client.chat([{"role": "user", "content": "Question"}])

    @patch("ergix_hormozi.ollama.requests.post")
    def test_calls_moonshot_kimi_k3_with_auth_and_reasoning(self, post):
        post.return_value = FakeResponse({"choices": [{"message": {"content": "A K3 answer."}}]})
        client = ModelClient(
            "http://localhost:11434",
            "kimi-k3",
            "embed",
            chat_provider="moonshot",
            chat_api_key="test-key",
            reasoning_effort="high",
        )

        answer = client.chat([{"role": "user", "content": "Question"}], json_mode=True)

        self.assertEqual(answer, "A K3 answer.")
        self.assertEqual(post.call_args.args[0], "https://api.moonshot.ai/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "kimi-k3")
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "high")
        self.assertEqual(post.call_args.kwargs["json"]["temperature"], 1)
        self.assertNotIn("response_format", post.call_args.kwargs["json"])

    def test_moonshot_key_error_is_actionable(self):
        client = ModelClient(
            "http://localhost:11434",
            "kimi-k3",
            "embed",
            chat_provider="moonshot",
        )
        with self.assertRaisesRegex(ModelClientError, "MOONSHOT_API_KEY"):
            client.chat([{"role": "user", "content": "Question"}])


if __name__ == "__main__":
    unittest.main()
