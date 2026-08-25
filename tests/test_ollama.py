import unittest
from unittest.mock import patch

from ergix_hormozi.ollama import OllamaClient, OllamaError


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
        client = OllamaClient("http://localhost:11434", "qwen", "embed")
        self.assertEqual(client.chat([{"role": "user", "content": "Question"}]), "A direct answer.")
        self.assertIs(post.call_args.kwargs["json"]["think"], False)

    @patch("ergix_hormozi.ollama.requests.post")
    def test_rejects_empty_model_answer(self, post):
        post.return_value = FakeResponse({"message": {"content": "", "thinking": "unfinished"}})
        client = OllamaClient("http://localhost:11434", "qwen", "embed")
        with self.assertRaises(OllamaError):
            client.chat([{"role": "user", "content": "Question"}])


if __name__ == "__main__":
    unittest.main()
