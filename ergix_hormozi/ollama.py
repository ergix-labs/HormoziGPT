from __future__ import annotations

from typing import Any, Iterable

import requests


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, chat_model: str, embedding_model: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                detail = f": {exc.response.text[:500]}"
            raise OllamaError(f"Ollama request failed at {path}{detail}") from exc

    def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            # Qwen 3.x can otherwise spend the entire response budget in the
            # separate `thinking` field and return an empty application answer.
            "think": False,
            "options": {"temperature": 0.25},
        }
        if json_mode:
            payload["format"] = "json"
        result = self._post("/api/chat", payload)
        try:
            content = str(result["message"]["content"]).strip()
        except (KeyError, TypeError) as exc:
            raise OllamaError("Ollama returned no chat message") from exc
        if not content:
            raise OllamaError("Ollama returned an empty chat message")
        return content

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        items = [str(text) for text in texts]
        if not items:
            return []
        result = self._post("/api/embed", {"model": self.embedding_model, "input": items})
        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(items):
            raise OllamaError("Ollama returned an invalid embedding batch")
        return [[float(value) for value in vector] for vector in embeddings]
