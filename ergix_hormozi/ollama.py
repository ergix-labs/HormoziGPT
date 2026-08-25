from __future__ import annotations

from typing import Any, Iterable

import requests


class ModelClientError(RuntimeError):
    pass


class ModelClient:
    """Moonshot chat plus local Ollama embeddings.

    The Ollama chat path remains available for development and offline fallback.
    """

    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embedding_model: str,
        timeout: int = 300,
        *,
        chat_provider: str = "ollama",
        chat_base_url: str = "https://api.moonshot.ai/v1",
        chat_api_key: str = "",
        reasoning_effort: str = "high",
    ):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.chat_provider = chat_provider.strip().lower()
        self.chat_base_url = chat_base_url.rstrip("/")
        self.chat_api_key = chat_api_key
        self.reasoning_effort = reasoning_effort

    def _post(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        service: str,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                detail = f": {exc.response.text[:500]}"
            raise ModelClientError(f"{service} request failed{detail}") from exc

    def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        if self.chat_provider == "moonshot":
            return self._moonshot_chat(messages, json_mode=json_mode)
        if self.chat_provider != "ollama":
            raise ModelClientError(f"Unsupported chat provider: {self.chat_provider}")

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
        result = self._post(
            f"{self.base_url}/api/chat",
            payload,
            service="Ollama chat",
        )
        try:
            content = str(result["message"]["content"]).strip()
        except (KeyError, TypeError) as exc:
            raise ModelClientError("Ollama returned no chat message") from exc
        if not content:
            raise ModelClientError("Ollama returned an empty chat message")
        return content

    def _moonshot_chat(self, messages: list[dict[str, str]], json_mode: bool) -> str:
        if not self.chat_api_key:
            raise ModelClientError(
                "Moonshot API key is missing. Pull `kimi` from Infisical in the Ergix harness "
                "or set MOONSHOT_API_KEY in .env, then restart the app."
            )
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            # Kimi K3 currently accepts only temperature 1.
            "temperature": 1,
            "reasoning_effort": self.reasoning_effort,
        }
        # Moonshot structured-output support varies by model. Daily-motion
        # prompts explicitly demand JSON and the application validates it.
        result = self._post(
            f"{self.chat_base_url}/chat/completions",
            payload,
            service="Moonshot chat",
            headers={
                "Authorization": f"Bearer {self.chat_api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            content = str(result["choices"][0]["message"]["content"]).strip()
        except (IndexError, KeyError, TypeError) as exc:
            raise ModelClientError("Moonshot returned no chat message") from exc
        if not content:
            raise ModelClientError("Moonshot returned an empty chat message")
        return content

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        items = [str(text) for text in texts]
        if not items:
            return []
        result = self._post(
            f"{self.base_url}/api/embed",
            {"model": self.embedding_model, "input": items},
            service="Ollama embeddings",
        )
        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(items):
            raise ModelClientError("Ollama returned an invalid embedding batch")
        return [[float(value) for value in vector] for vector in embeddings]


# Backwards-compatible names for integrations built against the original fork.
OllamaClient = ModelClient
OllamaError = ModelClientError
