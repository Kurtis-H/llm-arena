"""The system under test: either a model with a system prompt, or a real chat endpoint."""

from __future__ import annotations

import httpx

from llm_arena.providers import Provider, ProviderError


class ModelTarget:
    def __init__(self, provider: Provider, model: str, system_prompt: str, temperature: float = 0.7):
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature

    @property
    def name(self) -> str:
        return f"model:{self.model}"

    def send(self, message: str) -> str:
        return self.provider.chat(
            self.model,
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": message}],
            temperature=self.temperature,
        )


class HttpTarget:
    """Any JSON chat API, e.g. vuln-shop's POST /api/chat {"message": ...} -> {"reply": ...}."""

    def __init__(self, url: str, message_field: str = "message", reply_field: str = "reply", timeout: float = 120):
        self.url = url
        self.message_field = message_field
        self.reply_field = reply_field
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"http:{self.url}"

    def send(self, message: str) -> str:
        try:
            resp = httpx.post(self.url, json={self.message_field: message}, timeout=self.timeout)
            resp.raise_for_status()
            return str(resp.json()[self.reply_field])
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise ProviderError(f"{self.url}: {exc}") from exc
