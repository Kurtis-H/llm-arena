"""Model providers: where chat completions come from.

- OllamaProvider: real local models on your own GPU (free).
- FakeProvider: deterministic stand-ins for tests and demos (see fakes.py).
- RecordingProvider: wraps another provider to record responses to a "cassette" file,
  or replay them later. CI replays real model output without needing a GPU.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import httpx

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class Provider(Protocol):
    def chat(self, model: str, messages: list[Message], temperature: float = 0.0) -> str: ...


class ProviderError(Exception):
    pass


class OllamaProvider:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, model: str, messages: list[Message], temperature: float = 0.0) -> str:
        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise ProviderError(f"ollama/{model}: {exc}") from exc


def _key(model: str, messages: list[Message], temperature: float) -> str:
    blob = json.dumps({"m": model, "msgs": messages, "t": temperature}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class RecordingProvider:
    """Record or replay calls. The same request asked 3 times (for consistency checks)
    is stored as 3 separate responses and replayed in the same order."""

    def __init__(self, path: str | Path, mode: str, inner: Provider | None = None):
        if mode not in ("record", "replay"):
            raise ValueError("mode must be 'record' or 'replay'")
        if mode == "record" and inner is None:
            raise ValueError("recording needs a real provider to record from")
        self.path = Path(path)
        self.mode = mode
        self.inner = inner
        self.store: dict[str, list[str]] = {}
        if self.path.exists():
            self.store = json.loads(self.path.read_text(encoding="utf-8"))
        elif mode == "replay":
            raise ProviderError(f"cassette not found: {self.path}")
        self.seen: dict[str, int] = defaultdict(int)

    def chat(self, model: str, messages: list[Message], temperature: float = 0.0) -> str:
        key = _key(model, messages, temperature)
        n = self.seen[key]
        self.seen[key] += 1
        recorded = self.store.get(key, [])
        if self.mode == "replay":
            if n >= len(recorded):
                raise ProviderError(f"no recording for this request (call #{n + 1} to {model}); re-record the cassette")
            return recorded[n]
        assert self.inner is not None  # guaranteed by __init__ in record mode
        reply = self.inner.chat(model, messages, temperature)
        self.store.setdefault(key, [])
        if n < len(self.store[key]):
            self.store[key][n] = reply
        else:
            self.store[key].append(reply)
        return reply

    def save(self) -> None:
        if self.mode == "record":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.store, indent=1, sort_keys=True), encoding="utf-8")
