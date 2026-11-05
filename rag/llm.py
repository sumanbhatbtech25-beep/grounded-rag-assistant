"""Thin LLM client: local Ollama by default, or any OpenAI-compatible API.
Supports both one-shot `generate` and token `stream`ing."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import requests

import config


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class LLM:
    def __init__(self, backend: str = config.LLM_BACKEND, model: str = config.LLM_MODEL,
                 temperature: float = 0.0):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.last_usage: tuple[int | None, int | None] = (None, None)

    @staticmethod
    def _messages(system: str, user: str) -> list[dict]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def generate(self, system: str, user: str) -> LLMResult:
        text = "".join(self.stream(system, user))
        return LLMResult(text, *self.last_usage)

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield the answer piece by piece. Token counts land in `self.last_usage`."""
        self.last_usage = (None, None)
        messages = self._messages(system, user)

        if self.backend == "ollama":
            with requests.post(f"{config.OLLAMA_URL}/api/chat", json={
                "model": self.model, "messages": messages, "stream": True,
                "options": {"temperature": self.temperature},
            }, stream=True, timeout=600) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    j = json.loads(line)
                    if j.get("error"):
                        raise RuntimeError(j["error"])
                    piece = j.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if j.get("done"):
                        self.last_usage = (j.get("prompt_eval_count"), j.get("eval_count"))
            return

        if self.backend == "openai":
            with requests.post(f"{config.OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                               headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                               json={"model": self.model, "messages": messages, "stream": True,
                                     "temperature": self.temperature,
                                     "stream_options": {"include_usage": True}},
                               stream=True, timeout=600) as r:
                r.raise_for_status()
                for raw in r.iter_lines():
                    line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    j = json.loads(data)
                    if j.get("usage"):
                        self.last_usage = (j["usage"].get("prompt_tokens"), j["usage"].get("completion_tokens"))
                    for ch in j.get("choices", []):
                        piece = (ch.get("delta") or {}).get("content")
                        if piece:
                            yield piece
            return

        raise ValueError(f"Unknown LLM backend: {self.backend}")


def ollama_models() -> list[str]:
    """Names of locally pulled Ollama models, or [] if Ollama isn't running."""
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=2)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []
