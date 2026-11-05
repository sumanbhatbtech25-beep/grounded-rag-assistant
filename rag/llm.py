"""Thin LLM client: local Ollama by default, or any OpenAI-compatible API."""
from __future__ import annotations

from dataclasses import dataclass

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

    def generate(self, system: str, user: str) -> LLMResult:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.backend == "ollama":
            r = requests.post(f"{config.OLLAMA_URL}/api/chat", json={
                "model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": self.temperature},
            }, timeout=600)
            r.raise_for_status()
            j = r.json()
            return LLMResult(j["message"]["content"], j.get("prompt_eval_count"), j.get("eval_count"))

        if self.backend == "openai":
            r = requests.post(f"{config.OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                              headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
                              json={"model": self.model, "messages": messages,
                                    "temperature": self.temperature}, timeout=600)
            r.raise_for_status()
            j = r.json()
            usage = j.get("usage", {})
            return LLMResult(j["choices"][0]["message"]["content"],
                             usage.get("prompt_tokens"), usage.get("completion_tokens"))

        raise ValueError(f"Unknown LLM backend: {self.backend}")


def ollama_models() -> list[str]:
    """Names of locally pulled Ollama models, or [] if Ollama isn't running."""
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=2)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []
