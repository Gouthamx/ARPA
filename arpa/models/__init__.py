"""LLM backends for ARPA.

Both clients expose the same surface used by the agents:
  - chat(messages, model=None, temperature=..., format_json=...)
  - generate(prompt, system=None, ...)
  - general_model / code_model properties
  - extract_json(text)  [static]

Use `get_llm_client()` to pick a backend from settings (ARPA_LLM_BACKEND).
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from arpa.core.config import ARPASettings, get_settings
from arpa.models.gemini_client import GeminiClient, GeminiError
from arpa.models.ollama_client import OllamaClient

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    """Structural type implemented by both Ollama and Gemini clients."""

    @property
    def general_model(self) -> str: ...

    @property
    def code_model(self) -> str: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = ...,
        temperature: float = ...,
        format_json: bool = ...,
    ) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = ...,
        system: str | None = ...,
        temperature: float = ...,
        format_json: bool = ...,
    ) -> str: ...

    def complete_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        model: str | None = ...,
        system: str | None = ...,
        temperature: float = ...,
    ) -> T: ...


def get_llm_client(
    settings: ARPASettings | None = None,
    *,
    backend: str | None = None,
) -> LLMClient:
    """Instantiate the configured LLM backend.

    backend: "ollama" or "gemini". Falls back to settings.llm_backend.
    """
    settings = settings or get_settings()
    backend = (backend or settings.llm_backend or "ollama").lower()
    if backend == "gemini":
        return GeminiClient(settings)
    if backend == "ollama":
        return OllamaClient(settings)
    raise ValueError(f"Unknown LLM backend: {backend!r} (expected 'ollama' or 'gemini')")


__all__ = [
    "GeminiClient",
    "GeminiError",
    "LLMClient",
    "OllamaClient",
    "get_llm_client",
]
