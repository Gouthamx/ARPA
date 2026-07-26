"""Unified Ollama client for ARPA agents."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from arpa.core.config import ARPASettings, get_settings
from arpa.core.retry import request_with_retry
from arpa.models._structured import parse_into_schema, render_schema_instructions

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaError(RuntimeError):
    """Raised when the Ollama API returns an unrecoverable error."""


class OllamaClient:
    def __init__(self, settings: ARPASettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.ollama_base_url.rstrip("/")

    @property
    def general_model(self) -> str:
        return self.settings.ollama_general_model

    @property
    def code_model(self) -> str:
        return self.settings.ollama_code_model

    @property
    def _max_retries(self) -> int:
        # Reuse the Gemini retry budget knob so both backends share one setting.
        return max(1, getattr(self.settings, "gemini_max_retries", 3))

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to Ollama with retries on transient transport/5xx errors."""
        url = f"{self.base_url}{endpoint}"
        timeout = self.settings.ollama_timeout_s

        def _do() -> httpx.Response:
            with httpx.Client(timeout=timeout) as client:
                return client.post(url, json=payload)

        resp = request_with_retry(_do, max_retries=self._max_retries, label=f"Ollama {endpoint}")
        if resp is None:
            raise OllamaError(f"Ollama request to {endpoint} failed after retries")
        resp.raise_for_status()
        return resp.json()

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        format_json: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        model = model or self.settings.ollama_general_model
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        if format_json:
            payload["format"] = "json"

        data = self._post("/api/generate", payload)
        return data.get("response", "")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        format_json: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        model = model or self.settings.ollama_general_model
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if format_json:
            payload["format"] = "json"

        data = self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    @staticmethod
    def extract_json(text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, including fenced code blocks."""
        text = text.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence:
            text = fence.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def complete_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
    ) -> T:
        """Return a validated ``schema`` instance from a schema-constrained call."""
        system_prompt = render_schema_instructions(schema)
        if system:
            system_prompt = f"{system}\n\n{system_prompt}"
        raw = self.generate(
            prompt,
            model=model,
            system=system_prompt,
            temperature=temperature,
            format_json=True,
        )
        data = self.extract_json(raw)
        return parse_into_schema(data, schema)
