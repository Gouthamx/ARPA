"""Gemini (Google Generative Language API) client for ARPA agents.

Drop-in alternative to OllamaClient. Uses the REST API via httpx so no extra
SDK dependency is required. Exposes the same surface the agents rely on:
  - chat(messages, model=..., temperature=..., format_json=...)
  - generate(prompt, system=..., ...)
  - general_model / code_model properties
  - extract_json(text)  [static]
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from arpa.core.config import ARPASettings, get_settings
from arpa.core.retry import RETRYABLE_STATUS_CODES, compute_backoff, parse_retry_after
from arpa.models._structured import parse_into_schema, render_schema_instructions

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GeminiError(RuntimeError):
    """Raised when the Gemini API returns an unrecoverable error."""


class GeminiClient:
    """Minimal Gemini REST client mirroring the OllamaClient interface."""

    def __init__(self, settings: ARPASettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.gemini_base_url.rstrip("/")
        self.api_key = self.settings.gemini_api_key
        if not self.api_key:
            raise GeminiError(
                "No Gemini API key configured. Set ARPA_GEMINI_API_KEY or pass it "
                "via ARPASettings(gemini_api_key=...)."
            )

    @property
    def general_model(self) -> str:
        return self.settings.gemini_general_model

    @property
    def code_model(self) -> str:
        return self.settings.gemini_code_model

    # ------------------------------------------------------------------ #
    # Public API (mirrors OllamaClient)
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.2,
        format_json: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages,
            model=model,
            temperature=temperature,
            format_json=format_json,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        format_json: bool = False,
    ) -> str:
        model = model or self.general_model
        system_instruction, contents = self._convert_messages(messages)

        generation_config: dict[str, Any] = {"temperature": temperature}
        if format_json:
            generation_config["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        return self._post_generate(model, payload)

    def complete_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
    ) -> T:
        """Return a validated ``schema`` instance from a schema-constrained call.

        Drives JSON mode with schema instructions injected into the system
        prompt, then validates the decoded JSON locally with Pydantic.
        """
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

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _convert_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Translate OpenAI-style messages to Gemini contents + systemInstruction."""
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                system_parts.append(text)
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:  # user (and anything else) maps to user
                contents.append({"role": "user", "parts": [{"text": text}]})
        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    def _post_generate(self, model: str, payload: dict[str, Any]) -> str:
        url = f"{self.base_url}/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        timeout = self.settings.gemini_timeout_s
        max_retries = self.settings.gemini_max_retries

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 429:
                    body = resp.text[:300]
                    logger.error(
                        "Gemini rate limited (429) on %s — not retrying: %s",
                        model,
                        body,
                    )
                    raise GeminiError(
                        f"Gemini quota/rate-limit exceeded for model '{model}': {body}"
                    )
                # Other transient server statuses (overloaded / unavailable / gateway
                # timeout) — retry with backoff before giving up.
                if resp.status_code in RETRYABLE_STATUS_CODES:
                    delay = compute_backoff(attempt, retry_after=parse_retry_after(resp))
                    logger.warning(
                        "Gemini server error (%d) on %s; retry %d/%d in %.1fs",
                        resp.status_code,
                        model,
                        attempt,
                        max_retries,
                        delay,
                    )
                    if attempt < max_retries:
                        time.sleep(delay)
                        continue
                    raise GeminiError(
                        f"Gemini unavailable (HTTP {resp.status_code}) for model "
                        f"'{model}' after {max_retries} attempts: {resp.text[:300]}"
                    )
                resp.raise_for_status()
                return self._extract_response_text(resp.json())
            except httpx.HTTPStatusError as exc:
                # Non-retryable HTTP error (4xx other than 429).
                body = exc.response.text[:300] if exc.response is not None else ""
                logger.error("Gemini HTTP error on %s: %s %s", model, exc, body)
                raise GeminiError(f"Gemini request failed: {exc} {body}") from exc
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "Gemini transport error on %s (attempt %d/%d): %s",
                    model,
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(compute_backoff(attempt, cap=10.0))
                    continue
        raise GeminiError(f"Gemini request failed after retries: {last_exc}")

    @staticmethod
    def _extract_response_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates", [])
        if not candidates:
            block = data.get("promptFeedback", {}).get("blockReason")
            if block:
                raise GeminiError(f"Gemini blocked the prompt: {block}")
            return ""
        candidate = candidates[0]
        finish = candidate.get("finishReason")
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if "text" in p)
        if not text and finish and finish not in ("STOP", "MAX_TOKENS"):
            raise GeminiError(f"Gemini returned no text (finishReason={finish})")
        return text

    @staticmethod
    def extract_json(text: str) -> dict[str, Any]:
        """Parse JSON from model output, tolerating fenced code blocks."""
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
