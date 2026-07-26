"""Groq LLM client for ARPA pipeline.

Mirrors the existing OpenRouterClient interface (same method signatures)
so it can be dropped into arpa/models/__init__.py's factory function
with zero changes elsewhere in the codebase.

Why Groq instead of OpenRouter free tier:
- 30 requests/minute, up to 14,400 requests/day, no credit card required
- Runs Llama 3.3 70B / Qwen3 32B, both far more capable than local 7B models
- Free-tier model list is stable (unlike OpenRouter's rotating :free catalog)

Setup:
1. Sign up at https://console.groq.com (no credit card needed)
2. Create an API key in the console
3. Set GROQ_API_KEY in your .env
4. pip install openai   (Groq uses the OpenAI-compatible client)
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Dict, Optional, TypeVar

from pydantic import BaseModel

try:
    import httpx
    from openai import APIError, OpenAI, RateLimitError
except ImportError:
    raise ImportError(
        "Groq client requires 'openai' package. Install with: pip install openai"
    )

from arpa.core.config import ARPASettings, get_settings
from arpa.core.rate_limiter import GlobalRateLimiter
from arpa.models._structured import parse_into_schema, render_schema_instructions

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Safe defaults for Groq's free tier (30 RPM). We stay under that with
# margin so bursts from retries don't tip us over.
MIN_SECONDS_BETWEEN_CALLS = 2.5
MAX_JSON_REPAIR_ATTEMPTS = 1  # No retry - if JSON is malformed, fail fast
MAX_RATE_LIMIT_RETRIES = 5


class GroqError(RuntimeError):
    """Raised when Groq API returns an unrecoverable error."""


class GroqClient:
    """Drop-in replacement for OpenRouterClient, targeting Groq's free API.
    
    Usage:
        client = GroqClient(api_key="gsk_...", model="llama-3.3-70b-versatile")
        result = client.extract_json(system_prompt, user_prompt, schema_hint)
    """

    def __init__(self, settings: ARPASettings | None = None) -> None:
        self.settings = settings or get_settings()
        api_key = getattr(self.settings, "groq_api_key", None)
        if not api_key:
            raise GroqError(
                "No Groq API key configured. Set ARPA_GROQ_API_KEY."
            )
        
        self._model = getattr(self.settings, "groq_model", "llama-3.3-70b-versatile")
        self._temperature = 0.1
        self._timeout = 60
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0),
        )
        
        # Use global rate limiter instead of per-instance tracking
        self.rate_limiter = GlobalRateLimiter(min_interval=2.2)

    @property
    def general_model(self) -> str:
        return self._model

    @property
    def code_model(self) -> str:
        return self._model

    # ------------------------------------------------------------------ #
    # Core chat call with rate-limit backoff
    # ------------------------------------------------------------------ #

    def _call(
        self,
        messages: list[dict],
        json_mode: bool = True,
        max_tokens: int | None = None,
        frequency_penalty: float | None = None,
    ) -> str:
        """Single chat completion call with exponential backoff on 429s.

        Returns the raw text content of the model's response.
        """
        last_error: Optional[Exception] = None

        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            # Use global rate limiter - ensures ALL calls across all instances respect limit
            self.rate_limiter.wait()

            try:
                kwargs: Dict[str, Any] = dict(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                )

                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens

                if frequency_penalty is not None:
                    kwargs["frequency_penalty"] = frequency_penalty

                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
                
            except RateLimitError as e:
                last_error = e
                # Respect Retry-After if the SDK/exception exposes it,
                # otherwise fall back to exponential backoff with jitter.
                retry_after = getattr(e, "retry_after", None)
                if retry_after:
                    wait = float(retry_after)
                else:
                    base = min(2 ** attempt, 30)
                    wait = base + random.uniform(0, 0.5 * base)
                
                logger.warning(
                    "Groq rate limit hit (attempt %d/%d). Waiting %.1fs.",
                    attempt + 1, MAX_RATE_LIMIT_RETRIES, wait,
                )
                time.sleep(wait)
                
            except APIError as e:
                logger.error("Groq API error: %s", e)
                raise GroqError(f"Groq API error: {e}") from e
        
        raise GroqError(
            f"Groq rate limit retries exhausted after {MAX_RATE_LIMIT_RETRIES} attempts"
        ) from last_error

    # ------------------------------------------------------------------ #
    # JSON extraction with validation + repair loop
    # ------------------------------------------------------------------ #

    def extract_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask the model for structured JSON output, validating and repairing
        malformed responses automatically (same pattern as the OpenRouter
        client's retry-with-repair logic).

        Args:
            system_prompt: instructions, ideally including the exact schema.
            user_prompt: the extraction task / paper content.
            schema_hint: optional string reminder appended to repair prompts.

        Returns:
            Parsed JSON as a dict.

        Raises:
            ValueError if valid JSON can't be produced after all attempts.
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt(system_prompt)},
            {"role": "user", "content": user_prompt},
        ]

        raw_output = self._call(messages, json_mode=True)

        for attempt in range(MAX_JSON_REPAIR_ATTEMPTS):
            parsed = self._try_parse(raw_output)
            if parsed is not None:
                return parsed

            logger.warning(
                "Malformed JSON on attempt %d/%d. Requesting repair.",
                attempt + 1, MAX_JSON_REPAIR_ATTEMPTS,
            )

            repair_messages = messages + [
                {"role": "assistant", "content": raw_output},
                {
                    "role": "user",
                    "content": (
                        "That was not valid JSON. Return ONLY the corrected, "
                        "valid JSON object — no prose, no markdown code fences, "
                        "no explanation."
                        + (f"\n\nExpected schema:\n{schema_hint}" if schema_hint else "")
                    ),
                },
            ]
            raw_output = self._call(repair_messages, json_mode=True)

        raise ValueError(
            f"Failed to obtain valid JSON after {MAX_JSON_REPAIR_ATTEMPTS} repair attempts. "
            f"Last raw output: {raw_output[:500]}"
        )

    # ------------------------------------------------------------------ #
    # Public interface matching OpenRouterClient
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        format_json: bool = False,
        max_tokens: int | None = None,
        frequency_penalty: float | None = None,
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
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        format_json: bool = False,
        max_tokens: int | None = None,
        frequency_penalty: float | None = None,
    ) -> str:
        # Make a copy to avoid modifying original
        messages = [msg.copy() for msg in messages]

        # For JSON, add strict instructions
        if format_json:
            json_instruction = (
                "\n\nCRITICAL: You MUST respond with valid JSON only. "
                "Do not include any text before or after the JSON. "
                "Do not use markdown fences. Return raw JSON directly."
            )

            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += json_instruction
            else:
                messages.insert(0, {
                    "role": "system",
                    "content": "You are an assistant that returns valid JSON." + json_instruction
                })

        # Call with JSON validation and repair
        raw_output = self._call(
            messages,
            json_mode=format_json,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
        )
        
        # If JSON mode, validate and repair if needed
        if format_json:
            for attempt in range(MAX_JSON_REPAIR_ATTEMPTS):
                parsed = self._try_parse(raw_output)
                if parsed is not None:
                    return json.dumps(parsed)
                
                logger.warning(
                    "Malformed JSON on attempt %d/%d. Requesting repair.",
                    attempt + 1, MAX_JSON_REPAIR_ATTEMPTS,
                )
                
                repair_messages = messages + [
                    {"role": "assistant", "content": raw_output},
                    {
                        "role": "user",
                        "content": (
                            "That was not valid JSON. Return ONLY the corrected, "
                            "valid JSON object — no prose, no markdown code fences, "
                            "no explanation."
                        ),
                    },
                ]
                raw_output = self._call(repair_messages, json_mode=True)
            
            raise GroqError(
                f"Failed to obtain valid JSON after {MAX_JSON_REPAIR_ATTEMPTS} repair attempts. "
                f"Last raw output: {raw_output[:500]}"
            )
        
        return raw_output

    def chat_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> T:
        """Chat with structured output."""
        schema_instructions = render_schema_instructions(schema)
        
        messages = [msg.copy() for msg in messages]
        
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += f"\n\n{schema_instructions}"
        else:
            messages.insert(0, {"role": "system", "content": schema_instructions})
        
        raw_response = self.chat(
            messages,
            model=model,
            temperature=temperature,
            format_json=True,
        )
        
        # Parse JSON string to dict before passing to schema validator
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as e:
            raise GroqError(f"Failed to parse JSON: {e}\nContent: {raw_response[:500]}")
        
        return parse_into_schema(data, schema)

    def complete_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
    ) -> T:
        """Generate structured output."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat_structured(
            messages,
            schema,
            model=model,
            temperature=temperature,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_system_prompt(base_prompt: str) -> str:
        return (
            base_prompt.strip()
            + "\n\nRespond with ONLY a valid JSON object. "
            "Do not include markdown code fences, explanations, or any text "
            "outside the JSON object."
        )

    @staticmethod
    def _try_parse(text: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse text as JSON, stripping common wrapper artifacts."""
        if not text:
            return None
        
        cleaned = text.strip()
        
        # Strip markdown code fences if the model added them despite instructions
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def extract_json(text: str) -> Dict[str, Any]:
        """Extract JSON from text (for compatibility with other clients)."""
        parsed = GroqClient._try_parse(text)
        if parsed is not None:
            return parsed
        raise ValueError(f"No valid JSON in: {text[:200]}")
