"""OpenRouter API client for ARPA agents with JSON repair logic.

OpenRouter provides access to many LLMs through a unified OpenAI-compatible API.
This client includes retry-with-repair for robust JSON extraction.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from arpa.core.config import ARPASettings, get_settings
from arpa.models._structured import parse_into_schema, render_schema_instructions

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter API returns an unrecoverable error."""


class OpenRouterClient:
    """OpenRouter API client with JSON repair logic."""

    def __init__(self, settings: ARPASettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = self.settings.openrouter_api_key
        if not self.api_key:
            raise OpenRouterError(
                "No OpenRouter API key configured. Set ARPA_OPENROUTER_API_KEY."
            )

    @property
    def general_model(self) -> str:
        return self.settings.openrouter_general_model

    @property
    def code_model(self) -> str:
        return self.settings.openrouter_code_model

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
        model = model or self.general_model

        # Make a copy to avoid modifying original
        messages = [msg.copy() for msg in messages]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        
        # For JSON, add strict instructions and enable JSON mode
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
            
            # Enable JSON mode
            payload["response_format"] = {"type": "json_object"}
            payload["messages"] = messages
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/arpa-project",
        }
        
        # JSON extraction with minimal retry (just 1 attempt, no repair)
        max_retries = 1  # No retry - if JSON is malformed, fail fast
        last_content = None
        
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0)) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    
                    data = resp.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"]
                        last_content = content
                        
                        # Validate JSON if requested
                        if format_json:
                            try:
                                json.loads(content)
                                return content
                            except json.JSONDecodeError as e:
                                logger.warning(
                                    f"Attempt {attempt + 1}/{max_retries}: Invalid JSON. "
                                    f"Error: {str(e)[:100]}"
                                )
                                
                                # Try repair on next attempt
                                if attempt < max_retries - 1:
                                    payload["messages"] = [{
                                        "role": "system",
                                        "content": "Fix the following to be valid JSON. Return ONLY the corrected JSON."
                                    }, {
                                        "role": "user",
                                        "content": f"Fix to valid JSON:\n\n{content}"
                                    }]
                                    continue
                                else:
                                    raise OpenRouterError(
                                        f"Invalid JSON after {max_retries} attempts.\n"
                                        f"Content: {content[:300]}"
                                    )
                        else:
                            return content
                    else:
                        raise OpenRouterError(f"Unexpected response: {data}")
                        
            except httpx.HTTPStatusError as exc:
                body = exc.response.text[:500] if exc.response else ""
                
                # Handle rate limiting with exponential backoff
                if exc.response and exc.response.status_code == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        # Exponential backoff: 10s, 20s, 40s
                        wait_time = 10 * (2 ** attempt)
                    
                    logger.warning(
                        f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}"
                    )
                    time.sleep(wait_time)
                    continue
                
                logger.error(f"HTTP error: {exc} {body}")
                
                # Don't retry on 404/401
                if exc.response.status_code in (404, 401):
                    raise OpenRouterError(f"Request failed: {exc}\n{body}") from exc
                
                if attempt < max_retries - 1:
                    continue
                raise OpenRouterError(f"Request failed: {exc}") from exc
                
            except Exception as exc:
                logger.error(f"Error: {exc}")
                if attempt < max_retries - 1:
                    continue
                raise OpenRouterError(f"Request failed: {exc}") from exc
        
        raise OpenRouterError(f"Failed after {max_retries} attempts. Last: {last_content[:200] if last_content else 'None'}")

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
        
        return parse_into_schema(raw_response, schema)

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

    @staticmethod
    def extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from text."""
        text = text.strip()
        
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # From markdown
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
        
        # Find JSON object
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"No valid JSON in: {text[:200]}")
