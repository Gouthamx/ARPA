"""NVIDIA NIM API client for ARPA agents.

NVIDIA NIM provides free access to 100+ models through an OpenAI-compatible API.
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
from arpa.core.rate_limiter import GlobalRateLimiter
from arpa.core.api_logger import get_api_logger
from arpa.models._structured import parse_into_schema, render_schema_instructions

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class NvidiaError(RuntimeError):
    """Raised when NVIDIA NIM API returns an unrecoverable error."""


class NvidiaClient:
    """NVIDIA NIM API client with JSON repair logic."""

    def __init__(self, settings: ARPASettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = getattr(self.settings, "nvidia_base_url", "https://integrate.api.nvidia.com/v1")
        self.api_key = getattr(self.settings, "nvidia_api_key", None)
        if not self.api_key:
            raise NvidiaError(
                "No NVIDIA API key configured. Set ARPA_NVIDIA_API_KEY."
            )
        
        # Use global rate limiter
        self.rate_limiter = GlobalRateLimiter(min_interval=2.5)
        
        # Get API logger
        self.api_logger = get_api_logger("nvidia_api_debug.log")
        
        # Call counter for unique IDs
        self.call_counter = 0

    # Read the real settings fields so ARPA_NVIDIA_GENERAL_MODEL /
    # ARPA_NVIDIA_CODE_MODEL in .env are actually honoured. These previously
    # looked up a "nvidia_model" field that does not exist on ARPASettings, so
    # getattr silently fell through to the hardcoded default and every .env
    # model override was ignored -- the config said one thing and the client
    # did another. `nvidia_model` is still accepted as a fallback so an older
    # .env that sets it keeps working.
    @property
    def general_model(self) -> str:
        return (
            getattr(self.settings, "nvidia_general_model", None)
            or getattr(self.settings, "nvidia_model", None)
            or "meta/llama-3.3-70b-instruct"
        )

    @property
    def code_model(self) -> str:
        return (
            getattr(self.settings, "nvidia_code_model", None)
            or getattr(self.settings, "nvidia_model", None)
            or "meta/llama-3.3-70b-instruct"
        )

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
            # 4096 is enough for the small structured-extraction payloads
            # (schema JSON is a few KB at most). Code generation calls a
            # full model.py/train.py file back as a JSON string and needs
            # much more headroom, or the JSON gets truncated mid-string and
            # fails to parse -- callers that expect long output should pass
            # an explicit max_tokens override.
            "max_tokens": max_tokens if max_tokens is not None else 4096,
        }
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        
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
            
            payload["messages"] = messages
        
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # JSON extraction with minimal retry (just 1 attempt, no repair)
        max_retries = 1  # No retry - if JSON is malformed, fail fast
        last_content = None
        last_error: str | None = None

        for attempt in range(max_retries):
            started_at = time.monotonic()
            try:
                # Use global rate limiter
                self.rate_limiter.wait()

                # Log the request
                logger.info("=" * 80)
                logger.info(f"NVIDIA API CALL - Attempt {attempt + 1}/{max_retries}")
                logger.info(f"Model: {payload['model']}")
                logger.info(f"Temperature: {payload['temperature']}")
                logger.info(f"Format JSON: {format_json}")
                logger.info("-" * 80)
                logger.info("MESSAGES:")
                for i, msg in enumerate(payload['messages']):
                    logger.info(f"  [{i}] Role: {msg['role']}")
                    content_preview = msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content']
                    logger.info(f"      Content: {content_preview}")
                logger.info("-" * 80)
                
                timeout = httpx.Timeout(
                    connect=10.0,
                    read=self.settings.nvidia_timeout_s,
                    write=10.0,
                    pool=10.0
                )
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    
                    data = resp.json()
                    
                    # Log the response
                    logger.info("RESPONSE:")
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"]["content"]
                        last_content = content
                        
                        logger.info(f"  Status: SUCCESS")
                        logger.info(f"  Content Length: {len(content)} characters")
                        logger.info(f"  Content Preview: {content[:300]}...")
                        logger.info("=" * 80)
                        
                        # Validate and extract JSON if requested
                        if format_json:
                            try:
                                # Try to extract JSON from the content
                                parsed = self.extract_json(content)
                                
                                logger.info("JSON EXTRACTION:")
                                logger.info(f"  Status: SUCCESS")
                                logger.info(f"  Parsed Keys: {list(parsed.keys())}")
                                logger.info(f"  Parsed JSON: {json.dumps(parsed, indent=2)[:500]}...")
                                logger.info("=" * 80)
                                
                                # Return as JSON string
                                return json.dumps(parsed)
                            except (json.JSONDecodeError, ValueError) as e:
                                logger.error("JSON EXTRACTION FAILED:")
                                logger.error(f"  Error: {str(e)}")
                                logger.error(f"  Raw Content: {content[:500]}")
                                logger.error("=" * 80)
                                
                                # Try repair on next attempt
                                if attempt < max_retries - 1:
                                    logger.warning(f"Will retry with repair prompt...")
                                    payload["messages"] = [{
                                        "role": "system",
                                        "content": "Fix the following to be valid JSON. Return ONLY the corrected JSON."
                                    }, {
                                        "role": "user",
                                        "content": f"Fix to valid JSON:\n\n{content}"
                                    }]
                                    continue
                                else:
                                    raise NvidiaError(
                                        f"Invalid JSON after {max_retries} attempts.\n"
                                        f"Content: {content[:300]}"
                                    )
                        else:
                            return content
                    else:
                        logger.error(f"RESPONSE ERROR: Unexpected format - {data}")
                        logger.error("=" * 80)
                        raise NvidiaError(f"Unexpected response: {data}")
                        
            except httpx.HTTPStatusError as exc:
                body = exc.response.text[:500] if exc.response is not None else ""
                status = exc.response.status_code if exc.response is not None else "?"
                elapsed = time.monotonic() - started_at
                last_error = f"HTTP {status} after {elapsed:.1f}s: {body or exc}"
                logger.error(
                    f"NVIDIA API FAILED | model={payload['model']} | status={status} "
                    f"| elapsed={elapsed:.1f}s | body={body}"
                )

                # Handle rate limiting with exponential backoff
                if exc.response is not None and status == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        # Exponential backoff: 10s, 20s, 40s
                        wait_time = 10 * (2 ** attempt)

                    # Only sleep if a further attempt actually remains.
                    # max_retries is 1, so `continue` here used to fall straight
                    # out of the loop into a generic "Failed after 1 attempts.
                    # Last: None" -- burning the wait and discarding the fact
                    # that we were rate limited at all.
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Rate limited (429). Waiting {wait_time}s before retry "
                            f"{attempt + 1}/{max_retries}"
                        )
                        time.sleep(wait_time)
                        continue
                    raise NvidiaError(
                        f"Rate limited (429) and no retries remain: {body}"
                    ) from exc

                # Don't retry on 404/401 -- bad key or unknown model won't heal.
                if status in (404, 401, 403):
                    raise NvidiaError(f"Request failed: {exc}\n{body}") from exc

                if attempt < max_retries - 1:
                    continue
                raise NvidiaError(f"Request failed: {exc}\n{body}") from exc

            except Exception as exc:
                elapsed = time.monotonic() - started_at
                last_error = f"{type(exc).__name__} after {elapsed:.1f}s: {exc}"
                # Name the exception type explicitly: a bare str() on
                # httpx.ReadTimeout is empty, which is precisely how a wall of
                # timeouts previously showed up as a blank, unattributable error.
                logger.error(
                    f"NVIDIA API FAILED | model={payload['model']} "
                    f"| {type(exc).__name__} | elapsed={elapsed:.1f}s | {exc}"
                )
                if attempt < max_retries - 1:
                    continue
                raise NvidiaError(f"Request failed: {last_error}") from exc

        # Reached only when every attempt used `continue`. Report the last real
        # error rather than a content-free "Last: None".
        raise NvidiaError(
            f"Failed after {max_retries} attempt(s). Last error: {last_error or 'unknown'}"
        )

    def chat_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> T:
        """Chat with structured output."""
        # Generate unique call ID
        self.call_counter += 1
        call_id = f"call_{self.call_counter}_{schema.__name__}"
        
        # Log the request
        self.api_logger.log_request(
            call_id=call_id,
            schema_name=schema.__name__,
            messages=messages,
            model=model or self.general_model
        )
        
        schema_instructions = render_schema_instructions(schema)
        
        messages = [msg.copy() for msg in messages]
        
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += f"\n\n{schema_instructions}"
        else:
            messages.insert(0, {"role": "system", "content": schema_instructions})
        
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║ STRUCTURED EXTRACTION REQUEST" + " " * 48 + "║")
        logger.info("╠" + "=" * 78 + "╣")
        logger.info(f"║ Call ID: {call_id:<69} ║")
        logger.info(f"║ Target Schema: {schema.__name__:<62} ║")
        logger.info(f"║ Model: {(model or self.general_model):<69} ║")
        logger.info("╚" + "=" * 78 + "╝")
        
        import time
        start_time = time.time()
        
        raw_response = self.chat(
            messages,
            model=model,
            temperature=temperature,
            format_json=True,
        )
        
        response_time = time.time() - start_time
        
        # Parse JSON string to dict before passing to schema validator
        try:
            data = json.loads(raw_response)
            
            # Log the response
            self.api_logger.log_response(
                call_id=call_id,
                response_text=raw_response,
                parsed_data=data,
                response_time_s=response_time
            )
            
            logger.info("╔" + "=" * 78 + "╗")
            logger.info("║ SCHEMA VALIDATION" + " " * 59 + "║")
            logger.info("╠" + "=" * 78 + "╣")
            logger.info(f"║ Validating against: {schema.__name__:<59} ║")
            logger.info(f"║ Data keys: {str(list(data.keys())[:5]):<66} ║")
            logger.info("╚" + "=" * 78 + "╝")
            
            result = parse_into_schema(data, schema)
            
            logger.info("✓ Schema validation PASSED")
            logger.info("=" * 80)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error("╔" + "=" * 78 + "╗")
            logger.error("║ JSON PARSE ERROR" + " " * 61 + "║")
            logger.error("╠" + "=" * 78 + "╣")
            logger.error(f"║ Error: {str(e)[:70]:<70} ║")
            logger.error(f"║ Content preview: {raw_response[:60]:<60} ║")
            logger.error("╚" + "=" * 78 + "╝")
            raise NvidiaError(f"Failed to parse JSON: {e}\nContent: {raw_response[:500]}")
        except Exception as e:
            # Log validation error with detailed analysis
            try:
                self.api_logger.log_validation_error(
                    call_id=call_id,
                    schema=schema,
                    error=e,
                    actual_data=data
                )
            except:
                pass  # Don't fail if logging fails
            
            logger.error("╔" + "=" * 78 + "╗")
            logger.error("║ VALIDATION ERROR" + " " * 61 + "║")
            logger.error("╠" + "=" * 78 + "╣")
            logger.error(f"║ Call ID: {call_id:<69} ║")
            logger.error(f"║ Schema: {schema.__name__:<68} ║")
            logger.error(f"║ Error: {str(e)[:70]:<70} ║")
            logger.error("╚" + "=" * 78 + "╝")
            logger.error(f"Full error: {e}")
            logger.error(f"Data that failed validation: {json.dumps(data, indent=2)[:500]}")
            logger.error("=" * 80)
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
