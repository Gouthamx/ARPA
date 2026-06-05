"""Shared helpers for schema-constrained ("structured") LLM completions.

Both the Gemini and Ollama clients expose ``complete_structured(prompt, schema)``
which returns a validated Pydantic model instance. We deliberately drive this
through plain JSON mode (``format_json=True``) plus prompt-injected schema
instructions, then validate locally with Pydantic, rather than relying on each
provider's native "response schema" feature. That keeps a single, provider-
agnostic code path and tolerates schemas with free-form ``dict`` fields (e.g. a
preprocessing step's ``parameters``) that strict response-schema validators on
the provider side reject.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_STRUCTURED_PREAMBLE = (
    "You must respond with a single JSON object that validates against the JSON "
    "Schema below. Output JSON only — no prose, no markdown fences.\n"
    "Rules:\n"
    "  * Use null for any field whose value is NOT explicitly stated in the "
    "provided text. Never invent, guess, or fill in defaults.\n"
    "  * Only include values you can ground in the text.\n"
    "  * Omit optional list items rather than fabricating them.\n\n"
    "JSON Schema:\n"
)


def render_schema_instructions(schema: type[BaseModel]) -> str:
    """Build the system-prompt fragment describing the required JSON schema."""
    js = json.dumps(schema.model_json_schema(), indent=2)
    return f"{_STRUCTURED_PREAMBLE}{js}"


def parse_into_schema(data: object, schema: type[T]) -> T:
    """Validate raw decoded JSON into ``schema``.

    Tolerates a top-level wrapper key (some models nest the object under a single
    key such as ``{"result": {...}}``).
    """
    if isinstance(data, dict):
        try:
            return schema.model_validate(data)
        except ValidationError:
            if len(data) == 1:
                inner = next(iter(data.values()))
                if isinstance(inner, dict):
                    return schema.model_validate(inner)
            raise
    raise ValueError(f"Expected a JSON object for {schema.__name__}, got {type(data).__name__}")
