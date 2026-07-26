"""Schema helpers for flexible LLM output validation.

Provides type annotations that accept either plain values or ConfidenceField-wrapped
objects, automatically unwrapping to the plain value. This makes schemas tolerant of
LLM formatting inconsistencies.
"""

from typing import Any, Optional

from pydantic import BeforeValidator
from typing_extensions import Annotated

from arpa.core.confidence import ConfidenceField


def unwrap_confidence_field(v: Any) -> Any:
    """
    Accept either a plain value, a {"value": ..., "confidence": ...} dict
    (the shape LLM JSON responses come back in), or an actual ConfidenceField
    instance (the shape code constructing these schemas directly - e.g.
    tests and fixtures - tends to pass). Always returns the underlying plain
    value. Passes through None and already-plain values unchanged.
    """
    if isinstance(v, ConfidenceField):
        return v.value
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


# Flexible type annotations that auto-unwrap ConfidenceField dicts
FlexibleStr = Annotated[Optional[str], BeforeValidator(unwrap_confidence_field)]
FlexibleInt = Annotated[Optional[int], BeforeValidator(unwrap_confidence_field)]
FlexibleFloat = Annotated[Optional[float], BeforeValidator(unwrap_confidence_field)]
FlexibleBool = Annotated[Optional[bool], BeforeValidator(unwrap_confidence_field)]
FlexibleList = Annotated[Optional[list], BeforeValidator(unwrap_confidence_field)]
