"""Schema helpers for flexible LLM output validation.

Provides type annotations that accept either plain values or ConfidenceField-wrapped
objects, automatically unwrapping to the plain value. This makes schemas tolerant of
LLM formatting inconsistencies.
"""

import json
from typing import Any, Optional

from pydantic import BeforeValidator
from typing_extensions import Annotated

from arpa.core.confidence import ConfidenceField


# Strings models emit to mean "absent" when a JSON null was wanted. Left as
# literal strings these fail every non-str field ("mixed_precision.value:
# Input should be a valid boolean, input_value='None'") and, because
# ExtractionAgent turns a ValidationError into an empty placeholder pass, one
# such token discards an entire pass worth of correctly-extracted fields.
_NULL_TOKENS = {"none", "null", "n/a", "na", "nil", "not specified", "unspecified", ""}


def _is_null_token(v: Any) -> bool:
    return isinstance(v, str) and v.strip().lower() in _NULL_TOKENS


def unwrap_confidence_field(v: Any) -> Any:
    """
    Accept either a plain value, a {"value": ..., "confidence": ...} dict
    (the shape LLM JSON responses come back in), or an actual ConfidenceField
    instance (the shape code constructing these schemas directly - e.g.
    tests and fixtures - tends to pass). Always returns the underlying plain
    value. Passes through None and already-plain values unchanged.

    Also normalizes the "absent" string tokens above to a real None, both at
    the top level and inside a ConfidenceField envelope.
    """
    if isinstance(v, ConfidenceField):
        v = v.value
    elif isinstance(v, dict) and "value" in v:
        v = v["value"]
    if _is_null_token(v):
        return None
    return v


def parse_stringified_list(v: Any) -> Any:
    """Turn a list that arrived as a string back into a list.

    Models emit shape fields as `"[3, 224, 224]"` about as often as
    `[3, 224, 224]` -- JSON-in-JSON, usually when the surrounding value is
    quoted for the confidence envelope. Left as a string it fails
    `list[int]` ("input_shape.value: Input should be a valid list,
    input_value='[3, 224, 224]'"), and since one bad field fails the whole
    pass, VGG lost its entire architecture extraction to nine of these.

    Handles both JSON (`"[3, 224, 224]"`) and bare comma-separated forms
    (`"3, 224, 224"`). Anything unparseable is returned untouched so the
    validation error still surfaces rather than being masked.
    """
    if not isinstance(v, str):
        return v
    text = v.strip()
    if not text:
        return v
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            inner = text[1:-1].strip()
            parsed = _split_scalars(inner) if inner else []
        return parsed if isinstance(parsed, list) else v
    # Bare "3, 224, 224" -- only when it really is a scalar sequence, so a
    # sentence containing a comma is never silently shredded into a list.
    if "," in text:
        parts = _split_scalars(text)
        if parts is not None:
            return parts
    return v


def _split_scalars(text: str) -> Optional[list]:
    """Split "3, 224, 224" into [3, 224, 224]; None if any part isn't numeric."""
    out: list = []
    for chunk in text.split(","):
        chunk = chunk.strip().strip("'\"")
        if not chunk:
            return None
        try:
            out.append(int(chunk))
        except ValueError:
            try:
                out.append(float(chunk))
            except ValueError:
                return None
    return out


def coerce_to_str(v: Any) -> Any:
    """Stringify a scalar that arrived where prose was expected.

    Descriptive fields (notes, test_time_augmentation) are declared `str`, but
    a model asked "did you use test-time augmentation?" reasonably answers
    `True` rather than a sentence. Rejecting that costs the whole pass; keeping
    it as "True" costs nothing and preserves the answer. Containers are left
    alone so a genuinely wrong shape still surfaces rather than being flattened
    into a useless repr.
    """
    if isinstance(v, bool) or isinstance(v, (int, float)):
        return str(v)
    return v


# Flexible type annotations that auto-unwrap ConfidenceField dicts
FlexibleStr = Annotated[
    Optional[str],
    BeforeValidator(coerce_to_str),
    BeforeValidator(unwrap_confidence_field),
]
FlexibleInt = Annotated[Optional[int], BeforeValidator(unwrap_confidence_field)]
FlexibleFloat = Annotated[Optional[float], BeforeValidator(unwrap_confidence_field)]
FlexibleBool = Annotated[Optional[bool], BeforeValidator(unwrap_confidence_field)]
FlexibleList = Annotated[Optional[list], BeforeValidator(unwrap_confidence_field)]
