"""Confidence-scored fields shared across ARPA agents."""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, field_validator

T = TypeVar("T")


class ConfidenceLevel(str, Enum):
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    ASSUMED = "assumed"


class ConfidenceField(BaseModel, Generic[T]):
    """A value with explicit uncertainty provenance."""

    value: T | None = None  # Allow None for missing values
    confidence: ConfidenceLevel | str | None = None  # Allow None or string, will be converted
    source: str | list[str] | None = None
    evidence: str | None = None
    alternatives: list[T] | None = None
    warning: str | None = None
    replacement_logic: str | None = None

    @field_validator("evidence", mode="before")
    @classmethod
    def join_evidence_lists(cls, v: Any) -> Any:
        """Accept several supporting quotes, not just one.

        `source` was already declared `str | list[str]`, but `evidence` was
        `str` alone -- and a model quoting two sentences returns a list. One
        such field ("evidence: ['Dense Convolutional Network (DenseNet)']")
        failed validation and, because a ValidationError degrades the whole
        pass to an empty placeholder, took DenseNet's entire codegen plan with
        it.

        Joined rather than truncated to the first item: every quote the model
        offered is provenance worth keeping, and this field is read as prose.
        """
        if isinstance(v, (list, tuple)):
            parts = [str(item).strip() for item in v if item is not None and str(item).strip()]
            return "; ".join(parts) if parts else None
        return v

    def is_high_uncertainty(self) -> bool:
        conf = self.get_confidence()
        return conf == ConfidenceLevel.ASSUMED
    
    def get_value(self) -> T | None:
        """Get the value, returning None if not set."""
        return self.value
    
    def get_confidence(self) -> ConfidenceLevel:
        """Get the confidence level, defaulting to ASSUMED if None or invalid."""
        if self.confidence is None:
            return ConfidenceLevel.ASSUMED
        if isinstance(self.confidence, ConfidenceLevel):
            return self.confidence
        # Try to convert string to ConfidenceLevel
        try:
            return ConfidenceLevel(self.confidence)
        except (ValueError, AttributeError):
            return ConfidenceLevel.ASSUMED


class ConfidenceSummary(BaseModel):
    confirmed: int = 0
    inferred: int = 0
    assumed: int = 0

    def record(self, level: ConfidenceLevel) -> None:
        if level == ConfidenceLevel.CONFIRMED:
            self.confirmed += 1
        elif level == ConfidenceLevel.INFERRED:
            self.inferred += 1
        else:
            self.assumed += 1

    @property
    def total(self) -> int:
        return self.confirmed + self.inferred + self.assumed

    def as_dict(self) -> dict[str, int]:
        return {
            "confirmed": self.confirmed,
            "inferred": self.inferred,
            "assumed": self.assumed,
        }
