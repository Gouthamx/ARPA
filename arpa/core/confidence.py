"""Confidence-scored fields shared across ARPA agents."""

from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ConfidenceLevel(str, Enum):
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    ASSUMED = "assumed"


class ConfidenceField(BaseModel, Generic[T]):
    """A value with explicit uncertainty provenance."""

    value: T
    confidence: ConfidenceLevel
    source: str | list[str] = ""
    evidence: str | None = None
    alternatives: list[T] | None = None
    warning: str | None = None
    replacement_logic: str | None = None

    def is_high_uncertainty(self) -> bool:
        return self.confidence == ConfidenceLevel.ASSUMED


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
