"""Tests for confidence schema."""

from arpa.core.confidence import ConfidenceField, ConfidenceLevel, ConfidenceSummary


def test_confidence_summary_counts():
    summary = ConfidenceSummary()
    summary.record(ConfidenceLevel.CONFIRMED)
    summary.record(ConfidenceLevel.INFERRED)
    summary.record(ConfidenceLevel.ASSUMED)
    assert summary.total == 3
    assert summary.as_dict() == {"confirmed": 1, "inferred": 1, "assumed": 1}


def test_confidence_field_high_uncertainty():
    field = ConfidenceField(
        value=0.1,
        confidence=ConfidenceLevel.ASSUMED,
        warning="HIGH UNCERTAINTY",
    )
    assert field.is_high_uncertainty()
