"""Shared ARPA state and sub-schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from arpa.core.confidence import ConfidenceField, ConfidenceLevel, ConfidenceSummary


class DatasetDescription(BaseModel):
    """Dataset section extracted from a paper (input to Dataset Agent)."""

    name: str
    split_description: str | None = None
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None
    input_shape: list[int] | None = None  # e.g. [3, 32, 32]
    num_classes: int | None = None
    transform_description: str | None = None
    raw_context: str | None = None


class ExtractedPreprocessStep(BaseModel):
    """A single preprocessing/augmentation step as extracted from paper text.

    This is the Gemini-facing contract: parameters are captured as a free-form
    dict so any augmentation (named or not) can be represented without a
    hardcoded vocabulary. ``confidence`` reuses the shared provenance enum.
    """

    name: str = Field(description="Transform name, e.g. 'RandomCrop' or 'Normalize'")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Exact parameters stated in the paper, e.g. {'size': 32, 'padding': 4}",
    )
    confidence: ConfidenceLevel = ConfidenceLevel.ASSUMED
    evidence: str | None = Field(
        default=None,
        description="Verbatim phrase from the text supporting this step, or null",
    )


class ExtractedDatasetInfo(BaseModel):
    """Structured dataset/preprocessing facts extracted from a paper by the LLM.

    Contract between the semantic extraction layer (Gemini) and the Dataset
    Agent. Every field defaults to ``None``/empty so the model can faithfully
    signal "not stated in the text" instead of inventing a value.
    """

    name: str | None = Field(default=None, description="Canonical dataset name, or null")
    aliases: list[str] = Field(
        default_factory=list, description="Other names/abbreviations used for the dataset"
    )
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None
    num_classes: int | None = None
    input_shape: list[int] | None = Field(
        default=None, description="Channels-Height-Width, e.g. [3, 32, 32], or null"
    )
    split_description: str | None = None
    preprocess_steps: list[ExtractedPreprocessStep] = Field(default_factory=list)
    notes: str | None = Field(
        default=None, description="Any extra grounding detail; null if none"
    )

    def to_description(self, raw_context: str | None = None) -> "DatasetDescription":
        """Project the extracted facts onto the agent's DatasetDescription."""
        transform_desc = None
        if self.preprocess_steps:
            parts = []
            for step in self.preprocess_steps:
                if step.parameters:
                    params = ", ".join(f"{k}={v}" for k, v in step.parameters.items())
                    parts.append(f"{step.name}({params})")
                else:
                    parts.append(step.name)
            transform_desc = "; ".join(parts)
        return DatasetDescription(
            name=self.name or "unknown",
            split_description=self.split_description,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            transform_description=transform_desc,
            raw_context=(raw_context or "")[:8000] or None,
        )


class PreprocessStep(BaseModel):
    """One preprocessing step with confidence stamp."""

    name: str
    code_snippet: str
    confidence: ConfidenceLevel
    source: str | list[str] = ""
    evidence: str | None = None
    warning: str | None = None


class DatasetSpec(BaseModel):
    """Verified dataset specification produced by the Dataset Agent."""

    dataset_name: str
    registry_source: Literal["huggingface", "paperswithcode", "torchvision", "tfds"]
    registry_id: str
    loading_code: str
    preprocess_steps: list[PreprocessStep] = Field(default_factory=list)
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None
    input_shape: list[int] | None = None
    num_classes: int | None = None
    verified: bool = False
    verification_log: str | None = None
    resolution_notes: str | None = None


class MethodologySpec(BaseModel):
    """Partial methodology spec — dataset section used by Dataset Agent."""

    dataset_description: DatasetDescription | None = None
    # Full spec grows with Extraction Agent; only dataset fields needed here.


class DatasetAgentResult(BaseModel):
    """Output bundle from a Dataset Agent run."""

    spec: DatasetSpec | None = None
    loading_code: str | None = None
    verified: bool = False
    escalated: bool = False
    escalation_reason: str | None = None
    preprocess_confidence: ConfidenceSummary = Field(default_factory=ConfidenceSummary)
    resolution_attempts: list[str] = Field(default_factory=list)
    verify_attempts: int = 0


# Re-export for agents that need hyperparameter fields later
__all__ = [
    "ConfidenceField",
    "ConfidenceLevel",
    "ConfidenceSummary",
    "DatasetAgentResult",
    "DatasetDescription",
    "DatasetSpec",
    "ExtractedDatasetInfo",
    "ExtractedPreprocessStep",
    "MethodologySpec",
    "PreprocessStep",
]
