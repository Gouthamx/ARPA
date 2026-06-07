"""Shared ARPA state and sub-schemas."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

from arpa.core.confidence import ConfidenceField, ConfidenceLevel, ConfidenceSummary


class DatasetDescription(BaseModel):
    """Dataset section extracted from a paper, used by the Dataset Agent."""

    name: str
    split_description: str | None = None
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None
    input_shape: list[int] | None = None  # e.g. [3, 32, 32]
    num_classes: int | None = None
    transform_description: str | None = None
    raw_context: str | None = None


class CodegenMissingDetail(BaseModel):
    """A missing or ambiguous detail that can block faithful code generation.
    
    RAG Integration:
        When a detail is missing from the paper but can be filled from external
        sources (knowledge base, domain standards), use:
        - proposed_default: The suggested value from RAG
        - default_source: Attribution for the RAG value
        
        The CodeGen agent can then decide to use, question, or ignore the default.
    """

    field: str
    reason: str
    severity: Literal["critical", "important", "optional"] = "important"
    suggested_resolution: str | None = None
    evidence: str | None = None
    proposed_default: Any | None = Field(
        default=None,
        description="RAG-supplied default value from knowledge base or domain standards",
    )
    default_source: str | None = Field(
        default=None,
        description="Source attribution for proposed_default (e.g. 'ARPA KB: He et al. 2015')",
    )


class ExtractedPreprocessStep(BaseModel):
    """A single preprocessing/augmentation step as extracted from paper text."""

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
    """Structured dataset/preprocessing facts extracted from a paper by the LLM."""

    name: str | None = Field(default=None, description="Canonical dataset name, or null")
    aliases: list[str] = Field(
        default_factory=list,
        description="Other names/abbreviations used for the dataset",
    )
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None
    num_classes: int | None = None
    input_shape: list[int] | None = Field(
        default=None,
        description="Channels-Height-Width, e.g. [3, 32, 32], or null",
    )
    split_description: str | None = None
    preprocess_steps: list[ExtractedPreprocessStep] = Field(default_factory=list)
    notes: str | None = Field(
        default=None,
        description="Any extra grounding detail; null if none",
    )

    def to_description(self, raw_context: str | None = None) -> DatasetDescription:
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


class ModelComponentSpec(BaseModel):
    """A model component or layer-level implementation fact."""

    name: str
    kind: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    source: str | list[str] = ""
    evidence: str | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.ASSUMED


class ArchitectureSpec(BaseModel):
    """Model architecture details needed by the CodeGen Agent."""

    model_name: ConfidenceField[str] | None = None
    architecture_family: ConfidenceField[str] | None = None
    backbone: ConfidenceField[str] | None = None
    layers: ConfidenceField[list[str]] | None = None
    hidden_dim: ConfidenceField[int] | None = None
    activation: ConfidenceField[str] | None = None
    normalization: ConfidenceField[str] | None = None
    dropout: ConfidenceField[float] | None = None
    input_shape: ConfidenceField[list[int]] | None = None
    output_dim: ConfidenceField[int] | None = None
    components: list[ModelComponentSpec] = Field(default_factory=list)
    forward_pass: ConfidenceField[str] | None = None
    initialization: ConfidenceField[str] | None = None
    pretrained_weights: ConfidenceField[str] | None = None
    notes: str | None = None


class TrainingSpec(BaseModel):
    """Training hyperparameters and objective details."""

    optimizer: ConfidenceField[str] | None = None
    learning_rate: ConfidenceField[float] | None = None
    batch_size: ConfidenceField[int] | None = None
    epochs: ConfidenceField[int] | None = None
    weight_decay: ConfidenceField[float] | None = None
    momentum: ConfidenceField[float] | None = None
    scheduler: ConfidenceField[str] | None = None
    loss_function: ConfidenceField[str] | None = None
    gradient_clip: ConfidenceField[float] | None = None
    seed: ConfidenceField[int] | None = None
    early_stopping: ConfidenceField[str] | None = None
    optimizer_parameters: dict[str, ConfidenceField[Any]] = Field(default_factory=dict)
    scheduler_parameters: dict[str, ConfidenceField[Any]] = Field(default_factory=dict)
    regularization: list[ConfidenceField[str]] = Field(default_factory=list)
    augmentation_policy: list[ConfidenceField[str]] = Field(default_factory=list)
    mixed_precision: ConfidenceField[bool] | None = None
    gradient_accumulation_steps: ConfidenceField[int] | None = None
    checkpoint_selection: ConfidenceField[str] | None = None
    notes: str | None = None


class EvaluationSpec(BaseModel):
    """Evaluation protocol and paper-reported target metric."""

    task_type: ConfidenceField[str] | None = None
    metric_name: ConfidenceField[str] | None = None
    reported_metric: ConfidenceField[float] | None = None
    metric_direction: ConfidenceField[Literal["higher_is_better", "lower_is_better"]] | None = None
    eval_split: ConfidenceField[str] | None = None
    protocol: ConfidenceField[str] | None = None
    secondary_metrics: list[ConfidenceField[str]] = Field(default_factory=list)
    test_time_augmentation: ConfidenceField[str] | None = None
    aggregation: ConfidenceField[str] | None = None
    notes: str | None = None


class BenchmarkExperimentSpec(BaseModel):
    """One reported experiment/baseline result from a paper benchmark table."""

    model_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    dataset_metric: float | None = None
    comparison_metric: float | None = None
    metric_name: str = "test_accuracy"
    dataset_name: str | None = None
    comparison_dataset_name: str | None = None
    repeats: int | None = None
    aggregation: str | None = None
    source: str | list[str] = ""
    evidence: str | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.CONFIRMED


class ImplementationSpec(BaseModel):
    """Implementation/environment facts that affect generated runnable code."""

    framework: ConfidenceField[str] | None = None
    language: ConfidenceField[str] | None = None
    dependencies: list[ConfidenceField[str]] = Field(default_factory=list)
    hardware: ConfidenceField[str] | None = None
    training_time: ConfidenceField[str] | None = None
    official_code_url: ConfidenceField[str] | None = None
    entrypoint_hint: ConfidenceField[str] | None = None
    config_format: ConfidenceField[str] | None = None
    reproducibility_notes: list[ConfidenceField[str]] = Field(default_factory=list)
    notes: str | None = None


class CodegenFileSpec(BaseModel):
    """A file the CodeGen Agent should produce."""

    path: str
    purpose: str
    required_symbols: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class CodegenPlanSpec(BaseModel):
    """Code generation plan inferred from the methodology."""

    target_task: ConfidenceField[str] | None = None
    framework: ConfidenceField[str] | None = None
    entrypoint: ConfidenceField[str] | None = None
    model_class_name: ConfidenceField[str] | None = None
    train_function_name: ConfidenceField[str] | None = None
    eval_function_name: ConfidenceField[str] | None = None
    files: list[CodegenFileSpec] = Field(default_factory=list)
    required_runtime_checks: list[str] = Field(default_factory=list)
    unsupported_reasons: list[str] = Field(default_factory=list)
    notes: str | None = None


class DatasetTaskPass(BaseModel):
    """Pass 1: dataset, task, and reported metric."""

    dataset_description: DatasetDescription | None = None
    evaluation: EvaluationSpec | None = None
    assumptions_needed: list[CodegenMissingDetail] = Field(default_factory=list)
    notes: str | None = None


class ArchitecturePass(BaseModel):
    """Pass 2: architecture and model internals."""

    architecture: ArchitectureSpec | None = None
    assumptions_needed: list[CodegenMissingDetail] = Field(default_factory=list)
    notes: str | None = None


class TrainingEvalPass(BaseModel):
    """Pass 3: training and evaluation hyperparameters."""

    training: TrainingSpec | None = None
    evaluation: EvaluationSpec | None = None
    assumptions_needed: list[CodegenMissingDetail] = Field(default_factory=list)
    notes: str | None = None


class ImplementationPlanPass(BaseModel):
    """Pass 4: implementation, codegen plan, and remaining missing details."""

    implementation: ImplementationSpec | None = None
    codegen_plan: CodegenPlanSpec | None = None
    assumptions_needed: list[CodegenMissingDetail] = Field(default_factory=list)
    notes: str | None = None


class MethodologySpec(BaseModel):
    """Paper methodology facts needed to generate and evaluate reproduction code."""

    dataset_description: DatasetDescription | None = None
    architecture: ArchitectureSpec | None = None
    training: TrainingSpec | None = None
    evaluation: EvaluationSpec | None = None
    benchmark_experiments: list[BenchmarkExperimentSpec] = Field(default_factory=list)
    implementation: ImplementationSpec | None = None
    codegen_plan: CodegenPlanSpec | None = None
    assumptions_needed: list[CodegenMissingDetail] = Field(default_factory=list)
    extraction_notes: str | None = None

    def confidence_fields(self) -> Iterable[ConfidenceField[Any]]:
        """Yield every confidence-stamped field in the methodology tree."""

        def walk(value: Any) -> Iterable[ConfidenceField[Any]]:
            if isinstance(value, ConfidenceField):
                yield value
            elif isinstance(value, BaseModel):
                for child_name in value.__class__.model_fields:
                    yield from walk(getattr(value, child_name))
            elif isinstance(value, list):
                for item in value:
                    yield from walk(item)
            elif isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)

        yield from walk(self)

    def confidence_summary(self) -> ConfidenceSummary:
        """Count confirmed/inferred/assumed fields across the spec."""
        summary = ConfidenceSummary()

        def walk(value: Any) -> None:
            if isinstance(value, ConfidenceField):
                summary.record(value.confidence)
            elif isinstance(value, ConfidenceLevel):
                summary.record(value)
            elif isinstance(value, BaseModel):
                for child_name in value.__class__.model_fields:
                    walk(getattr(value, child_name))
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)

        walk(self)
        return summary


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


__all__ = [
    "ArchitectureSpec",
    "ArchitecturePass",
    "BenchmarkExperimentSpec",
    "CodegenFileSpec",
    "CodegenMissingDetail",
    "CodegenPlanSpec",
    "ConfidenceField",
    "ConfidenceLevel",
    "ConfidenceSummary",
    "DatasetAgentResult",
    "DatasetDescription",
    "DatasetSpec",
    "DatasetTaskPass",
    "EvaluationSpec",
    "ExtractedDatasetInfo",
    "ExtractedPreprocessStep",
    "ImplementationPlanPass",
    "ImplementationSpec",
    "MethodologySpec",
    "ModelComponentSpec",
    "PreprocessStep",
    "TrainingEvalPass",
    "TrainingSpec",
]
