"""Shared ARPA state and sub-schemas."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from arpa.core.confidence import ConfidenceField, ConfidenceLevel, ConfidenceSummary
from arpa.models.schema_helpers import FlexibleStr, FlexibleInt, FlexibleFloat, FlexibleBool, FlexibleList


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
    
    @field_validator('input_shape', mode='before')
    @classmethod
    def parse_input_shape(cls, v):
        """Handle LLM returning string like '[3, 224, 224]' instead of actual list."""
        if v is None:
            return None
        # Already a list
        if isinstance(v, list):
            return v
        # String representation of a list - parse it
        if isinstance(v, str):
            import json
            try:
                # Try to parse as JSON
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            # Try eval as last resort (for strings like "[3, 224, 224]")
            try:
                parsed = eval(v)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
        return None


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
    source: str | list[str] | None = None
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
    source: str | list[str] | None = None
    evidence: str | None = None
    confidence: ConfidenceLevel | None = None  # Plain enum, not ConfidenceField
    
    @field_validator('confidence', mode='before')
    @classmethod
    def unwrap_confidence(cls, v):
        """If LLM returns ConfidenceField object, extract just the enum value."""
        if v is None:
            return ConfidenceLevel.ASSUMED
        if isinstance(v, dict) and 'value' in v:
            return v['value']
        if hasattr(v, 'value'):
            return v.value
        return v


class ArchitectureSpec(BaseModel):
    """Model architecture details needed by the CodeGen Agent."""

    model_config = ConfigDict(extra='ignore')  # Ignore unknown fields from LLM
    
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
    notes: FlexibleStr = None  # Changed to FlexibleStr for auto-unwrapping
    
    @field_validator('layers', mode='before')
    @classmethod
    def handle_layers(cls, v):
        """Handle layers field - can be:
        1. None
        2. ConfidenceField[list[str]] (dict with 'value' key containing list)
        3. List of plain strings
        4. List of ConfidenceField dicts (COMMON ERROR - unwrap each)
        5. List of ModelComponentSpec-like dicts (extract 'name' field from each)
        """
        if v is None:
            return None
        
        # Already a proper ConfidenceField dict
        if isinstance(v, dict) and 'value' in v:
            return v
        
        # List of items - check what kind
        if isinstance(v, list):
            # Empty list
            if len(v) == 0:
                return {"value": [], "confidence": "assumed", "source": None, "evidence": None}
            
            # Check first item to determine list type
            first = v[0]
            
            # List of ConfidenceField dicts - unwrap each and collect values
            if isinstance(first, dict) and 'value' in first and 'confidence' in first:
                unwrapped = [item['value'] if isinstance(item, dict) and 'value' in item else item for item in v]
                return {
                    "value": unwrapped,
                    "confidence": first.get('confidence', 'assumed'),
                    "source": first.get('source'),
                    "evidence": first.get('evidence'),
                    "alternatives": None,
                    "warning": None,
                    "replacement_logic": None
                }
            
            # List of ModelComponentSpec-like dicts (has 'name', 'kind', 'parameters')
            # Extract just the layer names and move full objects to components field
            if isinstance(first, dict) and 'name' in first and 'kind' in first:
                layer_names = [item['name'] if isinstance(item, dict) and 'name' in item else str(item) for item in v]
                return {
                    "value": layer_names,
                    "confidence": "assumed",
                    "source": "extracted from component specs",
                    "evidence": None,
                    "alternatives": None,
                    "warning": "Full component specs should be in architecture.components field",
                    "replacement_logic": None
                }
            
            # List of plain strings - wrap whole list
            return {
                "value": v,
                "confidence": "assumed",
                "source": "auto-wrapped from plain list",
                "evidence": None,
                "alternatives": None,
                "warning": None,
                "replacement_logic": None
            }
        
        # Single string somehow - wrap as single-item list
        if isinstance(v, str):
            return {
                "value": [v],
                "confidence": "assumed",
                "source": "auto-wrapped from string",
                "evidence": None,
                "alternatives": None,
                "warning": None,
                "replacement_logic": None
            }
        
        # Unknown format - return as-is and let Pydantic handle
        return v
    
    @field_validator('model_name', 'architecture_family', 'backbone', 'activation', 'normalization', 'forward_pass', 'initialization', 'pretrained_weights', mode='before')
    @classmethod
    def wrap_plain_strings(cls, v):
        """Auto-wrap plain strings into ConfidenceField objects."""
        if v is None or isinstance(v, dict) or isinstance(v, ConfidenceField):
            return v
        return {
            "value": v,
            "confidence": "assumed",
            "source": "auto-wrapped from plain value",
            "evidence": None,
            "alternatives": None,
            "warning": None,
            "replacement_logic": None
        }
    
    @field_validator('hidden_dim', 'output_dim', mode='before')
    @classmethod
    def wrap_plain_ints(cls, v):
        """Auto-wrap plain integers into ConfidenceField objects."""
        if v is None or isinstance(v, dict) or isinstance(v, ConfidenceField):
            return v
        return {
            "value": v,
            "confidence": "assumed",
            "source": "auto-wrapped from plain value",
            "evidence": None,
            "alternatives": None,
            "warning": None,
            "replacement_logic": None
        }
    
    @field_validator('dropout', mode='before')
    @classmethod
    def wrap_plain_floats(cls, v):
        """Auto-wrap plain floats into ConfidenceField objects."""
        if v is None or isinstance(v, dict) or isinstance(v, ConfidenceField):
            return v
        return {
            "value": v,
            "confidence": "assumed",
            "source": "auto-wrapped from plain value",
            "evidence": None,
            "alternatives": None,
            "warning": None,
            "replacement_logic": None
        }
    
    @field_validator('layers', 'input_shape', mode='before')
    @classmethod
    def wrap_plain_lists(cls, v):
        """Auto-wrap plain lists into ConfidenceField objects."""
        if v is None or isinstance(v, dict) or isinstance(v, ConfidenceField):
            return v
        return {
            "value": v,
            "confidence": "assumed",
            "source": "auto-wrapped from plain value",
            "evidence": None,
            "alternatives": None,
            "warning": None,
            "replacement_logic": None
        }
    
    @field_validator('components', mode='before')
    @classmethod
    def unwrap_components(cls, v):
        """Handle LLM wrapping components list in dict like {"items": [...]} or {"components": [...]}."""
        if v is None:
            return []
        # If it's a dict with 'items' or 'components' key, extract the list
        if isinstance(v, dict):
            if 'items' in v:
                return v['items']
            if 'components' in v:
                return v['components']
            # Dict without expected keys - return empty list
            return []
        # Already a list
        if isinstance(v, list):
            return v
        return []


class TrainingSpec(BaseModel):
    """Training hyperparameters and objective details."""

    model_config = ConfigDict(extra='ignore')  # Ignore unknown fields from LLM
    
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
    optimizer_parameters: dict[str, ConfidenceField[Any]] | None = Field(default=None)
    scheduler_parameters: dict[str, ConfidenceField[Any]] | None = Field(default=None)
    regularization: list[ConfidenceField[str]] | None = Field(default=None)
    augmentation_policy: list[ConfidenceField[str]] | None = Field(default=None)
    mixed_precision: ConfidenceField[bool] | None = None
    gradient_accumulation_steps: ConfidenceField[int] | None = None
    checkpoint_selection: ConfidenceField[str] | None = None
    notes: str | None = None
    
    @field_validator('optimizer', 'scheduler', 'loss_function', 'early_stopping', 'checkpoint_selection', mode='before')
    @classmethod
    def wrap_plain_strings(cls, v):
        """Auto-wrap plain strings into ConfidenceField objects, or unwrap double-wrapped."""
        if v is None:
            return None
        # If already a proper ConfidenceField dict, return as-is
        if isinstance(v, dict) and "value" in v and "confidence" in v:
            return v
        # If it's a ConfidenceField object
        if isinstance(v, ConfidenceField):
            return v
        # If it's a dict but missing required fields, might be double-wrapped
        if isinstance(v, dict) and "value" in v:
            return v
        # Plain string - wrap it
        if isinstance(v, str):
            return {
                "value": v,
                "confidence": "assumed",
                "source": "auto-wrapped from plain value",
                "evidence": None,
                "alternatives": None,
                "warning": None,
                "replacement_logic": None
            }
        # Something else unexpected - pass through and let Pydantic handle it
        return v
    
    @field_validator('learning_rate', 'weight_decay', 'momentum', 'gradient_clip', mode='before')
    @classmethod
    def wrap_plain_floats(cls, v):
        """Auto-wrap plain floats into ConfidenceField objects."""
        if v is None or isinstance(v, dict) or isinstance(v, ConfidenceField):
            return v
        return {
            "value": v,
            "confidence": "assumed",
            "source": "auto-wrapped from plain value",
            "evidence": None,
            "alternatives": None,
            "warning": None,
            "replacement_logic": None
        }
    
    @field_validator('batch_size', 'epochs', 'seed', 'gradient_accumulation_steps', mode='before')
    @classmethod
    def wrap_plain_ints(cls, v):
        """Auto-wrap plain integers into ConfidenceField objects."""
        if v is None or isinstance(v, dict) or isinstance(v, ConfidenceField):
            return v
        return {
            "value": v,
            "confidence": "assumed",
            "source": "auto-wrapped from plain value",
            "evidence": None,
            "alternatives": None,
            "warning": None,
            "replacement_logic": None
        }
    
    @field_validator('mixed_precision', mode='before')
    @classmethod
    def wrap_plain_bools(cls, v):
        """Auto-wrap plain booleans into ConfidenceField objects."""
        if v is None or isinstance(v, dict) or isinstance(v, ConfidenceField):
            return v
        return {
            "value": v,
            "confidence": "assumed",
            "source": "auto-wrapped from plain value",
            "evidence": None,
            "alternatives": None,
            "warning": None,
            "replacement_logic": None
        }
    
    @field_validator('optimizer_parameters', 'scheduler_parameters', mode='before')
    @classmethod
    def handle_none_dicts(cls, v):
        """Convert None to empty dict and handle nested None values in dict."""
        if v is None:
            return {}
        if not isinstance(v, dict):
            return {}
        # Handle dict with None values - convert them to empty ConfidenceField-like dicts
        result = {}
        for key, val in v.items():
            if val is None:
                # Skip None values entirely
                continue
            result[key] = val
        return result if result else {}
    
    @field_validator('regularization', 'augmentation_policy', mode='before')
    @classmethod
    def handle_none_lists(cls, v):
        """Convert None to empty list, and unwrap ConfidenceField dicts to plain lists."""
        # Handle None
        if v is None:
            return []
        # Handle ConfidenceField-wrapped dict {"value": [...], "confidence": ...}
        if isinstance(v, dict) and "value" in v:
            val = v["value"]
            return val if val is not None else []
        # Handle plain dict that's not a ConfidenceField (like {"weight_decay": {...}})
        # This shouldn't be a list field, but LLM might structure it wrong - skip it
        if isinstance(v, dict) and "value" not in v:
            return []
        # Already a list
        if isinstance(v, list):
            return v
        return []


class EvaluationSpec(BaseModel):
    """Evaluation protocol and paper-reported target metric.
    
    Follows the principle: categorical/definitional fields are plain values,
    measured/numeric fields use ConfidenceField for uncertainty tracking.
    """

    model_config = ConfigDict(extra='ignore')  # Ignore unknown fields from LLM
    
    # CATEGORICAL fields - FlexibleStr accepts either plain or ConfidenceField-wrapped
    task_type: FlexibleStr = None
    metric_name: FlexibleStr = None
    metric_direction: FlexibleStr = None  # Changed from Literal to allow flexibility
    eval_split: FlexibleStr = None
    protocol: FlexibleStr = None
    aggregation: FlexibleStr = None
    
    # MEASURED/NUMERIC field - ConfidenceField (value could be uncertain/estimated)
    reported_metric: ConfidenceField[float] | None = None
    
    # COMPLEX fields - FlexibleList/FlexibleStr for auto-unwrapping
    secondary_metrics: FlexibleList = None
    test_time_augmentation: FlexibleStr = None
    notes: FlexibleStr = None
    
    @field_validator('reported_metric', mode='before')
    @classmethod
    def wrap_plain_numbers(cls, v):
        """Auto-wrap plain numbers into ConfidenceField objects, handle double-wrapping."""
        if v is None:
            return None
        # If it's already a ConfidenceField object
        if isinstance(v, ConfidenceField):
            return v
        # If it's a dict, check for double-wrapping: {"value": {"value": 85, "confidence": ...}}
        if isinstance(v, dict):
            if "value" in v:
                inner = v["value"]
                # Check if inner value is itself a ConfidenceField dict
                if isinstance(inner, dict) and "value" in inner:
                    # Double-wrapped - return the outer wrapper
                    return {
                        "value": inner.get("value"),
                        "confidence": inner.get("confidence", "assumed"),
                        "source": inner.get("source"),
                        "evidence": inner.get("evidence"),
                        "alternatives": None,
                        "warning": None,
                        "replacement_logic": None
                    }
                # Normal single wrap - return as-is
                return v
            # Dict without 'value' key - shouldn't happen but pass through
            return v
        # Plain number received - wrap it
        if isinstance(v, (int, float)):
            return {
                "value": v,
                "confidence": "assumed",
                "source": "auto-wrapped from plain value",
                "evidence": None,
                "alternatives": None,
                "warning": None,
                "replacement_logic": None
            }
        return v
    
    @field_validator('secondary_metrics', mode='before')
    @classmethod
    def handle_secondary_metrics(cls, v):
        """Handle secondary_metrics - unwrap if needed, then ensure list."""
        # First unwrap if it's a ConfidenceField dict
        if isinstance(v, dict) and "value" in v:
            v = v["value"]
        # Now ensure it's a list
        if v is None:
            return []
        if not isinstance(v, list):
            return [v]
        return v


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
    source: str | list[str] | None = None
    evidence: str | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.CONFIRMED


class ImplementationSpec(BaseModel):
    """Implementation/environment facts that affect generated runnable code.
    
    Follows the principle: categorical/definitional fields are plain values,
    measured/uncertain fields use ConfidenceField.
    """

    model_config = ConfigDict(extra='ignore')  # Ignore unknown fields from LLM
    
    # CATEGORICAL fields - plain values (either correct or not), but tolerate
    # a ConfidenceField/dict wrapper from the LLM (or from callers constructing
    # this directly) via FlexibleStr's auto-unwrapping.
    framework: FlexibleStr = None
    language: FlexibleStr = None
    hardware: str | None = None
    official_code_url: str | None = None
    config_format: str | None = None

    # DESCRIPTIVE/UNCERTAIN fields - ConfidenceField
    training_time: ConfidenceField[str] | None = None
    entrypoint_hint: ConfidenceField[str] | None = None
    
    # COMPLEX fields
    dependencies: list[str] | None = Field(default=None)
    reproducibility_notes: list[str] | None = Field(default=None)
    notes: str | None = None
    
    @field_validator('training_time', 'entrypoint_hint', mode='before')
    @classmethod
    def wrap_plain_strings(cls, v):
        """Auto-wrap plain strings into ConfidenceField objects."""
        if v is None or isinstance(v, dict) or isinstance(v, ConfidenceField):
            return v
        return {
            "value": v,
            "confidence": "assumed",
            "source": "auto-wrapped from plain value",
            "evidence": None,
            "alternatives": None,
            "warning": None,
            "replacement_logic": None
        }
    
    @field_validator('dependencies', 'reproducibility_notes', mode='before')
    @classmethod
    def handle_none_lists(cls, v):
        """Convert None to empty list for list fields."""
        return v if v is not None else []


class CodegenFileSpec(BaseModel):
    """A file the CodeGen Agent should produce."""

    path: str
    purpose: str
    required_symbols: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class CodegenPlanSpec(BaseModel):
    """Code generation plan inferred from the methodology.
    
    Follows the principle: categorical/definitional fields are plain values,
    uncertain/inferred fields use ConfidenceField.
    """

    model_config = ConfigDict(extra='ignore')  # Ignore unknown fields from LLM

    # CATEGORICAL fields - plain values (either correct or not), but tolerate
    # a ConfidenceField/dict wrapper via FlexibleStr's auto-unwrapping.
    target_task: FlexibleStr = None
    framework: FlexibleStr = None
    entrypoint: FlexibleStr = None
    
    # NAMING fields - ConfidenceField (might be inferred/uncertain)
    model_class_name: ConfidenceField[str] | None = None
    train_function_name: ConfidenceField[str] | None = None
    eval_function_name: ConfidenceField[str] | None = None
    
    # COMPLEX fields
    files: list[CodegenFileSpec] = Field(default_factory=list)
    required_runtime_checks: list[str] = Field(default_factory=list)
    unsupported_reasons: list[str] = Field(default_factory=list)
    notes: str | None = None
    
    @field_validator('model_class_name', 'train_function_name', 'eval_function_name', mode='before')
    @classmethod
    def wrap_plain_strings(cls, v):
        """Auto-wrap plain strings into ConfidenceField objects."""
        if v is None or isinstance(v, dict) or isinstance(v, ConfidenceField):
            return v
        return {
            "value": v,
            "confidence": "assumed",
            "source": "auto-wrapped from plain value",
            "evidence": None,
            "alternatives": None,
            "warning": None,
            "replacement_logic": None
        }
    
    @field_validator('files', 'required_runtime_checks', 'unsupported_reasons', mode='before')
    @classmethod
    def handle_none_lists(cls, v):
        """Convert None to empty list for list fields."""
        return v if v is not None else []


class DatasetTaskPass(BaseModel):
    """Pass 1: dataset, task, and reported metric."""

    model_config = ConfigDict(extra='ignore')
    
    dataset_description: DatasetDescription | None = None
    evaluation: EvaluationSpec | None = None
    assumptions_needed: list[CodegenMissingDetail] = Field(default_factory=list)
    notes: str | None = None


class ArchitecturePass(BaseModel):
    """Pass 2: architecture and model internals."""

    model_config = ConfigDict(extra='ignore')
    
    architecture: ArchitectureSpec | None = None
    assumptions_needed: list[CodegenMissingDetail] = Field(default_factory=list)
    notes: str | None = None


class TrainingEvalPass(BaseModel):
    """Pass 3: training and evaluation hyperparameters."""

    model_config = ConfigDict(extra='ignore')
    
    training: TrainingSpec | None = None
    evaluation: EvaluationSpec | None = None
    assumptions_needed: list[CodegenMissingDetail] = Field(default_factory=list)
    notes: str | None = None


class ImplementationPlanPass(BaseModel):
    """Pass 4: implementation, codegen plan, and remaining missing details."""

    model_config = ConfigDict(extra='ignore')
    
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
                # Use get_confidence() which handles None by defaulting to ASSUMED
                summary.record(value.get_confidence())
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
