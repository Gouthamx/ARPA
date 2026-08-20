"""Shared ARPA state and sub-schemas."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from arpa.core.confidence import ConfidenceField, ConfidenceLevel, ConfidenceSummary
from arpa.models.schema_helpers import (
    FlexibleStr,
    FlexibleInt,
    FlexibleFloat,
    FlexibleBool,
    FlexibleList,
    _is_null_token,
    join_if_list,
    parse_stringified_list,
    unwrap_confidence_field,
)


# Prose fields a model may answer with several quotes instead of one. Declared
# once and attached per class: the same defect turned up in eight fields across
# five classes, and one `evidence: ['Dense Convolutional Network (DenseNet)']`
# discarded DenseNet's entire codegen plan, because a ValidationError anywhere
# degrades the whole pass to an empty placeholder.
_PROSE_FIELDS = ("evidence", "reason", "suggested_resolution", "purpose")


def _join_prose_fields():
    return field_validator(*_PROSE_FIELDS, mode="before", check_fields=False)(
        classmethod(lambda cls, v: join_if_list(v))
    )


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

    _join_prose = _join_prose_fields()

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


class PassFailure(BaseModel):
    """Record of an extraction pass that raised an exception."""
    
    pass_label: str  # e.g. "dataset/task", "architecture", etc.
    exception_type: str  # e.g. "TimeoutError", "httpx.ReadTimeout"
    exception_message: str


class ExtractedPreprocessStep(BaseModel):
    """A single preprocessing/augmentation step as extracted from paper text."""

    _join_prose = _join_prose_fields()

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
    notes: FlexibleStr = Field(
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

    _join_prose = _join_prose_fields()

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

    _join_prose = _join_prose_fields()

    name: str
    kind: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    source: str | list[str] | None = None
    evidence: str | None = None
    confidence: ConfidenceLevel | None = None  # Plain enum, not ConfidenceField
    
    @field_validator('input_shape', 'output_shape', mode='before')
    @classmethod
    def parse_shape(cls, v):
        """Accept a shape that arrived as a string, e.g. "[3, 224, 224]".

        VGG's architecture pass produced nine of these in one response and
        lost the whole extraction to them. Unwrap before parsing -- the string
        is usually inside a confidence envelope, so parsing first would look
        at a dict and do nothing.

        A shape that is prose rather than numbers ("varies with input") is
        dropped to None instead of raising: it carries no usable shape either
        way, and this field is optional, so failing here would discard every
        other correctly-extracted field in the pass to preserve a value we
        could not have used.
        """
        v = unwrap_confidence_field(v)
        if v is None:
            return None
        parsed = parse_stringified_list(v)
        if isinstance(parsed, str):
            return None
        return parsed

    @field_validator('confidence', mode='before')
    @classmethod
    def unwrap_confidence(cls, v):
        """If LLM returns ConfidenceField object, extract just the enum value.

        Anything that is not a recognised level falls back to ASSUMED rather
        than raising. Models occasionally misalign fields and put a component
        description here -- MobileNetV2 sent confidence='224x224 conv2d' --
        and passing that through to the enum failed the whole pass, discarding
        all seven correctly-extracted components to preserve a value that was
        never usable. ASSUMED is also the honest reading: a garbled confidence
        marker is no evidence of confidence, and it is already the default for
        a missing one.
        """
        if v is None:
            return ConfidenceLevel.ASSUMED
        if isinstance(v, dict) and 'value' in v:
            v = v['value']
        elif isinstance(v, ConfidenceLevel):
            return v
        elif hasattr(v, 'value'):
            v = v.value

        if isinstance(v, ConfidenceLevel):
            return v
        try:
            return ConfidenceLevel(v)
        except (ValueError, TypeError):
            return ConfidenceLevel.ASSUMED


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

        Cases 3-5 are also reached when the list arrives already wrapped in a
        ConfidenceField, which is what models actually return when the prompt
        asks for confidence/evidence per field. That used to fall straight
        through the "already a proper ConfidenceField" check below with a
        list-of-dicts still sitting in `value`, and Pydantic then rejected it
        against `list[str]` ("layers.value.0 Input should be a valid string").
        Every architecture pass on a real paper died that way -- the model had
        extracted the layers correctly (ResNet conv stacks, DeiT's
        patch_embedding, SimCLR's augmentation module) and the spec was
        discarded at the door, leaving CodeGenAgent to invent a generic CNN.
        So unwrap the envelope first and normalize what's inside, rather than
        trusting any dict that merely has a 'value' key.
        """
        if v is None:
            return None

        # ConfidenceField envelope: keep its provenance, but normalize the
        # payload through the same list handling as a bare list.
        if isinstance(v, dict) and 'value' in v:
            inner = v['value']
            if inner is None or isinstance(inner, str) or not isinstance(inner, list):
                normalized = cls.handle_layers(inner) if inner is not None else None
                inner_value = normalized['value'] if isinstance(normalized, dict) else inner
            else:
                normalized = cls.handle_layers(inner)
                inner_value = normalized['value'] if isinstance(normalized, dict) else inner
            return {**v, 'value': inner_value}

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
    
    @field_validator('input_shape', mode='before')
    @classmethod
    def wrap_plain_lists(cls, v):
        """Auto-wrap plain lists into ConfidenceField objects.

        Parses a stringified shape ("[3, 224, 224]") first, at the top level
        and inside an envelope -- wrapping without parsing just relocates the
        string and it still fails `list[int]`.

        Deliberately does NOT cover 'layers'. Pydantic v2 runs mode='before'
        validators in reverse definition order, so this one (defined later)
        ran *first* and wrapped a raw layers list into {'value': [...]},
        after which handle_layers' early return for dicts skipped all of its
        shape normalization -- making its list-of-component-dicts branch
        unreachable and failing every real architecture extraction. handle_layers
        does its own wrapping, so listing 'layers' here is both redundant and
        actively harmful.
        """
        def as_shape(raw):
            """Parsed list, or None when it is prose we cannot use.

            Mirrors ModelComponentSpec.parse_shape: an unusable shape must not
            cost the pass every other field it extracted.
            """
            parsed = parse_stringified_list(raw)
            return None if isinstance(parsed, str) else parsed

        if v is None:
            return v
        if isinstance(v, ConfidenceField):
            return v
        if isinstance(v, dict):
            if "value" in v:
                return {**v, "value": as_shape(v["value"])}
            return v
        return {
            "value": as_shape(v),
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
    # Iteration/step-based schedules need their own home. Plenty of papers
    # never state epochs at all -- ResNet trains "for up to 60x10^4
    # iterations" -- and with only `epochs` on offer the model put that count
    # there, yielding epochs=60000 for a paper that names no epoch count and
    # whose real figure is 600,000 iterations. Two different units silently
    # sharing one field produces a training script off by orders of magnitude.
    max_iterations: ConfidenceField[int] | None = None
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
    # FlexibleStr, not bare str: models wrap notes in a ConfidenceField as
    # readily as any other field, and a plain `str` rejected that outright
    # ("training.notes: Input should be a valid string, input_value={'value':
    # 'None', 'confidence': 'assumed'}"), taking the entire training/eval pass
    # down with it. Matches ArchitectureSpec.notes and EvaluationSpec.notes,
    # which are already FlexibleStr.
    notes: FlexibleStr = None
    
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
    
    @field_validator('batch_size', 'epochs', 'max_iterations', 'seed',
                     'gradient_accumulation_steps', mode='before')
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
        """Auto-wrap plain booleans into ConfidenceField objects.

        Also maps the string tokens models use for "absent" ("None", "N/A",
        ...) onto a real None, at the top level and inside an envelope. A
        literal 'None' string previously reached Pydantic's bool parser and
        raised ("mixed_precision.value: Input should be a valid boolean,
        input_value='None'"), which discarded DeiT's entire training/eval pass
        over a field that simply wasn't stated in the paper.
        """
        if v is None or _is_null_token(v):
            return None
        if isinstance(v, ConfidenceField):
            return None if _is_null_token(v.value) else v
        if isinstance(v, dict):
            if "value" in v and _is_null_token(v["value"]):
                return None
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
        """Convert None to empty dict and handle nested None values in dict.

        Values are `ConfidenceField[Any]`, so a plain entry like
        `{"scheduler_type": "step"}` -- the natural way to answer, and what
        models actually send -- failed as "scheduler_parameters.scheduler_type:
        Input should be a valid dictionary or instance of ConfidenceField[Any]"
        and cost the whole pass. Bare values are wrapped here instead.
        """
        if v is None:
            return {}
        if not isinstance(v, dict):
            return {}
        # Handle dict with None values - convert them to empty ConfidenceField-like dicts
        result = {}
        for key, val in v.items():
            if val is None or _is_null_token(val):
                # Skip None values entirely
                continue
            if isinstance(val, ConfidenceField) or (isinstance(val, dict) and "value" in val):
                result[key] = val
                continue
            result[key] = {
                "value": val,
                "confidence": "assumed",
                "source": "auto-wrapped from plain value",
                "evidence": None,
                "alternatives": None,
                "warning": None,
                "replacement_logic": None,
            }
        return result if result else {}
    
    @field_validator('regularization', 'augmentation_policy', mode='before')
    @classmethod
    def handle_none_lists(cls, v):
        """Convert None to empty list, and unwrap ConfidenceField dicts to plain lists.

        These are `list[ConfidenceField[str]]` fields, but a model asked for a
        single regularizer or augmentation policy naturally answers with one
        value rather than a list -- as a bare scalar ("L2"), or wrapped
        ("{'value': 'L2', 'confidence': 'confirmed'}"). Both used to be lost:
        the wrapped form unwrapped straight to the scalar and then failed
        Pydantic's list check ("Input should be a valid list, input_value='L2'"),
        killing the whole training/eval pass on SimCLR and DeiT; the bare form
        silently became [] further down, dropping a real extracted value
        without any error at all. Scalars are now promoted to single-item
        lists, which is what they mean.
        """
        def _as_element(item, envelope=None):
            """Coerce one entry into a ConfidenceField-shaped dict.

            The field is `list[ConfidenceField[str]]`, so a bare "L2" in the
            list is just as invalid as a bare "L2" instead of the list --
            both fail as `regularization.0`. Plain entries are therefore
            wrapped, inheriting the outer envelope's provenance when they were
            promoted out of one.
            """
            if isinstance(item, dict):
                return item
            base = {"confidence": "assumed", "source": "auto-wrapped from plain value"}
            if envelope:
                base = {k: val for k, val in envelope.items() if k != "value"} or base
            return {**base, "value": item}

        # Handle None
        if v is None:
            return []
        # Handle ConfidenceField-wrapped dict {"value": [...], "confidence": ...}
        if isinstance(v, dict) and "value" in v:
            val = v["value"]
            if val is None:
                return []
            # Preserve the envelope's provenance when promoting a lone scalar,
            # so a confirmed value doesn't silently lose its evidence.
            if not isinstance(val, list):
                return [_as_element(val, envelope=v)]
            return [_as_element(item, envelope=v) for item in val]
        # Handle plain dict that's not a ConfidenceField (like {"weight_decay": {...}})
        # This shouldn't be a list field, but LLM might structure it wrong - skip it
        if isinstance(v, dict) and "value" not in v:
            return []
        # Already a list
        if isinstance(v, list):
            return [_as_element(item) for item in v]
        # Bare scalar ("L2") -- a single value, not an absent one.
        if isinstance(v, (str, int, float)):
            return [_as_element(v)]
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

    _join_prose = _join_prose_fields()

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
    # FlexibleStr, not bare str -- see the note on TrainingSpec.notes. Models
    # wrap `notes` in a ConfidenceField as readily as any other field, and on
    # the four *Pass wrappers that meant one stray envelope failed the whole
    # pass and discarded everything else it had extracted.
    notes: FlexibleStr = None
    
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

    _join_prose = _join_prose_fields()

    path: str
    purpose: str
    required_symbols: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)

    @field_validator('required_symbols', 'depends_on', mode='before')
    @classmethod
    def none_means_empty(cls, v):
        """Treat an explicit null as "nothing", which is what it means here.

        The default_factory does not cover this: it applies when the key is
        absent, and models send `"depends_on": null` for a file that depends
        on nothing -- a perfectly reasonable answer that Pydantic rejects for
        a `list[str]`. VGG's codegen plan died on exactly one such entry
        ("codegen_plan.files.3.depends_on: Input should be a valid list,
        input_value=None"), discarding the plan for all the other files with
        it. A lone string is likewise promoted, since a single dependency is
        as natural to write bare as in a list.
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v


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
    # FlexibleStr, not bare str -- see the note on TrainingSpec.notes. Models
    # wrap `notes` in a ConfidenceField as readily as any other field, and on
    # the four *Pass wrappers that meant one stray envelope failed the whole
    # pass and discarded everything else it had extracted.
    notes: FlexibleStr = None
    
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
    # FlexibleStr, not bare str -- see the note on TrainingSpec.notes. Models
    # wrap `notes` in a ConfidenceField as readily as any other field, and on
    # the four *Pass wrappers that meant one stray envelope failed the whole
    # pass and discarded everything else it had extracted.
    notes: FlexibleStr = None


class ArchitecturePass(BaseModel):
    """Pass 2: architecture and model internals."""

    model_config = ConfigDict(extra='ignore')
    
    architecture: ArchitectureSpec | None = None
    assumptions_needed: list[CodegenMissingDetail] = Field(default_factory=list)
    # FlexibleStr, not bare str -- see the note on TrainingSpec.notes. Models
    # wrap `notes` in a ConfidenceField as readily as any other field, and on
    # the four *Pass wrappers that meant one stray envelope failed the whole
    # pass and discarded everything else it had extracted.
    notes: FlexibleStr = None


class TrainingEvalPass(BaseModel):
    """Pass 3: training and evaluation hyperparameters."""

    model_config = ConfigDict(extra='ignore')
    
    training: TrainingSpec | None = None
    evaluation: EvaluationSpec | None = None
    assumptions_needed: list[CodegenMissingDetail] = Field(default_factory=list)
    # FlexibleStr, not bare str -- see the note on TrainingSpec.notes. Models
    # wrap `notes` in a ConfidenceField as readily as any other field, and on
    # the four *Pass wrappers that meant one stray envelope failed the whole
    # pass and discarded everything else it had extracted.
    notes: FlexibleStr = None


class ImplementationPlanPass(BaseModel):
    """Pass 4: implementation, codegen plan, and remaining missing details."""

    model_config = ConfigDict(extra='ignore')
    
    implementation: ImplementationSpec | None = None
    codegen_plan: CodegenPlanSpec | None = None
    assumptions_needed: list[CodegenMissingDetail] = Field(default_factory=list)
    # FlexibleStr, not bare str -- see the note on TrainingSpec.notes. Models
    # wrap `notes` in a ConfidenceField as readily as any other field, and on
    # the four *Pass wrappers that meant one stray envelope failed the whole
    # pass and discarded everything else it had extracted.
    notes: FlexibleStr = None


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
    pass_failures: list[PassFailure] = Field(default_factory=list)
    extraction_notes: str | None = None
    
    def all_passes_failed(self) -> bool:
        """Return True if all four extraction passes raised exceptions."""
        return len(self.pass_failures) >= 4

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
    "PassFailure",
    "PreprocessStep",
    "TrainingEvalPass",
    "TrainingSpec",
]
