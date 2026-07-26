"""Methodology extraction agent.

The extractor runs four focused structured passes:

1. dataset + task + reported metric
2. architecture and model internals
3. training/evaluation hyperparameters
4. implementation/codegen plan + remaining missing assumptions

Each pass returns a typed partial schema, then deterministic merge logic combines
the pieces into one MethodologySpec for downstream Dataset/CodeGen agents.
"""

from __future__ import annotations

import re
from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel

from arpa.core.confidence import ConfidenceField, ConfidenceLevel
from arpa.core.config import ARPASettings, get_settings
from arpa.core.state import (
    ArchitecturePass,
    BenchmarkExperimentSpec,
    CodegenMissingDetail,
    DatasetTaskPass,
    ImplementationPlanPass,
    MethodologySpec,
    TrainingEvalPass,
)
from arpa.knowledge import ComponentKnowledgeBase
from arpa.models import LLMClient, get_llm_client
from arpa.tools.paper_extractor import PaperSectionExtractor

T = TypeVar("T", bound=BaseModel)

_SYSTEM = (
    "You are a meticulous ML research engineer extracting implementation details "
    "from supervised learning papers. Return only facts grounded in the provided "
    "text. Do not invent defaults. If a value is missing, return null for that "
    "field and mention it in assumptions_needed if it is important for codegen.\n\n"
    "CRITICAL: Respond with valid JSON only. No prose, no markdown fences, no "
    "explanations. Just the raw JSON object."
)

_CONFIDENCE_RULES = """For fields that require confidence tracking (ConfidenceField):
You can use EITHER plain values OR explicit confidence objects:

Plain value format (auto-wrapped with "assumed" confidence):
  "optimizer": "Adam"
  "learning_rate": 0.001

Explicit confidence format (preferred when you have evidence):
  "optimizer": {
    "value": "Adam",
    "confidence": "confirmed",
    "source": "Section 3.2",
    "evidence": "We use the Adam optimizer"
  }

Confidence levels:
  - "confirmed": explicitly stated in the paper
  - "inferred": clearly implied but not directly stated
  - "assumed": standard convention (avoid for numeric hyperparameters)

Both formats are valid. Use explicit format when you have source/evidence.
"""

_PASS1_PROMPT = """Pass 1/4: extract dataset, task, and reported metric.

{confidence_rules}

Extract:
  - dataset name, aliases if visible in prose, split sizes, input shape, class count.
  - preprocessing/augmentation stated near the dataset description.
  - task type, primary metric, reported metric value, metric direction, eval split.

Rules:
  - DatasetDescription fields are plain values, not ConfidenceField objects.
  - EvaluationSpec: categorical fields (task_type, metric_name, metric_direction, 
    eval_split, protocol, aggregation) are plain strings, NOT ConfidenceField objects.
  - EvaluationSpec: reported_metric should be a ConfidenceField object since it's a
    measured/numeric value with uncertainty.
  - Convert image shapes to [channels, height, width] when the text states size
    and channel information.
  - Store percentages as the number written in the paper, e.g. 93.4 for 93.4%.
  - If multiple datasets/metrics are reported, pick the primary reproduction
    target and mention ambiguity in assumptions_needed.

Paper excerpts:
---
{context}
---
"""

_PASS2_PROMPT = """Pass 2/4: extract architecture and model internals needed for codegen.

{confidence_rules}

Extract:
  - model name, architecture family, backbone, pretrained weights.
  - layer/block/component structure, including convolution/kernel/stride/padding,
    pooling, normalization, activation, residual/skip connections, attention,
    classifier head, hidden dimensions, output dimensions.
  - forward-pass description and initialization details.

For components:
  - name: local component name, e.g. "stem", "residual_block_1", "classifier".
  - kind: component type, e.g. "conv", "residual_block", "mlp", "attention".
  - parameters: exact parameters from text/tables only.
  - confidence/evidence: provenance for the component.

Put codegen-critical gaps in assumptions_needed, especially:
  - model architecture underspecified
  - layer order missing
  - classifier head missing
  - pretrained/frozen backbone ambiguity

Paper excerpts:
---
{context}
---
"""

_PASS3_PROMPT = """Pass 3/4: extract training and evaluation hyperparameters.

{confidence_rules}

Extract:
  - optimizer, learning rate, optimizer parameters/betas/momentum.
  - batch size, epochs/steps, weight decay, scheduler and milestones/warmup.
  - loss function and loss parameters, label smoothing, class weights, temperature.
  - regularization, augmentation policies, mixed precision, gradient accumulation,
    gradient clipping, seed, checkpoint selection.
  - evaluation protocol details not already captured: top-k, macro/micro averaging,
    validation/test split, best-vs-last checkpoint, test-time augmentation.

Rules:
  - Do not infer standard hyperparameters. Missing numeric values should remain null.
  - Capture scheduler/optimizer parameters in their dict fields when stated.
  - EvaluationSpec: categorical fields (task_type, metric_name, metric_direction,
    eval_split, protocol, aggregation) are plain strings, NOT ConfidenceField objects.
  - EvaluationSpec: reported_metric is a ConfidenceField object (measured value).
  - Put missing critical training fields in assumptions_needed.

Paper excerpts:
---
{context}
---
"""

_PASS4_PROMPT = """Pass 4/4: extract implementation details and produce a codegen plan.

{confidence_rules}

Extract:
  - implementation framework/language/dependencies.
  - hardware, training time, official code URL, config format, reproducibility notes.
  - a practical file plan for a PyTorch reproduction codebase.

The codegen plan should include:
  - target task and framework.
  - entrypoint, model class name, train/eval function names.
  - files to generate, their purpose, required symbols, and dependencies.
  - runtime checks needed before full training, such as dataset batch shape,
    forward pass, loss computation, and one optimizer step.
  - unsupported reasons if the paper appears out of scope.

Rules:
  - ImplementationSpec: categorical fields (framework, language, hardware, 
    official_code_url, config_format) are plain strings, NOT ConfidenceField objects.
  - ImplementationSpec: training_time and entrypoint_hint use ConfidenceField (uncertain values).
  - CodegenPlanSpec: categorical fields (target_task, framework, entrypoint) are plain strings.
  - CodegenPlanSpec: naming fields (model_class_name, train_function_name, eval_function_name) 
    use ConfidenceField objects since they're inferred.
  - dependencies and reproducibility_notes are plain lists of strings, NOT ConfidenceField lists.

Finally, list all remaining codegen-critical missing details in assumptions_needed.
Be strict: if CodeGen would need to choose a value, record the missing detail.

Paper excerpts:
---
{context}
---
"""


class ExtractionAgent:
    """Extract a typed MethodologySpec from full or reduced paper text."""

    def __init__(
        self,
        settings: ARPASettings | None = None,
        *,
        llm: LLMClient | None = None,
        backend: str | None = None,
        section_extractor: PaperSectionExtractor | None = None,
        kb: ComponentKnowledgeBase | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or get_llm_client(self.settings, backend=backend)
        self.section_extractor = section_extractor or PaperSectionExtractor(
            max_blocks=28,
            max_chars=32000,
            per_block_chars=2400,
        )
        self.kb = kb or ComponentKnowledgeBase()

    def run(self, paper_text: str, *, reduce_first: bool = True) -> MethodologySpec:
        """Return a codegen-focused methodology spec for the paper."""
        context = self._prepare_context(paper_text, reduce_first=reduce_first)

        partials: list[BaseModel] = []
        for label, schema, prompt in (
            ("dataset/task", DatasetTaskPass, _PASS1_PROMPT),
            ("architecture", ArchitecturePass, _PASS2_PROMPT),
            ("training/eval", TrainingEvalPass, _PASS3_PROMPT),
            ("implementation/codegen", ImplementationPlanPass, _PASS4_PROMPT),
        ):
            partial = self._extract_pass(label, schema, prompt, context)

            # RAG enrichment: add KB definitions for architecture components
            if label == "architecture" and isinstance(partial, ArchitecturePass):
                partial = self._enrich_architecture_with_kb(partial)

            partials.append(partial)

        spec = self._merge_passes(partials)
        spec.benchmark_experiments = self._extract_benchmark_experiments(context)
        if spec.dataset_description and not spec.dataset_description.raw_context:
            spec.dataset_description.raw_context = context[:8000]

        summary = spec.confidence_summary()
        logger.info(
            "Extracted methodology: confidence confirmed={} inferred={} assumed={} missing={}",
            summary.confirmed,
            summary.inferred,
            summary.assumed,
            len(spec.assumptions_needed),
        )
        return spec

    def _extract_benchmark_experiments(self, context: str) -> list[BenchmarkExperimentSpec]:
        """Extract scikit-learn-style benchmark table rows from paper text.

        This deterministic pass catches dataset papers such as Fashion-MNIST where
        the reproducible methodology is a suite of classical baseline experiments,
        not a single neural architecture.
        """
        if "Table 3" not in context or "Test Accuracy" not in context:
            return []

        experiments: list[BenchmarkExperimentSpec] = []
        current_model: str | None = None
        protocol = self._extract_experiment_protocol(context)
        source = "Table 3"

        known_models = (
            "DecisionTreeClassifier",
            "ExtraTreeClassifier",
            "GaussianNB",
            "GradientBoostingClassifier",
            "KNeighborsClassifier",
            "LinearSVC",
            "LogisticRegression",
            "MLPClassifier",
            "PassiveAggressiveClassifier",
            "Perceptron",
            "RandomForestClassifier",
            "SGDClassifier",
            "SVC",
        )

        for raw_line in context.splitlines():
            line = self._normalize_table_text(raw_line)
            match = re.search(r"\s(0\.\d{3})\s+0\s*\.\s*(\d{3})\s*$", line)
            if not match:
                continue

            fashion_accuracy = float(match.group(1))
            mnist_accuracy = float(f"0.{match.group(2)}")
            left = line[: match.start()].strip()
            if not left:
                continue

            model_name = current_model
            params_text = left
            for candidate in known_models:
                if left.startswith(candidate):
                    model_name = candidate
                    params_text = left[len(candidate):].strip()
                    current_model = candidate
                    break

            if model_name is None:
                continue

            experiments.append(
                BenchmarkExperimentSpec(
                    model_name=model_name,
                    parameters=self._parse_parameter_text(params_text),
                    dataset_metric=fashion_accuracy,
                    comparison_metric=mnist_accuracy,
                    metric_name="test_accuracy",
                    dataset_name="Fashion-MNIST",
                    comparison_dataset_name="MNIST",
                    repeats=protocol.get("repeats"),
                    aggregation=protocol.get("aggregation"),
                    source=source,
                    evidence=raw_line.strip(),
                )
            )

        if experiments:
            logger.info("Extracted {} benchmark experiment row(s)", len(experiments))
        return experiments

    @staticmethod
    def _normalize_table_text(text: str) -> str:
        replacements = {
            "Classiï¬er": "Classifier",
            "classiï¬er": "classifier",
            "ï¬": "fi",
            "ï¬‚": "fl",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return " ".join(text.split())

    @staticmethod
    def _extract_experiment_protocol(context: str) -> dict[str, Any]:
        protocol: dict[str, Any] = {}
        if re.search(r"repeated\s+5\s+times", context, flags=re.IGNORECASE):
            protocol["repeats"] = 5
        if re.search(r"average\s+accuracy\s+on\s+the\s+test\s+set", context, flags=re.IGNORECASE):
            protocol["aggregation"] = "average"
        return protocol

    @staticmethod
    def _parse_parameter_text(text: str) -> dict[str, Any]:
        if not text:
            return {}

        params: dict[str, Any] = {}
        for key, raw_value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^=]+?)(?=\s+[A-Za-z_][A-Za-z0-9_]*=|$)", text):
            value = raw_value.strip()
            params[key] = _coerce_parameter_value(value)
        return params

    def _prepare_context(self, paper_text: str, *, reduce_first: bool) -> str:
        context = paper_text
        if reduce_first:
            report = self.section_extractor.extract_report(paper_text)
            if report.text.strip():
                context = report.text
            else:
                logger.warning("Methodology reduction produced no text; using raw input.")
        return context[:32000]

    def _extract_pass(
        self,
        label: str,
        schema: type[T],
        prompt_template: str,
        context: str,
    ) -> T:
        prompt = prompt_template.format(
            confidence_rules=_CONFIDENCE_RULES,
            context=context,
        )
        try:
            return self.llm.complete_structured(
                prompt,
                schema,
                model=self.llm.general_model,
                system=_SYSTEM,
            )
        except Exception as exc:
            logger.error("Methodology extraction pass '{}' failed: {}", label, exc)
            missing = CodegenMissingDetail(
                field=label,
                reason=f"{label} extraction failed",
                severity="critical",
                suggested_resolution="Retry extraction or inspect paper text manually.",
                evidence=str(exc),
            )
            if schema is DatasetTaskPass:
                return schema(assumptions_needed=[missing])  # type: ignore[return-value]
            if schema is ArchitecturePass:
                return schema(assumptions_needed=[missing])  # type: ignore[return-value]
            if schema is TrainingEvalPass:
                return schema(assumptions_needed=[missing])  # type: ignore[return-value]
            return schema(assumptions_needed=[missing])  # type: ignore[return-value]

    def _merge_passes(self, partials: list[BaseModel]) -> MethodologySpec:
        spec = MethodologySpec()
        notes: list[str] = []
        missing: list[CodegenMissingDetail] = []

        for partial in partials:
            if isinstance(partial, DatasetTaskPass):
                spec.dataset_description = self._choose(spec.dataset_description, partial.dataset_description)
                spec.evaluation = self._merge_model(spec.evaluation, partial.evaluation)
                missing.extend(partial.assumptions_needed)
                if partial.notes:
                    notes.append(f"dataset/task: {partial.notes}")
            elif isinstance(partial, ArchitecturePass):
                spec.architecture = self._merge_model(spec.architecture, partial.architecture)
                missing.extend(partial.assumptions_needed)
                if partial.notes:
                    notes.append(f"architecture: {partial.notes}")
            elif isinstance(partial, TrainingEvalPass):
                spec.training = self._merge_model(spec.training, partial.training)
                spec.evaluation = self._merge_model(spec.evaluation, partial.evaluation)
                missing.extend(partial.assumptions_needed)
                if partial.notes:
                    notes.append(f"training/eval: {partial.notes}")
            elif isinstance(partial, ImplementationPlanPass):
                spec.implementation = self._merge_model(spec.implementation, partial.implementation)
                spec.codegen_plan = self._merge_model(spec.codegen_plan, partial.codegen_plan)
                missing.extend(partial.assumptions_needed)
                if partial.notes:
                    notes.append(f"implementation/codegen: {partial.notes}")

        spec.assumptions_needed = self._dedupe_missing(missing)
        spec.extraction_notes = "\n".join(notes) if notes else None
        return spec

    def _merge_model(self, current: T | None, incoming: T | None) -> T | None:
        if current is None:
            return incoming
        if incoming is None:
            return current

        for field_name in incoming.__class__.model_fields:
            new_value = getattr(incoming, field_name)
            old_value = getattr(current, field_name, None)
            merged = self._merge_value(old_value, new_value)
            setattr(current, field_name, merged)
        return current

    def _merge_value(self, old_value: Any, new_value: Any) -> Any:
        if old_value is None:
            return new_value
        if new_value is None:
            return old_value
        if isinstance(old_value, ConfidenceField) and isinstance(new_value, ConfidenceField):
            return self._prefer_confidence_field(old_value, new_value)
        if isinstance(old_value, BaseModel) and isinstance(new_value, BaseModel):
            return self._merge_model(old_value, new_value)
        if isinstance(old_value, list) and isinstance(new_value, list):
            return self._merge_lists(old_value, new_value)
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            merged = dict(old_value)
            for key, value in new_value.items():
                merged[key] = self._merge_value(merged.get(key), value)
            return merged
        return old_value

    @staticmethod
    def _choose(old_value: T | None, new_value: T | None) -> T | None:
        return old_value if old_value is not None else new_value

    @staticmethod
    def _prefer_confidence_field(
        old_value: ConfidenceField[Any],
        new_value: ConfidenceField[Any],
    ) -> ConfidenceField[Any]:
        rank = {"confirmed": 3, "inferred": 2, "assumed": 1}
        # Use get_confidence() method which handles None safely
        old_conf = old_value.get_confidence() if old_value else ConfidenceLevel.ASSUMED
        new_conf = new_value.get_confidence() if new_value else ConfidenceLevel.ASSUMED
        
        # Get rank values, handling string or enum
        old_rank = rank.get(old_conf.value if hasattr(old_conf, 'value') else old_conf, 0)
        new_rank = rank.get(new_conf.value if hasattr(new_conf, 'value') else new_conf, 0)
        
        if new_rank > old_rank:
            return new_value
        return old_value

    @staticmethod
    def _merge_lists(old_value: list[Any], new_value: list[Any]) -> list[Any]:
        merged = list(old_value)
        seen = {_stable_key(item) for item in merged}
        for item in new_value:
            key = _stable_key(item)
            if key not in seen:
                merged.append(item)
                seen.add(key)
        return merged

    @staticmethod
    def _dedupe_missing(items: list[CodegenMissingDetail]) -> list[CodegenMissingDetail]:
        out: list[CodegenMissingDetail] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            key = (item.field.lower(), item.reason.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    def _enrich_architecture_with_kb(self, arch_pass: ArchitecturePass) -> ArchitecturePass:
        """Enrich architecture components with KB definitions.
        
        When a component is mentioned by name but not fully defined in the paper,
        lookup the standard definition from the knowledge base and add it as a
        proposed default in the missing details.
        
        This follows RAG principles:
        - Paper-stated values always win (never modify existing components)
        - KB values go into proposed_default, not into spec fields
        - Mark as INFERRED confidence (it's a standard definition, not paper-stated)
        - Clear source attribution
        
        Args:
            arch_pass: ArchitecturePass from Pass 2 extraction
        
        Returns:
            Enriched ArchitecturePass with KB suggestions in assumptions_needed
        """
        if not arch_pass.architecture or not arch_pass.architecture.components:
            return arch_pass

        enriched_count = 0

        for component in arch_pass.architecture.components:
            # Skip if component already has detailed parameters
            if component.parameters and len(component.parameters) > 2:
                continue

            # Try KB lookup by name and kind
            try:
                kb_entry = self.kb.lookup_component(component.name, component.kind)
                if kb_entry:
                    # Add KB definition as a missing detail with proposed default
                    arch_pass.assumptions_needed.append(
                        CodegenMissingDetail(
                            field=f"architecture.component.{component.name}",
                            reason=(
                                f"Paper mentions '{component.name}' but does not provide "
                                f"implementation details. Using standard definition."
                            ),
                            severity="important",
                            proposed_default=kb_entry.canonical_implementation,
                            default_source=f"ARPA Knowledge Base: {kb_entry.reference}",
                            evidence=kb_entry.definition,
                            suggested_resolution=(
                                f"Use standard implementation: {kb_entry.canonical_implementation}"
                            ),
                        )
                    )
                    enriched_count += 1

                    logger.debug(
                        "KB enrichment: '{}' → {} ({})",
                        component.name,
                        kb_entry.canonical_implementation,
                        kb_entry.reference,
                    )
            except Exception as exc:
                # KB lookup should never crash extraction
                logger.warning(
                    "KB lookup failed for component '{}': {}",
                    component.name,
                    exc,
                )

        if enriched_count > 0:
            logger.info(
                "Knowledge base enriched {}/{} architecture components",
                enriched_count,
                len(arch_pass.architecture.components),
            )

        return arch_pass


def _stable_key(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return repr(value)


def _coerce_parameter_value(value: str) -> Any:
    value = value.strip()
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = [part.strip() for part in inner.split(",")]
        return [_coerce_parameter_value(part) for part in parts]
    return value


def run_extraction_agent(paper_text: str, **kwargs) -> MethodologySpec:
    """Convenience entrypoint for graph nodes and scripts."""
    return ExtractionAgent(**kwargs).run(paper_text)
