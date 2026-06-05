"""
Dataset Agent — Innovation 1: autonomous dataset acquisition.

Pipeline:
  1. Name extraction from paper dataset description
  2. Registry resolution (HF → PWC → torchvision → TFDS)
  3. Preprocessing / loading code generation with confidence stamps
  4. Docker sandbox verification (max retries)
  5. Verified DatasetSpec output
"""

from __future__ import annotations

from loguru import logger

from arpa.core.confidence import ConfidenceLevel, ConfidenceSummary
from arpa.core.config import ARPASettings, get_settings
from arpa.core.state import (
    DatasetAgentResult,
    DatasetDescription,
    DatasetSpec,
    ExtractedDatasetInfo,
    MethodologySpec,
    PreprocessStep,
)
from arpa.models import LLMClient, OllamaClient, get_llm_client
from arpa.tools.dataset_extractor import DatasetExtractor
from arpa.tools.dataset_metadata import DatasetMetadataResolver
from arpa.tools.dataset_tools import DatasetResolver, build_loading_code_skeleton
from arpa.tools.docker_tools import DatasetSandboxVerifier, VerificationExpectations

EXTRACTION_SYSTEM = """You are an ML research engineer extracting dataset details from papers.
Return valid JSON only. Be precise about splits, shapes, and transforms mentioned in the text."""

EXTRACTION_PROMPT = """Extract dataset information from this paper excerpt.

Paper context:
{context}

Return JSON:
{{
  "name": "canonical dataset name",
  "split_description": "how train/val/test are defined",
  "train_size": null or integer,
  "val_size": null or integer,
  "test_size": null or integer,
  "input_shape": [C, H, W] or null,
  "num_classes": null or integer,
  "transform_description": "augmentation and preprocessing steps described",
  "confidence_notes": {{
    "name": "confirmed|inferred|assumed",
    "transforms": "confirmed|inferred|assumed"
  }}
}}"""

PREPROCESS_SYSTEM = """You generate PyTorch dataset loading code for paper reproduction.
Stamp each transform with confidence: confirmed (in paper), inferred (standard for dataset), assumed (default).
Return valid JSON only."""

PREPROCESS_PROMPT = """Generate a complete Python module for loading this dataset.

Dataset: {dataset_name}
Registry: {registry_source} / {registry_id}
Paper transforms: {transform_description}
Expected train size: {train_size}
Expected val size: {val_size}
Expected input shape (CHW): {input_shape}
Num classes: {num_classes}

Base loader skeleton (extend this, do not remove load_splits):
```python
{skeleton}
```

Return JSON:
{{
  "loading_code": "full python module as string",
  "preprocess_steps": [
    {{
      "name": "RandomHorizontalFlip",
      "code_snippet": "transforms.RandomHorizontalFlip()",
      "confidence": "confirmed|inferred|assumed",
      "source": "Section X or standard CIFAR practice",
      "evidence": "quote or rationale",
      "warning": null or "HIGH UNCERTAINTY ..."
    }}
  ]
}}"""


class DatasetAgent:
    """Autonomous dataset resolution, codegen, and sandbox verification."""

    def __init__(
        self,
        settings: ARPASettings | None = None,
        ollama: OllamaClient | None = None,
        resolver: DatasetResolver | None = None,
        verifier: DatasetSandboxVerifier | None = None,
        *,
        llm: LLMClient | None = None,
        backend: str | None = None,
        extractor: DatasetExtractor | None = None,
        metadata_resolver: DatasetMetadataResolver | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        # Backend selection priority:
        #   1. explicit `llm` client
        #   2. legacy `ollama` client argument (back-compat)
        #   3. `backend` override ("ollama" | "gemini")
        #   4. settings.llm_backend
        if llm is not None:
            self.llm = llm
        elif ollama is not None:
            self.llm = ollama
        else:
            self.llm = get_llm_client(self.settings, backend=backend)
        # Keep `self.ollama` as an alias for backward compatibility.
        self.ollama = self.llm
        self.resolver = resolver or DatasetResolver()
        # Layer 2 semantic extractor + dynamic property resolver.
        self.extractor = extractor or DatasetExtractor(self.llm)
        self.metadata_resolver = metadata_resolver or DatasetMetadataResolver()
        self.verifier = verifier or DatasetSandboxVerifier(
            image=self.settings.docker_sandbox_image,
            timeout_s=self.settings.docker_verify_timeout_s,
            work_dir=self.settings.ensure_work_dir(),
        )
        self.max_verify_retries = self.settings.dataset_verify_max_retries

    def run(
        self,
        methodology: MethodologySpec | None = None,
        *,
        paper_context: str | None = None,
        dataset_description: DatasetDescription | None = None,
        use_llm_extraction: bool = True,
        use_docker: bool = True,
    ) -> DatasetAgentResult:
        """
        Execute the full dataset pipeline.

        Provide either `methodology.dataset_description`, `dataset_description`,
        or `paper_context` (raw text for LLM extraction).
        """
        result = DatasetAgentResult()

        # Step 1 — Name / metadata extraction (Layer 2 semantic extraction)
        desc = dataset_description
        if desc is None and methodology and methodology.dataset_description:
            desc = methodology.dataset_description
        if desc is None and paper_context and use_llm_extraction:
            desc = self._extract_description(paper_context)
        if desc is None:
            result.escalated = True
            result.escalation_reason = "No dataset description available for extraction"
            return result

        logger.info("Dataset agent: resolving '{}'", desc.name)

        # Step 2 — Registry resolution (name -> loader)
        resolution, log = self.resolver.resolve(desc.name)
        result.resolution_attempts = log
        if resolution is None:
            result.escalated = True
            result.escalation_reason = (
                f"Could not resolve dataset '{desc.name}' in any registry"
            )
            return result

        # Step 2.5 — Dynamic property enrichment (HF card -> PWC -> paper values).
        desc = self._enrich_with_metadata(desc)

        skeleton = build_loading_code_skeleton(resolution)

        # Step 3 — Preprocessing code generation
        loading_code, steps, preprocess_summary = self._generate_loading_code(
            desc, resolution, skeleton, paper_context or desc.raw_context or ""
        )
        result.preprocess_confidence = preprocess_summary
        result.loading_code = loading_code

        # Step 4 — Docker verification with retries
        expectations = VerificationExpectations(
            train_size=desc.train_size,
            val_size=desc.val_size,
            input_shape=desc.input_shape,
            num_classes=desc.num_classes,
        )

        verified = False
        verify_log = ""
        last_code = loading_code

        for attempt in range(1, self.max_verify_retries + 1):
            result.verify_attempts = attempt
            verification = self.verifier.verify(
                last_code,
                expectations,
                use_docker=use_docker,
            )
            verify_log = verification.raw_log
            if verification.passed:
                verified = True
                loading_code = last_code
                break

            logger.warning(
                "Verification attempt {}/{} failed: {}",
                attempt,
                self.max_verify_retries,
                verification.failed_checks or verification.error,
            )
            if attempt < self.max_verify_retries:
                last_code = self._regenerate_after_failure(
                    last_code,
                    verification,
                    desc,
                    resolution,
                )

        if not verified:
            result.escalated = True
            result.escalation_reason = (
                f"Dataset verification failed after {self.max_verify_retries} attempts"
            )
            result.loading_code = last_code

        # Step 5 — Output DatasetSpec
        result.spec = DatasetSpec(
            dataset_name=desc.name,
            registry_source=resolution.source,
            registry_id=resolution.registry_id,
            loading_code=loading_code,
            preprocess_steps=steps,
            train_size=desc.train_size,
            val_size=desc.val_size,
            test_size=desc.test_size,
            input_shape=desc.input_shape,
            num_classes=desc.num_classes,
            verified=verified,
            verification_log=verify_log[-4000:] if verify_log else None,
            resolution_notes="\n".join(log),
        )
        result.verified = verified
        return result

    def _extract_description(self, paper_context: str) -> DatasetDescription:
        """Layer 2 semantic extraction: full paper text -> DatasetDescription.

        Delegates to :class:`DatasetExtractor`, which runs Layer-1 structural
        reduction and then a schema-constrained LLM call. Falls back to a
        name-only description if extraction yields nothing usable.
        """
        info: ExtractedDatasetInfo = self.extractor.extract(paper_context)
        desc = info.to_description(raw_context=paper_context)
        if not info.name:
            logger.warning("Semantic extraction did not identify a dataset name.")
        return desc

    def _enrich_with_metadata(self, desc: DatasetDescription) -> DatasetDescription:
        """Fill missing properties from live registries, deferring to paper values.

        Paper-stated values always win; live metadata only fills gaps. This keeps
        the agent faithful to the paper while recovering specs the paper omitted.
        """
        try:
            meta = self.metadata_resolver.resolve(desc.name)
        except Exception as exc:  # noqa: BLE001 - metadata must never break the run
            logger.warning("Metadata resolution failed for '{}': {}", desc.name, exc)
            return desc

        # Paper values take precedence; live values fill only what the paper omitted.
        if desc.num_classes is None and meta.num_classes is not None:
            desc.num_classes = meta.num_classes
        if desc.train_size is None and meta.train_size is not None:
            desc.train_size = meta.train_size
        if desc.val_size is None and meta.val_size is not None:
            desc.val_size = meta.val_size
        if desc.test_size is None and meta.test_size is not None:
            desc.test_size = meta.test_size
        if meta.sources:
            logger.info("Enriched '{}' from live metadata: {}", desc.name, meta.sources)
        return desc

    def _generate_loading_code(
        self,
        desc: DatasetDescription,
        resolution,
        skeleton: str,
        context: str,
    ) -> tuple[str, list[PreprocessStep], ConfidenceSummary]:
        summary = ConfidenceSummary()
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": PREPROCESS_SYSTEM},
                    {
                        "role": "user",
                        "content": PREPROCESS_PROMPT.format(
                            dataset_name=desc.name,
                            registry_source=resolution.source,
                            registry_id=resolution.registry_id,
                            transform_description=desc.transform_description or "standard train/test",
                            train_size=desc.train_size,
                            val_size=desc.val_size,
                            input_shape=desc.input_shape,
                            num_classes=desc.num_classes,
                            skeleton=skeleton,
                        )
                        + (f"\n\nAdditional paper context:\n{context[:6000]}" if context else ""),
                    },
                ],
                model=self.llm.code_model,
                format_json=True,
            )
            data = self.llm.extract_json(raw)
            code = data.get("loading_code", skeleton)
            steps = [
                PreprocessStep(
                    name=s.get("name", "step"),
                    code_snippet=s.get("code_snippet", ""),
                    confidence=ConfidenceLevel(s.get("confidence", "assumed")),
                    source=s.get("source", ""),
                    evidence=s.get("evidence"),
                    warning=s.get("warning"),
                )
                for s in data.get("preprocess_steps", [])
            ]
            for step in steps:
                summary.record(step.confidence)
            return code, steps, summary
        except Exception as exc:
            logger.warning("LLM codegen failed, using skeleton: {}", exc)
            summary.record(ConfidenceLevel.ASSUMED)
            return skeleton, [], summary

    def _regenerate_after_failure(
        self,
        code: str,
        verification,
        desc: DatasetDescription,
        resolution,
    ) -> str:
        failed = verification.failed_checks
        err = verification.error or ""
        prompt = f"""Fix this PyTorch dataset loader. Verification failed.

Failed checks: {failed}
Error: {err}

Dataset: {desc.name} ({resolution.source}/{resolution.registry_id})
Expected: train={desc.train_size}, val={desc.val_size}, shape={desc.input_shape}

Current code:
```python
{code}
```

Return JSON: {{"loading_code": "fixed full module"}}"""
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": PREPROCESS_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=self.llm.code_model,
                format_json=True,
            )
            data = self.llm.extract_json(raw)
            return data.get("loading_code", code)
        except Exception:
            return code


def run_dataset_agent(
    methodology: MethodologySpec | None = None,
    **kwargs,
) -> DatasetAgentResult:
    """Convenience entrypoint for LangGraph nodes."""
    return DatasetAgent().run(methodology, **kwargs)
