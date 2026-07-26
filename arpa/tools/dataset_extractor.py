"""Layer 2 — semantic dataset extraction (LLM-driven).

Takes the structurally-reduced paper blocks from
:class:`arpa.tools.paper_extractor.PaperSectionExtractor` and asks the LLM to
read them and return a validated :class:`ExtractedDatasetInfo`. There are no
hardcoded dataset names or preprocessing keywords here — the model reads the
prose and reports what is actually stated, returning ``null`` for anything that
is not.
"""

from __future__ import annotations

from loguru import logger

from arpa.core.state import ExtractedDatasetInfo
from arpa.models import LLMClient
from arpa.tools.paper_extractor import PaperSectionExtractor

_EXTRACTION_SYSTEM = (
    "You are a meticulous ML research engineer. You read excerpts from a paper "
    "and extract ONLY the dataset and preprocessing facts that are explicitly "
    "stated. You never infer specific numeric values (sizes, class counts, image "
    "dimensions, normalization constants) that are not written in the text — for "
    "those, return null. For augmentation steps, you may mark a step's confidence "
    "as 'inferred' when the text clearly implies it without giving parameters, "
    "but you must not fabricate the parameter values themselves."
)

_EXTRACTION_PROMPT = """Read the following excerpts from a machine learning paper and extract the dataset and preprocessing details.

For each preprocessing/augmentation step, capture:
  - name: the transform name (e.g. "RandomCrop", "RandomHorizontalFlip", "Normalize", "Resize")
  - parameters: ONLY parameters explicitly stated, as a dict (e.g. {{"size": 32, "padding": 4}}). Empty dict if none stated.
  - confidence: "confirmed" if the step AND its use are explicitly in the text; "inferred" if strongly implied but not fully specified; "assumed" only if it is a near-universal default you are flagging as such.
  - evidence: the verbatim phrase from the text that supports it, or null.

Rules:
  - Identify the dataset(s) the paper trains/evaluates on. If the paper uses MULTIPLE datasets, set "name" to the PRIMARY one (the first named or most central to the experiments) and list every other dataset name in "aliases". Never return null for name when at least one dataset is named anywhere in the text.
  - Only include the steps and parameters that are stated. Return null for any split size, class count, or image dimension not explicitly stated. DO NOT guess from a dataset's reputation.
  - When the paper reports specs for multiple datasets, report the split sizes / class count / input_shape that belong to the PRIMARY dataset named in "name". If they are not stated for that dataset, return null.
  - input_shape must be [channels, height, width]. Determine channels from the text: grayscale/single-channel = 1, RGB/color = 3. Map informal phrasings to the array, e.g. "28x28 grayscale" -> [1, 28, 28]; "224x224 RGB crops" -> [3, 224, 224]; "32x32 color images" -> [3, 32, 32]. Only return null if neither the spatial size nor channel information is stated anywhere in the text.
  - Do not include preprocessing steps that are not mentioned or clearly implied by the text. Architecture/training details (e.g. ReLU, MaxPool, Dropout, Softmax, optimizer) are NOT data preprocessing — exclude them.

Paper excerpts:
---
{context}
---
"""


class DatasetExtractor:
    """Semantic extraction of dataset facts from reduced paper text via an LLM."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        section_extractor: PaperSectionExtractor | None = None,
    ) -> None:
        self.llm = llm
        self.section_extractor = section_extractor or PaperSectionExtractor()

    def extract(self, paper_text: str, *, reduce_first: bool = True) -> ExtractedDatasetInfo:
        """Extract :class:`ExtractedDatasetInfo` from full or pre-reduced paper text.

        Args:
            paper_text: Full paper text, or an already dataset-focused snippet.
            reduce_first: When True (default) run Layer-1 structural reduction
                before sending to the LLM. Set False if ``paper_text`` is already
                a focused snippet (e.g. a hand-written .txt).
        """
        context = paper_text
        if reduce_first:
            report = self.section_extractor.extract_report(paper_text)
            if report.text.strip():
                context = report.text
            else:
                logger.warning("Structural reduction produced no text; using raw input.")

        context = context[:16000]
        prompt = _EXTRACTION_PROMPT.format(context=context)

        try:
            info = self.llm.complete_structured(
                prompt,
                ExtractedDatasetInfo,
                model=self.llm.general_model,
                system=_EXTRACTION_SYSTEM,
            )
        except Exception as exc:
            logger.error("Semantic dataset extraction failed: {}", exc)
            return ExtractedDatasetInfo()

        logger.info(
            "Extracted dataset='{}' classes={} train={} val={} test={} steps={}",
            info.name,
            info.num_classes,
            info.train_size,
            info.val_size,
            info.test_size,
            len(info.preprocess_steps),
        )
        return info
