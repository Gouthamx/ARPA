"""PDF -> clean dataset-relevant ``.txt`` pipeline.

This sits *before* the Dataset Agent. The agent only ever consumes a ``.txt``
file; this module is responsible for turning a research-paper PDF into one:

    PDF  --(pypdf text extraction)-->  raw text
         --(PaperSectionExtractor)--->  focused dataset/preprocessing snippet
         --(write)-------------------->  <run_dir>/<stem>_dataset_sections.txt

The extracted ``.txt`` is persisted in the runs output directory so it is
inspectable for debugging when the agent resolves the wrong dataset. Plain-text
input bypasses this module entirely and is left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from arpa.tools.paper_extractor import PaperSectionExtractor, StructuralReport


@dataclass
class PdfConversionResult:
    """Outcome of converting a PDF into a dataset-relevant text file."""

    txt_path: Path
    text: str
    num_pages: int
    raw_char_count: int
    report: StructuralReport


def _read_pdf_text(pdf_path: Path) -> tuple[str, int]:
    """Extract raw text from every page of a PDF using pypdf.

    Returns ``(concatenated_text, num_pages)``. Raises ``RuntimeError`` with an
    actionable message if pypdf is missing or the PDF cannot be parsed.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "pypdf is required to read PDF input. Install it with 'pip install pypdf' "
            "or pass a pre-extracted .txt file instead."
        ) from exc

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:
            logger.warning("Failed to extract text from page {} of '{}': {}", i + 1, pdf_path.name, exc)
            pages.append("")

    text = "\n".join(pages)
    if not text.strip():
        raise RuntimeError(
            f"No extractable text found in '{pdf_path}'. The PDF may be scanned/image-only; "
            "OCR is not supported. Provide a text-based PDF or a .txt file."
        )
    return text, len(reader.pages)


class PdfToTextPipeline:
    """Convert a paper PDF into a focused dataset-relevant ``.txt`` file."""

    def __init__(
        self,
        extractor: PaperSectionExtractor | None = None,
        *,
        max_chars: int = 16000,
    ) -> None:
        self.extractor = extractor or PaperSectionExtractor(max_chars=max_chars)

    def convert(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        *,
        output_name: str | None = None,
    ) -> PdfConversionResult:
        """Convert ``pdf_path`` to a dataset-relevant ``.txt`` inside ``output_dir``.

        The output filename defaults to ``<pdf-stem>_dataset_sections.txt``.
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        txt_name = output_name or f"{pdf_path.stem}_dataset_sections.txt"
        txt_path = output_dir / txt_name

        logger.info("Converting PDF '{}' -> dataset sections", pdf_path.name)
        raw_text, num_pages = _read_pdf_text(pdf_path)
        logger.info("Extracted raw text from {} page(s) ({} chars)", num_pages, len(raw_text))

        report = self.extractor.extract_report(raw_text)
        if not report.text.strip():
            raise RuntimeError(
                f"Could not isolate any candidate section in '{pdf_path.name}'. "
                "The paper text may be unreadable (e.g. scanned/image-only)."
            )

        txt_path.write_text(report.text, encoding="utf-8")
        logger.info("Wrote candidate sections to '{}' ({} chars)", txt_path, report.char_count)

        return PdfConversionResult(
            txt_path=txt_path,
            text=report.text,
            num_pages=num_pages,
            raw_char_count=len(raw_text),
            report=report,
        )


def is_pdf(path: str | Path) -> bool:
    """Return True if ``path`` has a .pdf suffix (case-insensitive)."""
    return Path(path).suffix.lower() == ".pdf"
