"""Layer 1 — structural noise removal for research papers.

This layer does **not** understand dataset semantics. Its only job is to take a
full paper (15-40 pages of extracted text) and throw away the parts that
*structurally* cannot contain dataset/experimental information, returning a
handful of candidate prose blocks for the semantic layer (Gemini) to read.

Design rules:
  * No content keyword lists. We never look for "cifar", "normalize", "augment",
    etc. Filtering is by document structure only.
  * Drop blocks by section title only for a small, universal set of structural
    sections (references, acknowledgements, author contributions, broader
    impact, appendix) — these never carry dataset facts.
  * Drop blocks that are not prose: equation-dense, symbol-dense, table/figure
    dumps, or too short to be a paragraph.
  * Bias the surviving set toward the middle of the paper (methodology +
    experiments), where dataset/setup descriptions live, and cap the count so
    the semantic layer receives ~10-15 blocks, not the whole paper.

The semantic meaning ("which dataset, which augmentations") is extracted
downstream by :mod:`arpa.tools.dataset_extractor`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

# The ONLY hardcoded content rule allowed in Layer 1: structural section titles
# that never contain dataset information. Matched as a prefix on the (lowercased)
# section title.
DROP_SECTION_TITLES: tuple[str, ...] = (
    "reference",
    "references",
    "bibliography",
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "author contribution",
    "author contributions",
    "contributions",
    "broader impact",
    "ethics statement",
    "ethical statement",
    "appendix",
    "supplementary",
    "supplemental",
    "funding",
    "conflict of interest",
    "declaration",
)

# Once we pass into any of these trailing sections, everything after is dropped
# (papers put references/appendices last, and nothing dataset-relevant follows).
TERMINAL_SECTION_TITLES: tuple[str, ...] = (
    "reference",
    "references",
    "bibliography",
    "appendix",
    "supplementary",
    "supplemental",
)

# Structural thresholds (no content semantics).
_MIN_PROSE_CHARS = 120          # shorter blocks are headers/captions/noise
_MAX_NONALPHA_RATIO = 0.42      # equation/symbol-dense blocks exceed this
_MIN_LETTER_RATIO = 0.55        # prose is mostly letters+spaces
_MAX_DIGIT_RATIO = 0.22         # tables/number dumps exceed this
_MIN_AVG_WORD_LEN = 2.2         # symbol soup has tiny "words"
_MIN_WORDS = 25

_NUMBERED_HEADER_RE = re.compile(
    r"^\s*((?:\d+|[A-Z])(?:\.\d+){0,3})\.?\s+([A-Z][A-Za-z0-9 ,\-/&:]{2,80})\s*$"
)
_CAPS_HEADER_RE = re.compile(r"^\s*([A-Z][A-Z0-9 ,\-/&:]{3,60})\s*$")


@dataclass
class PaperBlock:
    """A titled chunk of paper text."""

    title: str
    body: str
    order: int
    dropped_reason: str | None = None

    @property
    def text(self) -> str:
        if self.title:
            return f"{self.title}\n{self.body}".strip()
        return self.body.strip()


@dataclass
class StructuralReport:
    """Diagnostics for a structural-extraction pass."""

    blocks: list[PaperBlock] = field(default_factory=list)
    total_blocks: int = 0
    kept_titles: list[str] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (title, reason)
    char_count: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks).strip()


def _clean_text(raw: str) -> str:
    """Normalise whitespace, de-hyphenate line breaks, drop obvious page noise."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Repair scanned-PDF artifact where characters are separated by slashes,
    # e.g. "Mo di/ ed" / "/2/0x/2/0". Conservative: only collapse "/" that sit
    # between word characters or digits.
    text = re.sub(r"(?<=\w)/(?=\w)", "", text)
    # Join words split across line breaks: "augmenta-\ntion" -> "augmentation".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Strip bare page-number lines.
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    # Collapse runs of blank lines and intra-line whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in text.split("\n"))
    return text.strip()


def _looks_like_header(line: str) -> str | None:
    """Return a normalised header title if ``line`` looks like a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None
    m = _NUMBERED_HEADER_RE.match(stripped)
    if m:
        return f"{m.group(1)} {m.group(2)}".strip()
    if stripped.endswith((".", ",", ";")):
        return None
    m = _CAPS_HEADER_RE.match(stripped)
    if m:
        words = m.group(1).split()
        if 1 <= len(words) <= 8:
            return m.group(1).strip()
    return None


def _title_only(title: str) -> str:
    """Strip a leading section number, e.g. '4.1 Datasets' -> 'datasets'."""
    return re.sub(r"^\s*(?:\d+|[A-Z])(?:\.\d+)*\.?\s*", "", title).strip().lower()


def _segment_into_blocks(text: str) -> list[PaperBlock]:
    """Split cleaned text into titled blocks using detected section headers."""
    lines = text.split("\n")
    blocks: list[PaperBlock] = []
    current_title = ""
    current_lines: list[str] = []
    order = 0

    def flush() -> None:
        nonlocal order, current_lines, current_title
        body = "\n".join(current_lines).strip()
        if body or current_title:
            blocks.append(PaperBlock(title=current_title, body=body, order=order))
            order += 1
        current_lines = []

    for line in lines:
        header = _looks_like_header(line)
        if header is not None:
            flush()
            current_title = header
        else:
            current_lines.append(line)
    flush()

    if len(blocks) <= 1 and not any(b.title for b in blocks):
        return _segment_into_paragraphs(text)
    return blocks


def _segment_into_paragraphs(text: str) -> list[PaperBlock]:
    """Fallback segmentation: blank-line-delimited paragraphs."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [PaperBlock(title="", body=p, order=i) for i, p in enumerate(paragraphs)]


def _is_dropped_section(title: str) -> bool:
    t = _title_only(title)
    return any(t == drop or t.startswith(drop + " ") or t == drop + "s" for drop in DROP_SECTION_TITLES)


def _is_terminal_section(title: str) -> bool:
    t = _title_only(title)
    return any(t == term or t.startswith(term) for term in TERMINAL_SECTION_TITLES)


# Layer specifications as papers write them in architecture tables:
# "conv3-64", "3x3 conv, 64", "FC-4096", "maxpool", "1x1, 256".
_ARCH_TABLE_MARKERS = (
    re.compile(r"\bconv\d", re.I),                  # conv3-64, conv1
    re.compile(r"\bfc[-_ ]?\d{3,}", re.I),          # FC-4096
    re.compile(r"\b\d+\s*[x×]\s*\d+\b"),            # 3x3, 7×7
    re.compile(r"\b(max|avg|average|global)[-_ ]?pool", re.I),
    re.compile(r"\bsoft[-]?max\b", re.I),
    re.compile(r"\b(stride|padding|kernel|filters?)\b", re.I),
)
_MIN_ARCH_MARKERS = 2


def _looks_like_architecture_table(body: str) -> bool:
    """Does this block specify layers, even though it does not read as prose?

    The prose filter exists to drop tables, and that was right when this module
    only fed the dataset agent. It is wrong now: an architecture table is the
    single most valuable block in the paper. VGG's Table 1 was discarded as
    "low letter ratio (0.47)" -- the header row is "A A-LRN B C D E" and the
    body is conv3-64 / FC-4096 / maxpool -- so the pass that should describe
    the network never saw a single layer, and returned zero components while
    reporting success.

    Two distinct markers are required so ordinary prose mentioning a stride or
    a 3x3 kernel in passing is not mistaken for a specification.
    """
    hits = sum(1 for pattern in _ARCH_TABLE_MARKERS if pattern.search(body))
    return hits >= _MIN_ARCH_MARKERS


def _prose_quality(body: str) -> tuple[bool, str]:
    """Decide structurally whether ``body`` is readable prose (vs math/table/noise).

    Returns ``(is_prose, reason_if_not)``.
    """
    stripped = body.strip()
    if len(stripped) < _MIN_PROSE_CHARS:
        return False, f"too short ({len(stripped)} chars)"

    words = stripped.split()
    if len(words) < _MIN_WORDS:
        return False, f"too few words ({len(words)})"

    letters = sum(c.isalpha() for c in stripped)
    digits = sum(c.isdigit() for c in stripped)
    spaces = sum(c.isspace() for c in stripped)
    total = len(stripped)

    letter_ratio = letters / total
    digit_ratio = digits / total
    nonalpha_ratio = (total - letters - spaces) / total
    avg_word_len = letters / max(len(words), 1)

    if letter_ratio < _MIN_LETTER_RATIO:
        return False, f"low letter ratio ({letter_ratio:.2f})"
    if digit_ratio > _MAX_DIGIT_RATIO:
        return False, f"digit-dense ({digit_ratio:.2f}) — likely a table"
    if nonalpha_ratio > _MAX_NONALPHA_RATIO:
        return False, f"symbol-dense ({nonalpha_ratio:.2f}) — likely equations"
    if avg_word_len < _MIN_AVG_WORD_LEN:
        return False, f"fragmented tokens (avg len {avg_word_len:.1f})"

    return True, ""


class PaperSectionExtractor:
    """Layer 1: reduce a full paper to candidate prose blocks by structure alone."""

    def __init__(
        self,
        *,
        max_blocks: int = 20,
        max_chars: int = 32000,
        per_block_chars: int = 2000,
        always_keep_abstract: bool = True,
    ) -> None:
        self.max_blocks = max_blocks
        self.max_chars = max_chars
        self.per_block_chars = per_block_chars
        self.always_keep_abstract = always_keep_abstract

    def extract_report(self, paper_text: str) -> StructuralReport:
        """Reduce ``paper_text`` to candidate blocks and return diagnostics."""
        report = StructuralReport()
        cleaned = _clean_text(paper_text)
        if not cleaned:
            logger.warning("Paper text is empty after cleaning; nothing to extract.")
            return report

        blocks = _segment_into_blocks(cleaned)
        report.total_blocks = len(blocks)

        kept: list[PaperBlock] = []
        terminal_reached = False
        for block in blocks:
            if terminal_reached:
                report.dropped.append((block.title or f"#{block.order}", "after terminal section"))
                continue
            if block.title and _is_terminal_section(block.title):
                terminal_reached = True
                report.dropped.append((block.title, "terminal section (references/appendix)"))
                continue
            if block.title and _is_dropped_section(block.title):
                report.dropped.append((block.title, "structural drop-section"))
                continue
            is_prose, reason = _prose_quality(block.body)
            if not is_prose:
                # A layer specification is worth keeping precisely because it
                # is not prose. Checked only for blocks the prose test already
                # rejected, so nothing that used to survive is affected.
                if _looks_like_architecture_table(block.body):
                    logger.debug(
                        "  kept '{}' despite {}: reads as an architecture table",
                        block.title or f"#{block.order}",
                        reason,
                    )
                    kept.append(block)
                    continue
                report.dropped.append((block.title or f"#{block.order}", reason))
                continue
            kept.append(block)

        if not kept:
            logger.warning("No prose blocks survived structural filtering; returning best-effort.")
            kept = [b for b in blocks if b.body][: self.max_blocks]

        selected = self._select_central(kept, blocks)
        report.blocks = selected
        report.kept_titles = [b.title or f"(untitled #{b.order})" for b in selected]
        report.char_count = len(report.text)

        logger.info(
            "Structural filter kept {}/{} block(s) ({} chars): {}",
            len(selected),
            report.total_blocks,
            report.char_count,
            ", ".join(report.kept_titles) if report.kept_titles else "(none)",
        )
        for title, reason in report.dropped:
            logger.debug("  dropped '{}': {}", title, reason)
        return report

    def extract(self, paper_text: str) -> str:
        """Convenience wrapper returning only the reduced text."""
        return self.extract_report(paper_text).text

    def _select_central(
        self,
        kept: list[PaperBlock],
        all_blocks: list[PaperBlock],
    ) -> list[PaperBlock]:
        """Window the kept blocks and evenly sample across the whole document.

        Long blocks are sub-segmented into fixed-size, paragraph-aware windows so
        that no single block dominates the budget and — crucially — content in the
        *middle* of a giant block (where a dataset description may sit) is reachable.
        When the windowed text exceeds the budget we sample windows at an even
        stride across the entire document rather than taking a reading-order prefix,
        giving coverage of methodology and experiments alike. This is purely
        structural: no window content is inspected.
        """
        windows = self._to_windows(kept)

        if self.always_keep_abstract:
            abstract = self._find_abstract(all_blocks)
            if abstract is not None and not any(
                w.order == abstract.order for w in windows
            ):
                windows.insert(
                    0,
                    PaperBlock(
                        title=abstract.title or "Abstract",
                        body=abstract.body[: self.per_block_chars],
                        order=abstract.order,
                    ),
                )

        total = sum(len(w.text) for w in windows)
        if total <= self.max_chars:
            return windows

        # Evenly sample windows across the document to fit the char budget.
        avg = total / len(windows)
        keep_n = max(1, int(self.max_chars / max(avg, 1.0)))
        if keep_n >= len(windows):
            return windows
        stride = len(windows) / keep_n
        sampled = [windows[min(int(i * stride), len(windows) - 1)] for i in range(keep_n)]

        # De-dup preserving order, then enforce a hard budget.
        seen: set[int] = set()
        out: list[PaperBlock] = []
        budget = self.max_chars
        for w in sampled:
            key = id(w)
            if key in seen:
                continue
            seen.add(key)
            if budget <= 0:
                break
            out.append(w)
            budget -= len(w.text)
        return out

    def _to_windows(self, blocks: list[PaperBlock]) -> list[PaperBlock]:
        """Split each block into paragraph-aware windows of ~per_block_chars."""
        windows: list[PaperBlock] = []
        for block in blocks:
            if len(block.body) <= self.per_block_chars:
                windows.append(block)
                continue
            paras = re.split(r"\n\s*\n", block.body)
            buf = ""
            first = True
            for para in paras:
                if buf and len(buf) + len(para) > self.per_block_chars:
                    windows.append(
                        PaperBlock(
                            title=block.title if first else "",
                            body=buf.strip(),
                            order=block.order,
                        )
                    )
                    first = False
                    buf = ""
                # A single mega-paragraph still larger than the window: hard-split.
                while len(para) > self.per_block_chars:
                    windows.append(
                        PaperBlock(
                            title=block.title if first else "",
                            body=para[: self.per_block_chars],
                            order=block.order,
                        )
                    )
                    first = False
                    para = para[self.per_block_chars :]
                buf = f"{buf}\n\n{para}" if buf else para
            if buf.strip():
                windows.append(
                    PaperBlock(
                        title=block.title if first else "",
                        body=buf.strip(),
                        order=block.order,
                    )
                )
        return windows

    @staticmethod
    def _find_abstract(blocks: list[PaperBlock]) -> PaperBlock | None:
        for block in blocks:
            if "abstract" in block.title.lower():
                return block
        if blocks and blocks[0].body.lower().startswith("abstract"):
            return blocks[0]
        return None


def reduce_paper_to_blocks(paper_text: str, *, max_chars: int = 16000) -> str:
    """Module-level helper: structurally reduce a full paper to candidate blocks."""
    return PaperSectionExtractor(max_chars=max_chars).extract(paper_text)
