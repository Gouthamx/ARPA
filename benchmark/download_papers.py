"""Download the real arXiv papers used by the benchmark.

Difficulty is graded by how hard the PDF -> dataset extraction is:
  * easy   - dataset papers with a single, explicitly described benchmark.
  * medium - method papers built around one primary benchmark (ImageNet), with
             related work + equations adding noise.
  * hard   - multi-dataset transfer / fine-grained papers with abbreviated names
             and dataset details scattered across the paper.

Run:  python benchmark/download_papers.py
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
from loguru import logger

PAPERS_DIR = Path(__file__).parent / "papers"

# level -> (filename, arxiv_id, short description)
PAPERS: list[tuple[str, str, str, str]] = [
    ("easy", "easy1_fashion_mnist.pdf", "1708.07747", "Fashion-MNIST dataset paper"),
    ("easy", "easy2_emnist.pdf", "1702.05373", "EMNIST dataset paper"),
    ("medium", "medium1_resnet.pdf", "1512.03385", "Deep Residual Learning (ResNet)"),
    ("medium", "medium2_vgg.pdf", "1409.1556", "Very Deep ConvNets (VGG)"),
    ("hard", "hard1_simclr.pdf", "2002.05709", "SimCLR contrastive learning"),
    ("hard", "hard2_bilinear_cnn.pdf", "1504.07889", "Bilinear CNNs (fine-grained)"),
]

_HEADERS = {"User-Agent": "ARPA-benchmark/1.0 (research reproduction agent)"}


def _download(arxiv_id: str, dest: Path) -> bool:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    for attempt in range(1, 4):
        try:
            with httpx.Client(timeout=60.0, follow_redirects=True, headers=_HEADERS) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.content
            if not data.startswith(b"%PDF"):
                logger.warning("{}: response is not a PDF (got {} bytes)", arxiv_id, len(data))
                return False
            dest.write_bytes(data)
            logger.info("Downloaded {} -> {} ({:.0f} KB)", arxiv_id, dest.name, len(data) / 1024)
            return True
        except httpx.HTTPError as exc:
            logger.warning("{} attempt {}/3 failed: {}", arxiv_id, attempt, exc)
            time.sleep(min(2 ** attempt, 8))
    return False


def main() -> int:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for level, filename, arxiv_id, desc in PAPERS:
        dest = PAPERS_DIR / filename
        if dest.exists() and dest.stat().st_size > 10_000:
            logger.info("Skip (exists): {} [{}] {}", filename, level, desc)
            ok += 1
            continue
        logger.info("Fetching [{}] {} ({})", level, desc, arxiv_id)
        if _download(arxiv_id, dest):
            ok += 1
        time.sleep(1.0)  # be polite to arXiv
    logger.info("Downloaded/verified {}/{} papers into {}", ok, len(PAPERS), PAPERS_DIR.resolve())
    return 0 if ok == len(PAPERS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
