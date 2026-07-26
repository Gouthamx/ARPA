"""Paper registry for CodeGenAgent verification (10 papers).

Six real arXiv PDFs (see benchmark/download_papers.py) plus four text excerpts
for fast, reproducible runs without downloading PDFs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodegenPaper:
    key: str
    level: str
    source: Path
    kind: str  # "pdf" | "text"
    dataset_aliases: list[str]
    description: str
    is_benchmark: bool = False


_REPO = Path(__file__).resolve().parent.parent
_FIXTURES = _REPO / "tests" / "fixtures"
_PAPERS = _REPO / "benchmark" / "papers"

CODEGEN_PAPERS: list[CodegenPaper] = [
    CodegenPaper(
        key="easy1",
        level="easy",
        source=_PAPERS / "easy1_fashion_mnist.pdf",
        kind="pdf",
        dataset_aliases=["fashion-mnist", "fashion mnist", "fashionmnist"],
        description="Fashion-MNIST dataset paper",
        is_benchmark=True,
    ),
    CodegenPaper(
        key="easy2",
        level="easy",
        source=_PAPERS / "easy2_emnist.pdf",
        kind="pdf",
        dataset_aliases=["emnist", "extended mnist"],
        description="EMNIST dataset paper",
    ),
    CodegenPaper(
        key="medium1",
        level="medium",
        source=_PAPERS / "medium1_resnet.pdf",
        kind="pdf",
        dataset_aliases=["imagenet", "ilsvrc", "imagenet-1k"],
        description="Deep Residual Learning (ResNet)",
    ),
    CodegenPaper(
        key="medium2",
        level="medium",
        source=_PAPERS / "medium2_vgg.pdf",
        kind="pdf",
        dataset_aliases=["imagenet", "ilsvrc", "imagenet-1k"],
        description="Very Deep ConvNets (VGG)",
    ),
    CodegenPaper(
        key="hard1",
        level="hard",
        source=_PAPERS / "hard1_simclr.pdf",
        kind="pdf",
        dataset_aliases=["imagenet", "ilsvrc", "cifar-10", "cifar10"],
        description="SimCLR contrastive learning",
    ),
    CodegenPaper(
        key="hard2",
        level="hard",
        source=_PAPERS / "hard2_bilinear_cnn.pdf",
        kind="pdf",
        dataset_aliases=["cub", "cub-200", "aircraft", "cars", "dtd"],
        description="Bilinear CNNs (fine-grained)",
    ),
    CodegenPaper(
        key="fixture_cifar10",
        level="easy",
        source=_FIXTURES / "paper_cifar10_excerpt.txt",
        kind="text",
        dataset_aliases=["cifar-10", "cifar10"],
        description="Synthetic CIFAR-10 experiments excerpt",
    ),
    CodegenPaper(
        key="fixture_mnist",
        level="easy",
        source=_FIXTURES / "paper_mnist_excerpt.txt",
        kind="text",
        dataset_aliases=["mnist"],
        description="Synthetic MNIST experiments excerpt",
    ),
    CodegenPaper(
        key="fixture_svhn",
        level="medium",
        source=_FIXTURES / "paper_svhn_excerpt.txt",
        kind="text",
        dataset_aliases=["svhn", "street view house numbers"],
        description="Synthetic SVHN experiments excerpt",
    ),
    CodegenPaper(
        key="fixture_cifar100",
        level="medium",
        source=_FIXTURES / "paper_cifar100_excerpt.txt",
        kind="text",
        dataset_aliases=["cifar-100", "cifar100"],
        description="Synthetic CIFAR-100 experiments excerpt",
    ),
]

CODEGEN_PAPER_KEYS: tuple[str, ...] = tuple(p.key for p in CODEGEN_PAPERS)
