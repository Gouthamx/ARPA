"""Ground-truth expectations for the real (downloaded) benchmark papers.

Difficulty grading reflects how hard the PDF -> dataset extraction is, not the
paper's fame:

  easy   - dataset papers; a single benchmark is the explicit subject of the
           paper, with sizes/classes stated plainly.
  medium - method papers centred on ONE primary benchmark (ImageNet) but padded
           with related work, equations and architecture tables.
  hard   - multi-dataset transfer / fine-grained papers: several datasets, often
           abbreviated, with details scattered across sections.

Some real papers are legitimately ambiguous for a single-dataset schema:
  * EMNIST defines six splits with different class counts (62/47/47/26/10/10) and
    no single canonical size — we accept the family and any one valid variant.
  * Bilinear-CNN / SimCLR evaluate on several datasets — we accept any one of the
    correct primary datasets and reward surfacing the others as secondary.

Scoring (run_benchmark.py) is therefore tolerant: a field scores full marks when
the agent matches ANY accepted value, and is not penalised when the paper itself
does not pin a single value (encoded as ``flexible=True`` / ``None`` expectations).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GroundTruth:
    level: str
    pdf: str
    # Accepted dataset names (case-insensitive, fuzzy). A match against the
    # extracted name OR the resolved registry id counts.
    dataset_aliases: list[str]
    # Accepted values for each numeric field. Empty list = "paper does not pin a
    # single value" -> field is not scored (full marks regardless), because
    # penalising the agent for a genuinely ambiguous spec would be unfair.
    num_classes: list[int] = field(default_factory=list)
    train_size: list[int] = field(default_factory=list)
    test_size: list[int] = field(default_factory=list)
    input_shape_options: list[list[int]] = field(default_factory=list)
    expected_steps: list[str] = field(default_factory=list)
    secondary_datasets: list[str] = field(default_factory=list)
    notes: str = ""


GROUND_TRUTH: dict[str, GroundTruth] = {
    # ---------------------------- EASY ---------------------------------- #
    "easy1": GroundTruth(
        level="easy",
        pdf="benchmark/papers/easy1_fashion_mnist.pdf",
        dataset_aliases=["fashion-mnist", "fashion mnist", "fashionmnist"],
        num_classes=[10],
        train_size=[60000],
        test_size=[10000],
        input_shape_options=[[1, 28, 28]],
        expected_steps=[],  # the dataset paper does not prescribe augmentation
        notes="Single explicit dataset; 70k 28x28 grayscale images, 10 classes.",
    ),
    "easy2": GroundTruth(
        level="easy",
        pdf="benchmark/papers/easy2_emnist.pdf",
        dataset_aliases=["emnist", "extended mnist"],
        # Six splits: ByClass=62, ByMerge=47, Balanced=47, Letters=26, Digits/MNIST=10
        num_classes=[62, 47, 26, 10],
        train_size=[],   # varies by split (697932 / 88800 / 240000 / ...)
        test_size=[],
        input_shape_options=[[1, 28, 28]],
        expected_steps=[],
        notes="Multiple splits with different class counts; accept the family.",
    ),
    # --------------------------- MEDIUM --------------------------------- #
    "medium1": GroundTruth(
        level="medium",
        pdf="benchmark/papers/medium1_resnet.pdf",
        dataset_aliases=["imagenet", "ilsvrc", "imagenet-1k", "imagenet 2012"],
        num_classes=[1000],
        train_size=[1281167, 1280000],  # ~1.28M; paper says "1.28 million"
        test_size=[],  # paper reports val/test variably (50k val / 100k test)
        input_shape_options=[[3, 224, 224]],
        expected_steps=["randomcrop", "randomresizedcrop", "horizontalflip", "normalize"],
        secondary_datasets=["cifar-10", "cifar10"],
        notes="ResNet: primary ImageNet (also CIFAR-10 ablation); 224x224 crops.",
    ),
    "medium2": GroundTruth(
        level="medium",
        pdf="benchmark/papers/medium2_vgg.pdf",
        dataset_aliases=["imagenet", "ilsvrc", "imagenet-1k", "imagenet 2012"],
        num_classes=[1000],
        train_size=[1281167, 1280000],
        test_size=[],
        input_shape_options=[[3, 224, 224]],
        expected_steps=["randomcrop", "horizontalflip", "scale", "normalize"],
        notes="VGG: ImageNet (ILSVRC-2012), 224x224 crops, RGB mean subtraction.",
    ),
    # ---------------------------- HARD ---------------------------------- #
    "hard1": GroundTruth(
        level="hard",
        pdf="benchmark/papers/hard1_simclr.pdf",
        dataset_aliases=["imagenet", "ilsvrc", "imagenet-1k"],
        num_classes=[1000],
        train_size=[1281167, 1280000],
        test_size=[],
        input_shape_options=[[3, 224, 224]],
        expected_steps=["randomresizedcrop", "colorjitter", "grayscale", "horizontalflip", "gaussianblur", "normalize"],
        secondary_datasets=["cifar-10", "cifar10"],
        notes="SimCLR: contrastive aug pipeline; ImageNet primary, many transfer sets.",
    ),
    "hard2": GroundTruth(
        level="hard",
        pdf="benchmark/papers/hard2_bilinear_cnn.pdf",
        dataset_aliases=[
            "cub-200-2011", "cub 200 2011", "cub200", "cub", "caltech-ucsd birds", "birds",
            "fgvc-aircraft", "aircraft", "stanford cars", "cars", "dtd", "describable textures",
        ],
        num_classes=[200],  # CUB-200-2011 primary; aircraft=100, cars=196
        train_size=[5994, 5994],  # CUB train
        test_size=[5794],          # CUB test
        input_shape_options=[[3, 224, 224], [3, 448, 448]],
        expected_steps=[],
        secondary_datasets=["fgvc-aircraft", "stanford cars", "dtd"],
        notes="Bilinear-CNN: 4 fine-grained datasets; CUB-200-2011 is the headline.",
    ),
}
