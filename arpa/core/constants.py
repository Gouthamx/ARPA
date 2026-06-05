"""Universal fallback constants.

These are deliberately the ONLY hardcoded numeric conventions in ARPA. They are
genuine, dataset-agnostic defaults used strictly as a last resort when a paper
does not state normalization values and the dataset is unrecognized by any live
registry. They are NOT dataset-specific facts (e.g. CIFAR-10's exact channel
means live in the paper or HF card, never here).
"""

from __future__ import annotations

# ImageNet statistics — the de-facto universal default for 3-channel natural
# images when nothing else is known.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Generic single-channel fallback (e.g. grayscale).
GRAYSCALE_MEAN: tuple[float] = (0.5,)
GRAYSCALE_STD: tuple[float] = (0.5,)


def fallback_normalization(num_channels: int | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return (mean, std) fallback normalization for the given channel count."""
    if num_channels == 1:
        return GRAYSCALE_MEAN, GRAYSCALE_STD
    return IMAGENET_MEAN, IMAGENET_STD
