"""Dataset registry resolution.

Resolution strategy (best match across registries, not strict first-hit):

* **torchvision** and **tensorflow_datasets** are curated, offline lists that map
  directly to reliable, torch/tf-native loaders. They are queried first because an
  exact hit there produces a verifiable loader without any network round-trip.
* **HuggingFace Hub** is the long-tail catch-all (thousands of datasets) used for
  anything the curated lists do not cover — fine-grained recognition, domain
  adaptation, niche benchmarks, etc.
* **Papers With Code** is queried best-effort. Its public dataset API has been
  sunset and now redirects to HuggingFace; the resolver follows redirects and
  degrades gracefully (returns no candidates) instead of breaking the chain.

When nothing matches, the resolver returns ``None`` together with a log that
states exactly which registries were queried and why the match failed, so the
caller can surface an actionable escalation reason instead of silently failing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

import httpx
from loguru import logger

from arpa.core.retry import request_with_retry

RegistrySource = Literal["huggingface", "paperswithcode", "torchvision", "tfds"]

PWC_DATASETS_API = "https://paperswithcode.com/api/v1/datasets/"
PWC_API_HOST = "paperswithcode.com"
HF_DATASETS_API = "https://huggingface.co/api/datasets"

# HTTP behaviour for registry lookups. Transient failures (SSL handshake
# timeouts, resets) and transient statuses (429/5xx) are retried with backoff.
_HTTP_TIMEOUT = 25.0
_HTTP_RETRIES = 3


def _http_get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    label: str = "request",
) -> httpx.Response | None:
    """GET ``url`` retrying transient transport AND status errors with backoff.

    Returns the response (which may carry a non-retryable status) or ``None`` if
    every attempt failed at the transport level. Never raises.
    """
    def _do() -> httpx.Response:
        with httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
            return client.get(url, params=params, headers={"Accept": "application/json"})

    return request_with_retry(_do, max_retries=_HTTP_RETRIES, label=label)

# Minimum similarity for a fuzzy (HuggingFace / PWC) candidate to be kept.
_MIN_FUZZY_SCORE = 0.55
# Minimum similarity for a curated (torchvision / TFDS) candidate to be kept.
_MIN_CURATED_SCORE = 0.70
# A curated match at/above this score short-circuits resolution (no network call).
_CURATED_EARLY_RETURN = 0.90
# Higher = preferred when two candidates tie on match_score.
_SOURCE_PRIORITY: dict[RegistrySource, int] = {
    "torchvision": 3,
    "tfds": 2,
    "huggingface": 1,
    "paperswithcode": 0,
}

# Common informal / abbreviated spellings seen in papers, mapped to a canonical key.
DATASET_ALIASES: dict[str, list[str]] = {
    "cifar10": ["cifar-10", "cifar_10", "cifar 10"],
    "cifar100": ["cifar-100", "cifar_100", "cifar 100"],
    "mnist": ["modified_nist", "modified-nist", "modified nist"],
    "fashion_mnist": ["fashion-mnist", "fashionmnist", "fashion mnist"],
    "kmnist": ["kuzushiji-mnist", "kuzushiji mnist"],
    "emnist": ["extended-mnist", "extended mnist"],
    "imagenet": ["imagenet-1k", "imagenet1k", "ilsvrc2012", "ilsvrc-2012", "ilsvrc 2012", "imagenet 2012"],
    "svhn": ["street view house numbers", "svhn_cropped"],
    "stl10": ["stl-10", "stl 10"],
    "food101": ["food-101", "food 101"],
    "flowers102": ["flowers-102", "oxford flowers", "oxford-102-flowers", "oxford 102 flowers", "102 flowers"],
    "oxford_iiit_pet": ["oxford-iiit pet", "oxford pets", "oxford-iiit-pet", "oxford iiit pets"],
    "fgvc_aircraft": ["fgvc-aircraft", "fgvc aircraft", "aircraft", "fgvcaircraft"],
    "stanford_cars": ["stanford cars", "stanford-cars", "cars196", "cars 196"],
    "dtd": ["describable textures", "describable-textures"],
    "caltech101": ["caltech-101", "caltech 101", "caltech101"],
    "caltech256": ["caltech-256", "caltech 256"],
    "sun397": ["sun-397", "sun 397"],
    "eurosat": ["euro-sat", "euro sat"],
    "gtsrb": ["german traffic sign", "german-traffic-sign"],
    "country211": ["country-211", "country 211"],
    "rendered_sst2": ["rendered-sst2", "rendered sst2", "rendered sst-2"],
    "imagenette": ["image-nette", "imagenette2"],
    "cub": ["cub-200-2011", "cub 200 2011", "cub200", "caltech-ucsd birds", "caltech ucsd birds", "caltech_birds2011"],
    "tiny_imagenet": ["tiny-imagenet", "tiny imagenet", "tiny-imagenet-200"],
}


@dataclass
class DatasetResolution:
    """Result of resolving a paper's dataset name to a registry entry."""

    source: RegistrySource
    registry_id: str
    canonical_name: str
    metadata: dict
    match_score: float
    resolver_notes: str


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _last_segment(name: str) -> str:
    """Return the trailing path segment, e.g. ``uoft-cs/cifar10`` -> ``cifar10``."""
    return name.split("/")[-1]


def _expand_aliases(name: str) -> list[str]:
    """Generate query/match variants for a (possibly informal) dataset name."""
    normalized = _normalize_name(name)
    candidates = [normalized, name.lower().strip()]
    # Parenthetical text often holds the canonical name, e.g. "Modified NIST (MNIST)".
    for part in re.findall(r"\(([^)]+)\)", name):
        part = part.strip()
        if part:
            candidates.extend([part, _normalize_name(part)])
    for canonical, aliases in DATASET_ALIASES.items():
        all_names = [canonical, *aliases]
        if normalized == canonical or any(_normalize_name(a) == normalized for a in all_names):
            candidates.extend(all_names)
            break
    return list(dict.fromkeys(c for c in candidates if c))


def _best_similarity(candidates: list[str], target: str) -> float:
    """Best match of any query variant against ``target`` (and its last segment).

    Returns 1.0 for an exact normalized match and at least 0.9 when one name fully
    contains the other (handles ``cifar10`` vs ``uoft-cs/cifar10``).
    """
    targets = {_normalize_name(target), _normalize_name(_last_segment(target))}
    targets.discard("")
    best = 0.0
    for cand in candidates:
        nc = _normalize_name(cand)
        if not nc:
            continue
        for tgt in targets:
            if nc == tgt:
                return 1.0
            ratio = SequenceMatcher(None, nc, tgt).ratio()
            if nc in tgt or tgt in nc:
                ratio = max(ratio, 0.90)
            best = max(best, ratio)
    return best


class HuggingFaceResolver:
    """Long-tail resolver backed by the HuggingFace Hub datasets search API."""

    def search(self, dataset_name: str, limit: int = 25) -> list[DatasetResolution]:
        candidates = _expand_aliases(dataset_name)
        results: list[DatasetResolution] = []

        resp = _http_get_with_retry(
            HF_DATASETS_API,
            params={"search": dataset_name, "limit": limit},
            label=f"HuggingFace search {dataset_name!r}",
        )
        if resp is None:
            return []
        try:
            resp.raise_for_status()
            items = resp.json()
        except httpx.HTTPError as exc:
            logger.warning("HuggingFace search failed for {!r}: {}", dataset_name, exc)
            return []
        except ValueError as exc:  # JSON decode error
            logger.warning("HuggingFace returned non-JSON for {!r}: {}", dataset_name, exc)
            return []

        if not isinstance(items, list):
            return []

        best_near_miss: tuple[str, float] | None = None
        for item in items:
            did = item.get("id") or item.get("name", "")
            if not did:
                continue
            score = _best_similarity(candidates, did)
            if score < _MIN_FUZZY_SCORE:
                if best_near_miss is None or score > best_near_miss[1]:
                    best_near_miss = (did, score)
                continue
            results.append(
                DatasetResolution(
                    source="huggingface",
                    registry_id=did,
                    canonical_name=did,
                    metadata={
                        "downloads": item.get("downloads"),
                        "likes": item.get("likes"),
                        "tags": item.get("tags", []),
                    },
                    match_score=score,
                    resolver_notes=f"HF Hub match (score={score:.2f})",
                )
            )

        if not results and best_near_miss is not None:
            logger.debug(
                "HuggingFace closest near-miss for {!r}: {!r} (score={:.2f})",
                dataset_name,
                best_near_miss[0],
                best_near_miss[1],
            )

        # Rank by similarity, then popularity as a tiebreaker.
        results.sort(
            key=lambda r: (r.match_score, r.metadata.get("downloads") or 0),
            reverse=True,
        )
        return results


class PapersWithCodeResolver:
    """Best-effort PWC resolver.

    The public PWC dataset API has been sunset and now issues redirects to
    HuggingFace. This resolver follows redirects but verifies the response is
    still JSON from the PWC host; otherwise it logs and returns ``[]`` so the
    fallback chain keeps working.
    """

    def search(self, dataset_name: str, limit: int = 10) -> list[DatasetResolution]:
        candidates = _expand_aliases(dataset_name)

        try:
            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                resp = client.get(
                    PWC_DATASETS_API,
                    params={"q": dataset_name},
                    headers={"Accept": "application/json"},
                )
        except httpx.HTTPError as exc:
            logger.warning("Papers With Code request failed for {!r}: {}", dataset_name, exc)
            return []

        # Detect that the API redirected us off-host (PWC -> HuggingFace, etc.).
        final_host = resp.url.host or ""
        if PWC_API_HOST not in final_host:
            logger.info(
                "Papers With Code API unavailable (redirected to {}); skipping.",
                final_host or resp.url,
            )
            return []

        if resp.status_code != 200:
            logger.info("Papers With Code returned HTTP {}; skipping.", resp.status_code)
            return []

        content_type = resp.headers.get("content-type", "")
        if "application/json" not in content_type:
            logger.info(
                "Papers With Code returned non-JSON ({}); skipping.", content_type or "unknown"
            )
            return []

        try:
            payload = resp.json()
        except ValueError as exc:
            logger.warning("Papers With Code JSON decode failed for {!r}: {}", dataset_name, exc)
            return []

        items = payload.get("results", payload) if isinstance(payload, dict) else payload
        if not isinstance(items, list) or not items:
            return []

        results: list[DatasetResolution] = []
        for item in items[:limit]:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            slug = item.get("slug") or _normalize_name(name)
            if not name:
                continue
            score = _best_similarity(candidates, name)
            if score < _MIN_FUZZY_SCORE:
                continue
            results.append(
                DatasetResolution(
                    source="paperswithcode",
                    registry_id=slug,
                    canonical_name=name,
                    metadata={
                        "url": item.get("url"),
                        "description": item.get("description"),
                    },
                    match_score=score,
                    resolver_notes=f"PWC match (score={score:.2f})",
                )
            )

        results.sort(key=lambda r: r.match_score, reverse=True)
        return results


class TorchvisionResolver:
    """Match against the curated torchvision.datasets registry (see TORCHVISION_CLASS_MAP)."""

    def search(self, dataset_name: str) -> list[DatasetResolution]:
        candidates = _expand_aliases(dataset_name)
        results: list[DatasetResolution] = []

        for tv_name in TORCHVISION_CLASS_MAP:
            score = _best_similarity(candidates, tv_name)
            if score >= _MIN_CURATED_SCORE:
                results.append(
                    DatasetResolution(
                        source="torchvision",
                        registry_id=tv_name,
                        canonical_name=tv_name,
                        metadata={"module": f"torchvision.datasets.{tv_name}"},
                        match_score=score,
                        resolver_notes=f"torchvision.datasets.{tv_name}",
                    )
                )

        results.sort(key=lambda r: r.match_score, reverse=True)
        return results


class TFDSResolver:
    """Match against a curated subset of tensorflow_datasets builder names."""

    TFDS_DATASETS = [
        "cifar10",
        "cifar100",
        "mnist",
        "fashion_mnist",
        "kmnist",
        "emnist",
        "svhn_cropped",
        "stl10",
        "imagenet2012",
        "imagenette",
        "food101",
        "dtd",
        "oxford_flowers102",
        "oxford_iiit_pet",
        "caltech101",
        "caltech_birds2011",
        "cars196",
        "sun397",
        "eurosat",
        "patch_camelyon",
        "colorectal_histology",
        "cassava",
        "beans",
    ]

    def search(self, dataset_name: str) -> list[DatasetResolution]:
        candidates = _expand_aliases(dataset_name)
        results: list[DatasetResolution] = []

        for tfds_name in self.TFDS_DATASETS:
            score = _best_similarity(candidates, tfds_name)
            if score >= _MIN_CURATED_SCORE:
                results.append(
                    DatasetResolution(
                        source="tfds",
                        registry_id=tfds_name,
                        canonical_name=tfds_name,
                        metadata={"builder": f"tfds.load('{tfds_name}')"},
                        match_score=score,
                        resolver_notes=f"TFDS builder '{tfds_name}'",
                    )
                )

        results.sort(key=lambda r: r.match_score, reverse=True)
        return results


class DatasetResolver:
    """Resolve a paper's dataset name to the best registry match.

    Curated registries (torchvision, TFDS) are tried first; an exact hit there
    short-circuits the network lookups. Otherwise candidates from every registry
    are pooled and ranked by (match score, source reliability).
    """

    def __init__(self) -> None:
        self._hf = HuggingFaceResolver()
        self._pwc = PapersWithCodeResolver()
        self._tv = TorchvisionResolver()
        self._tfds = TFDSResolver()

    @staticmethod
    def _rank_key(res: DatasetResolution) -> tuple[float, int]:
        return (res.match_score, _SOURCE_PRIORITY.get(res.source, 0))

    def resolve(
        self,
        dataset_name: str,
        *,
        preferred_source: RegistrySource | None = None,
    ) -> tuple[DatasetResolution | None, list[str]]:
        """Resolve ``dataset_name``. Returns ``(best_match_or_None, attempt_log)``."""
        log: list[str] = []
        all_candidates: list[DatasetResolution] = []

        curated: list[tuple[str, Callable[[str], list[DatasetResolution]]]] = [
            ("torchvision", self._tv.search),
            ("tensorflow_datasets", self._tfds.search),
        ]
        network: list[tuple[str, Callable[[str], list[DatasetResolution]]]] = [
            ("HuggingFace Hub", self._hf.search),
            ("Papers With Code", self._pwc.search),
        ]

        # Phase 1 — curated offline registries.
        for label, search_fn in curated:
            log.append(f"Trying {label}...")
            matches = search_fn(dataset_name)
            if matches:
                log.append(f"  Found {len(matches)} candidate(s) on {label}")
                all_candidates.extend(matches)
            else:
                log.append(f"  No matches on {label}")

        if not preferred_source and all_candidates:
            best_curated = max(all_candidates, key=self._rank_key)
            if best_curated.match_score >= _CURATED_EARLY_RETURN:
                log.append(
                    f"  High-confidence match: {best_curated.registry_id} "
                    f"({best_curated.source}, score={best_curated.match_score:.2f})"
                )
                return best_curated, log

        # Phase 2 — network registries (long-tail + best-effort PWC).
        for label, search_fn in network:
            log.append(f"Trying {label}...")
            matches = search_fn(dataset_name)
            if matches:
                log.append(f"  Found {len(matches)} candidate(s) on {label}")
                all_candidates.extend(matches)
            else:
                log.append(f"  No matches on {label}")

        if not all_candidates:
            reason = (
                f"ESCALATE: no registry match for '{dataset_name}'. "
                "Queried torchvision, tensorflow_datasets and HuggingFace Hub "
                "(Papers With Code API is unavailable). No candidate cleared the "
                "matching threshold — the dataset name may be uncommon, misspelled, "
                "or require a manual registry id."
            )
            log.append(reason)
            return None, log

        if preferred_source:
            preferred = [c for c in all_candidates if c.source == preferred_source]
            if preferred:
                best = max(preferred, key=self._rank_key)
                log.append(
                    f"Preferred-source match: {best.registry_id} "
                    f"({best.source}, score={best.match_score:.2f})"
                )
                return best, log

        best = max(all_candidates, key=self._rank_key)
        log.append(
            f"Best overall match: {best.registry_id} "
            f"({best.source}, score={best.match_score:.2f})"
        )
        return best, log


# Maps torchvision registry ids to their real class names and split semantics.
# torchvision is inconsistent: some datasets use train=bool, some split="...",
# and a few have no split argument at all.
#   selector "train"  -> Class(root, train=True/False, ...)
#   selector "split"  -> Class(root, split="<train_split>"/"<test_split>", ...)
#   selector "none"   -> Class(root, ...)            (no train/test split argument)
TORCHVISION_CLASS_MAP: dict[str, dict] = {
    # train=bool
    "mnist": {"class": "MNIST", "selector": "train"},
    "fashion_mnist": {"class": "FashionMNIST", "selector": "train"},
    "kmnist": {"class": "KMNIST", "selector": "train"},
    "cifar10": {"class": "CIFAR10", "selector": "train"},
    "cifar100": {"class": "CIFAR100", "selector": "train"},
    "usps": {"class": "USPS", "selector": "train"},
    # split="..."
    "svhn": {"class": "SVHN", "selector": "split", "train_split": "train", "test_split": "test"},
    "stl10": {"class": "STL10", "selector": "split", "train_split": "train", "test_split": "test"},
    "imagenet": {"class": "ImageNet", "selector": "split", "train_split": "train", "test_split": "val"},
    "imagenette": {"class": "Imagenette", "selector": "split", "train_split": "train", "test_split": "val"},
    "food101": {"class": "Food101", "selector": "split", "train_split": "train", "test_split": "test"},
    "flowers102": {"class": "Flowers102", "selector": "split", "train_split": "train", "test_split": "test"},
    "oxford_iiit_pet": {
        "class": "OxfordIIITPet",
        "selector": "split",
        "train_split": "trainval",
        "test_split": "test",
    },
    "dtd": {"class": "DTD", "selector": "split", "train_split": "train", "test_split": "test"},
    "fgvc_aircraft": {
        "class": "FGVCAircraft",
        "selector": "split",
        "train_split": "trainval",
        "test_split": "test",
    },
    "stanford_cars": {
        "class": "StanfordCars",
        "selector": "split",
        "train_split": "train",
        "test_split": "test",
    },
    "gtsrb": {"class": "GTSRB", "selector": "split", "train_split": "train", "test_split": "test"},
    "country211": {
        "class": "Country211",
        "selector": "split",
        "train_split": "train",
        "test_split": "test",
    },
    "fer2013": {"class": "FER2013", "selector": "split", "train_split": "train", "test_split": "test"},
    "rendered_sst2": {
        "class": "RenderedSST2",
        "selector": "split",
        "train_split": "train",
        "test_split": "test",
    },
    "pcam": {"class": "PCAM", "selector": "split", "train_split": "train", "test_split": "test"},
    # no split argument
    "caltech101": {"class": "Caltech101", "selector": "none"},
    "caltech256": {"class": "Caltech256", "selector": "none"},
    "eurosat": {"class": "EuroSAT", "selector": "none"},
    "sun397": {"class": "SUN397", "selector": "none"},
    "semeion": {"class": "SEMEION", "selector": "none"},
}


def build_loading_code_skeleton(resolution: DatasetResolution) -> str:
    """Minimal loading code template before LLM refinement."""
    src = resolution.source
    rid = resolution.registry_id

    if src == "huggingface":
        return f'''"""Auto-generated dataset loader for {rid} (HuggingFace)."""
from datasets import load_dataset

def load_splits():
    ds = load_dataset("{rid}")
    train = ds.get("train") or ds[list(ds.keys())[0]]
    val = ds.get("validation") or ds.get("valid")
    test = ds.get("test")
    return train, val, test
'''

    if src == "torchvision":
        meta = TORCHVISION_CLASS_MAP.get(
            rid,
            {"class": "".join(p.capitalize() for p in rid.split("_")), "selector": "train"},
        )
        tv_class = meta["class"]
        selector = meta.get("selector", "train")
        if selector == "train":
            train_ctor = (
                f"datasets.{tv_class}(root=data_dir, train=True, download=True, transform=transform)"
            )
            test_ctor = (
                f"datasets.{tv_class}(root=data_dir, train=False, download=True, transform=transform)"
            )
        elif selector == "none":
            train_ctor = f"datasets.{tv_class}(root=data_dir, download=True, transform=transform)"
            test_ctor = "None"
        else:
            train_split = meta.get("train_split", "train")
            test_split = meta.get("test_split", "test")
            train_ctor = (
                f'datasets.{tv_class}(root=data_dir, split="{train_split}", '
                f"download=True, transform=transform)"
            )
            test_ctor = (
                f'datasets.{tv_class}(root=data_dir, split="{test_split}", '
                f"download=True, transform=transform)"
            )
        return f'''"""Auto-generated dataset loader for {rid} (torchvision)."""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def load_splits(data_dir="./data"):
    transform = get_transforms()
    train = {train_ctor}
    test = {test_ctor}
    return train, None, test
'''

    if src == "tfds":
        return f'''"""Auto-generated dataset loader for {rid} (TFDS)."""
import tensorflow_datasets as tfds

def load_splits():
    builder = tfds.builder("{rid}")
    builder.download_and_prepare()
    train = builder.as_dataset(split="train")
    test = builder.as_dataset(split="test")
    return train, None, test
'''

    return f'''"""Auto-generated stub for {rid} (PWC reference: {resolution.canonical_name})."""
# Resolve implementation from Papers With Code: {rid}
raise NotImplementedError("Implement loader for {resolution.canonical_name}")
'''
