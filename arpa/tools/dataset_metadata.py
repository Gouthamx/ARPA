"""Dynamic dataset property resolution.

Replaces the old hardcoded property registry. Given a dataset name (as extracted
from a paper) we look up its properties from live sources, in order:

  1. **HuggingFace Hub** dataset card (``cardData.dataset_info``) — provides split
     sizes (``splits[*].num_examples``) and class count (``features`` /
     ``class_label.names``) for most standard benchmarks.
  2. **Papers With Code** dataset API — best-effort fallback (its public API is
     largely sunset; we degrade gracefully).
  3. **The paper itself** — whatever the semantic extractor already pulled from
     the text. The paper frequently states split sizes and class counts directly.

Every network call is best-effort: on timeout / error / missing field we simply
fall through, and ultimately defer to the paper-extracted values. Nothing here is
hardcoded per-dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
from loguru import logger

from arpa.core.retry import request_with_retry

HF_SEARCH_API = "https://huggingface.co/api/datasets"
PWC_API_HOST = "paperswithcode.com"
PWC_DATASETS_API = "https://paperswithcode.com/api/v1/datasets/"

_HTTP_TIMEOUT = 20.0
_HTTP_RETRIES = 3


def _client() -> httpx.Client:
    return httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True)


def _get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict | None = None,
    label: str = "request",
) -> httpx.Response | None:
    """GET retrying transient transport AND status (429/5xx) errors with backoff."""
    return request_with_retry(
        lambda: client.get(url, params=params, headers={"Accept": "application/json"}),
        max_retries=_HTTP_RETRIES,
        label=label,
    )


@dataclass
class DatasetMetadata:
    """Resolved dataset properties with provenance per source."""

    canonical_name: str | None = None
    hf_id: str | None = None
    num_classes: int | None = None
    train_size: int | None = None
    val_size: int | None = None
    test_size: int | None = None
    sources: dict[str, str] = field(default_factory=dict)  # field -> source label
    notes: list[str] = field(default_factory=list)

    def merged_with_paper(
        self,
        *,
        num_classes: int | None,
        train_size: int | None,
        val_size: int | None,
        test_size: int | None,
    ) -> DatasetMetadata:
        """Fill any field still missing using paper-extracted values."""
        for fld, val in (
            ("num_classes", num_classes),
            ("train_size", train_size),
            ("val_size", val_size),
            ("test_size", test_size),
        ):
            if getattr(self, fld) is None and val is not None:
                setattr(self, fld, val)
                self.sources[fld] = "paper"
        return self


def _hf_dataset_info(client: httpx.Client, dataset_id: str) -> dict | None:
    resp = _get_with_retry(
        client,
        f"{HF_SEARCH_API}/{dataset_id}",
        params={"full": "true"},
        label=f"HF dataset-info {dataset_id!r}",
    )
    if resp is None:
        return None
    try:
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("HF dataset-info fetch failed for {!r}: {}", dataset_id, exc)
        return None


def _parse_hf_info(card_data: dict) -> tuple[int | None, dict[str, int]]:
    """Pull (num_classes, {split_name: size}) from a HF cardData.dataset_info."""
    info = card_data.get("dataset_info")
    # dataset_info may be a dict or a list of per-config dicts.
    if isinstance(info, list):
        info = info[0] if info else None
    if not isinstance(info, dict):
        return None, {}

    num_classes: int | None = None
    for feature in info.get("features", []) or []:
        dtype = feature.get("dtype")
        if isinstance(dtype, dict) and "class_label" in dtype:
            names = dtype["class_label"].get("names")
            if isinstance(names, dict) or isinstance(names, list):
                num_classes = len(names)
            break

    splits: dict[str, int] = {}
    for split in info.get("splits", []) or []:
        name = split.get("name")
        n = split.get("num_examples")
        if name and isinstance(n, int):
            splits[name] = n
    return num_classes, splits


def _name_matches(query: str, candidate_id: str) -> bool:
    """Loose check that a HF id corresponds to the queried dataset name."""
    q = query.lower().replace("-", "").replace("_", "").replace(" ", "")
    cand = candidate_id.split("/")[-1].lower().replace("-", "").replace("_", "")
    return q == cand or q in cand or cand in q


class DatasetMetadataResolver:
    """Resolve dataset properties from live sources with paper-value fallback."""

    def resolve(
        self,
        dataset_name: str,
        *,
        aliases: list[str] | None = None,
    ) -> DatasetMetadata:
        """Look up properties for ``dataset_name`` from HF, then PWC (best-effort)."""
        meta = DatasetMetadata(canonical_name=dataset_name)
        queries = [dataset_name, *(aliases or [])]

        hf_meta = self._from_huggingface(queries)
        if hf_meta is not None:
            return hf_meta

        pwc_meta = self._from_paperswithcode(queries)
        if pwc_meta is not None:
            return pwc_meta

        meta.notes.append("No live metadata found; will rely on paper-extracted values.")
        logger.info("No live metadata for {!r}; deferring to paper values.", dataset_name)
        return meta

    def _from_huggingface(self, queries: list[str]) -> DatasetMetadata | None:
        try:
            with _client() as client:
                for query in queries:
                    resp = _get_with_retry(
                        client,
                        HF_SEARCH_API,
                        params={"search": query, "limit": 10, "full": "true"},
                        label=f"HF metadata search {query!r}",
                    )
                    if resp is None:
                        continue
                    try:
                        resp.raise_for_status()
                        items = resp.json()
                    except (httpx.HTTPError, ValueError) as exc:
                        logger.debug("HF search failed for {!r}: {}", query, exc)
                        continue
                    if not isinstance(items, list) or not items:
                        continue

                    # Prefer a name-matching id; HF already sorts by relevance/downloads.
                    chosen = next(
                        (it for it in items if _name_matches(query, it.get("id", ""))),
                        items[0],
                    )
                    dataset_id = chosen.get("id")
                    if not dataset_id:
                        continue

                    card_data = chosen.get("cardData")
                    if not card_data:
                        full = _hf_dataset_info(client, dataset_id)
                        card_data = (full or {}).get("cardData", {})
                    if not card_data:
                        continue

                    num_classes, splits = _parse_hf_info(card_data)
                    if num_classes is None and not splits:
                        continue

                    meta = DatasetMetadata(
                        canonical_name=card_data.get("pretty_name") or dataset_id,
                        hf_id=dataset_id,
                    )
                    if num_classes is not None:
                        meta.num_classes = num_classes
                        meta.sources["num_classes"] = f"huggingface:{dataset_id}"
                    if "train" in splits:
                        meta.train_size = splits["train"]
                        meta.sources["train_size"] = f"huggingface:{dataset_id}"
                    for vkey in ("validation", "valid", "val", "dev"):
                        if vkey in splits:
                            meta.val_size = splits[vkey]
                            meta.sources["val_size"] = f"huggingface:{dataset_id}"
                            break
                    if "test" in splits:
                        meta.test_size = splits["test"]
                        meta.sources["test_size"] = f"huggingface:{dataset_id}"
                    logger.info(
                        "HF metadata for '{}' -> id={} classes={} train={} test={}",
                        query,
                        dataset_id,
                        meta.num_classes,
                        meta.train_size,
                        meta.test_size,
                    )
                    return meta
        except Exception as exc:  # noqa: BLE001 - never let metadata lookup break the agent
            logger.warning("HuggingFace metadata resolution errored: {}", exc)
        return None

    def _from_paperswithcode(self, queries: list[str]) -> DatasetMetadata | None:
        try:
            with _client() as client:
                for query in queries:
                    resp = _get_with_retry(
                        client,
                        PWC_DATASETS_API,
                        params={"q": query},
                        label=f"PWC metadata {query!r}",
                    )
                    if resp is None:
                        continue
                    if PWC_API_HOST not in (resp.url.host or ""):
                        logger.info("PWC API unavailable (redirected); skipping metadata.")
                        return None
                    if resp.status_code != 200 or "application/json" not in resp.headers.get(
                        "content-type", ""
                    ):
                        return None
                    try:
                        payload = resp.json()
                    except ValueError:
                        return None
                    items = payload.get("results", []) if isinstance(payload, dict) else payload
                    if not items:
                        continue
                    top = items[0]
                    meta = DatasetMetadata(
                        canonical_name=top.get("name") or query,
                        num_classes=top.get("num_classes"),
                    )
                    if meta.num_classes is not None:
                        meta.sources["num_classes"] = "paperswithcode"
                    meta.notes.append("Partial metadata from Papers With Code.")
                    return meta
        except Exception as exc:  # noqa: BLE001
            logger.debug("PWC metadata resolution errored: {}", exc)
        return None
