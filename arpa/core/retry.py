"""Shared HTTP retry policy for all ARPA network calls.

Centralizes the decision of *what is retryable* so every call site (LLM clients,
registry resolution, metadata lookups) behaves consistently:

  * Transport-level failures (connection reset/refused, SSL handshake timeout,
    read timeout, server disconnect) are transient and retried.
  * HTTP statuses that indicate a transient server-side condition — 408 Request
    Timeout, 425 Too Early, 429 Too Many Requests, and 500/502/503/504 (incl.
    "503 Service Unavailable") — are retried.
  * All other 4xx statuses are caller errors and are NOT retried.

Backoff is exponential with jitter, capped, and honours a server-provided
``Retry-After`` header when present.
"""

from __future__ import annotations

import random
import time
from typing import Callable

import httpx
from loguru import logger

# Statuses worth retrying: transient server / rate-limit conditions.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})


def parse_retry_after(resp: httpx.Response) -> float | None:
    """Return the server-requested delay in seconds from a Retry-After header."""
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)  # delta-seconds form
    except ValueError:
        try:
            from email.utils import parsedate_to_datetime

            when = parsedate_to_datetime(value)
            delta = (when.timestamp() - time.time()) if when else None
            return max(0.0, delta) if delta is not None else None
        except Exception:  # noqa: BLE001
            return None


def compute_backoff(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 30.0,
    retry_after: float | None = None,
) -> float:
    """Exponential backoff with full jitter, honouring Retry-After when given."""
    if retry_after is not None:
        return min(retry_after + random.uniform(0.0, 0.5), cap)
    exp = min(base * (2 ** (attempt - 1)), cap)
    return random.uniform(0.0, exp)


def request_with_retry(
    do_request: Callable[[], httpx.Response],
    *,
    max_retries: int = 3,
    label: str = "request",
    backoff_cap: float = 30.0,
) -> httpx.Response | None:
    """Execute ``do_request`` with retries on transient transport/status errors.

    ``do_request`` must perform a single HTTP call and return the response.

    Returns the response once it is non-retryable (success or a permanent error
    such as 4xx), or ``None`` if every attempt failed at the transport level.
    A response carrying a retryable status is still returned after the retry
    budget is exhausted, so callers can inspect/log it.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = do_request()
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = compute_backoff(attempt, cap=backoff_cap)
                logger.debug(
                    "{} transport error (attempt {}/{}): {}; retrying in {:.1f}s",
                    label, attempt, max_retries, exc, delay,
                )
                time.sleep(delay)
                continue
            logger.warning("{} failed after {} attempts: {}", label, max_retries, last_exc)
            return None

        if resp.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
            delay = compute_backoff(attempt, cap=backoff_cap, retry_after=parse_retry_after(resp))
            logger.warning(
                "{} got retryable HTTP {} (attempt {}/{}); retrying in {:.1f}s",
                label, resp.status_code, attempt, max_retries, delay,
            )
            time.sleep(delay)
            continue
        return resp

    return None
