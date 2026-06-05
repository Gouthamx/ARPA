"""Tests for the shared HTTP retry policy."""

from __future__ import annotations

import httpx
import pytest

from arpa.core.retry import (
    RETRYABLE_STATUS_CODES,
    compute_backoff,
    parse_retry_after,
    request_with_retry,
)


def _resp(status: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status_code=status, headers=headers or {}, request=httpx.Request("GET", "https://x"))


class TestRetryableMatrix:
    @pytest.mark.parametrize("code", [408, 425, 429, 500, 502, 503, 504])
    def test_transient_statuses_are_retried(self, code, monkeypatch):
        monkeypatch.setattr("arpa.core.retry.time.sleep", lambda *_: None)
        calls = {"n": 0}

        def do():
            calls["n"] += 1
            return _resp(code) if calls["n"] < 3 else _resp(200)

        resp = request_with_retry(do, max_retries=5, label="t")
        assert resp is not None and resp.status_code == 200
        assert calls["n"] == 3  # retried twice, then succeeded

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
    def test_client_errors_are_not_retried(self, code, monkeypatch):
        monkeypatch.setattr("arpa.core.retry.time.sleep", lambda *_: None)
        calls = {"n": 0}

        def do():
            calls["n"] += 1
            return _resp(code)

        resp = request_with_retry(do, max_retries=5, label="t")
        assert resp is not None and resp.status_code == code
        assert calls["n"] == 1  # no retry

    def test_transport_errors_are_retried_then_give_up(self, monkeypatch):
        monkeypatch.setattr("arpa.core.retry.time.sleep", lambda *_: None)
        calls = {"n": 0}

        def do():
            calls["n"] += 1
            raise httpx.ConnectTimeout("ssl handshake timed out")

        resp = request_with_retry(do, max_retries=3, label="t")
        assert resp is None
        assert calls["n"] == 3

    def test_transport_error_then_success(self, monkeypatch):
        monkeypatch.setattr("arpa.core.retry.time.sleep", lambda *_: None)
        calls = {"n": 0}

        def do():
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("refused")
            return _resp(200)

        resp = request_with_retry(do, max_retries=3, label="t")
        assert resp is not None and resp.status_code == 200
        assert calls["n"] == 2

    def test_retryable_status_exhausts_budget_returns_last(self, monkeypatch):
        monkeypatch.setattr("arpa.core.retry.time.sleep", lambda *_: None)

        def do():
            return _resp(503)

        resp = request_with_retry(do, max_retries=2, label="t")
        assert resp is not None and resp.status_code == 503


class TestBackoffAndRetryAfter:
    def test_503_is_in_retryable_set(self):
        assert 503 in RETRYABLE_STATUS_CODES

    def test_parse_retry_after_seconds(self):
        assert parse_retry_after(_resp(503, {"Retry-After": "5"})) == 5.0

    def test_parse_retry_after_absent(self):
        assert parse_retry_after(_resp(503)) is None

    def test_backoff_respects_retry_after(self):
        # With retry_after given, delay is ~retry_after (+ <=0.5 jitter), capped.
        assert 5.0 <= compute_backoff(1, retry_after=5.0, cap=30.0) <= 5.5

    def test_backoff_is_capped(self):
        assert compute_backoff(10, base=1.0, cap=4.0) <= 4.0
