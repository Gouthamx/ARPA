"""Regression tests for NvidiaClient's malformed/truncated JSON handling.

Two separate defects, both of which cost a whole extraction pass:

1. `max_retries = 1` made the repair branch unreachable -- `attempt <
   max_retries - 1` can never be true -- so the repair-prompt code below it
   was dead and a single bad response was fatal.

2. `finish_reason` was never inspected, so a response cut off at the token
   limit was reported as "Invalid JSON". That is actively misleading: it sent
   debugging toward the schema for two runs when the real cause was DenseNet's
   architecture pass running out of room mid-string ('"eviden'). A repair
   prompt cannot fix truncation either -- it resends the cut-off text and asks
   for more output, so it truncates again. Truncation needs a bigger budget;
   only genuine malformation needs repair.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest

from arpa.models.nvidia_client import (
    MAX_TOKENS_CEILING,
    STRUCTURED_MAX_TOKENS,
    NvidiaClient,
    NvidiaError,
)


class _Settings:
    nvidia_api_key = "test-key"
    nvidia_base_url = "http://test.invalid"
    nvidia_timeout_s = 60.0
    nvidia_general_model = "test-model"
    nvidia_code_model = "test-model"


def _response(content: str, finish_reason: str):
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    resp.json = lambda: {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}]
    }
    return resp


def _client_returning(*responses):
    """Client whose POSTs yield the given responses in order.

    Returns (client, http_client_mock, sent_payloads) so tests can assert on
    what was actually sent, not just what came back.
    """
    client = NvidiaClient(settings=_Settings())
    client.rate_limiter = MagicMock()
    sent: list[dict] = []
    stream = iter(responses)

    def post(url, headers=None, json=None):
        # Snapshot: chat() mutates one payload dict in place across retries, so
        # holding the reference would show every call the final values.
        sent.append(copy.deepcopy(json))
        return next(stream)

    http = MagicMock()
    http.__enter__ = lambda self: MagicMock(post=post)
    http.__exit__ = lambda *args: None
    return client, http, sent


class TestTruncationHandling:
    def test_truncated_response_retries_with_a_larger_budget(self):
        client, http, sent = _client_returning(
            _response('{"a": 1', "length"),   # cut off
            _response('{"a": 1}', "stop"),    # complete
        )
        with patch("httpx.Client", return_value=http):
            out = client.chat(
                [{"role": "user", "content": "x"}], format_json=True, max_tokens=4096
            )
        assert out == '{"a": 1}'
        assert [p["max_tokens"] for p in sent] == [4096, 8192], "budget must grow"

    def test_truncation_retry_resends_the_original_prompt(self):
        """Not a repair prompt: the request was fine, it just needed room."""
        client, http, sent = _client_returning(
            _response('{"a": 1', "length"),
            _response('{"a": 1}', "stop"),
        )
        with patch("httpx.Client", return_value=http):
            client.chat(
                [{"role": "user", "content": "extract this"}],
                format_json=True,
                max_tokens=4096,
            )
        assert sent[1]["messages"] == sent[0]["messages"]

    def test_persistent_truncation_names_the_real_cause(self):
        """"Invalid JSON" for a cut-off response is what misdirected debugging."""
        client, http, _ = _client_returning(
            _response('{"a": 1', "length"),
            _response('{"a": 1', "length"),
        )
        with patch("httpx.Client", return_value=http):
            with pytest.raises(NvidiaError) as excinfo:
                client.chat(
                    [{"role": "user", "content": "x"}], format_json=True, max_tokens=4096
                )
        message = str(excinfo.value)
        assert "truncated" in message.lower()
        assert "finish_reason=length" in message

    def test_budget_growth_is_capped(self):
        """A repetition loop must not escalate the budget without limit."""
        client, http, sent = _client_returning(
            _response("{", "length"),
            _response("{", "length"),
        )
        with patch("httpx.Client", return_value=http):
            with pytest.raises(NvidiaError):
                client.chat(
                    [{"role": "user", "content": "x"}],
                    format_json=True,
                    max_tokens=MAX_TOKENS_CEILING,
                )
        assert all(p["max_tokens"] <= MAX_TOKENS_CEILING for p in sent)


class TestMalformedJsonRepair:
    def test_malformed_json_triggers_the_repair_prompt(self):
        """The repair branch was dead code while max_retries was 1."""
        client, http, sent = _client_returning(
            _response("not json at all", "stop"),
            _response('{"b": 2}', "stop"),
        )
        with patch("httpx.Client", return_value=http):
            out = client.chat(
                [{"role": "user", "content": "x"}], format_json=True, max_tokens=4096
            )
        assert out == '{"b": 2}'
        assert "Fix" in sent[1]["messages"][0]["content"], "second call must be a repair"

    def test_valid_json_never_costs_a_second_call(self):
        client, http, sent = _client_returning(_response('{"ok": true}', "stop"))
        with patch("httpx.Client", return_value=http):
            out = client.chat([{"role": "user", "content": "x"}], format_json=True)
        assert out == '{"ok": true}'
        assert len(sent) == 1

    def test_persistent_malformation_is_not_reported_as_truncation(self):
        client, http, _ = _client_returning(
            _response("nope", "stop"),
            _response("still nope", "stop"),
        )
        with patch("httpx.Client", return_value=http):
            with pytest.raises(NvidiaError) as excinfo:
                client.chat([{"role": "user", "content": "x"}], format_json=True)
        message = str(excinfo.value).lower()
        assert "malformed" in message
        assert "truncated" not in message


class TestStructuredExtractionBudget:
    def test_structured_calls_do_not_silently_inherit_the_small_default(self):
        """chat_structured passed no max_tokens, so it used chat()'s 4096 --
        not enough for an architecture pass carrying confidence and evidence
        per component, which is how DenseNet ran off the end."""
        assert STRUCTURED_MAX_TOKENS > 4096

        from pydantic import BaseModel

        class Tiny(BaseModel):
            name: str

        client, http, sent = _client_returning(_response('{"name": "x"}', "stop"))
        with patch("httpx.Client", return_value=http):
            client.chat_structured([{"role": "user", "content": "x"}], Tiny)
        assert sent[0]["max_tokens"] == STRUCTURED_MAX_TOKENS
