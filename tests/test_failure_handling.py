"""Tests for the four defects behind the 1/10 verification run.

Covers: timeouts driven by config, extraction failure semantics (partial
degrades / total raises), and the harness treating a total wipeout as a
retryable transient error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arpa.agents.extraction_agent import ExtractionAgent
from arpa.core.config import ARPASettings
from arpa.core.state import (
    ArchitecturePass,
    DatasetTaskPass,
    ImplementationPlanPass,
    TrainingEvalPass,
)


# --------------------------------------------------------------------------
# Defect 1 -- read timeout must come from settings, never a hardcoded 180s
# --------------------------------------------------------------------------
class TestConfiguredTimeouts:
    def test_nvidia_read_timeout_follows_settings(self, monkeypatch):
        from arpa.models.nvidia_client import NvidiaClient

        settings = ARPASettings(nvidia_api_key="k", nvidia_timeout_s=777.0)
        client = NvidiaClient(settings)

        seen: dict = {}

        class FakeClient:
            def __init__(self, timeout=None, **kw):
                seen["timeout"] = timeout

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, *a, **kw):
                raise httpx.ReadTimeout("simulated")

        monkeypatch.setattr(httpx, "Client", FakeClient)
        monkeypatch.setattr(client.rate_limiter, "wait", lambda: None)

        with pytest.raises(Exception):
            client.chat([{"role": "user", "content": "hi"}])

        assert seen["timeout"].read == 777.0, "read timeout ignored settings"
        # A connect stall is a genuine fault and must stay short.
        assert seen["timeout"].connect == 10.0

    def test_no_hardcoded_180s_timeout_remains(self):
        offenders = [
            path
            for path in Path("arpa").rglob("*.py")
            if "read=180" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, f"hardcoded 180s read timeout still in {offenders}"


# --------------------------------------------------------------------------
# Defect 2 -- partial failure degrades, total failure raises
# --------------------------------------------------------------------------
class _LLM:
    """Fails the passes named in `fail_labels`, by schema."""

    general_model = "fake"
    code_model = "fake"

    def __init__(self, fail_schemas, exc=None):
        self.fail_schemas = fail_schemas
        self.exc = exc or httpx.ReadTimeout("simulated read timeout")

    def complete_structured(self, prompt, schema, **kw):
        if schema in self.fail_schemas:
            raise self.exc
        return schema()


ALL_PASSES = [DatasetTaskPass, ArchitecturePass, TrainingEvalPass, ImplementationPlanPass]


class TestExtractionFailureSemantics:
    def test_partial_failure_still_returns_a_spec(self):
        agent = ExtractionAgent(llm=_LLM([ArchitecturePass]))
        spec = agent.run("paper text", reduce_first=False)

        assert len(spec.pass_failures) == 1
        assert spec.pass_failures[0].pass_label == "architecture"
        # Existing graceful-degradation contract is preserved.
        assert any(d.field == "architecture" for d in spec.assumptions_needed)

    def test_total_failure_raises_instead_of_hollow_spec(self):
        agent = ExtractionAgent(llm=_LLM(ALL_PASSES))

        with pytest.raises(RuntimeError) as excinfo:
            agent.run("paper text", reduce_first=False)

        message = str(excinfo.value)
        assert "all 4 extraction passes failed" in message.lower()
        # The real cause must survive, not be flattened into a generic message.
        assert "ReadTimeout" in message
        assert isinstance(excinfo.value.__cause__, httpx.ReadTimeout)

    def test_three_of_four_failing_does_not_raise(self):
        """The wipeout guard must not fire on merely-bad extraction."""
        agent = ExtractionAgent(llm=_LLM(ALL_PASSES[:3]))
        spec = agent.run("paper text", reduce_first=False)
        assert len(spec.pass_failures) == 3

    def test_failures_survive_architecture_kb_enrichment(self):
        """Enrichment sits between the failure and the merge; it must not eat it."""
        agent = ExtractionAgent(llm=_LLM([ArchitecturePass]))
        spec = agent.run("paper text", reduce_first=False)
        assert [f.pass_label for f in spec.pass_failures] == ["architecture"]


# --------------------------------------------------------------------------
# Defect 3 -- the harness must retry a total wipeout
# --------------------------------------------------------------------------
class TestHarnessRetriesWipeout:
    def test_total_extraction_failure_is_transient(self):
        import verify_codegen_agent as V

        # Deliberately free of any other transient token (no "timeout", no
        # status code): this must match on the wipeout marker alone, or the
        # test would pass for the wrong reason.
        err = RuntimeError("All 4 extraction passes failed -- dataset/task: BadGateway: x")
        assert V.is_transient(err), "wipeout must be retryable on its own marker"

    def test_wipeout_marker_tolerates_the_pass_count(self):
        """The message embeds a count; the marker must not hardcode one."""
        import verify_codegen_agent as V

        for n in (2, 4, 7):
            assert V.is_transient(RuntimeError(f"All {n} extraction passes failed -- x: E: y"))

    def test_genuine_schema_error_is_not_transient(self):
        import verify_codegen_agent as V

        assert not V.is_transient(ValueError("unknown field 'foo' in schema"))

    def test_wipeout_is_actually_retried(self):
        import verify_codegen_agent as V

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("All 4 extraction passes failed -- x: BadGateway: y")
            return "ok"

        original = V.STAGE_RETRIES
        V.STAGE_RETRIES = 2
        V.time = type("T", (), {"sleep": staticmethod(lambda _s: None)})()
        try:
            assert V.run_stage("extraction", flaky) == "ok"
            assert calls["n"] == 2
        finally:
            V.STAGE_RETRIES = original
            import time as _real_time

            V.time = _real_time
