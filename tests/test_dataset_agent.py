"""Tests for Dataset Agent and registry resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from arpa.agents.dataset_agent import DatasetAgent
from arpa.core.state import DatasetDescription, MethodologySpec
from arpa.tools.dataset_tools import (
    DatasetResolution,
    DatasetResolver,
    TorchvisionResolver,
    build_loading_code_skeleton,
)
from arpa.tools.docker_tools import DatasetSandboxVerifier, VerificationExpectations


class TestDatasetResolver:
    def test_torchvision_resolves_cifar10(self):
        resolver = TorchvisionResolver()
        matches = resolver.search("CIFAR-10")
        assert matches
        assert matches[0].registry_id == "cifar10"
        assert matches[0].source == "torchvision"

    def test_fallback_hierarchy_prefers_hf_when_available(self):
        resolver = DatasetResolver()
        hf_match = DatasetResolution(
            source="huggingface",
            registry_id="cifar10",
            canonical_name="cifar10",
            metadata={},
            match_score=0.95,
            resolver_notes="test",
        )
        with patch.object(resolver._hf, "search", return_value=[hf_match]):
            with patch.object(resolver._pwc, "search", return_value=[]):
                with patch.object(resolver._tv, "search", return_value=[]):
                    with patch.object(resolver._tfds, "search", return_value=[]):
                        best, log = resolver.resolve("cifar10")
        assert best is not None
        assert best.source == "huggingface"
        assert any("HuggingFace" in line for line in log)

    def test_escalate_when_no_registry_match(self):
        resolver = DatasetResolver()
        with patch.object(resolver._hf, "search", return_value=[]):
            with patch.object(resolver._pwc, "search", return_value=[]):
                with patch.object(resolver._tv, "search", return_value=[]):
                    with patch.object(resolver._tfds, "search", return_value=[]):
                        best, log = resolver.resolve("nonexistent_dataset_xyz")
        assert best is None
        assert any("ESCALATE" in line for line in log)


class TestLoadingCodeSkeleton:
    def test_torchvision_skeleton_has_load_splits(self):
        res = DatasetResolution(
            source="torchvision",
            registry_id="cifar10",
            canonical_name="cifar10",
            metadata={},
            match_score=1.0,
            resolver_notes="",
        )
        code = build_loading_code_skeleton(res)
        assert "def load_splits" in code
        assert "CIFAR10" in code


class TestDatasetSandboxVerifier:
    def test_parse_verification_marker(self):
        verifier = DatasetSandboxVerifier()
        log = 'ARPA_VERIFY_RESULT:{"passed": true, "checks": [{"name": "x", "ok": true}], "error": null}'
        result = verifier._parse_log(log, exit_code=0)
        assert result.passed
        assert len(result.checks) == 1


class TestDatasetAgent:
    @pytest.fixture
    def mock_resolution(self):
        return DatasetResolution(
            source="torchvision",
            registry_id="cifar10",
            canonical_name="cifar10",
            metadata={},
            match_score=0.95,
            resolver_notes="torchvision match",
        )

    def test_run_with_prebuilt_description_escalates_on_resolution_failure(self):
        agent = DatasetAgent()
        desc = DatasetDescription(name="totally_fake_dataset_xyz_123")
        with patch.object(agent.resolver, "resolve", return_value=(None, ["ESCALATE"])):
            result = agent.run(dataset_description=desc)
        assert result.escalated
        assert "Could not resolve" in (result.escalation_reason or "")

    def test_run_success_path_without_docker(self, mock_resolution):
        agent = DatasetAgent()
        desc = DatasetDescription(
            name="cifar10",
            train_size=50000,
            input_shape=[3, 32, 32],
            num_classes=10,
            transform_description="normalize with mean 0.5",
        )
        skeleton = build_loading_code_skeleton(mock_resolution)
        fake_code = skeleton

        mock_verify = MagicMock()
        mock_verify.passed = True
        mock_verify.checks = [{"name": "train_split_exists", "ok": True}]
        mock_verify.error = None
        mock_verify.raw_log = 'ARPA_VERIFY_RESULT:{"passed": true, "checks": [], "error": null}'
        mock_verify.failed_checks = []

        from arpa.core.confidence import ConfidenceSummary

        with patch.object(agent.resolver, "resolve", return_value=(mock_resolution, ["ok"])):
            with patch.object(
                agent,
                "_generate_loading_code",
                return_value=(fake_code, [], ConfidenceSummary()),
            ):
                with patch.object(agent.verifier, "verify", return_value=mock_verify):
                    result = agent.run(
                        dataset_description=desc,
                        use_docker=False,
                    )

        assert result.spec is not None
        assert result.verified
        assert result.spec.registry_source == "torchvision"
        assert "load_splits" in result.spec.loading_code

    def test_run_from_methodology_spec(self, mock_resolution):
        agent = DatasetAgent()
        methodology = MethodologySpec(
            dataset_description=DatasetDescription(name="cifar10"),
        )
        with patch.object(agent.resolver, "resolve", return_value=(mock_resolution, [])):
            with patch.object(agent, "_generate_loading_code") as gen:
                from arpa.core.confidence import ConfidenceSummary

                gen.return_value = ("code", [], ConfidenceSummary())
                with patch.object(agent.verifier, "verify") as verify:
                    from arpa.tools.docker_tools import VerificationResult

                    verify.return_value = VerificationResult(passed=True, checks=[])
                    result = agent.run(
                        methodology=methodology,
                        use_docker=False,
                    )
        assert result.spec is not None

    def test_run_can_skip_loading_verification(self, mock_resolution):
        agent = DatasetAgent()
        desc = DatasetDescription(name="cifar10")

        from arpa.core.confidence import ConfidenceSummary

        with patch.object(agent.resolver, "resolve", return_value=(mock_resolution, [])):
            with patch.object(
                agent,
                "_generate_loading_code",
                return_value=("def load_splits():\n    return None, None, None\n", [], ConfidenceSummary()),
            ):
                with patch.object(agent.verifier, "verify") as verify:
                    result = agent.run(
                        dataset_description=desc,
                        verify_loading=False,
                    )

        verify.assert_not_called()
        assert result.spec is not None
        assert not result.escalated
        assert result.verify_attempts == 0
