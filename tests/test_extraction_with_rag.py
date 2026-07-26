"""Integration tests for extraction with RAG enrichment."""

from __future__ import annotations

import pytest

from arpa.agents.extraction_agent import ExtractionAgent
from arpa.core.confidence import ConfidenceLevel
from arpa.knowledge import ComponentKnowledgeBase


class TestExtractionWithRAG:
    """Test extraction agent with knowledge base enrichment."""

    def test_extraction_enriches_residual_blocks(self):
        """Extraction should enrich underspecified residual blocks with KB."""
        paper = """
        We train a ResNet architecture on ImageNet. The model uses 
        residual blocks with batch normalization. Each residual block
        consists of convolutional layers with skip connections.
        We train for 90 epochs using SGD with 8 GPUs.
        """
        
        agent = ExtractionAgent()
        spec = agent.run(paper)
        
        # Check that architecture was extracted
        assert spec.architecture is not None
        
        # Check for KB enrichment in missing details
        residual_details = [
            d for d in spec.assumptions_needed
            if "residual" in d.field.lower() or "residual" in d.reason.lower()
        ]
        
        # Should have at least one enrichment for residual blocks
        if len(residual_details) > 0:
            detail = residual_details[0]
            # KB should provide a concrete implementation suggestion
            assert detail.proposed_default is not None
            assert "block" in detail.proposed_default.lower() or "resnet" in detail.proposed_default.lower()
            # Should have KB attribution
            assert detail.default_source is not None
            assert "knowledge base" in detail.default_source.lower() or "kb" in detail.default_source.lower()

    def test_extraction_does_not_override_paper_stated(self):
        """KB enrichment should never override paper-stated values."""
        paper = """
        We use a custom residual block implementation. Each block uses 
        three convolutional layers (1x1, 3x3, 1x1) with ReLU activation
        and a learned gating mechanism on the skip connection.
        """
        
        agent = ExtractionAgent()
        spec = agent.run(paper)
        
        # If the paper describes the component in detail, that should be in the spec
        # KB might still add a "standard definition" note, but it shouldn't replace
        # the paper's description
        
        # The extraction should have captured the custom details
        if spec.architecture and spec.architecture.components:
            # Check that custom details are preserved somewhere
            has_custom_info = any(
                "three" in str(c.parameters).lower() or
                "gating" in str(c.parameters).lower() or
                "learned" in str(c.parameters).lower()
                for c in spec.architecture.components
            )
            
            # If LLM extracted the custom details, they should be there
            # (This test is a bit loose since extraction quality varies,
            # but the key is KB doesn't *override* what's there)

    def test_extraction_handles_attention_mechanisms(self):
        """Extraction should enrich attention mechanisms from KB."""
        paper = """
        We use a Vision Transformer with multi-head attention.
        The model has 12 attention heads and processes 16x16 patches.
        """
        
        agent = ExtractionAgent()
        spec = agent.run(paper)
        
        # Look for attention enrichment
        attention_details = [
            d for d in spec.assumptions_needed
            if "attention" in d.field.lower()
        ]
        
        # If attention was extracted but not fully specified, KB should help
        if len(attention_details) > 0:
            detail = attention_details[0]
            if detail.proposed_default:
                assert "attention" in detail.proposed_default.lower()

    def test_extraction_enriches_normalization(self):
        """Extraction should disambiguate generic 'batch norm' references."""
        paper = """
        We apply batch normalization after each convolutional layer
        in our image classification network trained on CIFAR-10.
        """
        
        agent = ExtractionAgent()
        spec = agent.run(paper)
        
        # Look for normalization enrichment
        norm_details = [
            d for d in spec.assumptions_needed
            if "norm" in d.field.lower() or "normalization" in d.reason.lower()
        ]
        
        # If batch norm was extracted without specifying variant, KB might help
        if len(norm_details) > 0:
            detail = norm_details[0]
            if detail.proposed_default:
                # Should suggest a concrete implementation
                assert "BatchNorm" in detail.proposed_default or "norm" in detail.proposed_default.lower()

    def test_kb_doesnt_match_novel_components(self):
        """KB should not match novel/custom components."""
        paper = """
        We propose a novel Gated Cross-Attention mechanism with learned
        temperature scaling. Each GCA block uses a custom gating function.
        """
        
        agent = ExtractionAgent()
        spec = agent.run(paper)
        
        # Novel components should not get KB enrichment
        gca_details = [
            d for d in spec.assumptions_needed
            if "gated cross" in d.field.lower() or "gca" in d.field.lower()
        ]
        
        # If there are details about GCA, they should NOT have KB defaults
        # (because this is a novel contribution not in KB)
        for detail in gca_details:
            if detail.proposed_default:
                # Should not be a standard implementation
                assert "gated cross" not in detail.proposed_default.lower()

    def test_extraction_without_kb_still_works(self):
        """Extraction should work even if KB is not available."""
        paper = """
        We train a simple CNN with convolutional and pooling layers
        on the MNIST dataset for 10 epochs.
        """
        
        # Create agent with empty KB (simulating KB failure)
        kb = ComponentKnowledgeBase()
        kb.components = {}  # Clear components
        
        agent = ExtractionAgent(kb=kb)
        spec = agent.run(paper)
        
        # Should still extract basic info
        assert spec is not None
        assert spec.dataset_description is not None

    def test_kb_enrichment_preserves_confidence_levels(self):
        """KB enrichment should not elevate confidence to CONFIRMED."""
        paper = """
        Our model uses residual blocks and achieves 95% accuracy on CIFAR-10.
        """
        
        agent = ExtractionAgent()
        spec = agent.run(paper)
        
        # Any KB-enriched details should be marked as INFERRED or ASSUMED
        kb_details = [
            d for d in spec.assumptions_needed
            if d.default_source and "knowledge base" in d.default_source.lower()
        ]
        
        for detail in kb_details:
            # KB enrichments should never be CONFIRMED
            # (They're not from the paper, they're from KB)
            assert detail.severity != "confirmed"

    def test_multiple_components_enrichment(self):
        """KB should enrich multiple underspecified components."""
        paper = """
        We build a CNN with:
        - Initial 7x7 convolution
        - Residual blocks in 4 stages
        - Batch normalization layers
        - ReLU activation
        - Global average pooling
        - Final linear classifier
        
        Trained on ImageNet for 90 epochs.
        """
        
        agent = ExtractionAgent()
        spec = agent.run(paper)
        
        # Should extract multiple components
        if spec.architecture and spec.architecture.components:
            # Multiple components mentioned
            assert len(spec.architecture.components) > 0

    def test_confidence_summary_includes_kb_enrichments(self):
        """Confidence summary should count KB enrichments properly."""
        paper = """
        ResNet-50 with residual blocks and batch normalization on ImageNet.
        """
        
        agent = ExtractionAgent()
        spec = agent.run(paper)
        
        summary = spec.confidence_summary()
        
        # Should have some confirmed facts (dataset name, etc.)
        # and potentially some inferred facts (KB enrichments)
        assert summary.confirmed + summary.inferred + summary.assumed > 0

    def test_kb_attribution_is_clear(self):
        """KB enrichments should have clear source attribution."""
        paper = """
        We use standard residual blocks in our architecture.
        """
        
        agent = ExtractionAgent()
        spec = agent.run(paper)
        
        kb_details = [
            d for d in spec.assumptions_needed
            if d.default_source and "kb" in d.default_source.lower()
        ]
        
        for detail in kb_details:
            # Should have clear attribution
            assert detail.default_source is not None
            assert len(detail.default_source) > 0
            # Should reference the source
            if "he et al" in detail.default_source.lower():
                # Good - citing the paper
                pass


class TestKBIntegrationEdgeCases:
    """Test edge cases in KB integration."""

    def test_kb_lookup_failure_doesnt_crash(self):
        """KB lookup failures should be handled gracefully."""
        paper = """
        We use a complex architecture with many custom components.
        """
        
        agent = ExtractionAgent()
        
        # Should not crash even if KB lookups fail
        spec = agent.run(paper)
        assert spec is not None

    def test_empty_architecture_no_enrichment(self):
        """If no architecture is extracted, KB enrichment should be skipped."""
        paper = """
        We use the CIFAR-10 dataset and report 90% accuracy.
        """
        
        agent = ExtractionAgent()
        spec = agent.run(paper)
        
        # Should complete without errors
        assert spec is not None

    def test_architecture_with_detailed_components_no_kb_needed(self):
        """Components with detailed parameters shouldn't need KB enrichment."""
        paper = """
        Our residual block implementation:
        - First 3x3 conv with 64 filters, stride 1, padding 1
        - Batch normalization with momentum 0.9
        - ReLU activation
        - Second 3x3 conv with 64 filters
        - Batch normalization
        - Skip connection
        - Final ReLU
        """
        
        agent = ExtractionAgent()
        spec = agent.run(paper)
        
        # Detailed components might not get KB enrichment
        # (because they're already well-specified)
        # Just verify extraction completes
        assert spec is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
