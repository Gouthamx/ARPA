"""Tests for the component knowledge base."""

from __future__ import annotations

import pytest

from arpa.knowledge import ComponentKnowledge, ComponentKnowledgeBase


class TestComponentKnowledgeBase:
    """Test suite for ComponentKnowledgeBase."""

    def test_kb_loads_components(self):
        """KB should load component definitions from YAML files."""
        kb = ComponentKnowledgeBase()
        
        assert kb.get_component_count() > 0, "KB should load at least one component"
        components = kb.get_all_components()
        assert len(components) > 0
        assert all(isinstance(c, ComponentKnowledge) for c in components)

    def test_exact_match_lookup(self):
        """KB should find components by exact name match."""
        kb = ComponentKnowledgeBase()
        
        # Residual block
        comp = kb.lookup_component("basic_block")
        assert comp is not None
        assert "residual" in comp.definition.lower()
        assert "torchvision" in comp.canonical_implementation.lower()
        
        # Attention
        comp = kb.lookup_component("multi_head_attention")
        assert comp is not None
        assert "attention" in comp.definition.lower()

    def test_alias_match_lookup(self):
        """KB should find components by alias."""
        kb = ComponentKnowledgeBase()
        
        # "residual block" is an alias for basic_block
        comp = kb.lookup_component("residual block")
        assert comp is not None
        assert "residual" in comp.definition.lower()
        
        # "batch norm" is an alias for batch_normalization
        comp = kb.lookup_component("batch norm")
        assert comp is not None
        assert "batch" in comp.definition.lower() or "BatchNorm" in comp.canonical_implementation

    def test_kind_lookup(self):
        """KB should find components by kind when name doesn't match."""
        kb = ComponentKnowledgeBase()
        
        # Looking up by kind
        comp = kb.lookup_component(name="my_residual_layer", kind="residual_block")
        assert comp is not None
        assert "residual" in comp.definition.lower()
        
        comp = kb.lookup_component(name="attention_1", kind="multi head attention")
        assert comp is not None
        assert "attention" in comp.definition.lower()

    def test_normalization_lookup(self):
        """KB should handle various normalization names."""
        kb = ComponentKnowledgeBase()
        
        # BatchNorm
        comp = kb.lookup_component("batch_normalization")
        assert comp is not None
        assert "BatchNorm2d" in comp.canonical_implementation
        
        # LayerNorm
        comp = kb.lookup_component("layer_normalization")
        assert comp is not None
        assert "LayerNorm" in comp.canonical_implementation
        
        # GroupNorm
        comp = kb.lookup_component("group_normalization")
        assert comp is not None
        assert "GroupNorm" in comp.canonical_implementation

    def test_activation_lookup(self):
        """KB should find activation functions."""
        kb = ComponentKnowledgeBase()
        
        activations = ["relu", "gelu", "swish", "sigmoid", "tanh"]
        for activation in activations:
            comp = kb.lookup_component(activation)
            assert comp is not None, f"Should find {activation}"
            assert comp.canonical_implementation
            assert comp.definition

    def test_pooling_lookup(self):
        """KB should find pooling layers."""
        kb = ComponentKnowledgeBase()
        
        # Max pooling
        comp = kb.lookup_component("max_pooling")
        assert comp is not None
        assert "MaxPool" in comp.canonical_implementation
        
        # Global average pooling
        comp = kb.lookup_component("global average pooling")
        assert comp is not None
        assert "Adaptive" in comp.canonical_implementation or "mean" in comp.forward_pass.lower()

    def test_case_insensitive_lookup(self):
        """KB lookup should be case insensitive."""
        kb = ComponentKnowledgeBase()
        
        # Different cases should all work
        variants = ["ReLU", "relu", "RELU", "ReLu"]
        for variant in variants:
            comp = kb.lookup_component(variant)
            assert comp is not None, f"Should find {variant}"

    def test_underscore_hyphen_handling(self):
        """KB should normalize underscores and hyphens."""
        kb = ComponentKnowledgeBase()
        
        # These should all match the same component
        variants = [
            "multi_head_attention",
            "multi-head-attention",
            "multi head attention",
            "multihead attention",
        ]
        
        results = [kb.lookup_component(v) for v in variants]
        assert all(r is not None for r in results), "All variants should match"
        
        # They should all return the same component
        assert len(set(r.name for r in results)) == 1

    def test_no_match_returns_none(self):
        """KB should return None when no match is found."""
        kb = ComponentKnowledgeBase()
        
        # Novel/custom components should not match
        comp = kb.lookup_component("my_custom_novel_attention_mechanism_with_gating")
        assert comp is None
        
        # Component names that are too different from standards
        comp = kb.lookup_component("xyz_custom_layer_unique")
        assert comp is None

    def test_component_has_required_fields(self):
        """All loaded components should have required fields."""
        kb = ComponentKnowledgeBase()
        
        for comp in kb.get_all_components():
            assert comp.name
            assert comp.definition
            assert comp.canonical_implementation
            assert comp.reference
            assert isinstance(comp.aliases, list)
            assert isinstance(comp.parameters, dict)

    def test_search_components(self):
        """KB should support searching across components."""
        kb = ComponentKnowledgeBase()
        
        # Search for attention-related components
        results = kb.search_components("attention")
        assert len(results) > 0
        assert all("attention" in c.name.lower() or "attention" in c.definition.lower() for c in results)
        
        # Search for normalization
        results = kb.search_components("normalization")
        assert len(results) > 0

    def test_substring_match_conservative(self):
        """KB should do substring matching only for longer queries."""
        kb = ComponentKnowledgeBase()
        
        # Long enough for substring match - "basic block" is in "resnet basic block"
        comp = kb.lookup_component("resnet basic block")
        assert comp is not None  # Should find basic_block via alias
        
        # Test that it matches basic patterns
        comp = kb.lookup_component("multi head self attention")
        # Should match either via alias or substring

    def test_multiple_component_types(self):
        """KB should load components from multiple YAML files."""
        kb = ComponentKnowledgeBase()
        
        # Should have residual blocks
        assert kb.lookup_component("basic_block") is not None
        
        # Should have attention mechanisms
        assert kb.lookup_component("multi_head_attention") is not None
        
        # Should have normalizations
        assert kb.lookup_component("batch_normalization") is not None
        
        # Should have pooling
        assert kb.lookup_component("max_pooling") is not None
        
        # Should have activations
        assert kb.lookup_component("relu") is not None


class TestComponentKnowledge:
    """Test suite for ComponentKnowledge dataclass."""

    def test_component_creation(self):
        """Should create ComponentKnowledge with all fields."""
        comp = ComponentKnowledge(
            name="test_component",
            aliases=["alias1", "alias2"],
            definition="Test definition",
            canonical_implementation="torch.nn.TestModule",
            parameters={"param1": "description"},
            pytorch_signature="TestModule(param1)",
            forward_pass="output = input",
            reference="Test et al., 2024",
            paper_url="https://example.com",
            confidence_when_used="inferred",
            domain_default="test domain",
        )
        
        assert comp.name == "test_component"
        assert len(comp.aliases) == 2
        assert comp.definition
        assert comp.canonical_implementation
        assert comp.confidence_when_used == "inferred"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
