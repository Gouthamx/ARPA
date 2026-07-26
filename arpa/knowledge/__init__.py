"""Knowledge base for standard ML components and training configurations.

Provides lookup for:
- Standard architecture components (residual blocks, attention, normalization, etc.)
- Component definitions and canonical implementations
- Future: Training standards for common domains (Phase 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


@dataclass
class ComponentKnowledge:
    """Knowledge base entry for a standard ML component."""

    name: str
    aliases: list[str]
    definition: str
    canonical_implementation: str
    parameters: dict[str, str]
    pytorch_signature: str
    forward_pass: str | None
    reference: str
    paper_url: str | None
    confidence_when_used: str
    domain_default: str | None = None


class ComponentKnowledgeBase:
    """Knowledge base for standard ML architecture components.
    
    Loads component definitions from YAML files and provides fuzzy lookup
    for enriching extracted methodology specs.
    
    Design principles:
    - Read-only: KB is for standard components that don't change
    - Fuzzy matching: Handle variations in naming (underscore, spaces, etc.)
    - Transparent: Always return source attribution for RAG values
    - Conservative: Only match well-known standard components
    """

    def __init__(self, kb_dir: Path | str | None = None) -> None:
        """Initialize knowledge base and load all component definitions.
        
        Args:
            kb_dir: Path to knowledge base directory. If None, uses default
                location at arpa/knowledge/components/
        """
        if kb_dir is None:
            kb_dir = Path(__file__).parent / "components"
        else:
            kb_dir = Path(kb_dir)

        self.kb_dir = kb_dir
        self.components: dict[str, ComponentKnowledge] = {}
        self._alias_map: dict[str, str] = {}  # alias → canonical name

        self._load_all_components()
        logger.info(
            "Loaded {} component definitions from knowledge base (kb_dir={})",
            len(self.components),
            self.kb_dir,
        )

    def _load_all_components(self) -> None:
        """Load all YAML files from the components directory."""
        if not self.kb_dir.exists():
            logger.warning(
                "Knowledge base directory not found: {}. No components loaded.",
                self.kb_dir,
            )
            return

        for yaml_file in self.kb_dir.glob("*.yaml"):
            try:
                self._load_component_file(yaml_file)
            except Exception as exc:
                logger.error(
                    "Failed to load component file {}: {}",
                    yaml_file.name,
                    exc,
                )

    def _load_component_file(self, yaml_file: Path) -> None:
        """Load component definitions from a single YAML file."""
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            logger.warning("Skipping invalid component file: {}", yaml_file.name)
            return

        for component_name, spec in data.items():
            if not isinstance(spec, dict):
                continue

            try:
                comp = ComponentKnowledge(
                    name=component_name,
                    aliases=spec.get("aliases", []),
                    definition=spec.get("definition", ""),
                    canonical_implementation=spec.get("canonical_implementation", ""),
                    parameters=spec.get("parameters", {}),
                    pytorch_signature=spec.get("pytorch_signature", ""),
                    forward_pass=spec.get("forward_pass"),
                    reference=spec.get("reference", ""),
                    paper_url=spec.get("paper_url"),
                    confidence_when_used=spec.get("confidence_when_used", "inferred"),
                    domain_default=spec.get("domain_default"),
                )

                # Store by canonical name
                normalized_name = self._normalize_key(component_name)
                self.components[normalized_name] = comp

                # Build alias map
                for alias in comp.aliases:
                    normalized_alias = self._normalize_key(alias)
                    self._alias_map[normalized_alias] = normalized_name

            except Exception as exc:
                logger.warning(
                    "Failed to parse component '{}' in {}: {}",
                    component_name,
                    yaml_file.name,
                    exc,
                )

    @staticmethod
    def _normalize_key(text: str) -> str:
        """Normalize component name/alias for fuzzy matching.
        
        Rules:
        - Convert to lowercase
        - Replace underscores, hyphens, and spaces with space
        - Strip whitespace
        - Collapse multiple spaces to single space
        
        Examples:
            "Residual_Block" → "residual block"
            "multi-head-attention" → "multi head attention"
            "BatchNorm2d" → "batchnorm2d"
        """
        return " ".join(
            text.lower()
            .replace("_", " ")
            .replace("-", " ")
            .strip()
            .split()
        )

    def lookup_component(
        self,
        name: str | None = None,
        kind: str | None = None,
    ) -> ComponentKnowledge | None:
        """Fuzzy lookup of a component by name or kind.
        
        Lookup strategy (in order):
        1. Exact match on normalized name
        2. Exact match on normalized kind
        3. Alias match on name
        4. Alias match on kind
        5. Substring match (conservative, only if query is >4 chars)
        
        Args:
            name: Component name from paper (e.g. "residual_block_1")
            kind: Component kind/type (e.g. "residual_block", "attention")
        
        Returns:
            ComponentKnowledge if found, None otherwise
        
        Examples:
            lookup_component("residual_block") → basic_block definition
            lookup_component(kind="multihead attention") → multi_head_attention
            lookup_component("my_residual_layer", "resblock") → basic_block
        """
        # Try exact match on name
        if name:
            name_norm = self._normalize_key(name)
            if name_norm in self.components:
                return self.components[name_norm]

        # Try exact match on kind
        if kind:
            kind_norm = self._normalize_key(kind)
            if kind_norm in self.components:
                return self.components[kind_norm]

        # Try alias match on name
        if name:
            name_norm = self._normalize_key(name)
            if name_norm in self._alias_map:
                canonical = self._alias_map[name_norm]
                return self.components[canonical]

        # Try alias match on kind
        if kind:
            kind_norm = self._normalize_key(kind)
            if kind_norm in self._alias_map:
                canonical = self._alias_map[kind_norm]
                return self.components[canonical]

        # Try substring matching (conservative)
        # Only match if query contains component name, not vice versa
        # This prevents "gated_cross_attention" from matching "cross_attention"
        if name and len(name) > 8:  # Longer threshold for substring matching
            name_norm = self._normalize_key(name)
            for comp_key, comp in self.components.items():
                # Only match if component key is IN the query
                # This prevents false positives like "gated cross attention" matching "cross attention"
                if len(comp_key) > 4 and comp_key in name_norm and comp_key != name_norm:
                    return comp

        if kind and len(kind) > 8:
            kind_norm = self._normalize_key(kind)
            for comp_key, comp in self.components.items():
                if len(comp_key) > 4 and comp_key in kind_norm and comp_key != kind_norm:
                    return comp

        return None

    def get_all_components(self) -> list[ComponentKnowledge]:
        """Return all loaded component definitions."""
        return list(self.components.values())

    def get_component_count(self) -> int:
        """Return the number of loaded component definitions."""
        return len(self.components)

    def search_components(self, query: str) -> list[ComponentKnowledge]:
        """Search for components matching query string.
        
        Args:
            query: Search string (searches name, aliases, definition)
        
        Returns:
            List of matching components
        """
        query_norm = self._normalize_key(query)
        matches = []

        for comp in self.components.values():
            # Check name
            if query_norm in self._normalize_key(comp.name):
                matches.append(comp)
                continue

            # Check aliases
            if any(query_norm in self._normalize_key(alias) for alias in comp.aliases):
                matches.append(comp)
                continue

            # Check definition
            if query_norm in self._normalize_key(comp.definition):
                matches.append(comp)

        return matches


# Singleton instance for easy access
_kb_instance: ComponentKnowledgeBase | None = None


def get_knowledge_base() -> ComponentKnowledgeBase:
    """Get or create singleton knowledge base instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = ComponentKnowledgeBase()
    return _kb_instance


__all__ = [
    "ComponentKnowledge",
    "ComponentKnowledgeBase",
    "get_knowledge_base",
]
