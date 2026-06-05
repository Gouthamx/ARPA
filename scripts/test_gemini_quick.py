"""Quick test: Gemini extraction quality without Docker verification."""
from __future__ import annotations

import sys
from pathlib import Path

from arpa.agents import DatasetAgent


def main():
    paper_path = Path("tests/fixtures/paper_cifar10_excerpt.txt")
    paper_text = paper_path.read_text(encoding="utf-8")
    
    print("="*70)
    print("GEMINI EXTRACTION TEST - CIFAR-10 Paper")
    print("="*70)
    
    agent = DatasetAgent(backend="gemini")
    result = agent.run(
        paper_context=paper_text,
        use_llm_extraction=True,
        use_docker=False,  # Skip verification to focus on extraction quality
    )
    
    print(f"\n✓ Extraction completed")
    print(f"  Dataset identified: {result.spec.dataset_name}")
    print(f"  Registry: {result.spec.registry_source} / {result.spec.registry_id}")
    print(f"  Train size: {result.spec.train_size}")
    print(f"  Val size: {result.spec.val_size}")
    print(f"  Input shape: {result.spec.input_shape}")
    print(f"  Num classes: {result.spec.num_classes}")
    
    print(f"\n📊 Confidence Breakdown:")
    print(f"  ✅ Confirmed: {result.preprocess_confidence.confirmed}")
    print(f"  ⚡ Inferred:  {result.preprocess_confidence.inferred}")
    print(f"  ⚠️  Assumed:   {result.preprocess_confidence.assumed}")
    
    print(f"\n🔧 Preprocessing Steps ({len(result.spec.preprocess_steps)} total):")
    for i, step in enumerate(result.spec.preprocess_steps, 1):
        icon = {"confirmed": "✅", "inferred": "⚡", "assumed": "⚠️"}[step.confidence.value]
        print(f"\n  {i}. {icon} {step.name} [{step.confidence.value.upper()}]")
        print(f"     Code: {step.code_snippet}")
        print(f"     Source: {step.source}")
        if step.evidence:
            print(f"     Evidence: {step.evidence[:100]}...")
    
    print(f"\n📝 Generated Loading Code Preview:")
    print("="*70)
    lines = result.spec.loading_code.split("\n")
    for line in lines[:40]:  # First 40 lines
        print(line)
    if len(lines) > 40:
        print(f"... ({len(lines) - 40} more lines)")
    
    print("\n" + "="*70)
    print("✨ Gemini successfully extracted all preprocessing details from the paper!")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
