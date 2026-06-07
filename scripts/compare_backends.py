"""Compare Ollama vs Gemini on the same dataset extraction task."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from arpa.agents import DatasetAgent
from arpa.agents.extraction_agent import ExtractionAgent
from arpa.core.state import DatasetDescription, MethodologySpec


def run_backend(backend: str, paper_path: Path | None, dataset_desc: DatasetDescription | None):
    """Run the agent with a specific backend and return timing + result."""
    print(f"\n{'='*60}")
    print(f"Running with backend: {backend.upper()}")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        paper_context = paper_path.read_text(encoding="utf-8") if paper_path else None
        
        # Build methodology
        methodology = None
        if paper_context:
            print(f"  Running ExtractionAgent...")
            extraction_agent = ExtractionAgent(backend=backend)
            methodology = extraction_agent.run(paper_context, reduce_first=True)
        elif dataset_desc:
            methodology = MethodologySpec(dataset_description=dataset_desc)
        
        print(f"  Running DatasetAgent...")
        agent = DatasetAgent(backend=backend)
        result = agent.run(
            methodology=methodology,
            use_docker=False,  # Skip docker for speed comparison
        )
        elapsed = time.time() - start
        
        print(f"\n✓ Completed in {elapsed:.2f}s")
        print(f"  Verified: {result.verified}")
        print(f"  Escalated: {result.escalated}")
        
        if result.spec:
            print(f"  Dataset: {result.spec.dataset_name}")
            print(f"  Registry: {result.spec.registry_source} / {result.spec.registry_id}")
            print(f"  Preprocess steps: {len(result.spec.preprocess_steps)}")
            print(f"\n  Confidence breakdown:")
            print(f"    Confirmed: {result.preprocess_confidence.confirmed}")
            print(f"    Inferred: {result.preprocess_confidence.inferred}")
            print(f"    Assumed: {result.preprocess_confidence.assumed}")
            
            print(f"\n  Preprocessing steps:")
            for step in result.spec.preprocess_steps:
                print(f"    [{step.confidence.value}] {step.name}")
                print(f"      Code: {step.code_snippet[:80]}...")
                if step.source:
                    print(f"      Source: {step.source}")
        
        if result.escalation_reason:
            print(f"  Escalation reason: {result.escalation_reason}")
        
        return {
            "backend": backend,
            "elapsed_s": elapsed,
            "verified": result.verified,
            "escalated": result.escalated,
            "dataset_name": result.spec.dataset_name if result.spec else None,
            "num_preprocess_steps": len(result.spec.preprocess_steps) if result.spec else 0,
            "confidence": result.preprocess_confidence.as_dict(),
            "result": result,
        }
    except Exception as exc:
        elapsed = time.time() - start
        print(f"\n✗ Failed after {elapsed:.2f}s: {exc}")
        import traceback
        traceback.print_exc()
        return {
            "backend": backend,
            "elapsed_s": elapsed,
            "error": str(exc),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare Ollama vs Gemini backends")
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path("tests/fixtures/paper_cifar10_excerpt.txt"),
        help="Paper excerpt for LLM extraction",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Skip LLM extraction, use known dataset name",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="ollama,gemini",
        help="Comma-separated list of backends to test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".arpa_runs/comparison.json"),
        help="Where to save comparison results",
    )
    args = parser.parse_args()
    
    # Prepare input
    paper_path = args.paper if args.paper.exists() and not args.dataset else None
    dataset_desc = None
    if args.dataset:
        dataset_desc = DatasetDescription(name=args.dataset)
    
    if not paper_path and not dataset_desc:
        print("Error: Provide --paper (existing file) or --dataset")
        return 1
    
    # Run each backend
    backends = [b.strip() for b in args.backends.split(",")]
    results = []
    
    for backend in backends:
        result = run_backend(backend, paper_path, dataset_desc)
        results.append(result)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for r in results:
        backend = r["backend"]
        if "error" in r:
            print(f"\n{backend.upper()}: FAILED")
            print(f"  Error: {r['error']}")
        else:
            print(f"\n{backend.upper()}: {r['elapsed_s']:.2f}s")
            print(f"  Dataset: {r['dataset_name']}")
            print(f"  Steps: {r['num_preprocess_steps']}")
            print(f"  Confidence: {r['confidence']}")
            print(f"  Verified: {r['verified']}, Escalated: {r['escalated']}")
    
    # Save detailed results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "result"}
        if "result" in r:
            entry["result_json"] = r["result"].model_dump()
        serializable.append(entry)
    
    args.output.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    print(f"\nDetailed results saved to: {args.output.resolve()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
