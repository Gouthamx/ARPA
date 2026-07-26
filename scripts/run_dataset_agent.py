"""Manual / integration runner for the Dataset Agent.

Usage:
  python scripts/run_dataset_agent.py --dataset cifar10
  python scripts/run_dataset_agent.py --paper path/to/dataset_section.txt
  python scripts/run_dataset_agent.py --paper path/to/full_paper.pdf

When ``--paper`` points at a PDF, the CLI first extracts the dataset/preprocessing
sections into a .txt file under the runs output directory, then feeds that .txt to
the Dataset Agent exactly as if it had been provided manually. Plain .txt input is
passed through unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from arpa.agents import DatasetAgent
from arpa.agents.extraction_agent import ExtractionAgent
from arpa.core.state import DatasetDescription, MethodologySpec
from arpa.tools.pdf_pipeline import PdfToTextPipeline, is_pdf


def _resolve_paper_to_txt(paper_path: Path, run_dir: Path) -> Path:
    """Return a .txt path for the agent, converting from PDF first if needed."""
    if is_pdf(paper_path):
        logger.info("PDF input detected; running PDF -> text extraction pipeline.")
        pipeline = PdfToTextPipeline()
        result = pipeline.convert(paper_path, run_dir)
        logger.info(
            "Using extracted sections: {} (kept: {})",
            result.txt_path,
            ", ".join(result.report.kept_titles) or "(none)",
        )
        return result.txt_path
    return paper_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ARPA Dataset Agent")
    parser.add_argument(
        "--paper",
        type=Path,
        help="Paper dataset section as .txt, or a full paper .pdf (auto-extracted)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Known dataset name (skips LLM extraction)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default=None,
        help="Comma-separated CHW, e.g. 3,32,32",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        default=None,
        help="Transform description from the paper",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Verify inside Docker sandbox (requires arpa-sandbox:latest)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM extraction/codegen (skeleton loader only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".arpa_runs/last_result.json"),
        help="Where to write the full result JSON",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ollama", "gemini"],
        default=None,
        help="LLM backend to use (overrides ARPA_LLM_BACKEND env var)",
    )
    args = parser.parse_args()

    if not args.paper and not args.dataset:
        parser.error("Provide --paper or --dataset")

    paper_context: str | None = None
    dataset_description: DatasetDescription | None = None

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.paper:
        if not args.paper.exists():
            parser.error(f"Paper file not found: {args.paper}")
        try:
            txt_path = _resolve_paper_to_txt(args.paper, args.output.parent)
        except (RuntimeError, FileNotFoundError) as exc:
            logger.error("Could not prepare paper input: {}", exc)
            return 2
        paper_context = txt_path.read_text(encoding="utf-8")
    elif args.dataset:
        shape = None
        if args.input_shape:
            shape = [int(x.strip()) for x in args.input_shape.split(",")]
        dataset_description = DatasetDescription(
            name=args.dataset,
            train_size=args.train_size,
            val_size=args.val_size,
            num_classes=args.num_classes,
            input_shape=shape,
            transform_description=args.transforms,
        )

    # Build methodology from components
    methodology = None
    if paper_context and not args.no_llm:
        logger.info("Running ExtractionAgent to get methodology...")
        extraction_agent = ExtractionAgent(backend=args.backend)
        methodology = extraction_agent.run(paper_context, reduce_first=True)
    elif dataset_description:
        methodology = MethodologySpec(dataset_description=dataset_description)

    agent = DatasetAgent(backend=args.backend)
    result = agent.run(
        methodology=methodology,
        use_docker=args.use_docker,
    )

    args.output.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    print("=== Dataset Agent Result ===")
    print(f"verified : {result.verified}")
    print(f"escalated: {result.escalated}")
    if result.escalation_reason:
        print(f"reason   : {result.escalation_reason}")
    if result.spec:
        print(f"dataset  : {result.spec.dataset_name}")
        print(f"registry : {result.spec.registry_source} / {result.spec.registry_id}")
        print(f"attempts : {result.verify_attempts} verification attempt(s)")
        print("\n--- Resolution log ---")
        print(result.spec.resolution_notes or "(none)")
        print("\n--- Preprocess steps ---")
        for step in result.spec.preprocess_steps:
            conf_value = step.confidence.value if step.confidence and hasattr(step.confidence, 'value') else step.confidence
            if conf_value is None:
                conf_value = "assumed"
            print(f"  [{conf_value}] {step.name}: {step.code_snippet}")
        print("\n--- Loading code ---")
        print(result.spec.loading_code)
    print(f"\nFull result saved to: {args.output.resolve()}")
    return 0 if result.verified else 1


if __name__ == "__main__":
    sys.exit(main())
