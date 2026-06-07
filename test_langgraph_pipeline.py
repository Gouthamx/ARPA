"""Test the LangGraph ARPA pipeline on easy1 Fashion-MNIST paper."""

from pathlib import Path

from loguru import logger

from arpa.graph import run_arpa_pipeline


def main():
    """Run LangGraph pipeline on easy1 methodology."""
    logger.info("="*70)
    logger.info("Testing LangGraph ARPA Pipeline")
    logger.info("="*70)
    
    # Paths
    methodology_path = Path(".arpa_runs/methodology_easy1.txt")
    output_dir = Path(".arpa_runs/langgraph_easy1")
    
    if not methodology_path.exists():
        logger.error("Methodology file not found: {}", methodology_path)
        logger.info("Run ExtractionAgent first to generate it")
        return 1
    
    # Load paper text
    paper_text = methodology_path.read_text(encoding="utf-8")
    logger.info("Loaded paper text ({} chars)", len(paper_text))
    
    # Run pipeline
    logger.info("\n" + "="*70)
    logger.info("Running LangGraph Pipeline")
    logger.info("="*70 + "\n")
    
    result = run_arpa_pipeline(
        paper_text=paper_text,
        output_dir=output_dir,
        backend="gemini",
        use_docker=False,  # Skip Docker for faster testing
        interactive=False,  # Fully autonomous
        max_retries=2,
        thread_id="easy1_test",
    )
    
    # Report results
    logger.info("\n" + "="*70)
    logger.info("PIPELINE RESULTS")
    logger.info("="*70)
    
    logger.info("Success: {}", result.get("success", False))
    logger.info("Current phase: {}", result.get("current_phase", "unknown"))
    
    # Extraction results
    if result.get("methodology"):
        spec = result["methodology"]
        logger.info("\n--- Extraction ---")
        logger.info("Dataset: {}", spec.dataset_description.name if spec.dataset_description else "None")
        logger.info("Assumptions needed: {}", len(spec.assumptions_needed))
        if result.get("extraction_confidence"):
            conf = result["extraction_confidence"]
            logger.info("Confidence: {}C / {}I / {}A", conf.confirmed, conf.inferred, conf.assumed)
    
    # Dataset results
    if result.get("dataset_spec"):
        spec = result["dataset_spec"]
        logger.info("\n--- Dataset ---")
        logger.info("Dataset: {}", spec.dataset_name)
        logger.info("Registry: {} / {}", spec.registry_source, spec.registry_id)
        logger.info("Verified: {}", result.get("dataset_verified", False))
    
    # CodeGen results
    if result.get("generated_files"):
        files = result["generated_files"]
        logger.info("\n--- CodeGen ---")
        logger.info("Generated {} files:", len(files))
        for f in files:
            status = "✓" if f.verified else "✗"
            logger.info("  {} {}: {}", status, f.path, f.purpose)
    
    # Decision log
    if result.get("decision_log"):
        logger.info("\n--- Decision Log ---")
        for decision in result["decision_log"]:
            logger.info("  • {}", decision)
    
    # Error log
    if result.get("error_log"):
        logger.warning("\n--- Error Log ---")
        for error in result["error_log"]:
            logger.warning("  ! {}", error)
    
    # Escalation
    if result.get("escalation_reason"):
        logger.warning("\n--- Escalation ---")
        logger.warning("Reason: {}", result["escalation_reason"])
    
    logger.info("\n" + "="*70)
    if result.get("success"):
        logger.success("✓ Pipeline completed successfully!")
        logger.info("Files written to: {}", output_dir)
    else:
        logger.warning("⚠ Pipeline completed with issues")
    logger.info("="*70)
    
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
