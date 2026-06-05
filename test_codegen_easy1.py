"""Test CodeGen agent on easy1 Fashion-MNIST paper."""

from pathlib import Path

from loguru import logger

from arpa.agents.codegen_agent import CodeGenAgent


def main():
    """Run CodeGen on easy1 methodology."""
    logger.info("Starting CodeGen test on easy1 (Fashion-MNIST)")
    
    # Paths
    methodology_path = Path(".arpa_runs/methodology_easy1.txt")
    output_dir = Path(".arpa_runs/generated_easy1")
    
    if not methodology_path.exists():
        logger.error("Methodology file not found: {}", methodology_path)
        return
    
    # Run CodeGen with explicit backend
    agent = CodeGenAgent(backend="gemini")
    result = agent.run(
        methodology_path=methodology_path,
        output_dir=output_dir,
    )
    
    # Report results
    logger.info("=" * 60)
    logger.info("CODE GENERATION RESULTS")
    logger.info("=" * 60)
    
    if result.success:
        logger.success("✓ Code generation successful!")
        logger.info("Generated {} files:", len(result.files))
        for gen_file in result.files:
            status = "✓ verified" if gen_file.verified else "✗ syntax errors"
            logger.info("  - {} ({}): {}", gen_file.path, status, gen_file.purpose)
            if gen_file.syntax_errors:
                for err in gen_file.syntax_errors:
                    logger.warning("    Error: {}", err)
        
        logger.info("\nFiles written to: {}", output_dir.absolute())
        logger.info("\nGeneration log:")
        for log_entry in result.generation_log:
            logger.info("  - {}", log_entry)
    
    elif result.escalated:
        logger.error("✗ Code generation failed (escalated)")
        logger.error("Reason: {}", result.escalation_reason)
    
    else:
        logger.warning("⚠ Code generation completed with issues")
        logger.info("Generated {} files", len(result.files))
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
