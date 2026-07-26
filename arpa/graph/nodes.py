"""Graph nodes for ARPA pipeline.

Each node wraps an existing agent and adapts it to the LangGraph state.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from arpa.agents.codegen_agent import CodeGenAgent
from arpa.agents.dataset_agent import DatasetAgent
from arpa.agents.extraction_agent import ExtractionAgent
from arpa.graph.state import ARPAState


def extraction_node(state: ARPAState) -> dict:
    """
    Extract methodology from paper using ExtractionAgent.
    
    This is the entry point - runs 4-pass extraction + RAG enrichment.
    """
    logger.info("=== EXTRACTION NODE ===")
    
    paper_text = state["paper_text"]
    
    try:
        agent = ExtractionAgent(backend=state.get("backend", "gemini"))
        spec = agent.run(paper_text, reduce_first=True)
        summary = spec.confidence_summary()
        
        logger.info(
            "Extraction complete: {}C/{}I/{}A, {} missing details",
            summary.confirmed,
            summary.inferred,
            summary.assumed,
            len(spec.assumptions_needed),
        )
        
        return {
            "methodology": spec,
            "extraction_confidence": summary,
            "extraction_complete": True,
            "current_phase": "extraction",
            "decision_log": [
                f"Extracted: {summary.confirmed} confirmed, "
                f"{summary.inferred} inferred, {summary.assumed} assumed"
            ],
            "retry_count": 0,  # Reset retry counter
        }
    
    except Exception as exc:
        logger.error("Extraction failed: {}", exc)
        return {
            "extraction_complete": False,
            "escalation_reason": f"Extraction failed: {exc}",
            "error_log": [f"Extraction error: {exc}"],
        }


def dataset_node(state: ARPAState) -> dict:
    """
    Resolve dataset and generate loading code using DatasetAgent.
    
    Takes MethodologySpec from extraction, resolves to registry,
    generates loading code, and optionally verifies in Docker.
    """
    logger.info("=== DATASET NODE ===")
    
    methodology = state["methodology"]
    use_docker = state.get("use_docker", False)
    
    if not methodology:
        return {
            "dataset_escalated": True,
            "escalation_reason": "No methodology available for dataset resolution",
        }
    
    try:
        agent = DatasetAgent(backend=state.get("backend", "gemini"))
        result = agent.run(
            methodology=methodology,
            use_docker=use_docker,
            verify_loading=use_docker,
        )
        
        logger.info(
            "Dataset resolution: {} -> {}/{}",
            result.spec.dataset_name if result.spec else "None",
            result.spec.registry_source if result.spec else "None",
            result.spec.registry_id if result.spec else "None",
        )
        
        return {
            "dataset_result": result,
            "dataset_spec": result.spec,
            "dataset_verified": result.verified,
            "dataset_attempts": result.verify_attempts,
            "dataset_escalated": result.escalated,
            "escalation_reason": result.escalation_reason,
            "current_phase": "dataset",
            "decision_log": [
                f"Dataset: {result.spec.dataset_name if result.spec else 'failed'}, "
                f"verified={result.verified}, escalated={result.escalated}"
            ],
            "retry_count": 0,  # Reset retry counter
        }
    
    except Exception as exc:
        logger.error("Dataset resolution failed: {}", exc)
        return {
            "dataset_escalated": True,
            "escalation_reason": f"Dataset resolution failed: {exc}",
            "error_log": [f"Dataset error: {exc}"],
        }


def codegen_node(state: ARPAState) -> dict:
    """
    Generate model and training code using CodeGenAgent.
    
    Takes MethodologySpec and generates model.py + train.py.
    """
    logger.info("=== CODEGEN NODE ===")
    
    methodology = state["methodology"]
    output_dir = state.get("output_dir")
    
    if not methodology:
        return {
            "codegen_escalated": True,
            "escalation_reason": "No methodology available for code generation",
        }
    
    try:
        agent = CodeGenAgent(backend=state.get("backend", "gemini"))
        result = agent.run(
            methodology=methodology,
            output_dir=output_dir,
        )
        
        logger.info(
            "CodeGen complete: {} files, success={}",
            len(result.files),
            result.success,
        )
        
        # Check syntax errors
        syntax_errors = []
        for file in result.files:
            if not file.verified:
                syntax_errors.extend(file.syntax_errors)
        
        return {
            "codegen_result": result,
            "generated_files": result.files,
            "codegen_success": result.success and not syntax_errors,
            "codegen_escalated": result.escalated,
            "escalation_reason": result.escalation_reason,
            "current_phase": "codegen",
            "decision_log": [
                f"CodeGen: {len(result.files)} files generated, "
                f"success={result.success}, syntax_errors={len(syntax_errors)}"
            ],
            "retry_count": 0,  # Reset retry counter
            "success": result.success and not syntax_errors,
        }
    
    except Exception as exc:
        logger.error("Code generation failed: {}", exc)
        return {
            "codegen_escalated": True,
            "escalation_reason": f"Code generation failed: {exc}",
            "error_log": [f"CodeGen error: {exc}"],
        }


def human_review_node(state: ARPAState) -> dict:
    """
    Pause for human review and input.
    
    This node is meant to be used with interrupt_before or interrupt_after
    so the graph pauses here. The user can then update the state with
    feedback and resume.
    
    In automated mode (no interruption), this node just logs the need for
    review and continues.
    """
    logger.warning("=== HUMAN REVIEW NODE ===")
    logger.warning("Pipeline paused for human review")
    
    spec = state.get("methodology")
    if spec:
        logger.info("Critical assumptions: {}", len(spec.assumptions_needed))
        for detail in spec.assumptions_needed:
            if detail.severity == "critical":
                logger.warning("  - {}: {}", detail.field, detail.reason)
    
    # Check if human has provided feedback
    if state.get("human_feedback"):
        logger.info("Human feedback received: {}", state["human_feedback"])
        return {
            "human_approved": True,
            "decision_log": ["Human review: approved with feedback"],
        }
    
    # In non-interactive mode, auto-approve
    if not state.get("interactive", False):
        logger.info("Non-interactive mode: auto-approving")
        return {
            "human_approved": True,
            "decision_log": ["Human review: auto-approved (non-interactive)"],
        }
    
    # Interactive mode: wait for feedback
    return {
        "human_approved": False,
        "decision_log": ["Human review: waiting for feedback"],
    }


def write_files_node(state: ARPAState) -> dict:
    """
    Write generated files to disk.
    
    This is a utility node that writes dataset_loader.py (from DatasetAgent)
    and all generated files (from CodeGenAgent) to the output directory.
    """
    logger.info("=== WRITE FILES NODE ===")
    
    output_dir = state.get("output_dir")
    if not output_dir:
        logger.warning("No output directory specified, skipping file write")
        return {}
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    written_files = []
    
    # Write dataset loader
    dataset_spec = state.get("dataset_spec")
    if dataset_spec and dataset_spec.loading_code:
        loader_path = output_path / "dataset_loader.py"
        loader_path.write_text(dataset_spec.loading_code, encoding="utf-8")
        written_files.append(str(loader_path))
        logger.info("Wrote {}", loader_path)
    
    # Write generated files
    for gen_file in state.get("generated_files", []):
        file_path = output_path / gen_file.path
        file_path.write_text(gen_file.content, encoding="utf-8")
        written_files.append(str(file_path))
        logger.info("Wrote {}", file_path)
    
    logger.success("Wrote {} files to {}", len(written_files), output_dir)
    
    return {
        "decision_log": [f"Wrote {len(written_files)} files to {output_dir}"],
        "current_phase": "done",
        "success": True,
    }
