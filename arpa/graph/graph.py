"""Main LangGraph graph construction for ARPA pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from loguru import logger

from arpa.graph.nodes import (
    codegen_node,
    dataset_node,
    extraction_node,
    human_review_node,
    write_files_node,
)
from arpa.graph.routing import codegen_router, confidence_router, dataset_router, human_review_router
from arpa.graph.state import ARPAState


def create_arpa_graph(*, checkpointer=None, interrupt_before: list[str] | None = None):
    """
    Create the ARPA pipeline state machine.
    
    Args:
        checkpointer: Optional checkpointer for persistence (default: MemorySaver())
        interrupt_before: List of node names to pause before (for human-in-the-loop)
    
    Returns:
        Compiled LangGraph app
    
    Example:
        >>> # Non-interactive mode (fully autonomous)
        >>> app = create_arpa_graph()
        >>> result = app.invoke({"paper_text": text, "output_dir": ".arpa_runs/out"})
        
        >>> # Interactive mode with human review checkpoints
        >>> app = create_arpa_graph(interrupt_before=["human_review"])
        >>> config = {"configurable": {"thread_id": "paper123"}}
        >>> # Run until human review needed
        >>> result = app.invoke({"paper_text": text}, config)
        >>> # User reviews and approves
        >>> app.update_state(config, {"human_approved": True})
        >>> # Resume from checkpoint
        >>> result = app.invoke(None, config)
    """
    # Create the graph with state schema
    workflow = StateGraph(ARPAState)
    
    # Add nodes
    workflow.add_node("extract", extraction_node)
    workflow.add_node("dataset", dataset_node)
    workflow.add_node("codegen", codegen_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("write_files", write_files_node)
    
    # Add retry nodes (same as main nodes but increment retry counter)
    def retry_extraction(state: ARPAState) -> dict:
        logger.warning("Retrying extraction (attempt {})", state.get("retry_count", 0) + 1)
        result = extraction_node(state)
        result["retry_count"] = state.get("retry_count", 0) + 1
        return result
    
    def retry_dataset(state: ARPAState) -> dict:
        logger.warning("Retrying dataset resolution (attempt {})", state.get("retry_count", 0) + 1)
        result = dataset_node(state)
        result["retry_count"] = state.get("retry_count", 0) + 1
        return result
    
    def retry_codegen(state: ARPAState) -> dict:
        logger.warning("Retrying code generation (attempt {})", state.get("retry_count", 0) + 1)
        result = codegen_node(state)
        result["retry_count"] = state.get("retry_count", 0) + 1
        return result
    
    workflow.add_node("re_extract", retry_extraction)
    workflow.add_node("retry_dataset", retry_dataset)
    workflow.add_node("retry_codegen", retry_codegen)
    
    # Set entry point
    workflow.set_entry_point("extract")
    
    # Add conditional edges with routing logic
    workflow.add_conditional_edges(
        "extract",
        confidence_router,
        {
            "dataset": "dataset",
            "human_review": "human_review",
            "re_extract": "re_extract",
        },
    )
    
    workflow.add_conditional_edges(
        "dataset",
        dataset_router,
        {
            "codegen": "codegen",
            "human_review": "human_review",
            "retry_dataset": "retry_dataset",
        },
    )
    
    workflow.add_conditional_edges(
        "codegen",
        codegen_router,
        {
            "write_files": "write_files",
            "human_review": "human_review",
            "retry_codegen": "retry_codegen",
        },
    )
    
    workflow.add_conditional_edges(
        "human_review",
        human_review_router,
        {
            "dataset": "dataset",
            "codegen": "codegen",
            "write_files": "write_files",
            "wait": "human_review",  # Loop back to wait for approval
            "end": END,
        },
    )
    
    # Retry nodes loop back to routers
    workflow.add_conditional_edges("re_extract", confidence_router, {
        "dataset": "dataset",
        "human_review": "human_review",
        "re_extract": "human_review",  # Max retries reached, escalate
    })
    
    workflow.add_conditional_edges("retry_dataset", dataset_router, {
        "codegen": "codegen",
        "human_review": "human_review",
        "retry_dataset": "human_review",  # Max retries reached, escalate
    })
    
    workflow.add_conditional_edges("retry_codegen", codegen_router, {
        "write_files": "write_files",
        "human_review": "human_review",
        "retry_codegen": "human_review",  # Max retries reached, escalate
    })
    
    # Write files always goes to END
    workflow.add_edge("write_files", END)
    
    # Compile with checkpointer
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )
    
    return app


def run_arpa_pipeline(
    paper_text: str,
    *,
    output_dir: str | Path | None = None,
    backend: str = "gemini",
    use_docker: bool = False,
    interactive: bool = False,
    max_retries: int = 2,
    thread_id: str = "default",
) -> dict:
    """
    Run the complete ARPA pipeline on a paper.
    
    This is a convenience function that creates the graph, runs it, and
    returns the final state.
    
    Args:
        paper_text: Full paper text or reduced sections
        output_dir: Where to write generated files
        backend: LLM backend ("gemini" or "ollama")
        use_docker: Whether to use Docker for dataset verification
        interactive: Whether to enable human-in-the-loop
        max_retries: Maximum retries per phase before escalating
        thread_id: Thread ID for checkpointing
    
    Returns:
        Final state dict containing all results
    
    Example:
        >>> result = run_arpa_pipeline(
        ...     paper_text=paper_content,
        ...     output_dir=".arpa_runs/paper123",
        ...     backend="gemini",
        ...     use_docker=True,
        ... )
        >>> print(f"Success: {result['success']}")
        >>> print(f"Files: {len(result['generated_files'])}")
    """
    # Create graph with optional human review breakpoints
    interrupt = ["human_review"] if interactive else []
    app = create_arpa_graph(interrupt_before=interrupt)
    
    # Prepare initial state
    initial_state: dict[str, Any] = {
        "paper_text": paper_text,
        "output_dir": str(output_dir) if output_dir else None,
        "backend": backend,
        "use_docker": use_docker,
        "interactive": interactive,
        "max_retries": max_retries,
        "retry_count": 0,
        "messages": [],
        "decision_log": [],
        "error_log": [],
        "current_phase": "start",
        "generated_files": [],
    }
    
    # Run the graph
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        logger.info("Starting ARPA pipeline (interactive={})", interactive)
        result = app.invoke(initial_state, config)
        
        # Log final state
        if result.get("success"):
            logger.success("Pipeline completed successfully")
        elif result.get("escalation_reason"):
            logger.warning("Pipeline escalated: {}", result["escalation_reason"])
        else:
            logger.warning("Pipeline completed with warnings")
        
        return result
    
    except Exception as exc:
        logger.exception("Pipeline failed with exception")
        return {
            **initial_state,
            "success": False,
            "escalation_reason": f"Pipeline exception: {exc}",
            "error_log": [str(exc)],
        }


def resume_arpa_pipeline(
    thread_id: str,
    *,
    human_feedback: str | None = None,
    human_approved: bool = False,
) -> dict:
    """
    Resume an interrupted pipeline after human review.
    
    Args:
        thread_id: Thread ID of the interrupted run
        human_feedback: Optional feedback to incorporate
        human_approved: Whether the human approved continuation
    
    Returns:
        Final state dict
    
    Example:
        >>> # Start pipeline in interactive mode
        >>> result = run_arpa_pipeline(text, interactive=True, thread_id="paper123")
        >>> # ... pipeline pauses at human_review ...
        >>> # Resume after review
        >>> result = resume_arpa_pipeline(
        ...     thread_id="paper123",
        ...     human_feedback="Use ResNet-18 not ResNet-50",
        ...     human_approved=True,
        ... )
    """
    app = create_arpa_graph(interrupt_before=["human_review"])
    config = {"configurable": {"thread_id": thread_id}}
    
    # Update state with human input
    updates = {
        "human_approved": human_approved,
    }
    if human_feedback:
        updates["human_feedback"] = human_feedback
    
    app.update_state(config, updates)
    
    # Resume execution
    logger.info("Resuming pipeline for thread_id={}", thread_id)
    result = app.invoke(None, config)
    
    return result
