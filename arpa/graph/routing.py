"""Conditional routing logic for ARPA graph."""

from __future__ import annotations

from loguru import logger

from arpa.graph.state import ARPAState


def confidence_router(state: ARPAState) -> str:
    """
    Route after extraction based on confidence and missing details.
    
    Decision tree:
    1. If critical missing details → re-extract with more context
    2. If too many assumptions (low confidence) → human review
    3. Otherwise → proceed to dataset resolution
    """
    summary = state.get("extraction_confidence")
    spec = state.get("methodology")
    
    if not summary or not spec:
        logger.error("Extraction incomplete, escalating")
        return "human_review"
    
    # Check for critical missing details
    critical_missing = [
        d for d in spec.assumptions_needed
        if d.severity == "critical"
    ]
    
    if critical_missing and state.get("retry_count", 0) < state.get("max_retries", 2):
        logger.warning("Found {} critical missing details, re-extracting", len(critical_missing))
        for detail in critical_missing[:3]:  # Log first 3
            logger.warning("  - {}: {}", detail.field, detail.reason)
        return "re_extract"
    
    # Check confidence balance
    # If assumed > (confirmed + inferred), we're guessing too much
    if summary.assumed > (summary.confirmed + summary.inferred):
        logger.warning(
            "Low extraction confidence: {}A > {}C + {}I, routing to human review",
            summary.assumed,
            summary.confirmed,
            summary.inferred,
        )
        return "human_review"
    
    # Good enough to proceed
    logger.info(
        "Extraction confidence acceptable: {}C/{}I/{}A",
        summary.confirmed,
        summary.inferred,
        summary.assumed,
    )
    return "dataset"


def dataset_router(state: ARPAState) -> str:
    """
    Route after dataset resolution based on verification status.
    
    Decision tree:
    1. If verified successfully → proceed to codegen
    2. If escalated and out of retries → human review
    3. If failed but have retries → retry dataset resolution
    4. Otherwise → proceed to codegen (best effort)
    """
    escalated = state.get("dataset_escalated", False)
    verified = state.get("dataset_verified", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    if verified:
        logger.success("Dataset verified, proceeding to codegen")
        return "codegen"
    
    if escalated:
        if retry_count >= max_retries:
            logger.error("Dataset escalated after {} retries, routing to human review", retry_count)
            return "human_review"
        else:
            logger.warning("Dataset escalated, retry {}/{}", retry_count + 1, max_retries)
            return "retry_dataset"
    
    # Not verified but not escalated - proceed with warning
    logger.warning("Dataset not verified but proceeding to codegen (best effort)")
    return "codegen"


def codegen_router(state: ARPAState) -> str:
    """
    Route after code generation based on success status.
    
    Decision tree:
    1. If successful → write files and finish
    2. If escalated and out of retries → human review
    3. If failed but have retries → retry codegen
    4. Otherwise → write files anyway (best effort)
    """
    success = state.get("codegen_success", False)
    escalated = state.get("codegen_escalated", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    if success:
        logger.success("CodeGen successful, writing files")
        return "write_files"
    
    if escalated:
        if retry_count >= max_retries:
            logger.error("CodeGen escalated after {} retries, routing to human review", retry_count)
            return "human_review"
        else:
            logger.warning("CodeGen escalated, retry {}/{}", retry_count + 1, max_retries)
            return "retry_codegen"
    
    # Not successful but not escalated - write files anyway
    logger.warning("CodeGen not fully successful but writing files (best effort)")
    return "write_files"


def human_review_router(state: ARPAState) -> str:
    """
    Route after human review based on approval status.
    
    Decision tree:
    1. If approved → continue to next phase
    2. If not approved and interactive → wait (should pause here)
    3. Otherwise → escalate/end
    """
    approved = state.get("human_approved", False)
    current_phase = state.get("current_phase", "extraction")
    
    if approved:
        # Continue based on where we came from
        if current_phase == "extraction":
            logger.info("Human approved extraction, proceeding to dataset")
            return "dataset"
        elif current_phase == "dataset":
            logger.info("Human approved dataset, proceeding to codegen")
            return "codegen"
        elif current_phase == "codegen":
            logger.info("Human approved codegen, writing files")
            return "write_files"
        else:
            logger.warning("Human approved from unknown phase, proceeding to dataset")
            return "dataset"
    
    # Not approved - if interactive, this should pause
    if state.get("interactive", False):
        logger.info("Waiting for human approval")
        return "wait"
    
    # Non-interactive and not approved - end
    logger.warning("Human review required but not approved, ending pipeline")
    return "end"
