"""State definition for ARPA LangGraph pipeline."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import add_messages

from arpa.agents.codegen_agent import CodegenResult, GeneratedFile
from arpa.core.confidence import ConfidenceSummary
from arpa.core.state import DatasetAgentResult, DatasetSpec, MethodologySpec


class ARPAState(TypedDict, total=False):
    """
    Complete state for the ARPA pipeline graph.
    
    This combines all agent outputs and control flow state into one unified
    state object that flows through the LangGraph nodes.
    """
    
    # ========== Input ==========
    paper_text: str  # Full paper text or reduced sections
    paper_path: str | None  # Original PDF/txt path for reference
    
    # ========== Extraction Phase ==========
    methodology: MethodologySpec | None  # Output from ExtractionAgent
    extraction_confidence: ConfidenceSummary | None  # Confidence breakdown
    extraction_complete: bool  # Flag for routing
    
    # ========== Dataset Phase ==========
    dataset_result: DatasetAgentResult | None  # Full result from DatasetAgent
    dataset_spec: DatasetSpec | None  # Shortcut to result.spec
    dataset_verified: bool  # Shortcut to result.verified
    dataset_attempts: int  # Number of verification attempts
    dataset_escalated: bool  # Escalation flag
    
    # ========== CodeGen Phase ==========
    codegen_result: CodegenResult | None  # Full result from CodeGenAgent
    generated_files: list[GeneratedFile]  # Shortcut to result.files
    codegen_success: bool  # Shortcut to result.success
    codegen_escalated: bool  # Escalation flag
    
    # ========== Control Flow ==========
    current_phase: str  # "extraction" | "dataset" | "codegen" | "verification" | "done"
    human_feedback: str | None  # Human input for review nodes
    human_approved: bool  # Approval flag
    escalation_reason: str | None  # Why escalation happened
    should_retry: bool  # Whether to retry current phase
    retry_count: int  # Number of retries for current phase
    max_retries: int  # Maximum retries before giving up
    
    # ========== Observability ==========
    messages: Annotated[list, add_messages]  # LLM interaction log
    decision_log: list[str]  # Why each routing decision was made
    error_log: list[str]  # Errors encountered during execution
    
    # ========== Final Output ==========
    success: bool  # Overall pipeline success
    output_dir: str | None  # Where files were written
