"""Advanced LangGraph tests: interactive mode, checkpointing, and visualization."""

from pathlib import Path

from loguru import logger

from arpa.graph import create_arpa_graph


def test_autonomous_mode():
    """Test 1: Fully autonomous mode (no human intervention)."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Autonomous Mode")
    logger.info("="*70)
    
    from arpa.graph import run_arpa_pipeline
    
    methodology_path = Path(".arpa_runs/methodology_easy1.txt")
    paper_text = methodology_path.read_text(encoding="utf-8")
    
    result = run_arpa_pipeline(
        paper_text=paper_text,
        output_dir=".arpa_runs/test_autonomous",
        backend="gemini",
        use_docker=False,
        interactive=False,  # Fully autonomous
        max_retries=2,
        thread_id="test_autonomous",
    )
    
    assert result["success"], "Autonomous pipeline should succeed"
    assert len(result["generated_files"]) > 0, "Should generate files"
    logger.success("✓ Test 1 passed: Autonomous mode works")


def test_checkpointing():
    """Test 2: Checkpointing - run, inspect state, resume."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Checkpointing & State Inspection")
    logger.info("="*70)
    
    from langgraph.checkpoint.memory import MemorySaver
    
    methodology_path = Path(".arpa_runs/methodology_easy1.txt")
    paper_text = methodology_path.read_text(encoding="utf-8")
    
    # Create graph with checkpointer
    checkpointer = MemorySaver()
    app = create_arpa_graph(checkpointer=checkpointer)
    
    # Run pipeline
    config = {"configurable": {"thread_id": "test_checkpoint"}}
    initial_state = {
        "paper_text": paper_text,
        "output_dir": ".arpa_runs/test_checkpoint",
        "backend": "gemini",
        "use_docker": False,
        "max_retries": 2,
        "messages": [],
        "decision_log": [],
        "error_log": [],
        "generated_files": [],
    }
    
    result = app.invoke(initial_state, config)
    
    # Inspect checkpointed state
    logger.info("Inspecting checkpointed state...")
    state_snapshot = app.get_state(config)
    logger.info("Checkpoint: {}", state_snapshot.next)
    logger.info("Values keys: {}", list(state_snapshot.values.keys()))
    
    assert result["success"], "Checkpointed pipeline should succeed"
    logger.success("✓ Test 2 passed: Checkpointing works")


def test_conditional_routing():
    """Test 3: Test conditional routing logic."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Conditional Routing")
    logger.info("="*70)
    
    from arpa.core.confidence import ConfidenceLevel, ConfidenceSummary
    from arpa.core.state import CodegenMissingDetail, MethodologySpec
    from arpa.graph.routing import confidence_router
    
    # Test case 1: High confidence → proceed to dataset
    summary1 = ConfidenceSummary()
    summary1.record(ConfidenceLevel.CONFIRMED)
    summary1.record(ConfidenceLevel.CONFIRMED)
    summary1.record(ConfidenceLevel.INFERRED)
    
    state1 = {
        "extraction_confidence": summary1,
        "methodology": MethodologySpec(assumptions_needed=[]),
        "retry_count": 0,
    }
    
    route1 = confidence_router(state1)
    assert route1 == "dataset", f"Expected 'dataset' but got '{route1}'"
    logger.info("✓ High confidence → dataset")
    
    # Test case 2: Too many assumptions → human review
    summary2 = ConfidenceSummary()
    summary2.record(ConfidenceLevel.ASSUMED)
    summary2.record(ConfidenceLevel.ASSUMED)
    summary2.record(ConfidenceLevel.ASSUMED)
    summary2.record(ConfidenceLevel.CONFIRMED)
    
    state2 = {
        "extraction_confidence": summary2,
        "methodology": MethodologySpec(assumptions_needed=[]),
        "retry_count": 0,
    }
    
    route2 = confidence_router(state2)
    assert route2 == "human_review", f"Expected 'human_review' but got '{route2}'"
    logger.info("✓ Low confidence → human_review")
    
    # Test case 3: Critical missing → re-extract
    summary3 = ConfidenceSummary()
    summary3.record(ConfidenceLevel.CONFIRMED)
    
    missing3 = CodegenMissingDetail(
        field="model_architecture",
        reason="Not specified in paper",
        severity="critical",
    )
    
    state3 = {
        "extraction_confidence": summary3,
        "methodology": MethodologySpec(assumptions_needed=[missing3]),
        "retry_count": 0,
        "max_retries": 2,
    }
    
    route3 = confidence_router(state3)
    assert route3 == "re_extract", f"Expected 're_extract' but got '{route3}'"
    logger.info("✓ Critical missing → re_extract")
    
    logger.success("✓ Test 3 passed: Conditional routing works correctly")


def test_retry_logic():
    """Test 4: Test retry and escalation logic."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Retry & Escalation")
    logger.info("="*70)
    
    from arpa.graph.routing import dataset_router
    
    # Test case 1: Verified → proceed to codegen
    state1 = {
        "dataset_verified": True,
        "dataset_escalated": False,
    }
    route1 = dataset_router(state1)
    assert route1 == "codegen", f"Expected 'codegen' but got '{route1}'"
    logger.info("✓ Verified → codegen")
    
    # Test case 2: Escalated with retries → retry
    state2 = {
        "dataset_verified": False,
        "dataset_escalated": True,
        "retry_count": 0,
        "max_retries": 2,
    }
    route2 = dataset_router(state2)
    assert route2 == "retry_dataset", f"Expected 'retry_dataset' but got '{route2}'"
    logger.info("✓ Escalated (retries left) → retry_dataset")
    
    # Test case 3: Escalated, out of retries → human review
    state3 = {
        "dataset_verified": False,
        "dataset_escalated": True,
        "retry_count": 2,
        "max_retries": 2,
    }
    route3 = dataset_router(state3)
    assert route3 == "human_review", f"Expected 'human_review' but got '{route3}'"
    logger.info("✓ Escalated (no retries) → human_review")
    
    logger.success("✓ Test 4 passed: Retry logic works correctly")


def test_graph_visualization():
    """Test 5: Generate graph visualization."""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Graph Visualization")
    logger.info("="*70)
    
    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False
        logger.warning("PIL not available, skipping graph visualization")
    
    app = create_arpa_graph()
    
    # Get mermaid diagram
    try:
        mermaid = app.get_graph().draw_mermaid()
        logger.info("Mermaid diagram generated ({} chars)", len(mermaid))
        
        # Save to file
        mermaid_path = Path(".arpa_runs/graph_diagram.mmd")
        mermaid_path.write_text(mermaid, encoding="utf-8")
        logger.info("Saved mermaid diagram to: {}", mermaid_path)
        
        # Try to generate PNG if PIL available
        if has_pil:
            try:
                png_data = app.get_graph().draw_mermaid_png()
                png_path = Path(".arpa_runs/graph_diagram.png")
                png_path.write_bytes(png_data)
                logger.success("Saved graph PNG to: {}", png_path)
            except Exception as exc:
                logger.warning("Could not generate PNG: {}", exc)
        
        logger.success("✓ Test 5 passed: Graph visualization works")
    
    except Exception as exc:
        logger.error("Graph visualization failed: {}", exc)


def main():
    """Run all advanced tests."""
    logger.info("="*70)
    logger.info("LangGraph Advanced Tests")
    logger.info("="*70)
    
    tests = [
        ("Autonomous Mode", test_autonomous_mode),
        ("Checkpointing", test_checkpointing),
        ("Conditional Routing", test_conditional_routing),
        ("Retry Logic", test_retry_logic),
        ("Graph Visualization", test_graph_visualization),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as exc:
            logger.error("Test failed: {}", name)
            logger.exception(exc)
            failed += 1
    
    logger.info("\n" + "="*70)
    logger.info("TEST RESULTS")
    logger.info("="*70)
    logger.info("Passed: {} / {}", passed, len(tests))
    logger.info("Failed: {}", failed)
    
    if failed == 0:
        logger.success("✓ All tests passed!")
        return 0
    else:
        logger.error("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
