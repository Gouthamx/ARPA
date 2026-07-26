"""
Simple ARPA Pipeline Verification Script

Tests the complete ARPA pipeline end-to-end:
1. PDF → Text extraction
2. Methodology extraction (ExtractionAgent)
3. Dataset resolution (DatasetAgent)
4. Code generation (CodeGenAgent)

Outputs a simple text report showing which steps succeeded/failed.

Usage:
    python verify_codegen_agent.py --quick                    # Fast test (2 papers, ~30 min with Ollama)
    python verify_codegen_agent.py --backend ollama            # All 10 papers (~3 hours with Ollama)
    python verify_codegen_agent.py --backend gemini            # All 10 papers (~15 min with Gemini)
    python verify_codegen_agent.py --backend openrouter        # All 10 papers (~15 min with OpenRouter)
    python verify_codegen_agent.py --backend nvidia            # All 10 papers (~15 min with NVIDIA NIM)
    python verify_codegen_agent.py --backend groq              # All 10 papers (~15 min with Groq)
    python verify_codegen_agent.py --papers easy1 easy2        # Test specific papers
    
Time estimates (Ollama on 15.6GB RAM):
    - Quick mode (2 papers):  ~30 minutes
    - Full test (10 papers):  ~2.5-3 hours
    - Gemini is 10x faster but requires API key
"""

import argparse
import concurrent.futures
import json
import time
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.error

from loguru import logger

from arpa.agents.codegen_agent import CodeGenAgent
from arpa.agents.dataset_agent import DatasetAgent
from arpa.agents.extraction_agent import ExtractionAgent
from arpa.core.config import get_settings
from arpa.tools.pdf_pipeline import PdfToTextPipeline

# Hard timeout (seconds) for any single agent call. If a call hangs past
# this, we give up on that STAGE (not the whole run) and record it as a
# timeout failure, instead of the script hanging until someone hits Ctrl+C.
STAGE_TIMEOUT_SECONDS = 240


def run_with_timeout(func, timeout_seconds, *args, **kwargs):
    """
    Run func(*args, **kwargs) with a hard wall-clock timeout.

    This exists purely so a hung LLM call can never block the whole
    verification run -- no changes to arpa's agents are needed for this.
    If the call doesn't return in time, we raise TimeoutError and move on.
    Note: the underlying thread may keep running in the background if the
    call truly never returns -- that's fine for a verification script,
    it doesn't block anything else, and the process exits with it at the end.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout_seconds)


# Paper definitions
PAPERS = {
    "easy1": {"title": "Fashion-MNIST", "arxiv_id": "1708.07747", "difficulty": "Easy"},
    "easy2": {"title": "EMNIST", "arxiv_id": "1702.05373", "difficulty": "Easy"},
    "easy3": {"title": "Kuzushiji-MNIST", "arxiv_id": "1812.01718", "difficulty": "Easy"},
    "medium1": {"title": "ResNet", "arxiv_id": "1512.03385", "difficulty": "Medium"},
    "medium2": {"title": "VGG", "arxiv_id": "1409.1556", "difficulty": "Medium"},
    "medium3": {"title": "DenseNet", "arxiv_id": "1608.06993", "difficulty": "Medium"},
    "medium4": {"title": "MobileNetV2", "arxiv_id": "1801.04381", "difficulty": "Medium"},
    "hard1": {"title": "SimCLR", "arxiv_id": "2002.05709", "difficulty": "Hard"},
    "hard2": {"title": "Bilinear CNN", "arxiv_id": "1504.07889", "difficulty": "Hard"},
    "hard3": {"title": "DeiT", "arxiv_id": "2012.12877", "difficulty": "Hard"},
}


def download_paper(arxiv_id: str, output_path: Path) -> bool:
    """Download paper from arXiv if not present."""
    if output_path.exists():
        return True
    
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            content = response.read()
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)
        return True
        
    except Exception as e:
        logger.error(f"Download failed for {arxiv_id}: {e}")
        return False


def test_paper(paper_key: str, paper_meta: dict, papers_dir: Path, output_dir: Path, backend: str) -> dict:
    """Test ARPA pipeline on a single paper."""
    result = {
        "paper": paper_key,
        "title": paper_meta["title"],
        "difficulty": paper_meta["difficulty"],
        "pdf_found": False,
        "extraction_success": False,
        "dataset_success": False,
        "codegen_success": False,
        "error": None,
        "time_seconds": 0,
    }
    
    start = time.time()
    
    try:
        # Check PDF
        print(f"    [1/5] Checking PDF...")
        pdf_path = papers_dir / f"{paper_key}_{paper_meta['arxiv_id']}.pdf"
        if not pdf_path.exists():
            result["error"] = f"PDF not found: {pdf_path.name}"
            return result
        result["pdf_found"] = True
        print(f"    ✓ PDF found ({pdf_path.stat().st_size // 1024}KB)")
        
        # Step 1: PDF → Text
        print(f"    [2/5] Converting PDF to text...")
        try:
            pipeline = PdfToTextPipeline()
            pdf_result = pipeline.convert(pdf_path, output_dir / paper_key)
            print(f"    ✓ Extracted {len(pdf_result.text):,} characters")
        except Exception as e:
            result["error"] = f"PDF extraction failed: {str(e)[:100]}"
            print(f"    ✗ PDF extraction failed: {e}")
            return result
        
        # Step 2: Extract methodology (SLOWEST - 4 LLM passes)
        print(f"    [3/5] Extracting methodology...")
        try:
            extraction_agent = ExtractionAgent(backend=backend)
            methodology = run_with_timeout(
                extraction_agent.run, STAGE_TIMEOUT_SECONDS, pdf_result.text
            )

            if methodology and methodology.dataset_description:
                result["extraction_success"] = True
                print(f"    ✓ Extracted methodology")
            else:
                result["error"] = "Extraction returned empty methodology"
                print(f"    ✗ Extraction failed: empty result")
                return result
        except concurrent.futures.TimeoutError:
            result["error"] = f"Extraction timed out after {STAGE_TIMEOUT_SECONDS}s"
            print(f"    ✗ Extraction TIMED OUT (>{STAGE_TIMEOUT_SECONDS}s) -- moving on")
            return result
        except Exception as e:
            result["error"] = f"Extraction failed: {str(e)[:100]}"
            print(f"    ✗ Extraction failed: {e}")
            return result

        # Step 3: Resolve dataset
        print(f"    [4/5] Resolving dataset...")
        try:
            dataset_agent = DatasetAgent(backend=backend)
            dataset_result = run_with_timeout(
                dataset_agent.run, STAGE_TIMEOUT_SECONDS, methodology
            )

            if dataset_result and dataset_result.spec:
                result["dataset_success"] = True
                print(f"    ✓ Dataset resolved: {dataset_result.spec.registry_source}")
            else:
                result["error"] = "Dataset agent returned empty result"
                print(f"    ✗ Dataset resolution failed: empty result")
                # Don't return - continue to codegen to see if it can work anyway
        except concurrent.futures.TimeoutError:
            result["error"] = f"Dataset resolution timed out after {STAGE_TIMEOUT_SECONDS}s"
            print(f"    ✗ Dataset resolution TIMED OUT (>{STAGE_TIMEOUT_SECONDS}s) -- continuing to CodeGen anyway")
        except Exception as e:
            result["error"] = f"Dataset resolution failed: {str(e)[:100]}"
            print(f"    ✗ Dataset resolution failed: {e}")
            # Don't return - continue to codegen to see if it can work anyway

        # Step 4: Generate code
        print(f"    [5/5] Generating code...")
        try:
            codegen_agent = CodeGenAgent(backend=backend)
            codegen_result = run_with_timeout(
                codegen_agent.run, STAGE_TIMEOUT_SECONDS, methodology
            )

            if codegen_result and codegen_result.files and len(codegen_result.files) > 0:
                result["codegen_success"] = True
                print(f"    ✓ Generated {len(codegen_result.files)} files")
            else:
                if not result["error"]:  # Only set if no previous error
                    result["error"] = "CodeGen returned no files"
                print(f"    ✗ Code generation failed: no files generated")
        except concurrent.futures.TimeoutError:
            if not result["error"]:
                result["error"] = f"CodeGen timed out after {STAGE_TIMEOUT_SECONDS}s"
            print(f"    ✗ CodeGen TIMED OUT (>{STAGE_TIMEOUT_SECONDS}s) -- moving to next paper")
        except Exception as e:
            if not result["error"]:  # Only set if no previous error
                result["error"] = f"CodeGen failed: {str(e)[:100]}"
            print(f"    ✗ Code generation failed: {e}")

    except Exception as e:
        # Catch-all: whatever goes wrong for this paper (including anything
        # not anticipated above), record it and let the run continue to the
        # next paper instead of crashing the whole script.
        result["error"] = f"Unexpected error: {str(e)[:100]}"
        print(f"    ✗ UNEXPECTED ERROR: {e}")
        logger.exception(f"Unexpected error testing {paper_key}")
    
    result["time_seconds"] = time.time() - start
    return result


def generate_summary(results: list[dict], output_path: Path):
    """Generate simple text summary."""
    lines = []
    
    lines.append("=" * 80)
    lines.append("ARPA PIPELINE VERIFICATION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Papers Tested: {len(results)}")
    lines.append("")
    
    # Count successes
    extraction_ok = sum(1 for r in results if r["extraction_success"])
    dataset_ok = sum(1 for r in results if r["dataset_success"])
    codegen_ok = sum(1 for r in results if r["codegen_success"])
    fully_working = sum(1 for r in results if r["extraction_success"] and r["dataset_success"] and r["codegen_success"])
    
    lines.append("OVERALL RESULTS:")
    lines.append(f"  Complete Pipeline Working: {fully_working}/{len(results)} papers")
    lines.append(f"  Extraction Agent:          {extraction_ok}/{len(results)} papers")
    lines.append(f"  Dataset Agent:             {dataset_ok}/{len(results)} papers")
    lines.append(f"  CodeGen Agent:             {codegen_ok}/{len(results)} papers")
    lines.append("")
    
    # Per-paper breakdown
    lines.append("PAPER-BY-PAPER BREAKDOWN:")
    lines.append("-" * 80)
    
    for r in results:
        status = "✓ PASS" if (r["extraction_success"] and r["dataset_success"] and r["codegen_success"]) else "✗ FAIL"
        lines.append(f"\n[{r['paper']}] {r['title']} ({r['difficulty']}) - {status} ({r['time_seconds']:.1f}s)")
        
        if r.get("error"):
            lines.append(f"  ERROR: {r['error']}")
            continue
        
        lines.append(f"  PDF Found:       {'✓' if r['pdf_found'] else '✗'}")
        lines.append(f"  Extraction:      {'✓' if r['extraction_success'] else '✗'}")
        lines.append(f"  Dataset:         {'✓' if r['dataset_success'] else '✗'}")
        lines.append(f"  CodeGen:         {'✓' if r['codegen_success'] else '✗'}")
    
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY:")
    lines.append(f"The ARPA pipeline is {'WORKING CORRECTLY' if fully_working == len(results) else 'PARTIALLY WORKING'}")
    lines.append(f"{fully_working} out of {len(results)} papers completed successfully")
    lines.append("=" * 80)
    
    # Write to file
    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    
    # Print to console
    print("\n" + text)


def main():
    global STAGE_TIMEOUT_SECONDS

    parser = argparse.ArgumentParser(description="Verify ARPA pipeline end-to-end")
    parser.add_argument("--backend", choices=["ollama", "gemini", "openrouter", "nvidia", "groq"], default=None)
    parser.add_argument("--papers", nargs="+", default=None, help="Specific papers to test")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Quick test with only 2 papers (easy1, medium1)")
    parser.add_argument("--stage-timeout", type=int, default=STAGE_TIMEOUT_SECONDS,
                         help=f"Max seconds to wait for any single agent call before giving up "
                              f"on that stage and moving on (default: {STAGE_TIMEOUT_SECONDS})")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any previously saved partial results and start over from scratch")
    args = parser.parse_args()
    STAGE_TIMEOUT_SECONDS = args.stage_timeout

    settings = get_settings()
    if args.backend:
        settings.llm_backend = args.backend
    
    papers_dir = Path("papers")
    output_dir = Path(".arpa_runs") / "codegen_verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        stale_partial = output_dir / "partial_results.jsonl"
        if stale_partial.exists():
            stale_partial.unlink()
            print("--fresh: cleared previous partial results, starting over.\n")
    
    # Download missing papers
    if not args.skip_download:
        print("Checking papers...")
        for paper_key, paper_meta in PAPERS.items():
            pdf_path = papers_dir / f"{paper_key}_{paper_meta['arxiv_id']}.pdf"
            if not pdf_path.exists():
                print(f"  Downloading {paper_key}...")
                download_paper(paper_meta['arxiv_id'], pdf_path)
                time.sleep(3)  # Be nice to arXiv
    
    # Select papers to test
    if args.quick:
        papers_to_test = ["easy1", "medium1"]  # Quick test with just 2 papers
        print("⚡ QUICK MODE: Testing only 2 papers (easy1, medium1)")
    elif args.papers:
        papers_to_test = args.papers
    else:
        papers_to_test = list(PAPERS.keys())
    
    print(f"\n{'=' * 80}")
    print(f"TESTING ARPA PIPELINE")
    print(f"{'=' * 80}")
    print(f"Backend: {settings.llm_backend}")
    print(f"Papers: {len(papers_to_test)}")
    print(f"Per-stage timeout: {STAGE_TIMEOUT_SECONDS}s (use --stage-timeout to change)")
    
    if settings.llm_backend == "ollama":
        est_time = len(papers_to_test) * 15  # ~15 minutes per paper with Ollama
        print(f"\n⚠ WARNING: Ollama is SLOW!")
        print(f"  Estimated time: {est_time} minutes ({est_time/60:.1f} hours)")
        print(f"  Each paper takes ~15-20 minutes with Ollama on 15.6GB RAM")
        print(f"  Consider using --quick flag or switching to Gemini/OpenRouter\n")
    elif settings.llm_backend in ("gemini", "openrouter", "nvidia", "groq"):
        est_time = len(papers_to_test) * 1.5  # ~1.5 minutes per paper with API
        print(f"  Estimated time: {est_time:.1f} minutes\n")
    
    print("")

    # Incremental results file -- every paper's result is written here the
    # moment it finishes, so a crash, hang, or manual stop never loses
    # progress already made. Re-running the script picks up where it left
    # off instead of starting over.
    partial_results_path = output_dir / "partial_results.jsonl"
    results = []
    already_done = set()

    if partial_results_path.exists():
        with open(partial_results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prior_result = json.loads(line)
                except json.JSONDecodeError:
                    continue
                results.append(prior_result)
                already_done.add(prior_result["paper"])
        if already_done:
            print(f"Resuming: {len(already_done)} paper(s) already completed "
                  f"({', '.join(sorted(already_done))}), skipping them.\n")

    # Test each paper
    for idx, paper_key in enumerate(papers_to_test, 1):
        if paper_key not in PAPERS:
            print(f"[{idx}/{len(papers_to_test)}] Unknown paper: {paper_key}, skipping...")
            continue

        if paper_key in already_done:
            print(f"[{idx}/{len(papers_to_test)}] Skipping {paper_key} (already completed in a prior run)")
            continue

        paper_meta = PAPERS[paper_key]
        print(f"[{idx}/{len(papers_to_test)}] Testing {paper_key}: {paper_meta['title']}...")

        result = test_paper(paper_key, paper_meta, papers_dir, output_dir, settings.llm_backend)
        results.append(result)

        # Save immediately -- don't wait until the whole batch finishes.
        with open(partial_results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        # Show quick status
        if result["codegen_success"]:
            print(f"  ✓ Pipeline completed successfully")
        else:
            print(f"  ✗ Pipeline failed at some step (recorded, continuing)")

        # Delay between papers
        if idx < len(papers_to_test):
            time.sleep(5)

    # Generate summary from ALL results (resumed + newly run)
    report_path = output_dir / f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    generate_summary(results, report_path)
    print(f"\n✓ Report saved to: {report_path}")
    print(f"✓ Raw per-paper results saved to: {partial_results_path}")


if __name__ == "__main__":
    main()