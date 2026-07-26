"""
Example: How to Run the Full ARPA Pipeline

This script shows how to run ARPA on a paper to generate reproduction code.
"""

from pathlib import Path
from arpa.graph import run_arpa_pipeline
from arpa.tools.pdf_pipeline import PdfToTextPipeline

def run_arpa_on_paper(pdf_path: str):
    """
    Run ARPA pipeline on a single paper PDF.
    
    Steps:
    1. Extract text from PDF
    2. Extract methodology (4 LLM passes)
    3. Resolve dataset from registries
    4. Generate reproduction code (model.py + train.py)
    5. Save generated files to output directory
    
    Args:
        pdf_path: Path to PDF file
    """
    
    # Step 1: Convert PDF to text
    print(f"📄 Converting PDF: {pdf_path}")
    pipeline = PdfToTextPipeline()
    pdf_result = pipeline.convert(Path(pdf_path), Path(".arpa_runs/temp"))
    
    print(f"✓ Extracted {len(pdf_result.text):,} characters\n")
    
    # Step 2-5: Run full ARPA pipeline
    print("🤖 Running ARPA pipeline...")
    print("   This will:")
    print("   1. Extract methodology (4 LLM passes)")
    print("   2. Resolve dataset")
    print("   3. Generate code (model.py + train.py)")
    print()
    
    result = run_arpa_pipeline(
        paper_text=pdf_result.text,
        output_dir=".arpa_runs/output",
        backend="nvidia",  # Options: "nvidia", "gemini", "ollama", "groq", "openrouter"
        use_docker=False,  # Set to True if you have Docker running
        interactive=False,  # Set to True for human-in-the-loop mode
    )
    
    # Show results
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    
    if result.get("success"):
        print("✅ SUCCESS: Pipeline completed!")
        print(f"\n📁 Generated files ({len(result['generated_files'])} files):")
        for f in result['generated_files']:
            print(f"   - {f.path} ({len(f.content)} chars)")
        
        if result.get("output_dir"):
            print(f"\n💾 Files saved to: {result['output_dir']}")
    else:
        print("❌ FAILED")
        if result.get("escalation_reason"):
            print(f"   Reason: {result['escalation_reason']}")
        if result.get("error_log"):
            print(f"   Errors: {result['error_log']}")
    
    print("\n" + "="*80)
    
    return result


def run_arpa_on_text(paper_text: str, output_dir: str = ".arpa_runs/output"):
    """
    Run ARPA pipeline on paper text (if you already have the text).
    
    Args:
        paper_text: Full text content of the paper
        output_dir: Where to save generated files
    """
    
    print("🤖 Running ARPA pipeline on provided text...")
    
    result = run_arpa_pipeline(
        paper_text=paper_text,
        output_dir=output_dir,
        backend="nvidia",  # Change to your preferred backend
        use_docker=False,  # Docker is optional
        interactive=False,
    )
    
    if result.get("success"):
        print(f"✅ Success! Generated {len(result['generated_files'])} files")
        print(f"💾 Saved to: {output_dir}")
    else:
        print(f"❌ Failed: {result.get('escalation_reason', 'Unknown error')}")
    
    return result


if __name__ == "__main__":
    """
    Example usage - Run ARPA on Fashion-MNIST paper
    """
    
    # Option 1: Run on a PDF file
    pdf_path = "papers/easy1_1708.07747.pdf"
    
    if Path(pdf_path).exists():
        print("Running ARPA on Fashion-MNIST paper...")
        result = run_arpa_on_paper(pdf_path)
    else:
        print(f"❌ PDF not found: {pdf_path}")
        print("\nTo run this example:")
        print("1. Make sure you have a PDF in papers/")
        print("2. Or use run_arpa_on_text() if you have the text already")
    
    # Option 2: Run on text directly (if you have the text)
    # paper_text = Path("path/to/paper.txt").read_text()
    # result = run_arpa_on_text(paper_text)
