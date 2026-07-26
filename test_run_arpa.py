"""Simple test: Run ARPA on Fashion-MNIST paper"""

from pathlib import Path
from arpa.graph import run_arpa_pipeline
from arpa.tools.pdf_pipeline import PdfToTextPipeline

# Extract text from Fashion-MNIST PDF
pdf_path = Path("papers/easy1_1708.07747.pdf")

if not pdf_path.exists():
    print(f"❌ PDF not found: {pdf_path}")
    print("Please make sure the PDF exists")
    exit(1)

print(f"📄 Extracting text from: {pdf_path}")
pipeline = PdfToTextPipeline()
pdf_result = pipeline.convert(pdf_path, Path(".arpa_runs/test_run"))

print(f"✓ Extracted {len(pdf_result.text):,} characters")
print()

# Run ARPA pipeline
print("🤖 Running ARPA pipeline...")
print("   Backend: NVIDIA NIM")
print("   Docker: DISABLED (testing without Docker)")
print()

result = run_arpa_pipeline(
    paper_text=pdf_result.text,
    output_dir=".arpa_runs/test_run/output",
    backend="nvidia",
    use_docker=False,  # Explicitly disable Docker
    interactive=False,
)

# Show results
print("\n" + "="*80)
print("RESULTS:")
print("="*80)

if result.get("success"):
    print("✅ SUCCESS!")
    print(f"\n📁 Generated Files ({len(result['generated_files'])}):")
    for f in result['generated_files']:
        print(f"   - {f.path} ({len(f.content)} chars)")
    print(f"\n💾 Saved to: {result['output_dir']}")
else:
    print("❌ FAILED")
    print(f"   Reason: {result.get('escalation_reason', 'Unknown')}")

print("="*80)
