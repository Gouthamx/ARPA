"""Verify CodeGenAgent on 10 papers: extraction -> code generation -> output checks.

Each paper is processed through:
  1. Paper text load (PDF via PdfToTextPipeline, or direct .txt excerpt)
  2. ExtractionAgent -> MethodologySpec
  3. CodeGenAgent -> model.py + train.py
  4. Structural verification (syntax, required files, non-empty output)

Usage:
    # Text fixtures only (no PDF download needed):
    python benchmark/verify_codegen.py --only-fixtures

    # All 10 papers (run benchmark/download_papers.py first for PDFs):
    python benchmark/verify_codegen.py

    # Single paper:
    python benchmark/verify_codegen.py --only fixture_cifar10

    # Dry-run with mocked LLM (no API key required):
    python benchmark/verify_codegen.py --mock --only-fixtures
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger

from arpa.agents.codegen_agent import CodeGenAgent
from arpa.agents.extraction_agent import ExtractionAgent
from arpa.tools.pdf_pipeline import PdfToTextPipeline
from benchmark.codegen_papers import CODEGEN_PAPERS, CodegenPaper
from benchmark.codegen_verify import (
    CodegenVerification,
    print_verification_report,
    verify_codegen_result,
)


def _load_paper_text(paper: CodegenPaper, run_dir: Path) -> str:
    if paper.kind == "text":
        return paper.source.read_text(encoding="utf-8")
    conversion = PdfToTextPipeline().convert(paper.source, run_dir)
    return conversion.text


def _run_one(
    paper: CodegenPaper,
    *,
    backend: str,
    out_root: Path,
    mock: bool,
) -> CodegenVerification:
    print(f"\n>>> [{paper.key}] ({paper.level}) {paper.description}", flush=True)
    start = time.time()
    run_dir = out_root / paper.key
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not paper.source.exists():
            raise FileNotFoundError(
                f"Missing {paper.source}. "
                + ("Run: python benchmark/download_papers.py" if paper.kind == "pdf" else "")
            )

        print(f"    [1/3] Loading paper ({paper.kind}) ...", flush=True)
        paper_text = _load_paper_text(paper, run_dir)
        print(f"          {len(paper_text)} chars", flush=True)

        print(f"    [2/3] ExtractionAgent ...", flush=True)
        if mock:
            methodology = _mock_methodology_for_paper(paper)
        else:
            methodology = ExtractionAgent(backend=backend).run(paper_text, reduce_first=True)

        dataset_name = None
        if methodology.dataset_description:
            dataset_name = methodology.dataset_description.name
            print(f"          dataset='{dataset_name}'", flush=True)

        print(f"    [3/3] CodeGenAgent ...", flush=True)
        
        # STEP 1 & 2: Log methodology and check sufficiency
        logger.info("=" * 80)
        logger.info(f"[{paper.key}] CodeGen Input (Methodology):")
        logger.info("=" * 80)
        try:
            methodology_dict = methodology.dict()
            logger.info(json.dumps(methodology_dict, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Could not serialize methodology: {e}")
            logger.info(f"Methodology: {methodology}")
        logger.info("")
        
        # Check sufficiency
        from arpa.core.codegen_readiness import check_codegen_readiness, format_readiness_report
        is_ready, gaps = check_codegen_readiness(methodology)
        readiness_report = format_readiness_report(methodology, gaps)
        logger.info(f"[{paper.key}] {readiness_report}")
        if not is_ready:
            logger.warning(f"[{paper.key}] ⚠️  Proceeding with {len(gaps)} sufficiency gap(s)")
        logger.info("=" * 80)
        logger.info("")
        
        codegen_out = run_dir / "generated"
        if mock:
            agent = CodeGenAgent(llm=_MockCodegenLLM(paper))
        else:
            agent = CodeGenAgent(backend=backend)

        result = agent.run(
            methodology=methodology,
            output_dir=codegen_out,
        )
        
        # Log CodeGen output
        logger.info(f"[{paper.key}] CodeGen Output:")
        logger.info(f"  Files: {len(result.files)}")
        logger.info(f"  Escalated: {result.escalated}")
        if result.escalated:
            logger.info(f"  Reason: {result.escalation_reason}")
        for f in result.files:
            logger.info(f"    -> {f.path}: verified={f.verified}, size={len(f.content)} chars")
        logger.info("")

        for f in result.files:
            print(f"          -> {f.path} verified={f.verified} ({len(f.content)} chars)", flush=True)

        vr = verify_codegen_result(
            paper.key,
            result,
            output_dir=codegen_out,
            elapsed_s=time.time() - start,
            dataset_name=dataset_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Verification failed for {}", paper.key)
        vr = CodegenVerification(
            paper_key=paper.key,
            success=False,
            escalated=True,
            escalation_reason=str(exc),
            files=[],
            elapsed_s=time.time() - start,
            error=str(exc),
        )

    status = "PASS" if vr.passed else "FAIL"
    print(f"<<< [{paper.key}] {status}  score={vr.score:.0f}/100  "
          f"time={vr.elapsed_s:.1f}s", flush=True)
    return vr


def _mock_methodology_for_paper(paper: CodegenPaper):
    """Build a minimal MethodologySpec for mock runs without an LLM."""
    from arpa.core.confidence import ConfidenceField, ConfidenceLevel
    from arpa.core.state import (
        ArchitectureSpec,
        BenchmarkExperimentSpec,
        DatasetDescription,
        EvaluationSpec,
        MethodologySpec,
        TrainingSpec,
    )

    name = paper.dataset_aliases[0].replace("-", " ").title()
    desc = DatasetDescription(
        name=name,
        train_size=60000,
        test_size=10000,
        input_shape=[3, 32, 32],
        num_classes=10,
    )
    spec = MethodologySpec(
        dataset_description=desc,
        architecture=ArchitectureSpec(
            model_name=ConfidenceField(value="SimpleCNN", confidence=ConfidenceLevel.CONFIRMED),
            notes="Mock architecture for verification.",
        ),
        training=TrainingSpec(
            optimizer=ConfidenceField(value="Adam", confidence=ConfidenceLevel.CONFIRMED),
            learning_rate=ConfidenceField(value=0.001, confidence=ConfidenceLevel.CONFIRMED),
            batch_size=ConfidenceField(value=64, confidence=ConfidenceLevel.CONFIRMED),
            epochs=ConfidenceField(value=10, confidence=ConfidenceLevel.CONFIRMED),
        ),
        evaluation=EvaluationSpec(
            metric_name=ConfidenceField(value="accuracy", confidence=ConfidenceLevel.CONFIRMED),
        ),
    )
    if paper.is_benchmark:
        spec.benchmark_experiments = [
            BenchmarkExperimentSpec(
                model_name="LinearSVC",
                parameters={"C": 1.0},
                dataset_metric=0.84,
                metric_name="test_accuracy",
            ),
        ]
    return spec


class _MockCodegenLLM:
    """Deterministic LLM stub for offline verification."""

    general_model = "mock-general"
    code_model = "mock-code"

    def __init__(self, paper: CodegenPaper) -> None:
        self.paper = paper
        self._call = 0

    @staticmethod
    def extract_json(text: str) -> dict:
        import json as _json

        return _json.loads(text)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        format_json: bool = False,
    ) -> str:
        import json as _json

        self._call += 1
        ds = self.paper.dataset_aliases[0]
        if self._call == 1:
            code = _BENCHMARK_MODEL_PY if self.paper.is_benchmark else _MODEL_PY.format(
                dataset=ds, num_classes=10
            )
        else:
            code = _TRAIN_PY
        return _json.dumps({"code": code})


_MODEL_PY = '''\
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """CNN for {dataset} ({num_classes} classes)."""

    def __init__(self, num_classes: int = {num_classes}) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
'''

_BENCHMARK_MODEL_PY = '''\
from sklearn.svm import LinearSVC
import numpy as np

def load_and_preprocess_data():
    """Load benchmark data (mock stub for verification)."""
    X = np.random.rand(100, 784)
    y = np.random.randint(0, 10, 100)
    return X, y

def train_and_evaluate_svc():
    X, y = load_and_preprocess_data()
    model = LinearSVC(C=1.0)
    model.fit(X, y)
    return model.score(X, y)

def main():
    acc = train_and_evaluate_svc()
    print(f"Accuracy: {acc:.4f}")
'''

_TRAIN_PY = '''\
import argparse
from model import SimpleCNN

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    model = SimpleCNN()
    print(f"Training for {{args.epochs}} epochs with {{model.__class__.__name__}}")

if __name__ == "__main__":
    main()
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CodeGenAgent on 10 papers")
    parser.add_argument("--backend", default="groq", choices=["gemini", "ollama", "groq", "nvidia", "openrouter"])
    parser.add_argument("--only", default=None, help="Run a single paper key")
    parser.add_argument(
        "--only-fixtures",
        action="store_true",
        help="Run only the 4 text fixture papers (no PDF download needed)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mocked LLM (no API key); still runs real agent logic",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/codegen_verification_results.json"),
    )
    args = parser.parse_args()

    if args.only:
        papers = [p for p in CODEGEN_PAPERS if p.key == args.only]
        if not papers:
            logger.error("Unknown paper key: {}", args.only)
            return 1
    elif args.only_fixtures:
        papers = [p for p in CODEGEN_PAPERS if p.kind == "text"]
    else:
        papers = list(CODEGEN_PAPERS)

    out_root = Path(".arpa_runs/codegen_verify")
    print("=" * 78, flush=True)
    print(f"CodeGenAgent verification — {len(papers)} paper(s)", flush=True)
    print(f"Backend: {'mock' if args.mock else args.backend}", flush=True)
    print("=" * 78, flush=True)

    results: list[CodegenVerification] = []
    for idx, paper in enumerate(papers, 1):
        print(f"\n[paper {idx}/{len(papers)}]", flush=True)
        results.append(
            _run_one(paper, backend=args.backend, out_root=out_root, mock=args.mock)
        )

    print_verification_report(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable = [r.to_dict() for r in results]
    args.output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    logger.info("Wrote verification results to {}", args.output.resolve())

    passed = sum(1 for r in results if r.passed)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
