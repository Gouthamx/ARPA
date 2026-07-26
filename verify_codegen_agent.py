"""
ARPA Pipeline Verification Script

Runs the complete ARPA pipeline end-to-end over a benchmark of 10 papers:

    1. PDF download (from arXiv)  ->  2. PDF -> text extraction
    3. Methodology extraction (ExtractionAgent)
    4. Dataset resolution      (DatasetAgent)
    5. Code generation         (CodeGenAgent)
    6. Syntax validation of the generated code

Generated code is written to disk under `verification_results/<paper>/generated/`
so you can inspect exactly what ARPA produced, and every generated file is
compiled (not executed) to confirm it is at least valid Python.

Usage:
    python verify_codegen_agent.py --backend nvidia            # all 10 papers (~20 min)
    python verify_codegen_agent.py --backend nvidia --quick    # 2 papers, fast smoke test
    python verify_codegen_agent.py --papers easy1 easy2        # specific papers
    python verify_codegen_agent.py --backend nvidia --fresh    # ignore saved progress
    python verify_codegen_agent.py --backend nvidia --verify-datasets   # opt in to real downloads

Dataset verification
--------------------
By default the DatasetAgent resolves the dataset and generates loading code but
does NOT download it. That is deliberate: 6 of the 10 benchmark papers train on
ImageNet, which is ~150GB and gated behind manual registration, so a real
download can never succeed in an automated run and would guarantee failures
that say nothing about whether ARPA works.

Pass --verify-datasets to opt into real downloads + Docker sandbox verification
(requires Docker running and the arpa-sandbox image built). Expect it to be much
slower, and expect the ImageNet-based papers to fail on download, not on logic.

Time estimates:
    nvidia / groq / gemini / openrouter :  ~1.5-2 min per paper
    ollama (local)                      : ~15-20 min per paper
"""

import argparse
import json
import os
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
import urllib.request

from loguru import logger

from arpa.agents.codegen_agent import CodeGenAgent
from arpa.agents.dataset_agent import DatasetAgent
from arpa.agents.extraction_agent import ExtractionAgent
from arpa.core.config import get_settings
from arpa.tools.pdf_pipeline import PdfToTextPipeline

# ---------------------------------------------------------------------------
# Timeouts
#
# Each pipeline stage gets its own budget because they are wildly different in
# cost. Extraction is 4 sequential LLM passes; codegen is 2 long generations
# that can legitimately run for minutes on a large file; dataset resolution is
# fast unless --verify-datasets turns on real downloads, which is why its
# budget jumps in that mode (see main()).
#
# A single flat 240s budget -- the previous behaviour -- was the main reason
# runs looked like failures: stages were being killed mid-flight while the API
# was still streaming a perfectly good response.
# ---------------------------------------------------------------------------
STAGE_TIMEOUTS = {
    "extraction": 900,
    "dataset": 600,
    "codegen": 900,
}

# How many times to attempt a stage that failed with a transient-looking error
# (rate limit, provider timeout, 5xx). Cloud LLM endpoints hiccup; one retry
# turns most of those into passes instead of false failures.
STAGE_RETRIES = 2

# Substrings that mark an error as worth retrying. Anything else is treated as
# a real, deterministic failure and is not retried (no point burning API calls
# re-running something that will fail identically).
TRANSIENT_ERROR_MARKERS = (
    "rate limit",
    "ratelimit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "temporarily",
    "overloaded",
    "unavailable",
    "connection",
    "read operation",
)


class StageTimeout(Exception):
    """Raised when a pipeline stage exceeds its wall-clock budget."""


def run_with_timeout(func, timeout_seconds, *args, **kwargs):
    """Run ``func(*args, **kwargs)`` with a hard wall-clock timeout.

    A hung LLM call must never block the whole verification run.

    A plain daemon thread is used rather than ThreadPoolExecutor, and that
    choice is load-bearing in two ways:

    * ``ThreadPoolExecutor`` used as a context manager calls
      ``shutdown(wait=True)`` on exit, which blocks until the worker finishes
      -- so on a real hang the timeout fires and the script freezes anyway.
    * Even with ``shutdown(wait=False)``, executor workers are non-daemon and
      Python joins every non-daemon thread at interpreter shutdown. The run
      would print its full report and then hang forever at exit, which looks
      exactly like the script still working.

    ``daemon=True`` sidesteps both: an abandoned stage keeps running harmlessly
    in the background, blocks nothing, and is torn down with the process.
    """
    box: dict = {}

    def target() -> None:
        try:
            box["value"] = func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True, name="arpa-stage")
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise StageTimeout(f"exceeded {timeout_seconds}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def is_transient(error: Exception) -> bool:
    """Heuristic: is this error worth retrying?"""
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in TRANSIENT_ERROR_MARKERS)


def run_stage(name: str, func, *args, **kwargs):
    """Run one pipeline stage with a timeout plus retries on transient errors.

    Returns the stage's return value, or raises the final exception if every
    attempt failed.
    """
    timeout = STAGE_TIMEOUTS[name]
    last_exc: Exception | None = None

    for attempt in range(1, STAGE_RETRIES + 1):
        try:
            return run_with_timeout(func, timeout, *args, **kwargs)
        except StageTimeout as exc:
            last_exc = exc
            # A timeout is worth exactly one more shot -- providers do
            # occasionally stall a single request -- but not more than that,
            # or a genuinely stuck stage costs timeout x retries of wall-clock
            # time for nothing.
            if attempt < STAGE_RETRIES:
                print(f"      ! {name} timed out ({timeout}s), retrying "
                      f"({attempt + 1}/{STAGE_RETRIES})...")
                time.sleep(5)
                continue
            raise
        except Exception as exc:  # noqa: BLE001 - stage errors are data, not crashes
            last_exc = exc
            if attempt < STAGE_RETRIES and is_transient(exc):
                backoff = 10 * attempt
                print(f"      ! {name} hit a transient error, retrying in "
                      f"{backoff}s ({attempt + 1}/{STAGE_RETRIES}): "
                      f"{str(exc)[:120]}")
                time.sleep(backoff)
                continue
            raise

    raise last_exc  # pragma: no cover - loop always returns or raises


# ---------------------------------------------------------------------------
# Benchmark papers
# ---------------------------------------------------------------------------
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

MIN_PDF_BYTES = 20_000  # anything smaller is an arXiv error page, not a paper


def download_paper(arxiv_id: str, output_path: Path, retries: int = 3) -> bool:
    """Download a paper from arXiv, validating that we actually got a PDF.

    arXiv intermittently serves rate-limit HTML or truncated bodies. Writing
    those to a .pdf silently poisons every later run, because the file exists
    so we never retry it -- and then PDF extraction fails with a confusing
    error. So content is checked for the %PDF magic bytes and a plausible size
    before it is written, and a bad existing file is re-downloaded.
    """
    if output_path.exists():
        head = output_path.read_bytes()[:5]
        if head.startswith(b"%PDF") and output_path.stat().st_size >= MIN_PDF_BYTES:
            return True
        print(f"      ! {output_path.name} is corrupt/incomplete, re-downloading")
        try:
            output_path.unlink()
        except OSError:
            pass

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=90) as response:
                content = response.read()

            if not content.startswith(b"%PDF") or len(content) < MIN_PDF_BYTES:
                raise ValueError(
                    f"got {len(content)} bytes that are not a PDF "
                    "(arXiv likely rate-limited us)"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(content)
            return True

        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Download attempt {attempt}/{retries} failed for {arxiv_id}: {exc}")
            if attempt < retries:
                time.sleep(5 * attempt)

    logger.error(f"Download failed for {arxiv_id} after {retries} attempts")
    return False


def check_generated_code(files, generated_dir: Path) -> dict:
    """Compile every generated file to confirm ARPA emitted valid Python.

    ``compile()`` parses without executing, so this is safe: it catches the
    failure mode that actually matters (truncated / malformed generations)
    without importing model code or touching the network.
    """
    report = {
        "file_count": len(files),
        "syntax_ok": 0,
        "syntax_failed": [],
        "quality_warnings": [],
        "written_to": str(generated_dir),
    }

    for gen_file in files:
        try:
            compile(gen_file.content, gen_file.path, "exec")
            report["syntax_ok"] += 1
        except SyntaxError as exc:
            report["syntax_failed"].append(f"{gen_file.path}: line {exc.lineno}: {exc.msg}")

        # Surfaced by the agent itself (e.g. runaway repetition), which
        # compile() cannot detect because the code is syntactically fine.
        for warning in getattr(gen_file, "quality_warnings", None) or []:
            report["quality_warnings"].append(f"{gen_file.path}: {warning}")

    return report


def test_paper(
    paper_key: str,
    paper_meta: dict,
    papers_dir: Path,
    output_dir: Path,
    backend: str,
    *,
    verify_datasets: bool,
) -> dict:
    """Run the full ARPA pipeline on a single paper."""
    result = {
        "paper": paper_key,
        "title": paper_meta["title"],
        "difficulty": paper_meta["difficulty"],
        "backend": backend,
        "pdf_found": False,
        "text_extracted": False,
        "extraction_success": False,
        "dataset_success": False,
        "dataset_verified": False,
        "codegen_success": False,
        "syntax_valid": False,
        "dataset_name": None,
        "registry_source": None,
        "code_report": None,
        "errors": [],
        "time_seconds": 0.0,
    }

    start = time.time()
    paper_out = output_dir / paper_key
    generated_dir = paper_out / "generated"

    def fail(stage: str, exc: Exception) -> None:
        """Record a stage failure without aborting the whole run."""
        label = "TIMEOUT" if isinstance(exc, StageTimeout) else type(exc).__name__
        result["errors"].append(f"{stage}: {label}: {str(exc)[:200]}")
        print(f"    x {stage} failed -> {label}: {str(exc)[:150]}")

    try:
        # -- Stage 1: PDF present -------------------------------------------
        print("    [1/5] Checking PDF...")
        pdf_path = papers_dir / f"{paper_key}_{paper_meta['arxiv_id']}.pdf"
        if not pdf_path.exists():
            result["errors"].append(f"pdf: not found ({pdf_path.name})")
            print(f"    x PDF not found: {pdf_path.name}")
            return result
        result["pdf_found"] = True
        print(f"    + PDF found ({pdf_path.stat().st_size // 1024}KB)")

        # -- Stage 2: PDF -> text -------------------------------------------
        print("    [2/5] Converting PDF to text...")
        try:
            pipeline = PdfToTextPipeline()
            pdf_result = pipeline.convert(pdf_path, paper_out)
            paper_text = pdf_result.text
            if not paper_text or len(paper_text) < 500:
                raise ValueError(f"extracted only {len(paper_text or '')} characters")
            result["text_extracted"] = True
            print(f"    + Extracted {len(paper_text):,} characters")
        except Exception as exc:  # noqa: BLE001
            fail("pdf_to_text", exc)
            return result

        # -- Stage 3: Methodology extraction (4 LLM passes) ------------------
        print("    [3/5] Extracting methodology...")
        try:
            extraction_agent = ExtractionAgent(backend=backend)
            methodology = run_stage("extraction", extraction_agent.run, paper_text)

            if not methodology or not methodology.dataset_description:
                raise ValueError("extraction returned no dataset description")

            result["extraction_success"] = True
            result["dataset_name"] = methodology.dataset_description.name
            print(f"    + Methodology extracted (dataset: {result['dataset_name']})")

            # Persist it so a downstream failure is debuggable without a rerun.
            paper_out.mkdir(parents=True, exist_ok=True)
            (paper_out / "methodology.json").write_text(
                methodology.model_dump_json(indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            fail("extraction", exc)
            return result  # nothing downstream can run without a methodology

        # -- Stage 4: Dataset resolution ------------------------------------
        # Non-fatal by design: codegen only needs the methodology, so a dataset
        # miss should not mask whether code generation itself works.
        suffix = " (with download + Docker verify)" if verify_datasets else ""
        print(f"    [4/5] Resolving dataset{suffix}...")
        try:
            dataset_agent = DatasetAgent(backend=backend)
            dataset_result = run_stage(
                "dataset",
                lambda: dataset_agent.run(
                    methodology,
                    use_docker=verify_datasets,
                    verify_loading=verify_datasets,
                ),
            )

            if dataset_result and dataset_result.spec:
                result["dataset_success"] = True
                result["registry_source"] = dataset_result.spec.registry_source
                result["dataset_verified"] = bool(dataset_result.spec.verified)
                tag = " [verified]" if result["dataset_verified"] else ""
                print(f"    + Dataset resolved via {result['registry_source']}{tag}")

                if dataset_result.spec.loading_code:
                    generated_dir.mkdir(parents=True, exist_ok=True)
                    (generated_dir / "dataset_loader.py").write_text(
                        dataset_result.spec.loading_code, encoding="utf-8"
                    )
            else:
                reason = getattr(dataset_result, "escalation_reason", None) or "empty result"
                result["errors"].append(f"dataset: {reason}")
                print(f"    x Dataset resolution failed: {reason}")
        except Exception as exc:  # noqa: BLE001
            fail("dataset", exc)

        # -- Stage 5: Code generation + syntax check ------------------------
        print("    [5/5] Generating code...")
        try:
            codegen_agent = CodeGenAgent(backend=backend)
            codegen_result = run_stage(
                "codegen",
                lambda: codegen_agent.run(methodology, output_dir=generated_dir),
            )

            if not (codegen_result and codegen_result.files):
                reason = getattr(codegen_result, "escalation_reason", None) or "no files generated"
                result["errors"].append(f"codegen: {reason}")
                print(f"    x Code generation failed: {reason}")
            else:
                result["codegen_success"] = True
                code_report = check_generated_code(codegen_result.files, generated_dir)
                result["code_report"] = code_report
                result["syntax_valid"] = (
                    code_report["syntax_ok"] == code_report["file_count"]
                    and not code_report["quality_warnings"]
                )

                print(f"    + Generated {code_report['file_count']} files "
                      f"({code_report['syntax_ok']} compile cleanly)")
                for problem in code_report["syntax_failed"]:
                    result["errors"].append(f"syntax: {problem}")
                    print(f"      ! syntax error -> {problem}")
                for warning in code_report["quality_warnings"]:
                    result["errors"].append(f"quality: {warning}")
                    print(f"      ! quality warning -> {warning}")
        except Exception as exc:  # noqa: BLE001
            fail("codegen", exc)

    except Exception as exc:  # noqa: BLE001 - last line of defence
        result["errors"].append(f"unexpected: {str(exc)[:200]}")
        print(f"    x UNEXPECTED ERROR: {exc}")
        logger.error(f"Unexpected error testing {paper_key}\n{traceback.format_exc()}")

    result["time_seconds"] = time.time() - start
    return result


def resolve_models(settings, backend: str) -> tuple[str | None, str | None]:
    """Return the (general, code) model names the run will actually use."""
    pairs = {
        "nvidia": ("nvidia_general_model", "nvidia_code_model"),
        "gemini": ("gemini_general_model", "gemini_code_model"),
        "openrouter": ("openrouter_general_model", "openrouter_code_model"),
        "ollama": ("ollama_general_model", "ollama_code_model"),
        "groq": ("groq_model", "groq_model"),
    }.get(backend)
    if not pairs:
        return None, None
    return getattr(settings, pairs[0], None), getattr(settings, pairs[1], None)


# Rough minutes per paper. A paper costs ~8 LLM calls: 4 extraction passes,
# ~2 dataset calls, and 2 code generations that request up to 8192 tokens and
# are by far the slowest. These numbers come from measured NVIDIA NIM runs
# (~95s per extraction call on llama-3.3-70b), not from guesswork -- an
# earlier flat "2 minutes per paper" estimate was off by roughly 6x and made
# a perfectly healthy run look like it had hung.
_BIG_MODEL_MARKERS = ("70b", "72b", "405b", "120b", "large", "opus")


def _is_big(model: str | None) -> bool:
    return any(marker in (model or "").lower() for marker in _BIG_MODEL_MARKERS)


def estimate_minutes_per_paper(
    backend: str, general_model: str | None, code_model: str | None
) -> float:
    """Price the two model roles separately.

    They are genuinely independent: extraction and dataset work run on
    general_model (~6 calls) while the two long code generations run on
    code_model. Charging one flat rate for the pair would hide the most useful
    tuning lever there is -- a small general_model with a large code_model
    cuts most of the wall-clock cost without giving up code quality.
    """
    if backend == "ollama":
        return 17.0
    # 4 extraction passes + ~2 dataset calls, on general_model.
    general_cost = 8.3 if _is_big(general_model) else 2.0
    # 2 generations requesting up to 8192 tokens, on code_model -- the
    # slowest calls in the run by a wide margin.
    code_cost = 6.0 if _is_big(code_model) else 1.3
    return round(general_cost + code_cost, 1)


def paper_passed(r: dict) -> bool:
    """A paper passes when the pipeline ran end to end AND the code compiles."""
    return bool(
        r.get("extraction_success")
        and r.get("dataset_success")
        and r.get("codegen_success")
        and r.get("syntax_valid")
    )


def generate_summary(results: list[dict], output_path: Path, *, verify_datasets: bool) -> None:
    """Write and print the verification report."""
    lines: list[str] = []
    total = len(results)

    def pct(n: int) -> str:
        return f"{(100.0 * n / total):.0f}%" if total else "n/a"

    lines.append("=" * 78)
    lines.append("ARPA PIPELINE VERIFICATION SUMMARY")
    lines.append("=" * 78)
    lines.append(f"Date            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Backend         : {results[0]['backend'] if results else 'n/a'}")
    lines.append(f"Papers tested   : {total}")
    lines.append(f"Dataset download: {'ON (Docker verify)' if verify_datasets else 'OFF (resolution only)'}")
    lines.append("")

    extraction_ok = sum(1 for r in results if r["extraction_success"])
    dataset_ok = sum(1 for r in results if r["dataset_success"])
    codegen_ok = sum(1 for r in results if r["codegen_success"])
    syntax_ok = sum(1 for r in results if r["syntax_valid"])
    passed = sum(1 for r in results if paper_passed(r))

    lines.append("STAGE RESULTS")
    lines.append("-" * 78)
    lines.append(f"  Extraction Agent   : {extraction_ok}/{total}  ({pct(extraction_ok)})")
    lines.append(f"  Dataset Agent      : {dataset_ok}/{total}  ({pct(dataset_ok)})")
    lines.append(f"  CodeGen Agent      : {codegen_ok}/{total}  ({pct(codegen_ok)})")
    lines.append(f"  Generated code OK  : {syntax_ok}/{total}  ({pct(syntax_ok)})")
    lines.append("")
    lines.append(f"  FULL PIPELINE PASS : {passed}/{total}  ({pct(passed)})")
    lines.append("")

    if verify_datasets:
        verified = sum(1 for r in results if r["dataset_verified"])
        lines.append(f"  Datasets verified in Docker: {verified}/{total}")
        lines.append("")

    lines.append("PAPER-BY-PAPER BREAKDOWN")
    lines.append("-" * 78)

    def mark(flag: bool) -> str:
        return "PASS" if flag else "FAIL"

    for r in results:
        status = "PASS" if paper_passed(r) else "FAIL"
        lines.append("")
        lines.append(f"[{r['paper']}] {r['title']} ({r['difficulty']}) "
                     f"-- {status}  ({r['time_seconds']:.1f}s)")
        lines.append(f"    PDF          : {mark(r['pdf_found'])}")
        lines.append(f"    PDF -> text  : {mark(r['text_extracted'])}")
        lines.append(f"    Extraction   : {mark(r['extraction_success'])}"
                     + (f"   dataset='{r['dataset_name']}'" if r["dataset_name"] else ""))
        lines.append(f"    Dataset      : {mark(r['dataset_success'])}"
                     + (f"   via {r['registry_source']}" if r["registry_source"] else ""))
        lines.append(f"    CodeGen      : {mark(r['codegen_success'])}")
        lines.append(f"    Code syntax  : {mark(r['syntax_valid'])}")

        report = r.get("code_report")
        if report:
            lines.append(f"    Files        : {report['syntax_ok']}/{report['file_count']} "
                         f"compile cleanly  ->  {report['written_to']}")

        for err in r["errors"]:
            lines.append(f"    ! {err}")

    lines.append("")
    lines.append("=" * 78)
    if total and passed == total:
        lines.append("RESULT: ARPA IS WORKING -- all papers completed the full pipeline")
        lines.append("        and produced syntactically valid code.")
    else:
        lines.append(f"RESULT: PARTIALLY WORKING -- {passed}/{total} papers passed.")
        lines.append("        See the per-paper errors above for what to fix.")
    lines.append("=" * 78)

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    print("\n" + text)


def main() -> None:
    global STAGE_RETRIES

    parser = argparse.ArgumentParser(
        description="Verify the ARPA pipeline end-to-end across benchmark papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "gemini", "openrouter", "nvidia", "groq"],
        default=None,
        help="LLM backend to use (defaults to ARPA_LLM_BACKEND from .env)",
    )
    parser.add_argument("--papers", nargs="+", default=None,
                        help=f"Specific papers to test. Choices: {', '.join(PAPERS)}")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test on 2 papers only (easy1, medium1)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Don't fetch missing PDFs from arXiv")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore saved partial results and start over")
    parser.add_argument("--verify-datasets", action="store_true",
                        help="Actually download datasets and verify loading code in Docker "
                             "(slow; ImageNet-based papers will fail on download)")
    parser.add_argument("--stage-timeout", type=int, default=None,
                        help="Override the per-stage timeout (seconds) for every stage")
    parser.add_argument("--retries", type=int, default=STAGE_RETRIES,
                        help=f"Attempts per stage on transient errors (default: {STAGE_RETRIES})")
    args = parser.parse_args()

    STAGE_RETRIES = max(1, args.retries)

    if args.stage_timeout:
        for stage in STAGE_TIMEOUTS:
            STAGE_TIMEOUTS[stage] = args.stage_timeout
    elif args.verify_datasets:
        # Real downloads (MNIST-family is ~100MB, plus Docker image pull and up
        # to dataset_verify_max_retries regeneration rounds) need far more than
        # the resolution-only budget.
        STAGE_TIMEOUTS["dataset"] = 2400

    # get_settings() constructs a fresh ARPASettings from the environment on
    # every call, so mutating a local copy would never reach the agents.
    # Setting the env var is what actually makes the choice stick process-wide,
    # including for any client constructed deep inside an agent.
    if args.backend:
        os.environ["ARPA_LLM_BACKEND"] = args.backend

    settings = get_settings()
    backend = args.backend or settings.llm_backend

    papers_dir = Path("papers")
    output_dir = Path("verification_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    partial_results_path = output_dir / "partial_results.jsonl"
    if args.fresh and partial_results_path.exists():
        partial_results_path.unlink()
        print("--fresh: cleared previous partial results, starting over.\n")

    # Select papers
    if args.quick:
        papers_to_test = ["easy1", "medium1"]
    elif args.papers:
        papers_to_test = args.papers
    else:
        papers_to_test = list(PAPERS)

    unknown = [p for p in papers_to_test if p not in PAPERS]
    if unknown:
        parser.error(f"Unknown paper(s): {', '.join(unknown)}. Choices: {', '.join(PAPERS)}")

    # -- Preflight: fail fast on bad config, before spending time on downloads
    key_attr = {
        "nvidia": "nvidia_api_key",
        "gemini": "gemini_api_key",
        "openrouter": "openrouter_api_key",
        "groq": "groq_api_key",
    }.get(backend)
    if key_attr and not getattr(settings, key_attr, None):
        parser.error(
            f"Backend '{backend}' selected but no API key found. "
            f"Set ARPA_{key_attr.upper()} in your .env file."
        )

    # -- Download papers -----------------------------------------------------
    if not args.skip_download:
        print(f"Checking {len(papers_to_test)} paper PDF(s)...")
        missing = []
        for paper_key in papers_to_test:
            meta = PAPERS[paper_key]
            pdf_path = papers_dir / f"{paper_key}_{meta['arxiv_id']}.pdf"
            existed = pdf_path.exists()
            if not existed:
                print(f"  downloading {paper_key} ({meta['title']})...")
            if not download_paper(meta["arxiv_id"], pdf_path):
                missing.append(paper_key)
            elif not existed:
                time.sleep(3)  # be polite to arXiv
        if missing:
            print(f"  ! could not download: {', '.join(missing)} "
                  "(they will be reported as failures)")
        print("  all PDFs ready.\n")

    print("=" * 78)
    print("ARPA PIPELINE VERIFICATION")
    print("=" * 78)
    print(f"Backend         : {backend}")
    print(f"Papers          : {len(papers_to_test)} ({', '.join(papers_to_test)})")
    print(f"Dataset download: {'ON  (Docker verify)' if args.verify_datasets else 'OFF (resolution only)'}")
    print("Stage timeouts  : " + ", ".join(f"{k}={v}s" for k, v in STAGE_TIMEOUTS.items()))
    print(f"Retries / stage : {STAGE_RETRIES}")

    general_model, code_model = resolve_models(settings, backend)
    if general_model or code_model:
        # Always show the models actually in use. A config bug once made the
        # client silently ignore the .env model settings and fall back to a
        # hardcoded default, so the run and the config disagreed with nobody
        # the wiser. Printing the resolved values makes that failure obvious.
        print(f"Models          : general={general_model}  code={code_model}")

    per_paper = estimate_minutes_per_paper(backend, general_model, code_model)
    est = len(papers_to_test) * per_paper
    print(f"Estimated time  : ~{est:.0f} min ({est / 60:.1f} h), "
          f"about {per_paper:.0f} min/paper")

    if per_paper >= 10:
        print("   Large models on a shared/free tier take 1.5-2 min per LLM call,")
        print("   and each paper needs roughly 8 of them. Progress is saved after")
        print("   every paper -- Ctrl+C is safe, re-run without --fresh to resume.")
    print("")

    # -- Resume from prior progress -----------------------------------------
    already_done: dict[str, dict] = {}
    if partial_results_path.exists():
        with open(partial_results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prior = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Later entries win, so re-running one paper supersedes its
                # older record instead of appearing twice in the report.
                already_done[prior["paper"]] = prior

    # Only carry forward papers that are part of THIS run, otherwise the
    # summary percentages would be computed over a different set of papers
    # than the one that was asked for.
    results: list[dict] = [
        already_done[p] for p in papers_to_test if p in already_done
    ]

    if results:
        names = ", ".join(r["paper"] for r in results)
        print(f"Resuming: skipping {len(results)} already-completed paper(s): {names}")
        print("(use --fresh to re-run everything)\n")

    # -- Run -----------------------------------------------------------------
    for idx, paper_key in enumerate(papers_to_test, 1):
        if paper_key in already_done:
            continue

        meta = PAPERS[paper_key]
        print(f"[{idx}/{len(papers_to_test)}] {paper_key}: {meta['title']} ({meta['difficulty']})")

        result = test_paper(
            paper_key, meta, papers_dir, output_dir, backend,
            verify_datasets=args.verify_datasets,
        )
        results.append(result)

        # Persist immediately: a crash or Ctrl+C never loses finished work.
        with open(partial_results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        verdict = "PASS" if paper_passed(result) else "FAIL"
        print(f"  => {verdict}  ({result['time_seconds']:.1f}s)\n")

        if idx < len(papers_to_test):
            time.sleep(3)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"verification_report_{stamp}.txt"
    generate_summary(results, report_path, verify_datasets=args.verify_datasets)

    print(f"\nReport            : {report_path}")
    print(f"Raw results       : {partial_results_path}")
    print(f"Generated code    : {output_dir}/<paper>/generated/")

    raise SystemExit(0 if results and all(paper_passed(r) for r in results) else 1)


if __name__ == "__main__":
    main()