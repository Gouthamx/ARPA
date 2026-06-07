"""Run the PDF -> dataset pipeline against the real-paper benchmark and score it.

For each paper we run the full pipeline exactly as a user would:
    PDF -> PdfToTextPipeline (Layer-1 structural reduction)
        -> DatasetAgent (Layer-2 Gemini extraction + registry resolution +
           live metadata enrichment)
    -> score the resulting DatasetSpec against known ground truth.

By default Docker verification is OFF: this benchmark measures the PDF->dataset
EXTRACTION + RESOLUTION quality, not sandbox execution (which needs large
downloads + network). Pass --use-docker to include it.

Usage:
    python benchmark/download_papers.py     # once, to fetch the PDFs
    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --levels easy
    python benchmark/run_benchmark.py --only hard2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from arpa.agents import DatasetAgent
from arpa.agents.extraction_agent import ExtractionAgent
from arpa.tools.pdf_pipeline import PdfToTextPipeline
from benchmark.ground_truth import GROUND_TRUTH, GroundTruth

# Field weights -> 0..100 per paper. "dataset" dominates because identifying the
# right dataset is the core of the task.
_WEIGHTS = {
    "dataset": 40,
    "num_classes": 12,
    "train_size": 12,
    "test_size": 8,
    "input_shape": 13,
    "steps": 15,
}


@dataclass
class FieldResult:
    name: str
    expected: object
    actual: object
    points: float
    max_points: float
    scored: bool = True

    @property
    def mark(self) -> str:
        if not self.scored:
            return "N/A "
        if self.points >= self.max_points:
            return "PASS"
        if self.points > 0:
            return "PART"
        return "FAIL"


@dataclass
class PaperResult:
    level: str
    key: str
    fields: list[FieldResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    error: str | None = None
    extracted_name: str | None = None
    aliases: list[str] = field(default_factory=list)
    registry: str | None = None
    verified: bool = False
    escalated: bool = False

    @property
    def score(self) -> float:
        return sum(f.points for f in self.fields if f.scored)

    @property
    def max_score(self) -> float:
        return sum(f.max_points for f in self.fields if f.scored)


def _norm(s: str) -> str:
    return "".join(c for c in str(s).lower() if c.isalnum())


def _name_matches(aliases: list[str], *candidates: str | None) -> bool:
    hay = [_norm(c) for c in candidates if c]
    for alias in aliases:
        a = _norm(alias)
        if a and any(a in h or h in a for h in hay):
            return True
    return False


def _score_choice(accepted: list[int], actual: int | None, max_points: float) -> tuple[float, bool]:
    """Score an int field. Empty 'accepted' => field not scored (returns scored=False)."""
    if not accepted:
        return 0.0, False
    return (max_points if actual in accepted else 0.0), True


def _score_shape(options: list[list[int]], actual: list[int] | None, max_points: float) -> tuple[float, bool]:
    if not options:
        return 0.0, False
    if not actual:
        return 0.0, True
    for opt in options:
        if list(actual) == list(opt):
            return max_points, True
    # partial: matching spatial dims (H,W) but wrong channels, or vice versa
    for opt in options:
        if len(actual) == len(opt) and actual[-2:] == opt[-2:]:
            return max_points * 0.5, True
    return 0.0, True


def _score_steps(expected: list[str], steps: list, max_points: float) -> tuple[float, bool, list[str]]:
    if not expected:
        return 0.0, False, []
    found = [_norm(getattr(s, "name", "")) for s in steps]
    matched = [e for e in expected if any(_norm(e) in f or f in _norm(e) for f in found if f)]
    return max_points * (len(matched) / len(expected)), True, matched


def _evaluate(key: str, gt: GroundTruth, result) -> PaperResult:
    pr = PaperResult(level=gt.level, key=key)
    spec = result.spec
    pr.verified = result.verified
    pr.escalated = result.escalated

    if spec is None:
        pr.error = result.escalation_reason or "no spec produced"
        pr.fields = [
            FieldResult("dataset", gt.dataset_aliases, None, 0, _WEIGHTS["dataset"]),
            FieldResult("num_classes", gt.num_classes, None, 0, _WEIGHTS["num_classes"],
                        scored=bool(gt.num_classes)),
            FieldResult("train_size", gt.train_size, None, 0, _WEIGHTS["train_size"],
                        scored=bool(gt.train_size)),
            FieldResult("test_size", gt.test_size, None, 0, _WEIGHTS["test_size"],
                        scored=bool(gt.test_size)),
            FieldResult("input_shape", gt.input_shape_options, None, 0, _WEIGHTS["input_shape"],
                        scored=bool(gt.input_shape_options)),
            FieldResult("steps", gt.expected_steps, [], 0, _WEIGHTS["steps"],
                        scored=bool(gt.expected_steps)),
        ]
        return pr

    pr.extracted_name = spec.dataset_name
    pr.registry = f"{spec.registry_source}/{spec.registry_id}"

    name_ok = _name_matches(gt.dataset_aliases, spec.dataset_name, spec.registry_id)
    pr.fields.append(FieldResult(
        "dataset", "|".join(gt.dataset_aliases[:3]),
        f"{spec.dataset_name} -> {pr.registry}",
        _WEIGHTS["dataset"] if name_ok else 0, _WEIGHTS["dataset"]))

    pts, scored = _score_choice(gt.num_classes, spec.num_classes, _WEIGHTS["num_classes"])
    pr.fields.append(FieldResult("num_classes", gt.num_classes, spec.num_classes, pts,
                                 _WEIGHTS["num_classes"], scored=scored))

    pts, scored = _score_choice(gt.train_size, spec.train_size, _WEIGHTS["train_size"])
    pr.fields.append(FieldResult("train_size", gt.train_size, spec.train_size, pts,
                                 _WEIGHTS["train_size"], scored=scored))

    pts, scored = _score_choice(gt.test_size, spec.test_size, _WEIGHTS["test_size"])
    pr.fields.append(FieldResult("test_size", gt.test_size, spec.test_size, pts,
                                 _WEIGHTS["test_size"], scored=scored))

    pts, scored = _score_shape(gt.input_shape_options, spec.input_shape, _WEIGHTS["input_shape"])
    pr.fields.append(FieldResult("input_shape", gt.input_shape_options, spec.input_shape, pts,
                                 _WEIGHTS["input_shape"], scored=scored))

    pts, scored, matched = _score_steps(gt.expected_steps, spec.preprocess_steps, _WEIGHTS["steps"])
    pr.fields.append(FieldResult("steps", gt.expected_steps, matched, pts,
                                 _WEIGHTS["steps"], scored=scored))
    return pr


def _run_one(key: str, gt: GroundTruth, *, use_docker: bool, out_dir: Path) -> PaperResult:
    print(f"\n>>> [{key}] ({gt.level}) starting: {Path(gt.pdf).name}", flush=True)
    logger.info("=== Benchmarking [{}] {} ===", key, gt.pdf)
    start = time.time()
    try:
        print(f"    [1/3] Layer-1: extracting PDF -> dataset sections ...", flush=True)
        txt_path = PdfToTextPipeline().convert(Path(gt.pdf), out_dir).txt_path
        paper_text = txt_path.read_text(encoding="utf-8")
        print(f"          reduced to {len(paper_text)} chars -> {txt_path.name}", flush=True)

        print(f"    [2/3] ExtractionAgent: extracting methodology ...", flush=True)
        extraction_agent = ExtractionAgent(backend="gemini")
        methodology = extraction_agent.run(paper_text, reduce_first=True)
        if methodology.dataset_description:
            print(f"          -> dataset='{methodology.dataset_description.name}'", flush=True)

        print(f"    [3/3] DatasetAgent: registry resolution + verification ...", flush=True)
        agent = DatasetAgent(backend="gemini")
        result = agent.run(
            methodology=methodology,
            use_docker=use_docker,
            verify_loading=use_docker,
        )

        if result.spec is not None:
            print(f"          -> dataset='{result.spec.dataset_name}'  "
                  f"registry={result.spec.registry_source}/{result.spec.registry_id}  "
                  f"classes={result.spec.num_classes} train={result.spec.train_size} "
                  f"test={result.spec.test_size} shape={result.spec.input_shape}", flush=True)
        else:
            print(f"          -> no spec produced (escalated: {result.escalation_reason})", flush=True)

        print(f"    Scoring against ground truth ...", flush=True)
        pr = _evaluate(key, gt, result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Benchmark run failed for {}", key)
        pr = PaperResult(level=gt.level, key=key, error=str(exc))
    pr.elapsed_s = time.time() - start
    pct = (pr.score / pr.max_score * 100) if pr.max_score else 0
    print(f"<<< [{key}] done in {pr.elapsed_s:.1f}s  score {pr.score:.0f}/{pr.max_score:.0f} "
          f"({pct:.0f}%)", flush=True)
    return pr


def _print_report(results: list[PaperResult]) -> None:
    print("\n" + "=" * 78)
    print("ARPA PDF -> DATASET PIPELINE BENCHMARK (real arXiv papers)")
    print("=" * 78)
    by_level: dict[str, list[PaperResult]] = {}
    for pr in results:
        by_level.setdefault(pr.level, []).append(pr)

    for level in ("easy", "medium", "hard"):
        for pr in by_level.get(level, []):
            pct = (pr.score / pr.max_score * 100) if pr.max_score else 0
            print(f"\n[{pr.level.upper()}:{pr.key}]  {pr.score:.0f}/{pr.max_score:.0f} ({pct:.0f}%)  "
                  f"time {pr.elapsed_s:.1f}s")
            if pr.error:
                print(f"  ERROR: {pr.error}")
            if pr.extracted_name:
                print(f"  extracted: {pr.extracted_name}   registry: {pr.registry}   "
                      f"verified={pr.verified} escalated={pr.escalated}")
            for f in pr.fields:
                print(f"    [{f.mark}] {f.name:12} exp={f.expected!s:34.34} "
                      f"got={f.actual!s:30.30} {f.points:.0f}/{f.max_points:.0f}"
                      f"{'' if f.scored else '  (not scored)'}")

    total = sum(p.score for p in results)
    total_max = sum(p.max_score for p in results)
    overall = (total / total_max * 100) if total_max else 0
    print("\n" + "-" * 78)
    for level in ("easy", "medium", "hard"):
        lp = by_level.get(level, [])
        if lp:
            s = sum(p.score for p in lp)
            m = sum(p.max_score for p in lp)
            print(f"  {level:7}: {s:.0f}/{m:.0f} ({(s/m*100) if m else 0:.0f}%)")
    print(f"  OVERALL: {total:.0f}/{total_max:.0f} ({overall:.0f}%)")
    print("-" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the PDF->dataset pipeline on real papers")
    parser.add_argument("--levels", default="easy,medium,hard")
    parser.add_argument("--only", default=None, help="Run a single key, e.g. hard2")
    parser.add_argument("--use-docker", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results.json"))
    args = parser.parse_args()

    out_dir = Path(".arpa_runs/benchmark")
    wanted_levels = {x.strip() for x in args.levels.split(",") if x.strip()}

    keys = [args.only] if args.only else [
        k for k, gt in GROUND_TRUTH.items() if gt.level in wanted_levels
    ]
    print("=" * 78, flush=True)
    print(f"ARPA benchmark starting — {len(keys)} paper(s): {', '.join(keys)}", flush=True)
    print(f"Docker verification: {'ON' if args.use_docker else 'OFF'}", flush=True)
    print("=" * 78, flush=True)
    results: list[PaperResult] = []
    for idx, key in enumerate(keys, 1):
        print(f"\n[paper {idx}/{len(keys)}]", flush=True)
        gt = GROUND_TRUTH.get(key)
        if gt is None:
            logger.warning("Unknown key '{}'; skipping", key)
            continue
        if not Path(gt.pdf).exists():
            logger.error("Missing PDF {} — run benchmark/download_papers.py first", gt.pdf)
            continue
        results.append(_run_one(key, gt, use_docker=args.use_docker, out_dir=out_dir))

    _print_report(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "key": pr.key, "level": pr.level, "score": pr.score, "max_score": pr.max_score,
            "elapsed_s": round(pr.elapsed_s, 2), "extracted_name": pr.extracted_name,
            "registry": pr.registry, "verified": pr.verified, "escalated": pr.escalated,
            "error": pr.error,
            "fields": [
                {"name": f.name, "expected": f.expected, "actual": str(f.actual),
                 "points": f.points, "max_points": f.max_points, "scored": f.scored}
                for f in pr.fields
            ],
        }
        for pr in results
    ]
    args.output.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote benchmark results to {}", args.output.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
