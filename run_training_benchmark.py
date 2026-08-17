"""Run the training stage across all benchmark papers and tabulate the result.

Stage 7 in isolation: it trains the code already sitting in
verification_results/<paper>/generated/ rather than regenerating it, so the
numbers describe code that has already been through syntax and smoke checks.
Regenerating first would change what is being measured and cost hours.

    python run_training_benchmark.py
    python run_training_benchmark.py --papers easy2 medium3 --max-steps 300
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from arpa.tools.train_runner import TrainingRunner

PAPERS = [
    ("easy1", "Fashion-MNIST"),
    ("easy2", "EMNIST"),
    ("easy3", "Kuzushiji-MNIST"),
    ("medium1", "ResNet"),
    ("medium2", "VGG"),
    ("medium3", "DenseNet"),
    ("medium4", "MobileNetV2"),
    ("hard1", "SimCLR"),
    ("hard2", "Bilinear CNN"),
    ("hard3", "DeiT"),
]

RESULTS_DIR = Path("verification_results")


def _plain(value):
    if isinstance(value, dict):
        return value.get("value")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=1500)
    args = parser.parse_args()

    selected = [p for p in PAPERS if not args.papers or p[0] in args.papers]
    runner = TrainingRunner(max_steps=args.max_steps, timeout_s=args.timeout)
    rows: list[dict] = []

    print(f"Training stage over {len(selected)} paper(s), {args.max_steps} steps each\n")

    for key, title in selected:
        methodology_path = RESULTS_DIR / key / "methodology.json"
        generated = RESULTS_DIR / key / "generated"
        if not methodology_path.exists():
            print(f"{key:9s} no methodology.json -- skipped")
            continue

        spec = json.loads(methodology_path.read_text(encoding="utf-8"))
        desc = spec.get("dataset_description") or {}
        evaluation = spec.get("evaluation") or {}

        outcome = runner.run(
            generated,
            desc.get("name"),
            reported_metric=_plain(evaluation.get("reported_metric")),
            num_classes=desc.get("num_classes"),
        )

        rows.append({
            "paper": key,
            "title": title,
            "dataset": outcome.dataset or desc.get("name"),
            "trains": outcome.trains,
            "learns": outcome.learns,
            "skipped_reason": outcome.skipped_reason,
            "steps": outcome.steps,
            "test_accuracy": outcome.test_accuracy,
            "chance_accuracy": outcome.chance_accuracy,
            "reported_metric": outcome.reported_metric,
            "gap": outcome.gap,
            "error": outcome.error,
            "summary": outcome.summary,
        })
        print(f"{key:9s} {outcome.summary}", flush=True)

    # -- report ---------------------------------------------------------
    trained = [r for r in rows if r["trains"]]
    learned = [r for r in trained if r["learns"]]
    skipped = [r for r in rows if r["skipped_reason"]]
    failed = [r for r in rows if not r["trains"] and not r["skipped_reason"]]

    lines = [
        "=" * 78,
        "ARPA TRAINING STAGE -- BENCHMARK RESULTS",
        "=" * 78,
        f"Date        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Papers      : {len(rows)}",
        f"Steps/paper : {args.max_steps}",
        "",
        f"  Attempted training : {len(trained)}/{len(rows)}",
        f"  Model learned      : {len(learned)}/{len(trained) if trained else 0}",
        f"  Skipped (by design): {len(skipped)}",
        f"  Failed             : {len(failed)}",
        "",
        "PER-PAPER",
        "-" * 78,
    ]
    for r in rows:
        lines.append(f"[{r['paper']}] {r['title']}")
        lines.append(f"    {r['summary']}")
        if r["test_accuracy"] is not None:
            lines.append(
                f"    accuracy {r['test_accuracy']:.1%} vs chance "
                f"{r['chance_accuracy']:.1%} over {r['steps']} steps"
            )
        lines.append("")

    lines += [
        "=" * 78,
        "Note: a capped run of a few hundred steps is not a reproduction of a",
        "paper trained for tens of epochs. 'learned' is the claim being made;",
        "the gap against the published figure is reported as context.",
        "=" * 78,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / f"training_report_{stamp}.txt").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "training_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: verification_results/training_report_{stamp}.txt")


if __name__ == "__main__":
    main()
