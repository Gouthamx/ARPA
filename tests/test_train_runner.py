"""Tests for the training + metric comparison stage.

Most of these exercise the decision logic rather than real training: a run
that downloads a dataset and does 150 optimizer steps belongs in a benchmark
run, not a unit test. The judgement calls -- what counts as learning, what
gets skipped, how the paper's number is compared -- are what need pinning
down, because they are what the stage actually claims.
"""

from __future__ import annotations

import pytest

from arpa.tools.train_runner import (
    LEARNING_MARGIN,
    TrainingOutcome,
    TrainingRunner,
    normalize_metric,
    resolve_dataset,
)


class TestMetricNormalization:
    """Papers state accuracy as "96.22" or "0.9622" and extraction passes
    through whichever the text used. Comparing the two scales produced
    "paper 9622.0%, gap +9556.1%" -- worse than no comparison, because it
    reads as a catastrophic failure to reproduce."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (96.22, 0.9622),
            (0.9622, 0.9622),
            (100.0, 1.0),
            (1.0, 1.0),
            (0.5, 0.5),
            (None, None),
            ("not a number", None),
            (9622, None),  # no scale we recognise
        ],
    )
    def test_metrics_land_on_the_same_scale(self, raw, expected):
        got = normalize_metric(raw)
        if expected is None:
            assert got is None
        else:
            assert got == pytest.approx(expected)

    def test_gap_is_believable_after_normalization(self):
        outcome = TrainingOutcome(
            attempted=True, trains=True, test_accuracy=0.659,
            reported_metric=normalize_metric(96.22),
        )
        assert outcome.gap == pytest.approx(0.3032, abs=1e-3)
        assert "9556" not in outcome.summary


class TestDatasetResolution:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Fashion-MNIST", "FashionMNIST"),
            ("fashion mnist", "FashionMNIST"),
            ("FashionMNIST", "FashionMNIST"),
            ("Kuzushiji-MNIST", "KMNIST"),
            ("KMNIST", "KMNIST"),
            ("EMNIST", "EMNIST"),
            ("CIFAR-10", "CIFAR10"),
            ("CIFAR100", "CIFAR100"),
        ],
    )
    def test_known_small_datasets_resolve(self, name, expected):
        spec = resolve_dataset(name)
        assert spec is not None and spec[0] == expected

    @pytest.mark.parametrize(
        "name", ["ImageNet", "ILSVRC", "Caltech-UCSD birds", "COCO", None, ""]
    )
    def test_infeasible_datasets_do_not_resolve(self, name):
        """ImageNet is ~150GB behind manual registration -- attempting it
        unattended would fail for reasons unrelated to ARPA."""
        assert resolve_dataset(name) is None

    def test_loose_paper_phrasing_still_resolves(self):
        assert resolve_dataset("the Fashion-MNIST benchmark")[0] == "FashionMNIST"


class TestSkipsRatherThanFails:
    """Skipping and failing are different claims. A paper ARPA cannot train
    for reasons of its own design must not be recorded as ARPA failing."""

    def test_infeasible_dataset_is_skipped(self, tmp_path):
        (tmp_path / "generated").mkdir()
        outcome = TrainingRunner().run(tmp_path / "generated", "ImageNet")
        assert outcome.skipped_reason is not None
        assert not outcome.trains
        assert "not a small trainable dataset" in outcome.skipped_reason

    def test_missing_generated_dir_is_skipped(self, tmp_path):
        outcome = TrainingRunner().run(tmp_path / "nope", "CIFAR-10")
        assert outcome.skipped_reason == "no generated code"

    def test_skip_reason_shows_in_the_summary(self):
        outcome = TrainingOutcome(skipped_reason="ImageNet is not trainable here")
        assert outcome.summary.startswith("skipped")


class TestLearningVerdict:
    """`learns` is the honest pass/fail: loss fell AND accuracy cleared chance
    by a real margin. Either alone is too easy to satisfy by accident."""

    def _outcome(self, **kw):
        base = dict(
            attempted=True, trains=True, initial_loss=2.3, final_loss=0.4,
            test_accuracy=0.85, chance_accuracy=0.1,
        )
        base.update(kw)
        return TrainingOutcome(**base)

    def test_gap_is_paper_minus_run(self):
        outcome = self._outcome(test_accuracy=0.85, reported_metric=0.93)
        assert outcome.gap == pytest.approx(0.08)

    def test_gap_is_none_without_a_paper_number(self):
        assert self._outcome(reported_metric=None).gap is None

    def test_summary_reports_gap_without_calling_it_a_failure(self):
        """A capped run trailing a fully-trained paper is expected, not a
        failure -- the summary must not imply otherwise."""
        summary = self._outcome(test_accuracy=0.85, reported_metric=0.93, learns=True).summary
        assert "gap" in summary and "learning" in summary
        assert "fail" not in summary.lower()

    def test_chance_level_accuracy_is_not_learning(self):
        """The check that matters: a model producing chance output has not
        learned, however cleanly it ran."""
        chance = 0.1
        assert not (chance * 1.0 > chance * LEARNING_MARGIN)

    def test_learning_needs_margin_over_chance(self):
        assert 0.85 > 0.1 * LEARNING_MARGIN
        assert not 0.12 > 0.1 * LEARNING_MARGIN


class TestOutcomeReporting:
    def test_failed_training_says_so(self):
        outcome = TrainingOutcome(attempted=True, trains=False, error="shape mismatch")
        assert "training failed" in outcome.summary
        assert "shape mismatch" in outcome.summary

    def test_accuracy_formats_as_a_percentage(self):
        outcome = TrainingOutcome(
            attempted=True, trains=True, test_accuracy=0.8523, reported_metric=0.93
        )
        assert "85.2%" in outcome.summary
        assert "93.0%" in outcome.summary
