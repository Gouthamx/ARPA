"""Tests for smoke execution of generated code.

The stage exists because `compile()` parses one file in isolation and never
resolves an import, so a train.py doing `from model import main` against a
model.py that defines no `main` compiles cleanly and dies on first run. That
defect sat inside a 10/10 benchmark result.
"""

from __future__ import annotations

import textwrap

import pytest

from arpa.tools.smoke_runner import CodeSmokeRunner, SmokeExpectations, SmokeResult


def _write(tmp_path, **files):
    d = tmp_path / "generated"
    d.mkdir(exist_ok=True)
    for name, src in files.items():
        (d / f"{name}.py").write_text(textwrap.dedent(src).strip() + "\n", encoding="utf-8")
    return d


GOOD_MODEL = """
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.fc = nn.Linear(12, num_classes)
        def forward(self, x):
            return self.fc(x.flatten(1))
"""


class TestCatchesWhatCompileCannot:
    def test_missing_symbol_in_cross_file_import_fails(self, tmp_path):
        """The exact easy1 defect: both files compile, train.py cannot run."""
        d = _write(
            tmp_path,
            model="def helper():\n    return 1",
            train="from model import main\n\nif __name__ == '__main__':\n    main()",
        )
        result = CodeSmokeRunner(timeout_s=90).run(d)
        assert not result.passed
        failed = {c["name"] for c in result.failed_checks}
        assert "import:train" in failed
        assert any("cannot import name" in c["detail"] for c in result.failed_checks)

    def test_syntax_error_in_a_non_codegen_file_is_caught(self, tmp_path):
        """dataset_loader.py comes from DatasetAgent, so check_generated_code()
        never compiled it -- medium4 shipped one broken on line 1."""
        d = _write(tmp_path, model=GOOD_MODEL, dataset_loader="\\ this is not python")
        result = CodeSmokeRunner(timeout_s=90).run(d)
        assert not result.passed
        assert any(c["name"] == "import:dataset_loader" for c in result.failed_checks)

    def test_healthy_code_passes_every_check(self, tmp_path):
        d = _write(tmp_path, model=GOOD_MODEL)
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=10)
        )
        assert result.passed, result.failed_checks
        names = {c["name"] for c in result.checks if c["ok"] and not c["skipped"]}
        assert {"import:model", "model_instantiates", "forward_pass", "backward_pass"} <= names

    def test_output_shape_mismatch_is_reported_but_not_fatal(self, tmp_path):
        """A head that does not match the paper's class count is worth saying,
        but it is not a "does it run" failure -- and treating it as one gave
        two wrong verdicts on correct architectures (a VAE returning a
        reconstruction, a SimCLR encoder returning a 128-dim projection).
        The mismatch must still be visible in the detail."""
        # Accepts num_classes and ignores it -- the SimCLR shape, where the
        # projection head is 128-wide by design regardless of class count.
        d = _write(
            tmp_path,
            model="""
                import torch.nn as nn

                class Encoder(nn.Module):
                    def __init__(self, num_classes=1000, proj_out_dim=128):
                        super().__init__()
                        self.proj = nn.Linear(12, proj_out_dim)
                    def forward(self, x):
                        return self.proj(x.flatten(1))
            """,
        )
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=1000)
        )
        assert result.passed, "a shape mismatch must not read as broken code"
        shape = next(c for c in result.checks if c["name"] == "output_shape")
        assert "128" in shape["detail"] and "1000" in shape["detail"]
        assert shape["skipped"], "a mismatch is surfaced as a note, not a pass"


class TestThirdPartyDependencyDistinction:
    def test_missing_third_party_module_is_skipped_not_failed(self, tmp_path):
        """A package absent from this machine says nothing about the generated
        code; failing on it would make the score measure the virtualenv."""
        d = _write(
            tmp_path,
            model=GOOD_MODEL,
            dataset_loader="import tensorflow_datasets as tfds\n\nDS = None",
        )
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=10)
        )
        assert result.passed, result.failed_checks
        assert "tensorflow_datasets" in result.missing_dependencies
        skipped = [c for c in result.checks if c["skipped"]]
        assert any("tensorflow_datasets" in c["detail"] for c in skipped)

    def test_missing_arpa_module_still_fails(self, tmp_path):
        """The exemption must not extend to ARPA's own files."""
        d = _write(tmp_path, train="import model\n\nX = model")
        result = CodeSmokeRunner(timeout_s=90).run(d)
        assert not result.passed
        assert any(c["name"] == "import:train" for c in result.failed_checks)

    def test_a_missing_dep_does_not_mask_a_real_defect(self, tmp_path):
        """Excusing an absent package must not excuse the code around it: a
        genuine fault in a module that imports fine still has to surface."""
        d = _write(
            tmp_path,
            model="""
                import torch.nn as nn

                class Net(nn.Module):
                    def __init__(self, num_classes=10):
                        super().__init__()
                        self.fc = nn.Linear(999, num_classes)   # wrong in_features
                    def forward(self, x):
                        return self.fc(x.flatten(1))
            """,
            dataset_loader="import timm\n",
        )
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=10)
        )
        assert "timm" in result.missing_dependencies
        assert not result.passed, "the shape mismatch is still a real failure"
        assert any(c["name"] == "forward_pass" for c in result.failed_checks)


class TestMultiOutputModels:
    """Returning several values is normal architecture, not a defect. The
    harness originally called .shape on whatever forward returned, which
    flagged a textbook VAE -- (recon, mu, logvar) -- as broken code."""

    VAE = """
        import torch, torch.nn as nn

        class VAE(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.enc = nn.Linear(12, 8)
                self.dec = nn.Linear(4, 12)
            def forward(self, x):
                h = self.enc(x.flatten(1))
                mu, logvar = h[:, :4], h[:, 4:]
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                return self.dec(z), mu, logvar
    """

    DICT_OUT = """
        import torch.nn as nn

        class Detector(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.fc = nn.Linear(12, num_classes)
            def forward(self, x):
                return {"logits": self.fc(x.flatten(1)), "features": x}
    """

    def test_tuple_returning_vae_passes(self, tmp_path):
        d = _write(tmp_path, model=self.VAE)
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=10)
        )
        assert result.passed, result.failed_checks

    def test_dict_returning_model_passes(self, tmp_path):
        d = _write(tmp_path, model=self.DICT_OUT)
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=10)
        )
        assert result.passed, result.failed_checks

    def test_output_shape_is_skipped_for_multi_output(self, tmp_path):
        """A VAE's first output is a reconstruction whose last dim is image
        width, not a class count -- asserting on it invents a failure."""
        d = _write(tmp_path, model=self.VAE)
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=10)
        )
        shape_check = next(c for c in result.checks if c["name"] == "output_shape")
        assert shape_check["skipped"]

    def test_single_tensor_output_is_still_shape_checked(self, tmp_path):
        """A plain classifier still gets its head compared -- the check is
        reported rather than skipped outright, it simply no longer fails the
        run."""
        d = _write(tmp_path, model=GOOD_MODEL)
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=999)
        )
        shape = next(c for c in result.checks if c["name"] == "output_shape")
        assert "999" in shape["detail"], "the paper's class count must be stated"

    def test_matching_head_is_confirmed(self, tmp_path):
        """The positive case: when the head does match, say so plainly."""
        d = _write(tmp_path, model=GOOD_MODEL)
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=10)
        )
        assert result.passed
        shape = next(c for c in result.checks if c["name"] == "output_shape")
        assert not shape["skipped"] and "matches" in shape["detail"]

    def test_forward_returning_no_tensor_still_fails(self, tmp_path):
        d = _write(
            tmp_path,
            model="""
                import torch.nn as nn

                class Net(nn.Module):
                    def __init__(self, num_classes=10):
                        super().__init__()
                        self.fc = nn.Linear(12, num_classes)
                    def forward(self, x):
                        return "not a tensor"
            """,
        )
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=10)
        )
        assert not result.passed
        assert any(c["name"] == "forward_pass" for c in result.failed_checks)


class TestSafety:
    def test_no_nn_module_is_skipped_not_failed(self, tmp_path):
        """Benchmark papers using sklearn estimators have no torch model."""
        d = _write(tmp_path, model="def train_svc():\n    return 0.89")
        result = CodeSmokeRunner(timeout_s=90).run(d)
        assert result.passed
        assert any(c["name"] == "model_instantiates" and c["skipped"] for c in result.checks)

    def test_hanging_import_is_killed_by_the_timeout(self, tmp_path):
        """Guards the case the timeout exists for: a module that starts
        downloading or training at import time."""
        d = _write(tmp_path, model="import time\ntime.sleep(60)")
        result = CodeSmokeRunner(timeout_s=5).run(d)
        assert not result.passed
        assert result.timed_out
        assert "timed out" in (result.error or "")

    def test_harness_file_is_cleaned_up(self, tmp_path):
        d = _write(tmp_path, model=GOOD_MODEL)
        CodeSmokeRunner(timeout_s=90).run(d)
        assert not (d / "_arpa_smoke.py").exists()

    def test_quarantined_files_are_not_executed(self, tmp_path):
        """UNVERIFIED_* files already failed an earlier check; running them
        would just re-report a known failure."""
        d = _write(tmp_path, model=GOOD_MODEL)
        (d / "UNVERIFIED_broken.py").write_text("this is not python\n", encoding="utf-8")
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=10)
        )
        assert result.passed, result.failed_checks

    def test_missing_directory_is_reported_not_raised(self, tmp_path):
        result = CodeSmokeRunner(timeout_s=30).run(tmp_path / "nope")
        assert not result.passed
        assert "no generated dir" in (result.error or "")


class TestSummary:
    @pytest.mark.parametrize(
        "checks, expected",
        [
            ([{"ok": True, "skipped": False}] * 3, "3/3 checks passed"),
            (
                [{"ok": True, "skipped": False}, {"ok": True, "skipped": True}],
                "1/1 checks passed (1 skipped)",
            ),
            (
                [{"ok": False, "skipped": False}, {"ok": True, "skipped": False}],
                "1/2 checks passed",
            ),
        ],
    )
    def test_skipped_checks_count_as_neither_pass_nor_total(self, checks, expected):
        """Counting them as passes produced nonsense like '6/3 checks passed'."""
        assert SmokeResult(passed=True, checks=checks).summary == expected


class TestModelSelection:
    """Which class the harness picks is the whole result. A file contains the
    model plus its building blocks, and sometimes a loss; picking wrong
    reports a defect in the wrong place."""

    def test_prefers_the_class_taking_num_classes(self, tmp_path):
        """Definition order is not reliable -- a DeiT file defines the model at
        line 34 and its Block at 165, other files do the reverse. Accepting
        num_classes is what distinguishes a classifier from a building block."""
        d = _write(
            tmp_path,
            model="""
                import torch.nn as nn

                class DeiT(nn.Module):
                    def __init__(self, num_classes=10):
                        super().__init__()
                        self.fc = nn.Linear(12, num_classes)
                    def forward(self, x):
                        return self.fc(x.flatten(1))

                class Block(nn.Module):
                    def __init__(self, dim=8):
                        super().__init__()
                        self.fc = nn.Linear(dim, dim)
                    def forward(self, x):
                        return self.fc(x)
            """,
        )
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=10)
        )
        assert result.passed, result.failed_checks
        chosen = next(c for c in result.checks if c["name"] == "model_instantiates")
        assert "DeiT" in chosen["detail"]

    def test_loss_modules_are_not_selected(self, tmp_path):
        """A loss is an nn.Module whose forward takes (pred, target); feeding
        it one tensor raises a TypeError that looks like a broken model."""
        d = _write(
            tmp_path,
            model="""
                import torch.nn as nn

                class Net(nn.Module):
                    def __init__(self, num_classes=10):
                        super().__init__()
                        self.fc = nn.Linear(12, num_classes)
                    def forward(self, x):
                        return self.fc(x.flatten(1))

                class ContrastiveLoss(nn.Module):
                    def __init__(self, num_classes=10):
                        super().__init__()
                    def forward(self, features, labels):
                        return features.sum()
            """,
        )
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=10)
        )
        assert result.passed, result.failed_checks

    def test_model_that_cannot_construct_is_reported_not_masked(self, tmp_path):
        """Falling back to a helper class that happens to build hides the real
        fault: a SimCLR model raising 'degrees should be a sequence of length
        2' was reported as a stride error inside its Sobel filter."""
        d = _write(
            tmp_path,
            model="""
                import torch.nn as nn

                class SobelFilter(nn.Module):
                    def __init__(self):
                        super().__init__()
                    def forward(self, x):
                        return x

                class RealModel(nn.Module):
                    def __init__(self, num_classes=10):
                        super().__init__()
                        raise ValueError("degrees should be a sequence of length 2")
                    def forward(self, x):
                        return x
            """,
        )
        result = CodeSmokeRunner(timeout_s=90).run(
            d, SmokeExpectations(input_shape=[12], num_classes=10)
        )
        assert not result.passed
        failure = next(c for c in result.failed_checks if c["name"] == "model_instantiates")
        assert "RealModel" in failure["detail"]
        assert "degrees" in failure["detail"]


class TestPdfBudget:
    """The PDF pipeline overrode the extractor's own char budget with a much
    smaller one, cutting a 12-page paper to under a quarter of its text and
    taking the architecture sections with it. Everything downstream was
    starved by that single number."""

    def test_pipeline_does_not_shrink_the_extractor_budget(self):
        import inspect

        from arpa.tools.paper_extractor import PaperSectionExtractor
        from arpa.tools.pdf_pipeline import PdfToTextPipeline

        pipeline_default = inspect.signature(
            PdfToTextPipeline.__init__
        ).parameters["max_chars"].default
        extractor_default = inspect.signature(
            PaperSectionExtractor.__init__
        ).parameters["max_chars"].default
        assert pipeline_default >= extractor_default, (
            "the pipeline must not silently truncate below what the extractor "
            f"would keep ({pipeline_default} < {extractor_default})"
        )
