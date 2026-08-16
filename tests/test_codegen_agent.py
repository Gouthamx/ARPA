"""Tests for CodeGenAgent verification on 10 papers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from arpa.agents.codegen_agent import CodeGenAgent, CodegenResult
from arpa.core.confidence import ConfidenceField, ConfidenceLevel
from arpa.core.state import (
    ArchitectureSpec,
    BenchmarkExperimentSpec,
    DatasetDescription,
    EvaluationSpec,
    MethodologySpec,
    TrainingSpec,
)
from benchmark.codegen_papers import CODEGEN_PAPERS, CODEGEN_PAPER_KEYS
from benchmark.codegen_verify import verify_codegen_result


# ---------------------------------------------------------------------------
# Shared mock LLM (mirrors benchmark/verify_codegen.py offline stub)
# ---------------------------------------------------------------------------

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
    print(f"Training for {args.epochs} epochs")

if __name__ == "__main__":
    main()
'''


class MockCodegenLLM:
    general_model = "mock-general"
    code_model = "mock-code"

    def __init__(self, *, benchmark: bool = False, dataset: str = "cifar10") -> None:
        self.benchmark = benchmark
        self.dataset = dataset
        self._call = 0

    @staticmethod
    def extract_json(text: str) -> dict:
        return json.loads(text)

    def chat(self, messages, *, model=None, temperature=0.1, format_json=False) -> str:
        self._call += 1
        if self.benchmark and self._call == 1:
            code = _BENCHMARK_MODEL_PY
        elif self._call == 1:
            code = _MODEL_PY.format(dataset=self.dataset, num_classes=10)
        else:
            code = _TRAIN_PY
        return json.dumps({"code": code})


def _methodology_for_paper(key: str) -> MethodologySpec:
    paper = next(p for p in CODEGEN_PAPERS if p.key == key)
    desc = DatasetDescription(
        name=paper.dataset_aliases[0],
        train_size=60000,
        test_size=10000,
        input_shape=[3, 32, 32],
        num_classes=10,
    )
    spec = MethodologySpec(
        dataset_description=desc,
        architecture=ArchitectureSpec(
            model_name=ConfidenceField(value="SimpleCNN", confidence=ConfidenceLevel.CONFIRMED),
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
            ),
        ]
    return spec


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestCodegenAgentUnit:
    def test_escalates_without_methodology(self):
        result = CodeGenAgent(llm=MockCodegenLLM()).run()
        assert result.escalated
        assert result.escalation_reason == "No methodology provided"
        assert not result.success

    def test_escalates_without_dataset(self):
        spec = MethodologySpec(dataset_description=None)
        result = CodeGenAgent(llm=MockCodegenLLM()).run(methodology=spec)
        assert result.escalated
        assert "dataset" in (result.escalation_reason or "").lower()

    def test_generates_verified_files(self, tmp_path: Path):
        spec = _methodology_for_paper("fixture_cifar10")
        agent = CodeGenAgent(llm=MockCodegenLLM(dataset="cifar10"))
        result = agent.run(methodology=spec, output_dir=tmp_path)

        assert result.success
        assert len(result.files) == 2
        paths = {f.path for f in result.files}
        assert paths == {"model.py", "train.py"}
        assert all(f.verified for f in result.files)
        assert (tmp_path / "model.py").exists()
        assert (tmp_path / "train.py").exists()

    def test_benchmark_paper_generates_sklearn_model(self):
        spec = _methodology_for_paper("easy1")
        result = CodeGenAgent(llm=MockCodegenLLM(benchmark=True)).run(methodology=spec)
        model = next(f for f in result.files if f.path == "model.py")
        assert "LinearSVC" in model.content
        assert model.verified


class TestCodegenVerification:
    def test_verify_passes_for_valid_result(self):
        spec = _methodology_for_paper("fixture_mnist")
        result = CodeGenAgent(llm=MockCodegenLLM(dataset="mnist")).run(methodology=spec)
        vr = verify_codegen_result("fixture_mnist", result, dataset_name="mnist")
        assert vr.passed
        assert vr.score == 100.0
        assert vr.has_model_py and vr.has_train_py

    def test_verify_fails_on_syntax_error(self):
        from arpa.agents.codegen_agent import GeneratedFile

        bad = CodegenResult(
            success=True,
            files=[
                GeneratedFile(path="model.py", content="def broken(", purpose="x"),
                GeneratedFile(path="train.py", content=_TRAIN_PY, purpose="y"),
            ],
        )
        bad.files[0].verified = False
        bad.files[0].syntax_errors.append("unexpected EOF")
        bad.files[1].verified = True

        vr = verify_codegen_result("test", bad)
        assert not vr.passed
        assert not vr.model_verified


@pytest.mark.parametrize("paper_key", CODEGEN_PAPER_KEYS)
def test_codegen_agent_all_ten_papers(paper_key: str, tmp_path: Path):
    """Each of the 10 papers produces valid model.py + train.py output."""
    paper = next(p for p in CODEGEN_PAPERS if p.key == paper_key)
    spec = _methodology_for_paper(paper_key)
    llm = MockCodegenLLM(benchmark=paper.is_benchmark, dataset=paper.dataset_aliases[0])
    out_dir = tmp_path / paper_key

    result = CodeGenAgent(llm=llm).run(methodology=spec, output_dir=out_dir)
    vr = verify_codegen_result(paper_key, result, output_dir=out_dir)

    assert result.success, f"{paper_key}: agent did not succeed"
    assert vr.passed, (
        f"{paper_key}: verification failed — "
        f"model_verified={vr.model_verified} train_verified={vr.train_verified}"
    )
    assert vr.score >= 95.0


@pytest.mark.parametrize("paper_key", CODEGEN_PAPER_KEYS)
def test_paper_sources_exist_or_are_fixtures(paper_key: str):
    """Paper registry points at real files (PDFs may be absent until downloaded)."""
    paper = next(p for p in CODEGEN_PAPERS if p.key == paper_key)
    if paper.kind == "text":
        assert paper.source.exists(), f"Missing fixture: {paper.source}"
    # PDF papers are optional in CI — existence checked at integration run time.


def test_ten_papers_registered():
    assert len(CODEGEN_PAPERS) == 10
    assert len(CODEGEN_PAPER_KEYS) == 10
    assert len(set(CODEGEN_PAPER_KEYS)) == 10


def test_verify_codegen_script_mock_mode(tmp_path: Path):
    """Integration entrypoint succeeds in --mock --only-fixtures mode."""
    from benchmark.verify_codegen import main

    out_json = tmp_path / "results.json"
    with patch(
        "sys.argv",
        [
            "verify_codegen",
            "--mock",
            "--only-fixtures",
            "--output",
            str(out_json),
        ],
    ):
        rc = main()

    assert rc == 0
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(data) == 4
    assert all(entry["passed"] for entry in data)


# ---------------------------------------------------------------------------
# Robustness regression tests -- these cover real failures observed running
# the pipeline against actual papers (easy1/Fashion-MNIST): a response that
# got cut off before closing its code fence, and a model that got stuck in a
# runaway repetition loop emitting the same templated line hundreds of times.
# ---------------------------------------------------------------------------


class TestExtractCodeUnclosedFence:
    def test_strips_leading_fence_even_without_closing_marker(self):
        from arpa.agents.codegen_agent import _extract_code

        # Mirrors what we actually saw: generation cut off mid-class-body,
        # with the opening ```python marker never closed.
        raw = "```python\nimport torch\n\nclass Model:\n    def __init__(self):\n"
        result = _extract_code(raw)

        assert not result.startswith("```")
        assert result.startswith("import torch")

    def test_closed_fence_still_extracts_normally(self):
        from arpa.agents.codegen_agent import _extract_code

        raw = "```python\nimport torch\n```"
        assert _extract_code(raw) == "import torch"

    def test_plain_code_with_no_fence_is_unaffected(self):
        from arpa.agents.codegen_agent import _extract_code

        raw = "import torch\nx = 1\n"
        assert _extract_code(raw) == raw.strip()


class TestRepetitionDetection:
    def test_flags_runaway_repeated_layer_definitions(self):
        from arpa.agents.codegen_agent import _detect_repetition_warning

        # The actual pathology observed: hundreds of near-identical layer
        # definitions differing only by a numeric suffix.
        lines = [f"    self.fc{i} = nn.Linear(10, 10)" for i in range(1, 600)]
        code = "class M(nn.Module):\n    def __init__(self):\n" + "\n".join(lines)

        warning = _detect_repetition_warning(code)

        assert warning is not None
        assert "repetition" in warning.lower()

    def test_normal_model_code_is_not_flagged(self):
        from arpa.agents.codegen_agent import _detect_repetition_warning

        assert _detect_repetition_warning(_MODEL_PY) is None
        assert _detect_repetition_warning(_TRAIN_PY) is None

    def test_short_files_are_never_flagged(self):
        from arpa.agents.codegen_agent import _detect_repetition_warning

        # A handful of genuinely repeated lines shouldn't false-positive on
        # tiny files -- the threshold requires a minimum line count.
        code = "\n".join(["x = 1"] * 5)
        assert _detect_repetition_warning(code) is None


class TestWriteFilesQuarantine:
    def test_clean_file_written_under_its_own_name(self, tmp_path: Path):
        from arpa.agents.codegen_agent import GeneratedFile

        agent = CodeGenAgent(llm=MockCodegenLLM())
        gen_file = GeneratedFile(
            path="model.py", content="import torch\n", purpose="x", verified=True
        )

        agent._write_files([gen_file], tmp_path)

        assert (tmp_path / "model.py").exists()
        assert not (tmp_path / "UNVERIFIED_model.py").exists()

    def test_unverified_file_is_quarantined(self, tmp_path: Path):
        from arpa.agents.codegen_agent import GeneratedFile

        agent = CodeGenAgent(llm=MockCodegenLLM())
        gen_file = GeneratedFile(
            path="model.py",
            content="def broken(",
            purpose="x",
            verified=False,
            syntax_errors=["unexpected EOF"],
        )

        agent._write_files([gen_file], tmp_path)

        assert not (tmp_path / "model.py").exists()
        assert (tmp_path / "UNVERIFIED_model.py").exists()

    def test_repetition_flagged_file_is_quarantined_even_if_syntactically_valid(
        self, tmp_path: Path
    ):
        from arpa.agents.codegen_agent import GeneratedFile

        agent = CodeGenAgent(llm=MockCodegenLLM())
        gen_file = GeneratedFile(
            path="model.py",
            content="import torch\n",
            purpose="x",
            verified=True,  # compiles fine
            quality_warnings=["Likely runaway repetition: ..."],
        )

        agent._write_files([gen_file], tmp_path)

        assert not (tmp_path / "model.py").exists()
        assert (tmp_path / "UNVERIFIED_model.py").exists()


class TestGenerateCodeFileDetectsRepetition:
    def test_repetition_in_llm_output_is_recorded_on_the_generated_file(self):
        """End-to-end through _generate_code_file: a mock LLM that returns
        the degenerate pattern should come back with quality_warnings set,
        even though the code is syntactically valid."""

        class RepeatingLLM:
            general_model = "mock-general"
            code_model = "mock-code"

            def chat(self, messages, *, model=None, temperature=0.1, format_json=False):
                lines = [f"        self.fc{i} = nn.Linear(10, 10)" for i in range(1, 600)]
                body = "\n".join(lines)
                return (
                    "```python\n"
                    "import torch.nn as nn\n\n"
                    "class M(nn.Module):\n"
                    "    def __init__(self):\n"
                    "        super().__init__()\n"
                    f"{body}\n"
                    "```"
                )

        agent = CodeGenAgent(llm=RepeatingLLM())
        result = CodegenResult()
        gen_file = agent._generate_code_file(
            prompt="generate a model",
            path="model.py",
            purpose="test",
            empty_warning="empty",
            error_label="failed",
            result=result,
        )

        assert gen_file is not None
        assert gen_file.verified  # it's syntactically valid Python
        assert gen_file.quality_warnings  # but flagged as degenerate


class TestSyntaxRepair:
    """The self-repair loop: a file that fails compile() gets one repair
    call with the exact SyntaxError fed back to the model before being
    accepted or given up on."""

    _BROKEN_MODEL_PY = (
        "import torch.nn as nn\n\n"
        "class Model(nn.Module)\n"  # missing colon -> SyntaxError
        "    def forward(self, x):\n"
        "        return x\n"
    )

    _FIXED_MODEL_PY = (
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return x\n"
    )

    def test_repairs_broken_code_on_syntax_error(self):
        class RepairsOnSecondCallLLM:
            general_model = "mock-general"
            code_model = "mock-code"

            def __init__(self):
                self.calls = 0

            def chat(self, messages, *, model=None, temperature=0.1, format_json=False):
                self.calls += 1
                code = (
                    TestSyntaxRepair._BROKEN_MODEL_PY
                    if self.calls == 1
                    else TestSyntaxRepair._FIXED_MODEL_PY
                )
                return f"```python\n{code}```"

        llm = RepairsOnSecondCallLLM()
        agent = CodeGenAgent(llm=llm)
        result = CodegenResult()
        gen_file = agent._generate_code_file(
            prompt="generate a model",
            path="model.py",
            purpose="test",
            empty_warning="empty",
            error_label="failed",
            result=result,
        )

        assert llm.calls == 2  # original generation + one repair attempt
        assert gen_file is not None
        assert gen_file.verified
        assert gen_file.syntax_errors == []
        # _extract_code() strips the fenced block, so compare stripped forms
        assert gen_file.content == TestSyntaxRepair._FIXED_MODEL_PY.strip()
        assert any("Repaired syntax error in model.py" in msg for msg in result.generation_log)

    def test_gives_up_cleanly_if_repair_also_fails(self):
        class AlwaysBrokenLLM:
            general_model = "mock-general"
            code_model = "mock-code"

            def __init__(self):
                self.calls = 0

            def chat(self, messages, *, model=None, temperature=0.1, format_json=False):
                self.calls += 1
                return f"```python\n{TestSyntaxRepair._BROKEN_MODEL_PY}```"

        llm = AlwaysBrokenLLM()
        agent = CodeGenAgent(llm=llm)
        result = CodegenResult()
        gen_file = agent._generate_code_file(
            prompt="generate a model",
            path="model.py",
            purpose="test",
            empty_warning="empty",
            error_label="failed",
            result=result,
        )

        assert llm.calls == 2  # still only one repair attempt, not a retry loop
        assert gen_file is not None
        assert not gen_file.verified
        assert gen_file.syntax_errors  # original error preserved
        # untouched, not corrupted (modulo the same strip() _extract_code() always applies)
        assert gen_file.content == TestSyntaxRepair._BROKEN_MODEL_PY.strip()
        assert any("Syntax repair did not fix model.py" in msg for msg in result.generation_log)

    def test_verified_file_never_triggers_a_repair_call(self):
        class CountingLLM:
            general_model = "mock-general"
            code_model = "mock-code"

            def __init__(self):
                self.calls = 0

            def chat(self, messages, *, model=None, temperature=0.1, format_json=False):
                self.calls += 1
                return f"```python\n{TestSyntaxRepair._FIXED_MODEL_PY}```"

        llm = CountingLLM()
        agent = CodeGenAgent(llm=llm)
        result = CodegenResult()
        gen_file = agent._generate_code_file(
            prompt="generate a model",
            path="model.py",
            purpose="test",
            empty_warning="empty",
            error_label="failed",
            result=result,
        )

        assert llm.calls == 1  # no repair call needed
        assert gen_file is not None
        assert gen_file.verified
