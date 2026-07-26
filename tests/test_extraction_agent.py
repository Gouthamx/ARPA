"""Tests for methodology extraction."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from arpa.agents.extraction_agent import ExtractionAgent
from arpa.core.confidence import ConfidenceField, ConfidenceLevel
from arpa.core.state import (
    ArchitecturePass,
    ArchitectureSpec,
    CodegenFileSpec,
    CodegenMissingDetail,
    CodegenPlanSpec,
    DatasetDescription,
    DatasetTaskPass,
    EvaluationSpec,
    ImplementationPlanPass,
    ImplementationSpec,
    MethodologySpec,
    ModelComponentSpec,
    TrainingEvalPass,
    TrainingSpec,
)

T = TypeVar("T", bound=BaseModel)


class FakeLLM:
    general_model = "fake-general"
    code_model = "fake-code"

    def __init__(self, outputs: dict[type[BaseModel], BaseModel]) -> None:
        self.outputs = outputs
        self.prompts: list[str] = []
        self.schemas: list[type[BaseModel]] = []

    def complete_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
    ) -> T:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        return self.outputs[schema]  # type: ignore[return-value]


def cf(value, level=ConfidenceLevel.CONFIRMED, evidence="evidence"):
    return ConfidenceField(value=value, confidence=level, evidence=evidence)


def test_methodology_confidence_summary_counts_nested_fields():
    spec = MethodologySpec(
        architecture=ArchitectureSpec(
            model_name=cf("ResNet-20"),
            components=[
                ModelComponentSpec(
                    name="classifier",
                    kind="linear",
                    parameters={"out_features": 10},
                    confidence=ConfidenceLevel.CONFIRMED,
                )
            ],
        ),
        training=TrainingSpec(
            learning_rate=cf(0.1),
            optimizer=cf("SGD", ConfidenceLevel.INFERRED),
        ),
        evaluation=EvaluationSpec(
            # metric_name is a FlexibleStr (categorical field, unwrapped to a
            # plain string by design - see EvaluationSpec docstring), so it
            # never shows up as a ConfidenceField in confidence_summary().
            # reported_metric is the actually confidence-tracked field here.
            reported_metric=cf(93.4, ConfidenceLevel.ASSUMED),
        ),
    )

    assert spec.confidence_summary().as_dict() == {
        "confirmed": 3,
        "inferred": 1,
        "assumed": 1,
    }


def test_extraction_agent_runs_four_passes_and_merges_results():
    outputs: dict[type[BaseModel], BaseModel] = {
        DatasetTaskPass: DatasetTaskPass(
            dataset_description=DatasetDescription(
                name="CIFAR-10",
                train_size=50000,
                test_size=10000,
                input_shape=[3, 32, 32],
                num_classes=10,
            ),
            evaluation=EvaluationSpec(
                task_type=cf("image classification"),
                metric_name=cf("accuracy"),
                reported_metric=cf(93.4),
            ),
        ),
        ArchitecturePass: ArchitecturePass(
            architecture=ArchitectureSpec(
                model_name=cf("ResNet-20"),
                architecture_family=cf("Residual Network"),
                components=[
                    ModelComponentSpec(
                        name="residual block",
                        kind="residual_block",
                        parameters={"convolutions": 2},
                        confidence=ConfidenceLevel.CONFIRMED,
                        evidence="basic residual block",
                    )
                ],
            ),
            assumptions_needed=[
                CodegenMissingDetail(
                    field="classifier_head",
                    reason="classifier head dimensions are not stated",
                    severity="important",
                )
            ],
        ),
        TrainingEvalPass: TrainingEvalPass(
            training=TrainingSpec(
                optimizer=cf("SGD"),
                learning_rate=cf(0.1),
                batch_size=cf(128),
                scheduler=cf("step decay"),
            ),
            evaluation=EvaluationSpec(
                eval_split=cf("test"),
                protocol=cf("single-crop evaluation"),
            ),
            assumptions_needed=[
                CodegenMissingDetail(
                    field="weight_decay",
                    reason="weight decay missing",
                    severity="critical",
                )
            ],
        ),
        ImplementationPlanPass: ImplementationPlanPass(
            implementation=ImplementationSpec(
                framework=cf("PyTorch", ConfidenceLevel.INFERRED),
                language=cf("Python", ConfidenceLevel.INFERRED),
            ),
            codegen_plan=CodegenPlanSpec(
                target_task=cf("image classification"),
                framework=cf("PyTorch", ConfidenceLevel.INFERRED),
                entrypoint=cf("train.py", ConfidenceLevel.ASSUMED),
                model_class_name=cf("ResNet20", ConfidenceLevel.INFERRED),
                files=[
                    CodegenFileSpec(
                        path="model.py",
                        purpose="model definition",
                        required_symbols=["ResNet20"],
                    ),
                    CodegenFileSpec(
                        path="train.py",
                        purpose="training entrypoint",
                        required_symbols=["main"],
                        depends_on=["model.py", "dataset_loader.py"],
                    ),
                ],
                required_runtime_checks=[
                    "dataset batch shape",
                    "forward pass",
                    "loss computation",
                    "one optimizer step",
                ],
            ),
            assumptions_needed=[
                CodegenMissingDetail(
                    field="weight_decay",
                    reason="weight decay missing",
                    severity="critical",
                )
            ],
        ),
    }
    llm = FakeLLM(outputs)
    agent = ExtractionAgent(llm=llm)

    result = agent.run(
        "We train a ResNet-20 on CIFAR-10 with SGD and report 93.4% accuracy.",
        reduce_first=False,
    )

    assert llm.schemas == [
        DatasetTaskPass,
        ArchitecturePass,
        TrainingEvalPass,
        ImplementationPlanPass,
    ]
    assert "Pass 1/4" in llm.prompts[0]
    assert "Pass 4/4" in llm.prompts[-1]
    assert result.dataset_description is not None
    assert result.dataset_description.raw_context is not None
    assert result.architecture is not None
    assert result.architecture.model_name is not None
    assert result.architecture.model_name.value == "ResNet-20"
    assert result.training is not None
    assert result.training.batch_size is not None
    assert result.training.batch_size.value == 128
    assert result.evaluation is not None
    assert result.evaluation.reported_metric is not None
    assert result.evaluation.protocol is not None
    assert result.evaluation.reported_metric.value == 93.4
    assert result.codegen_plan is not None
    assert len(result.codegen_plan.files) == 2
    assert {item.field for item in result.assumptions_needed} == {
        "classifier_head",
        "weight_decay",
        "architecture.component.residual block",  # KB enrichment
    }


def test_extraction_agent_records_failed_pass_as_missing_detail():
    class FailingLLM(FakeLLM):
        def complete_structured(
            self,
            prompt: str,
            schema: type[T],
            *,
            model: str | None = None,
            system: str | None = None,
            temperature: float = 0.1,
        ) -> T:
            if schema is ArchitecturePass:
                raise RuntimeError("model unavailable")
            return super().complete_structured(
                prompt,
                schema,
                model=model,
                system=system,
                temperature=temperature,
            )

    outputs: dict[type[BaseModel], BaseModel] = {
        DatasetTaskPass: DatasetTaskPass(),
        TrainingEvalPass: TrainingEvalPass(),
        ImplementationPlanPass: ImplementationPlanPass(),
    }
    agent = ExtractionAgent(llm=FailingLLM(outputs))

    result = agent.run("paper text", reduce_first=False)

    assert any(item.field == "architecture" for item in result.assumptions_needed)
    assert any(item.severity == "critical" for item in result.assumptions_needed)


def test_extraction_agent_extracts_benchmark_experiment_rows():
    outputs: dict[type[BaseModel], BaseModel] = {
        DatasetTaskPass: DatasetTaskPass(),
        ArchitecturePass: ArchitecturePass(),
        TrainingEvalPass: TrainingEvalPass(),
        ImplementationPlanPass: ImplementationPlanPass(),
    }
    agent = ExtractionAgent(llm=FakeLLM(outputs))

    paper = """
    3 Experiments
    We provide some classification results in Table 3 to form a benchmark on this data set.
    All algorithms are repeated 5 times by shuffling the training data and the average
    accuracy on the test set is reported.
    Table 3: Benchmark on Fashion-MNIST (Fashion) and MNIST.
    Test Accuracy
    Classifier Parameter Fashion MNIST
    SVC C=10 kernel=rbf 0.897 0 .973
    C=1 kernel=linear 0.839 0 .929
    MLPClassifier activation=relu hidden_layer_sizes=[100, 10] 0.870 0 .972
    """

    result = agent.run(paper, reduce_first=False)

    assert len(result.benchmark_experiments) == 3
    first = result.benchmark_experiments[0]
    assert first.model_name == "SVC"
    assert first.parameters == {"C": 10, "kernel": "rbf"}
    assert first.dataset_metric == 0.897
    assert first.comparison_metric == 0.973
    assert first.repeats == 5
    assert first.aggregation == "average"
    assert result.benchmark_experiments[1].model_name == "SVC"
    assert result.benchmark_experiments[2].parameters["hidden_layer_sizes"] == [100, 10]
