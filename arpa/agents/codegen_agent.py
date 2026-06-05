"""Code generation agent - Innovation 3: autonomous code generation.

Pipeline:
  1. Load MethodologySpec from extraction
  2. Generate dataset loading code (delegate to DatasetAgent)
  3. Generate model architecture code
  4. Generate training script
  5. Generate evaluation script
  6. Verify generated code compiles
  7. Return complete runnable codebase
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from arpa.core.config import ARPASettings, get_settings
from arpa.core.state import MethodologySpec
from arpa.models import LLMClient, get_llm_client


class GeneratedFile(BaseModel):
    """A single generated Python file."""

    path: str
    content: str
    purpose: str
    verified: bool = False
    syntax_errors: list[str] = Field(default_factory=list)


class CodegenResult(BaseModel):
    """Output bundle from a CodeGen Agent run."""

    files: list[GeneratedFile] = Field(default_factory=list)
    dataset_loading_code: str | None = None
    success: bool = False
    escalated: bool = False
    escalation_reason: str | None = None
    generation_log: list[str] = Field(default_factory=list)
    missing_details_resolved: dict[str, Any] = Field(default_factory=dict)


CODEGEN_SYSTEM = """You are an ML engineer generating complete, runnable PyTorch code from paper methodology.

Your code must:
- Be complete and syntactically correct Python
- Follow PyTorch best practices
- Include proper error handling
- Be well-documented with docstrings and comments
- Use the exact dataset, architecture, and hyperparameters from the paper
- Handle missing details gracefully with sensible defaults (marked with # DEFAULT comments)

Generate production-quality code, not research prototypes."""

MODEL_GEN_PROMPT = """Generate a complete PyTorch model definition file.

Dataset: {dataset_name}
Input shape: {input_shape}
Number of classes: {num_classes}

Architecture details:
{architecture_details}

Missing details that need defaults:
{missing_details}

Generate a complete model.py file with:
1. All necessary imports
2. A model class inheriting from nn.Module
3. Proper __init__ with all layers
4. Forward pass implementation
5. Docstrings explaining the architecture
6. Comments marking any DEFAULT assumptions

Return JSON: {{"code": "complete Python code as string"}}
"""

TRAIN_GEN_PROMPT = """Generate a complete PyTorch training script.

Dataset: {dataset_name}
Model architecture: {model_summary}

Training hyperparameters:
{training_details}

Evaluation:
{evaluation_details}

Missing details:
{missing_details}

Generate a complete train.py file with:
1. All necessary imports (including model, dataset_loader)
2. Argument parsing for hyperparameters
3. Dataset loading and preprocessing
4. Model initialization
5. Optimizer and loss function setup
6. Training loop with progress logging
7. Validation loop
8. Model checkpointing
9. Final evaluation on test set
10. Proper logging and error handling

Return JSON: {{"code": "complete Python code as string"}}
"""

DATASET_LOADER_PROMPT = """Generate a complete PyTorch dataset loading module.

Dataset: {dataset_name}
Input shape: {input_shape}
Num classes: {num_classes}
Train size: {train_size}
Test size: {test_size}

Preprocessing/transforms:
{transform_description}

Generate a complete dataset_loader.py file with:
1. Import from torchvision.datasets or implement custom Dataset
2. get_train_loader() function returning DataLoader
3. get_test_loader() function returning DataLoader
4. Proper transforms based on the paper
5. Data normalization if appropriate
6. Docstrings and comments

Return JSON: {{"code": "complete Python code as string"}}
"""


class CodeGenAgent:
    """Autonomous code generation from extracted methodology."""

    def __init__(
        self,
        settings: ARPASettings | None = None,
        *,
        llm: LLMClient | None = None,
        backend: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or get_llm_client(self.settings, backend=backend)
        # Use the code model from settings
        self.code_model = self.settings.gemini_code_model if backend != "ollama" else self.settings.ollama_code_model
        logger.info("CodeGenAgent initialized with code_model: {}", self.code_model)

    def run(
        self,
        methodology: MethodologySpec | None = None,
        *,
        methodology_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> CodegenResult:
        """Generate complete runnable code from methodology.

        Args:
            methodology: MethodologySpec object from extraction
            methodology_path: Path to JSON file containing MethodologySpec
            output_dir: Directory to write generated files (optional)

        Returns:
            CodegenResult with all generated files
        """
        result = CodegenResult()

        # Load methodology
        if methodology is None and methodology_path:
            methodology = self._load_methodology(methodology_path)
        
        if methodology is None:
            result.escalated = True
            result.escalation_reason = "No methodology provided"
            return result

        # Check if we have enough information
        if not methodology.dataset_description:
            result.escalated = True
            result.escalation_reason = "No dataset description available"
            return result

        logger.info("Starting code generation for dataset: {}", methodology.dataset_description.name)

        # Generate files in order
        try:
            # 1. Dataset loader
            dataset_file = self._generate_dataset_loader(methodology, result)
            if dataset_file:
                result.files.append(dataset_file)

            # 2. Model definition
            model_file = self._generate_model(methodology, result)
            if model_file:
                result.files.append(model_file)

            # 3. Training script
            train_file = self._generate_training_script(methodology, result)
            if train_file:
                result.files.append(train_file)

            # 4. Write files if output_dir provided
            if output_dir:
                self._write_files(result.files, Path(output_dir))

            result.success = True
            logger.info("Code generation complete: {} files generated", len(result.files))

        except Exception as exc:
            logger.error("Code generation failed: {}", exc)
            result.escalated = True
            result.escalation_reason = f"Code generation error: {exc}"

        return result

    def _load_methodology(self, path: str | Path) -> MethodologySpec:
        """Load MethodologySpec from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return MethodologySpec(**data)

    def _generate_dataset_loader(
        self,
        methodology: MethodologySpec,
        result: CodegenResult,
    ) -> GeneratedFile | None:
        """Generate dataset_loader.py file."""
        desc = methodology.dataset_description
        if not desc:
            return None

        logger.info("Generating dataset loader for {}", desc.name)

        prompt = DATASET_LOADER_PROMPT.format(
            dataset_name=desc.name,
            input_shape=desc.input_shape or [1, 28, 28],
            num_classes=desc.num_classes or 10,
            train_size=desc.train_size or "unknown",
            test_size=desc.test_size or "unknown",
            transform_description=desc.transform_description or "standard normalization",
        )

        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": CODEGEN_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=self.llm.code_model,
                format_json=True,
            )
            data = self.llm.extract_json(raw)
            code = data.get("code", "")

            if not code:
                logger.warning("Empty code generated for dataset loader")
                return None

            gen_file = GeneratedFile(
                path="dataset_loader.py",
                content=code,
                purpose="Dataset loading and preprocessing",
            )

            # Verify syntax
            self._verify_syntax(gen_file)
            result.generation_log.append("Generated dataset_loader.py")

            return gen_file

        except Exception as exc:
            logger.error("Dataset loader generation failed: {}", exc)
            result.generation_log.append(f"Dataset loader generation failed: {exc}")
            return None

    def _generate_model(
        self,
        methodology: MethodologySpec,
        result: CodegenResult,
    ) -> GeneratedFile | None:
        """Generate model.py file."""
        desc = methodology.dataset_description
        arch = methodology.architecture

        if not desc:
            return None

        # Check if this is a benchmark paper
        is_benchmark = (
            hasattr(methodology, 'benchmark_experiments') and 
            methodology.benchmark_experiments and 
            len(methodology.benchmark_experiments) > 0
        )

        if is_benchmark:
            logger.info("Detected benchmark paper - generating code for best performing model")
            return self._generate_benchmark_model(methodology, result)

        logger.info("Generating model architecture")

        # Collect architecture details
        arch_details = "No specific architecture mentioned in paper.\n"
        if arch and arch.notes:
            arch_details = arch.notes

        if arch and arch.components:
            arch_details += "\n\nComponents:\n"
            for comp in arch.components:
                arch_details += f"- {comp.name} ({comp.kind}): {comp.parameters}\n"

        # Collect missing details
        missing_details = []
        if methodology.assumptions_needed:
            for detail in methodology.assumptions_needed:
                if "architecture" in detail.field.lower() or "model" in detail.field.lower():
                    missing_details.append(
                        f"- {detail.field}: {detail.reason}"
                        + (f" [suggested: {detail.proposed_default}]" if detail.proposed_default else "")
                    )

        prompt = MODEL_GEN_PROMPT.format(
            dataset_name=desc.name,
            input_shape=desc.input_shape or [1, 28, 28],
            num_classes=desc.num_classes or 10,
            architecture_details=arch_details,
            missing_details="\n".join(missing_details) if missing_details else "None",
        )

        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": CODEGEN_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=self.code_model,
                format_json=True,
            )
            data = self.llm.extract_json(raw)
            code = data.get("code", "")

            if not code:
                logger.warning("Empty code generated for model")
                return None

            gen_file = GeneratedFile(
                path="model.py",
                content=code,
                purpose="Model architecture definition",
            )

            self._verify_syntax(gen_file)
            result.generation_log.append("Generated model.py")

            return gen_file

        except Exception as exc:
            logger.error("Model generation failed: {}", exc)
            result.generation_log.append(f"Model generation failed: {exc}")
            return None

    def _generate_benchmark_model(
        self,
        methodology: MethodologySpec,
        result: CodegenResult,
    ) -> GeneratedFile | None:
        """Generate model code for benchmark paper with multiple models."""
        desc = methodology.dataset_description
        
        # Find best performing model
        best_model = None
        best_metric = 0.0
        
        for exp in methodology.benchmark_experiments:
            if exp.dataset_metric and exp.dataset_metric > best_metric:
                best_metric = exp.dataset_metric
                best_model = exp
        
        if not best_model:
            logger.warning("No benchmark experiments found, falling back to simple model")
            return None
        
        logger.info(
            "Best model: {} with {}={:.4f}",
            best_model.model_name,
            best_model.metric_name,
            best_model.dataset_metric
        )
        
        prompt = f"""Generate PyTorch/sklearn code for implementing and training the best model from a benchmark paper.

Dataset: {desc.name}
Input shape: {desc.input_shape or [1, 28, 28]}
Number of classes: {desc.num_classes or 10}

Best performing model from benchmarks:
- Model: {best_model.model_name}
- Parameters: {best_model.parameters}
- Achieved accuracy: {best_model.dataset_metric} ({best_model.metric_name})
- Training setup: {best_model.repeats} runs with shuffled data, average reported

Additional benchmark results (for comparison):
{self._format_top_models(methodology.benchmark_experiments[:5])}

Generate a complete Python file that:
1. Imports necessary libraries (sklearn, pytorch, etc.)
2. Implements the {best_model.model_name} model with the exact parameters from the paper
3. Provides functions for training and evaluation
4. Includes proper data preprocessing (flatten images for sklearn models)
5. Supports training with cross-validation/shuffling as mentioned in paper
6. Logs metrics and compares against paper's reported accuracy
7. Includes docstrings and comments

The model should use sklearn if it's a classical ML model (like {best_model.model_name}),
or PyTorch if it's a deep learning model.

Return JSON: {{"code": "complete Python code as string"}}
"""

        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": CODEGEN_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=self.code_model,
                format_json=True,
            )
            data = self.llm.extract_json(raw)
            code = data.get("code", "")

            if not code:
                logger.warning("Empty code generated for benchmark model")
                return None

            gen_file = GeneratedFile(
                path="model.py",
                content=code,
                purpose=f"Best model from benchmark: {best_model.model_name}",
            )

            self._verify_syntax(gen_file)
            result.generation_log.append(f"Generated model.py for {best_model.model_name}")

            return gen_file

        except Exception as exc:
            logger.error("Benchmark model generation failed: {}", exc)
            result.generation_log.append(f"Benchmark model generation failed: {exc}")
            return None

    def _format_top_models(self, experiments: list) -> str:
        """Format top benchmark experiments for prompt."""
        lines = []
        for i, exp in enumerate(experiments[:5], 1):
            lines.append(
                f"{i}. {exp.model_name} (params: {exp.parameters}): "
                f"accuracy={exp.dataset_metric:.4f}"
            )
        return "\n".join(lines)

    def _generate_training_script(
        self,
        methodology: MethodologySpec,
        result: CodegenResult,
    ) -> GeneratedFile | None:
        """Generate train.py file."""
        desc = methodology.dataset_description
        training = methodology.training
        evaluation = methodology.evaluation

        if not desc:
            return None

        # Check if this is a benchmark paper
        is_benchmark = (
            hasattr(methodology, 'benchmark_experiments') and 
            methodology.benchmark_experiments and 
            len(methodology.benchmark_experiments) > 0
        )

        if is_benchmark:
            logger.info("Generating training script for benchmark paper")
            return self._generate_benchmark_training_script(methodology, result)

        logger.info("Generating training script")

        # Rest of the existing code for regular papers...
        # Collect training details
        training_details = "No specific training details in paper.\nUse sensible defaults.\n"
        if training:
            training_details = "Training configuration:\n"
            if training.optimizer:
                training_details += f"- Optimizer: {training.optimizer.value if hasattr(training.optimizer, 'value') else training.optimizer}\n"
            if training.learning_rate:
                training_details += f"- Learning rate: {training.learning_rate.value if hasattr(training.learning_rate, 'value') else training.learning_rate}\n"
            if training.batch_size:
                training_details += f"- Batch size: {training.batch_size.value if hasattr(training.batch_size, 'value') else training.batch_size}\n"
            if training.epochs:
                training_details += f"- Epochs: {training.epochs.value if hasattr(training.epochs, 'value') else training.epochs}\n"
            if training.loss_function:
                training_details += f"- Loss: {training.loss_function.value if hasattr(training.loss_function, 'value') else training.loss_function}\n"

        # Collect evaluation details
        eval_details = ""
        if evaluation:
            eval_details = "Evaluation:\n"
            if evaluation.metric_name:
                eval_details += f"- Metric: {evaluation.metric_name.value if hasattr(evaluation.metric_name, 'value') else evaluation.metric_name}\n"
            if evaluation.reported_metric:
                eval_details += f"- Target accuracy: {evaluation.reported_metric.value if hasattr(evaluation.reported_metric, 'value') else evaluation.reported_metric}\n"

        # Model summary
        model_summary = "SimpleCNN"  # Default
        if methodology.architecture and methodology.architecture.model_name:
            model_summary = (
                methodology.architecture.model_name.value
                if hasattr(methodology.architecture.model_name, "value")
                else str(methodology.architecture.model_name)
            )

        # Collect missing training details
        missing_details = []
        if methodology.assumptions_needed:
            for detail in methodology.assumptions_needed:
                if "training" in detail.field.lower() or "optimizer" in detail.field.lower():
                    missing_details.append(
                        f"- {detail.field}: {detail.reason}"
                        + (f" [suggested: {detail.proposed_default}]" if detail.proposed_default else "")
                    )

        prompt = TRAIN_GEN_PROMPT.format(
            dataset_name=desc.name,
            model_summary=model_summary,
            training_details=training_details,
            evaluation_details=eval_details,
            missing_details="\n".join(missing_details) if missing_details else "None",
        )

        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": CODEGEN_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=self.code_model,
                format_json=True,
            )
            data = self.llm.extract_json(raw)
            code = data.get("code", "")

            if not code:
                logger.warning("Empty code generated for training script")
                return None

            gen_file = GeneratedFile(
                path="train.py",
                content=code,
                purpose="Training and evaluation script",
            )

            self._verify_syntax(gen_file)
            result.generation_log.append("Generated train.py")

            return gen_file

        except Exception as exc:
            logger.error("Training script generation failed: {}", exc)
            result.generation_log.append(f"Training script generation failed: {exc}")
            return None

    def _generate_benchmark_training_script(
        self,
        methodology: MethodologySpec,
        result: CodegenResult,
    ) -> GeneratedFile | None:
        """Generate training script for benchmark paper that uses model.py."""
        desc = methodology.dataset_description
        
        # Find best performing model
        best_model = None
        best_metric = 0.0
        
        for exp in methodology.benchmark_experiments:
            if exp.dataset_metric and exp.dataset_metric > best_metric:
                best_metric = exp.dataset_metric
                best_model = exp
        
        if not best_model:
            logger.warning("No benchmark experiments found")
            return None
        
        prompt = f"""Generate a simple Python training script (train.py) for a benchmark experiment.

Dataset: {desc.name}
Best model: {best_model.model_name} with accuracy {best_model.dataset_metric}
Model parameters: {best_model.parameters}

IMPORTANT: The model.py file is already generated and contains:
- A main() function that loads data, trains the model, and evaluates it
- Functions: load_and_preprocess_fashion_mnist(), train_and_evaluate_svc()

Your train.py should be a SIMPLE wrapper that:
1. Import main from model.py
2. Call main() to run the training
3. That's it - keep it minimal

Example structure:
```python
from model import main

if __name__ == "__main__":
    main()
```

You can optionally add argparse for command-line flexibility or logging setup, but DO NOT:
- Create dummy model.py files
- Re-implement the model training logic
- Duplicate code from model.py

Return JSON: {{"code": "complete Python code as string"}}
"""

        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": CODEGEN_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=self.code_model,
                format_json=True,
            )
            data = self.llm.extract_json(raw)
            code = data.get("code", "")

            if not code:
                logger.warning("Empty code generated for benchmark training script")
                return None

            gen_file = GeneratedFile(
                path="train.py",
                content=code,
                purpose="Training script for benchmark model",
            )

            self._verify_syntax(gen_file)
            result.generation_log.append(f"Generated train.py for benchmark {best_model.model_name}")

            return gen_file

        except Exception as exc:
            logger.error("Benchmark training script generation failed: {}", exc)
            result.generation_log.append(f"Benchmark training script generation failed: {exc}")
            return None

    def _verify_syntax(self, gen_file: GeneratedFile) -> None:
        """Verify Python syntax of generated code."""
        try:
            compile(gen_file.content, gen_file.path, "exec")
            gen_file.verified = True
            logger.debug("Syntax verified: {}", gen_file.path)
        except SyntaxError as exc:
            gen_file.verified = False
            gen_file.syntax_errors.append(str(exc))
            logger.warning("Syntax error in {}: {}", gen_file.path, exc)

    def _write_files(self, files: list[GeneratedFile], output_dir: Path) -> None:
        """Write generated files to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for gen_file in files:
            file_path = output_dir / gen_file.path
            with open(file_path, "w") as f:
                f.write(gen_file.content)
            logger.info("Wrote {}", file_path)


def run_codegen_agent(
    methodology: MethodologySpec | None = None,
    **kwargs,
) -> CodegenResult:
    """Convenience entrypoint for graph nodes and scripts."""
    return CodeGenAgent().run(methodology, **kwargs)
