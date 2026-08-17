"""Code generation agent - Innovation 3: autonomous code generation.

Pipeline:
  1. Load MethodologySpec from extraction
  2. Generate model architecture code
  3. Generate training script
  4. Verify generated code compiles
  5. Return complete runnable codebase

Note: Dataset loading code should come from DatasetAgent.loading_code
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from arpa.core.config import ARPASettings, get_settings
from arpa.core.state import MethodologySpec
from arpa.models import LLMClient, get_llm_client

# Code files (model.py / train.py) are often several hundred lines. Asking
# the LLM to return that as a properly-escaped JSON string value (the old
# `{"code": "..."}` approach) is fragile: small/medium models routinely
# forget to escape a quote or newline somewhere in a long string, and if the
# response gets cut off by the token limit the JSON is left unterminated.
# Both failures make the whole response unparseable even though the code
# itself might be fine. Asking for a plain fenced code block sidesteps both
# problems -- there's nothing to escape, and a truncated response still
# yields partial code we can inspect instead of zero files.
CODEGEN_MAX_TOKENS = 8192

# Disabled (None => not sent to the API at all).
#
# This was 0.4, to discourage the exact-token-repeat degeneration we'd seen
# (a model stuck emitting `self.fcN = nn.Linear(10, 10)` for hundreds of
# lines). The risk the old comment here anticipated -- "too high can push the
# model into avoiding legitimately repeated tokens that are normal in code"
# -- turned out to be exactly what was happening, and it was the single
# biggest cause of unusable output.
#
# Measured 2026-08-15 on nvidia/nemotron-3-super-120b-a12b, same codegen
# prompt, 4 trials each:
#     frequency_penalty=0.4 -> 1/4 generations compiled
#     frequency_penalty=0.0 -> 4/4 generations compiled
# Failures were `unexpected indent` / `expected an indented block`, plus
# real runs producing arguments mangled into identifiers
# (`nn.Conv2d(in_channels=1_out_channels_32_kernel_size_3_padding_1)`) --
# i.e. the penalty was suppressing the commas, `=` and indentation whitespace
# that Python source repeats more than any other tokens.
#
# The runaway-repetition case this was guarding against is still caught, just
# detected instead of suppressed: `_detect_repetition_warning()` flags such a
# file and `_write_files()` quarantines it under an UNVERIFIED_ prefix. If
# repetition does come back, prefer re-enabling this at a much lower value
# (<=0.1) and re-running the A/B above rather than returning to 0.4.
CODEGEN_FREQUENCY_PENALTY = None


def _extract_code(raw: str) -> str:
    """Pull Python source out of an LLM response.

    Tries, in order:
      1. A fenced code block (```python ... ``` or ``` ... ```) -- the
         format we now ask for.
      2. A JSON object with a "code" key -- kept for backend compatibility
         (e.g. Groq's native JSON mode) and for any leftover old prompts.
      3. The raw text as-is, on the assumption the model just returned code
         with no wrapper at all -- but first stripping a leading, *unclosed*
         ```python marker. That happens when generation gets cut off (token
         limit, a repetition loop) before the model ever emits the closing
         fence: the regex above requires a close and correctly won't match,
         but leaving the literal backtick/language-tag line in the saved
         file is still pure noise on top of the file being incomplete.
    """
    if not raw:
        return ""
    text = raw.strip()

    fence = re.search(r"```(?:python)?\s*\n([\s\S]*?)```", text)
    if fence:
        candidate = fence.group(1).strip()
        if candidate:
            return candidate

    if text.startswith("{") or '"code"' in text:
        for candidate_text in (text, text[text.find("{"): text.rfind("}") + 1]):
            if not candidate_text:
                continue
            try:
                data = json.loads(candidate_text)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(data, dict) and data.get("code"):
                return data["code"]

    unclosed_opening_fence = re.match(r"```(?:python)?[ \t]*\n?", text)
    if unclosed_opening_fence:
        text = text[unclosed_opening_fence.end():]

    return text


# `super(Foo).__init__()` -- one argument, no `self`. Valid syntax, so it
# compiles and passes every static check, but it builds an *unbound* super
# object whose __init__ never runs against the instance. In an nn.Module that
# surfaces as "cannot assign module before Module.__init__() call" on the very
# first layer assignment. One benchmark paper shipped three of them.
_UNBOUND_SUPER = re.compile(r"super\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\.\s*__init__\s*\(\s*\)")


def _repair_known_defects(code: str) -> tuple[str, list[str]]:
    """Mechanically fix generated-code mistakes with one unambiguous correction.

    Only patterns that are *always* wrong and have a *single* right answer
    belong here -- this rewrites the model's output, so a guess would be worse
    than the defect. Anything needing judgement goes to the LLM repair pass
    instead.

    Returns the repaired code and a list of what was changed.
    """
    notes: list[str] = []
    if not code:
        return code, notes

    repaired, count = _UNBOUND_SUPER.subn("super().__init__()", code)
    if count:
        notes.append(
            f"rewrote {count} unbound super(Cls).__init__() call(s) to super().__init__()"
        )
    return repaired, notes


_REPEATED_LINE_MIN_COUNT = 20
_REPEATED_LINE_MIN_FRACTION = 0.15


def _detect_repetition_warning(code: str) -> str | None:
    """Best-effort detector for degenerate, runaway-repetition generations.

    Some models get stuck in a loop and emit the same templated line (e.g.
    ``self.fc137 = nn.Linear(10, 10)``) hundreds of times until the token
    limit cuts them off. That output is syntactically valid Python --
    compile() won't catch it -- but it has no basis in the paper and isn't
    usable. Detect it by normalizing out numeric suffixes/literals per line
    and checking whether any single normalized shape dominates the file.

    Returns a human-readable warning string if repetition looks likely,
    otherwise None. This is a heuristic, not a proof -- it's meant to catch
    the obvious case we've actually seen, not to be a general code quality
    linter.
    """
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    if len(lines) < _REPEATED_LINE_MIN_COUNT:
        return None

    normalized_counts: dict[str, int] = {}
    for ln in lines:
        normalized = re.sub(r"\d+", "#", ln)
        normalized_counts[normalized] = normalized_counts.get(normalized, 0) + 1

    worst_pattern, worst_count = max(normalized_counts.items(), key=lambda kv: kv[1])
    fraction = worst_count / len(lines)
    if worst_count >= _REPEATED_LINE_MIN_COUNT and fraction >= _REPEATED_LINE_MIN_FRACTION:
        return (
            f"Likely runaway repetition: {worst_count}/{len(lines)} lines "
            f"({fraction:.0%}) match the pattern {worst_pattern!r}"
        )
    return None


def _extract_value(field: Any) -> Any:
    """Safely extract value from a field that might be a ConfidenceField or plain value.
    
    Args:
        field: Can be None, a plain value, or a ConfidenceField object
        
    Returns:
        The underlying value, or None if field is None
    """
    if field is None:
        return None
    # Try to use get_value() method if available (ConfidenceField)
    if hasattr(field, 'get_value') and callable(field.get_value):
        return field.get_value()
    # Fallback to .value attribute
    if hasattr(field, 'value'):
        return field.value
    # Plain value
    return field


class GeneratedFile(BaseModel):
    """A single generated Python file."""

    path: str
    content: str
    purpose: str
    verified: bool = False
    syntax_errors: list[str] = Field(default_factory=list)
    # Non-syntax quality concerns (e.g. detected runaway repetition) that
    # compile() can't catch but that still make the file unusable. Populated
    # by _detect_repetition_warning() in _generate_code_file().
    quality_warnings: list[str] = Field(default_factory=list)


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
- Call super().__init__() -- with no arguments -- as the FIRST statement of
  every nn.Module subclass __init__. Never super(ClassName).__init__(), which
  is legal syntax but leaves the module uninitialised and fails on the first
  layer assignment.
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

Return ONLY the Python code in a single fenced code block, like this:
```python
<complete file contents here>
```
Do not include any prose, explanation, or JSON wrapper before or after the code block.
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

Return ONLY the Python code in a single fenced code block, like this:
```python
<complete file contents here>
```
Do not include any prose, explanation, or JSON wrapper before or after the code block.
"""

SYNTAX_REPAIR_PROMPT = """The Python file below does not compile -- it has a syntax error.

Syntax error: {error}

File contents:
```python
{code}
```

Return the COMPLETE corrected file with the syntax error fixed. Keep everything else -- logic,
structure, imports, comments -- unchanged; only fix what's necessary to make it valid Python.

Return ONLY the corrected Python code in a single fenced code block, like this:
```python
<complete corrected file contents here>
```
Do not include any prose, explanation, or JSON wrapper before or after the code block.
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
        logger.info("CodeGenAgent initialized with code_model: {}", self.llm.code_model)

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
            # Note: dataset_loader.py should come from DatasetAgent.loading_code
            # CodeGenAgent focuses on model.py and train.py
            
            # 1. Model definition
            model_file = self._generate_model(methodology, result)
            if model_file:
                result.files.append(model_file)

            # 2. Training script
            train_file = self._generate_training_script(methodology, result)
            if train_file:
                result.files.append(train_file)

            # 3. Write files if output_dir provided
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

        return self._generate_code_file(
            prompt=prompt,
            path="model.py",
            purpose="Model architecture definition",
            empty_warning="Empty code generated for model",
            error_label="Model generation failed",
            result=result,
        )

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

Return ONLY the Python code in a single fenced code block, like this:
```python
<complete file contents here>
```
Do not include any prose, explanation, or JSON wrapper before or after the code block.
"""

        return self._generate_code_file(
            prompt=prompt,
            path="model.py",
            purpose=f"Best model from benchmark: {best_model.model_name}",
            empty_warning="Empty code generated for benchmark model",
            error_label="Benchmark model generation failed",
            result=result,
            success_log=f"Generated model.py for {best_model.model_name}",
        )

    def _format_top_models(self, experiments: list) -> str:
        """Format top benchmark experiments for prompt."""
        lines = []
        for i, exp in enumerate(experiments[:5], 1):
            lines.append(
                f"{i}. {exp.model_name} (params: {exp.parameters}): "
                f"accuracy={exp.dataset_metric:.4f}"
            )
        return "\n".join(lines)

    def _generate_code_file(
        self,
        *,
        prompt: str,
        path: str,
        purpose: str,
        empty_warning: str,
        error_label: str,
        result: CodegenResult,
        success_log: str | None = None,
    ) -> GeneratedFile | None:
        """Template method for code generation workflow.
        
        Handles the common sequence: LLM call → extract code → verify → log.
        This eliminates duplication across the four generation methods.
        
        Args:
            prompt: The complete prompt to send to the LLM
            path: File path for the generated file (e.g., "model.py")
            purpose: Description of the file's purpose
            empty_warning: Log message if code extraction returns empty
            error_label: Prefix for error log messages
            result: CodegenResult to append logs to
            success_log: Custom success log message (defaults to "Generated {path}")
            
        Returns:
            GeneratedFile if successful, None otherwise
        """
        try:
            raw = self._chat_for_code(prompt)
            code = _extract_code(raw)

            if not code:
                logger.warning(empty_warning)
                return None

            # Applied before verification: these defects compile cleanly, so a
            # syntax check would pass them straight through to a crash at
            # import or first use.
            code, repairs = _repair_known_defects(code)
            for note in repairs:
                logger.info("{}: {}", path, note)
                result.generation_log.append(f"{path}: {note}")

            gen_file = GeneratedFile(
                path=path,
                content=code,
                purpose=purpose,
            )

            self._verify_syntax(gen_file)

            if not gen_file.verified:
                pre_repair_error = "; ".join(gen_file.syntax_errors)
                self._repair_syntax(gen_file)
                if gen_file.verified:
                    result.generation_log.append(
                        f"Repaired syntax error in {path}: {pre_repair_error}"
                    )
                else:
                    result.generation_log.append(
                        f"Syntax repair did not fix {path}: {pre_repair_error}"
                    )

            repetition_warning = _detect_repetition_warning(gen_file.content)
            if repetition_warning:
                gen_file.quality_warnings.append(repetition_warning)
                logger.warning("{}: {}", path, repetition_warning)

            log_msg = success_log or f"Generated {path}"
            result.generation_log.append(log_msg)

            return gen_file

        except Exception as exc:
            logger.error("{}: {}", error_label, exc)
            result.generation_log.append(f"{error_label}: {exc}")
            return None

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
                training_details += f"- Optimizer: {_extract_value(training.optimizer)}\n"
            if training.learning_rate:
                training_details += f"- Learning rate: {_extract_value(training.learning_rate)}\n"
            if training.batch_size:
                training_details += f"- Batch size: {_extract_value(training.batch_size)}\n"
            if training.epochs:
                training_details += f"- Epochs: {_extract_value(training.epochs)}\n"
            # Labelled distinctly from epochs on purpose: papers that schedule
            # by iteration (ResNet's "60x10^4 iterations") state no epoch count,
            # and passing the number through as epochs would be off by orders
            # of magnitude.
            if getattr(training, "max_iterations", None):
                training_details += (
                    f"- Max training iterations (steps, NOT epochs): "
                    f"{_extract_value(training.max_iterations)}\n"
                )
            if training.loss_function:
                training_details += f"- Loss: {_extract_value(training.loss_function)}\n"

        # Collect evaluation details
        eval_details = ""
        if evaluation:
            eval_details = "Evaluation:\n"
            if evaluation.metric_name:
                eval_details += f"- Metric: {_extract_value(evaluation.metric_name)}\n"
            if evaluation.reported_metric:
                eval_details += f"- Target accuracy: {_extract_value(evaluation.reported_metric)}\n"

        # Model summary
        model_summary = "SimpleCNN"  # Default
        if methodology.architecture and methodology.architecture.model_name:
            model_summary = str(_extract_value(methodology.architecture.model_name) or "SimpleCNN")

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

        return self._generate_code_file(
            prompt=prompt,
            path="train.py",
            purpose="Training and evaluation script",
            empty_warning="Empty code generated for training script",
            error_label="Training script generation failed",
            result=result,
        )

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

Return ONLY the Python code in a single fenced code block, like this:
```python
<complete file contents here>
```
Do not include any prose, explanation, or JSON wrapper before or after the code block.
"""

        return self._generate_code_file(
            prompt=prompt,
            path="train.py",
            purpose="Training script for benchmark model",
            empty_warning="Empty code generated for benchmark training script",
            error_label="Benchmark training script generation failed",
            result=result,
            success_log=f"Generated train.py for benchmark {best_model.model_name}",
        )

    def _chat_for_code(self, prompt: str) -> str:
        """Call the LLM for a full source file, asking for plain fenced code.

        Requests a larger max_tokens budget than the default (code files run
        much longer than the small structured-extraction payloads) and a
        frequency_penalty to discourage runaway repetition loops (we've seen
        a model get stuck emitting the same templated line, e.g.
        `self.fc137 = nn.Linear(10, 10)`, hundreds of times). Tolerates
        backends whose `chat()` doesn't accept either kwarg yet.
        """
        messages = [
            {"role": "system", "content": CODEGEN_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        try:
            return self.llm.chat(
                messages,
                model=self.llm.code_model,
                format_json=False,
                max_tokens=CODEGEN_MAX_TOKENS,
                frequency_penalty=CODEGEN_FREQUENCY_PENALTY,
            )
        except TypeError:
            pass
        try:
            return self.llm.chat(
                messages,
                model=self.llm.code_model,
                format_json=False,
                max_tokens=CODEGEN_MAX_TOKENS,
            )
        except TypeError:
            # Backend's chat() doesn't support max_tokens either -- fall back
            # to the bare minimum signature every backend implements.
            return self.llm.chat(
                messages,
                model=self.llm.code_model,
                format_json=False,
            )

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

    def _repair_syntax(self, gen_file: GeneratedFile) -> None:
        """One repair pass for a file that failed `_verify_syntax()`.

        Feeds the exact `SyntaxError` back to the model and asks for a
        corrected version, mirroring the retry-with-repair pattern already
        used for malformed JSON elsewhere (`groq_client.py`'s `_try_parse`
        repair loop, `nvidia_client.py`'s `extract_json` retry) -- same idea,
        applied to Python syntax instead: give the model the specific error
        rather than asking it to regenerate blind. Small models are far more
        likely to fix one named error in otherwise-correct code than to
        write several hundred lines syntactically perfect on the first try.

        Mutates `gen_file` in place on success. On any failure along the way
        (API error, empty response, still doesn't compile), `gen_file` is
        left exactly as `_verify_syntax()` left it -- no worse off than
        without this repair pass.
        """
        original_error = "; ".join(gen_file.syntax_errors) or "unknown syntax error"
        logger.info("Attempting syntax repair for {} ({})", gen_file.path, original_error)

        prompt = SYNTAX_REPAIR_PROMPT.format(error=original_error, code=gen_file.content)

        try:
            raw = self._chat_for_code(prompt)
            repaired_code = _extract_code(raw)
        except Exception as exc:  # noqa: BLE001 - a failed repair call is not fatal
            logger.warning("Syntax repair call failed for {}: {}", gen_file.path, exc)
            return

        if not repaired_code:
            logger.warning("Syntax repair for {} returned empty code; keeping original", gen_file.path)
            return

        try:
            compile(repaired_code, gen_file.path, "exec")
        except SyntaxError as exc:
            logger.warning("Syntax repair for {} still doesn't compile: {}", gen_file.path, exc)
            return

        logger.info("Syntax repair succeeded for {}", gen_file.path)
        gen_file.content = repaired_code
        gen_file.verified = True
        gen_file.syntax_errors = []

    def _write_files(self, files: list[GeneratedFile], output_dir: Path) -> None:
        """Write generated files to disk.

        A file that failed syntax verification, or that _detect_repetition_warning()
        flagged as a likely runaway-repetition degeneration, is still written
        (useful for debugging what the LLM actually produced) but under an
        `UNVERIFIED_` prefix with a loud log warning -- so broken/degenerate
        output can never be mistaken for the real `model.py`/`train.py` at a
        glance, the way it previously could be.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for gen_file in files:
            is_clean = gen_file.verified and not gen_file.quality_warnings
            file_path = output_dir / (gen_file.path if is_clean else f"UNVERIFIED_{gen_file.path}")

            with open(file_path, "w") as f:
                f.write(gen_file.content)

            if is_clean:
                logger.info("Wrote {}", file_path)
            else:
                problems = gen_file.syntax_errors + gen_file.quality_warnings
                logger.warning(
                    "{} did not pass verification ({}); wrote to {} instead of {} "
                    "so it isn't mistaken for usable output",
                    gen_file.path,
                    "; ".join(problems) or "unknown reason",
                    file_path.name,
                    gen_file.path,
                )


def run_codegen_agent(
    methodology: MethodologySpec | None = None,
    **kwargs,
) -> CodegenResult:
    """Convenience entrypoint for graph nodes and scripts."""
    return CodeGenAgent().run(methodology, **kwargs)
