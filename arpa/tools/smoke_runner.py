"""Smoke-execute generated code to see whether it actually runs.

`compile()` proves a file parses. It does not prove the code works, and the
gap between those is where the interesting failures live: it parses one file
in isolation and never resolves an import, so a `train.py` doing
`from model import main` against a `model.py` that defines no `main` compiles
cleanly and dies the moment anyone runs it. That exact case passed the
syntax check on a 10/10 benchmark run.

This runs the generated modules for a few steps and reports what happened:

    import:model        modules import at all (resolves cross-file symbols)
    import:train        train.py's imports of model.py actually exist
    model_instantiates  an nn.Module subclass can be constructed
    forward_pass        a synthetic batch flows through
    backward_pass       gradients propagate
    output_shape        the classifier head matches the paper's class count

Deliberately NOT a training run. Nothing here downloads a dataset or trains
to convergence -- 6 of the 10 benchmark papers use ImageNet (~150GB, gated),
so a real run could never succeed unattended and would say nothing about
whether the code is sound. Synthetic tensors of the paper's declared input
shape exercise the same code paths in seconds.

Execution happens in a subprocess so a hang, a crash, or an import that
starts downloading cannot take the harness down with it -- the timeout is
load-bearing, not decoration. Docker is used when available for isolation,
falling back to a local subprocess (mirroring DatasetSandboxVerifier, which
made the same trade for the same reason).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

RESULT_MARKER = "ARPA_SMOKE_RESULT:"

# Long enough for a CPU forward/backward on a large model, short enough that a
# module which starts downloading ImageNet at import time is cut off rather
# than left to fill the disk.
DEFAULT_TIMEOUT_S = 180

# Modules are imported in this order so a failure is attributed to the file
# that caused it: model.py first, then the files that import from it.
DEFAULT_MODULE_ORDER = ("model", "dataset_loader", "train")


@dataclass
class SmokeExpectations:
    """What the paper says the model should accept and produce."""

    input_shape: list[int] | None = None   # CHW, no batch dimension
    num_classes: int | None = None


@dataclass
class SmokeResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)
    error: str | None = None
    raw_log: str = ""
    timed_out: bool = False
    # Third-party packages this machine lacks. Reported so the gap is visible
    # and installable, without counting against the generated code.
    missing_dependencies: list[str] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[dict]:
        return [c for c in self.checks if not c.get("ok") and not c.get("skipped")]

    @property
    def summary(self) -> str:
        # Skipped checks carry ok=True so they never fail a run, so they have
        # to come out of the numerator as well as the denominator -- counting
        # them as passes produced nonsense like "6/3 checks passed".
        skipped = sum(1 for c in self.checks if c.get("skipped"))
        ok = sum(1 for c in self.checks if c.get("ok") and not c.get("skipped"))
        total = len(self.checks) - skipped
        base = f"{ok}/{total} checks passed"
        return f"{base} ({skipped} skipped)" if skipped else base


class CodeSmokeRunner:
    """Runs generated model/train code far enough to prove it executes."""

    def __init__(
        self,
        *,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        use_docker: bool = False,
        image: str = "arpa-sandbox:latest",
    ) -> None:
        self.timeout_s = timeout_s
        self.use_docker = use_docker
        self.image = image

    def run(
        self,
        generated_dir: Path,
        expectations: SmokeExpectations | None = None,
    ) -> SmokeResult:
        generated_dir = Path(generated_dir)
        if not generated_dir.is_dir():
            return SmokeResult(passed=False, error=f"no generated dir: {generated_dir}")

        modules = self._modules_present(generated_dir)
        if not modules:
            return SmokeResult(passed=False, error="no generated .py files to run")

        harness = self._build_harness(modules, expectations or SmokeExpectations())
        script_path = generated_dir / "_arpa_smoke.py"
        script_path.write_text(harness, encoding="utf-8")

        try:
            if self.use_docker and self._docker_available():
                return self._run_in_docker(generated_dir, script_path)
            return self._run_local(script_path)
        except subprocess.TimeoutExpired:
            # Usually means an import began real work -- downloading a dataset,
            # or training because a file lacked an `if __name__` guard.
            return SmokeResult(
                passed=False,
                error=f"timed out after {self.timeout_s}s (likely downloading or training on import)",
                timed_out=True,
            )
        finally:
            # The harness is a build artifact, not output worth keeping next to
            # the generated code someone is meant to read.
            script_path.unlink(missing_ok=True)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _modules_present(generated_dir: Path) -> list[str]:
        """Importable module names, ordered so dependencies come first.

        UNVERIFIED_* files are skipped: they are quarantined precisely because
        they failed an earlier check, and running them would report a failure
        that is already known.
        """
        available = {
            p.stem
            for p in generated_dir.glob("*.py")
            if not p.stem.startswith(("UNVERIFIED_", "_"))
        }
        ordered = [m for m in DEFAULT_MODULE_ORDER if m in available]
        ordered += sorted(available - set(ordered))
        return ordered

    @staticmethod
    def _docker_available() -> bool:
        try:
            proc = subprocess.run(
                ["docker", "info"], capture_output=True, timeout=15, text=True
            )
            return proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _run_local(self, script_path: Path) -> SmokeResult:
        env = dict(os.environ)
        # Several torch builds ship a second OpenMP runtime; without this the
        # interpreter aborts on import and every check looks like a code fault.
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        # Keep a stray CUDA init out of a check that only needs CPU tensors.
        env.setdefault("CUDA_VISIBLE_DEVICES", "")

        proc = subprocess.run(
            [sys.executable, script_path.name],
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            cwd=str(script_path.parent),
            env=env,
        )
        return self._parse_log(proc.stdout + proc.stderr, exit_code=proc.returncode)

    def _run_in_docker(self, generated_dir: Path, script_path: Path) -> SmokeResult:
        proc = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network", "none",     # a smoke check must never fetch data
                "-v", f"{generated_dir.resolve()}:/work",
                "-w", "/work",
                "-e", "KMP_DUPLICATE_LIB_OK=TRUE",
                self.image,
                "python", script_path.name,
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        return self._parse_log(proc.stdout + proc.stderr, exit_code=proc.returncode)

    def _parse_log(self, raw_log: str, exit_code: int = 0) -> SmokeResult:
        for line in raw_log.splitlines():
            if line.startswith(RESULT_MARKER):
                try:
                    payload = json.loads(line[len(RESULT_MARKER):])
                except json.JSONDecodeError:
                    break
                checks = payload.get("checks", [])
                real_failures = [
                    c for c in checks if not c.get("ok") and not c.get("skipped")
                ]
                return SmokeResult(
                    passed=not real_failures,
                    checks=checks,
                    error=payload.get("error"),
                    raw_log=raw_log,
                    missing_dependencies=payload.get("missing_dependencies", []),
                )

        # No marker means the harness itself died -- a segfault, an abort, or a
        # module that called sys.exit() on import. Surface the tail of the log
        # rather than a bare "no marker", which explains nothing.
        tail = "\n".join(raw_log.strip().splitlines()[-6:])
        return SmokeResult(
            passed=False,
            error=f"harness produced no result (exit={exit_code}): {tail[:400]}",
            raw_log=raw_log,
        )

    @staticmethod
    def _build_harness(modules: list[str], exp: SmokeExpectations) -> str:
        return textwrap.dedent(f'''
            """Generated by CodeSmokeRunner. Safe to delete."""
            import json, sys, traceback, importlib, inspect

            MODULES = {modules!r}
            # Names ARPA is responsible for, whether or not the file exists.
            # Checking only the files present would excuse a train.py that
            # imports a model.py which was never generated -- the absence is
            # the defect, so it must not be filed under "third-party".
            ARPA_MODULES = set(MODULES) | {set(DEFAULT_MODULE_ORDER)!r}
            INPUT_SHAPE = {exp.input_shape!r}
            NUM_CLASSES = {exp.num_classes!r}

            checks = []
            def check(name, ok, detail="", skipped=False):
                checks.append({{
                    "name": name, "ok": bool(ok),
                    "detail": str(detail)[:300], "skipped": bool(skipped),
                }})

            def emit():
                print("{RESULT_MARKER}" + json.dumps({{
                    "checks": checks,
                    "missing_dependencies": sorted(set(globals().get("missing_deps", []))),
                }}))

            # -- imports -------------------------------------------------
            # The check compile() cannot make: does `from model import X`
            # resolve to something that exists?
            loaded = {{}}
            missing_deps = []
            for name in MODULES:
                try:
                    loaded[name] = importlib.import_module(name)
                    check("import:" + name, True)
                except ModuleNotFoundError as exc:
                    absent = (getattr(exc, "name", "") or "").split(".")[0]
                    if absent and absent not in ARPA_MODULES:
                        # A third-party package this machine lacks (timm,
                        # tensorflow_datasets). That is a fact about the
                        # environment, not about the generated code -- failing
                        # here would make the score measure the virtualenv.
                        missing_deps.append(absent)
                        check("import:" + name, True,
                              "optional dependency not installed: " + absent,
                              skipped=True)
                    else:
                        # One of ARPA's own files is missing or unimportable.
                        check("import:" + name, False,
                              type(exc).__name__ + ": " + str(exc))
                except Exception as exc:
                    # ImportError (symbol absent), SyntaxError, anything else
                    # raised while executing the module -- all real defects.
                    check("import:" + name, False,
                          type(exc).__name__ + ": " + str(exc))

            try:
                import torch
                import torch.nn as nn
            except Exception as exc:
                check("torch_available", False, str(exc))
                emit(); sys.exit(0)

            # -- find a model -------------------------------------------
            candidates = []
            for mod in loaded.values():
                for _, obj in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(obj, nn.Module) and obj is not nn.Module:
                        if obj.__module__ in loaded:
                            candidates.append(obj)
            # Definition order, last first. Model files are written bottom-up
            # -- building blocks, then the model that composes them -- so the
            # last class defined is almost always the one to exercise.
            # Inheritance depth was the previous heuristic and picked wrong on
            # a SimCLR file whose classes were SobelFilter,
            # StochasticDataAugmentation, ResNet50ProjectionHead: it tested the
            # Sobel utility and reported the paper as broken.
            def definition_line(cls):
                """Where the class sits in its file.

                Read off __init__'s code object rather than
                inspect.getsourcelines, which needs to resolve the source file
                and returns nothing for a module imported this way -- every
                class then tied at -1 and the ranking silently degraded to
                alphabetical, picking `Block` over the VisionTransformer that
                composes it.
                """
                for attr in ("__init__", "forward"):
                    func = getattr(cls, attr, None)
                    code = getattr(func, "__code__", None)
                    if code is not None:
                        return code.co_firstlineno
                try:
                    return inspect.getsourcelines(cls)[1]
                except (OSError, TypeError):
                    return -1

            def takes_single_input(cls):
                """Can forward() be driven by one tensor?

                Loss functions and metrics are nn.Modules too, and a SimCLR
                file ends with SupervisedContrastiveLoss, whose forward wants
                (features, labels). Feeding it one tensor raises a TypeError
                that looks like a broken model but is really the wrong object
                under test. A model takes exactly one required input.
                """
                try:
                    params = list(inspect.signature(cls.forward).parameters.values())[1:]
                except (TypeError, ValueError):
                    return True
                required = [
                    p for p in params
                    if p.default is inspect.Parameter.empty
                    and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
                return len(required) <= 1

            testable = [c for c in candidates if takes_single_input(c)]
            if testable:
                candidates = testable

            def takes_num_classes(cls):
                """Does the constructor accept num_classes?

                The strongest available signal for "this is the paper's model
                rather than one of its building blocks": a classifier is
                parameterised by class count, a Block or PatchEmbed is not.
                Definition order was tried first and is not reliable -- a DeiT
                file defines the model at line 34 and its Block at 165, while
                other files do the reverse.
                """
                try:
                    return "num_classes" in inspect.signature(cls.__init__).parameters
                except (TypeError, ValueError):
                    return False

            def rank(cls):
                return (
                    1 if takes_num_classes(cls) else 0,
                    1 if cls.__module__ == "model" else 0,
                    definition_line(cls),
                )

            candidates.sort(key=rank, reverse=True)

            if not candidates:
                # Legitimate for benchmark papers that use sklearn estimators
                # rather than a torch module -- not a failure.
                check("model_instantiates", True, "no nn.Module found (sklearn paper?)", skipped=True)
                check("forward_pass", True, "no model", skipped=True)
                check("backward_pass", True, "no model", skipped=True)
                emit(); sys.exit(0)

            def try_construct(cls):
                for kwargs in ({{}}, {{"num_classes": NUM_CLASSES}} if NUM_CLASSES else {{}}):
                    if kwargs.get("num_classes") is None and kwargs:
                        continue
                    try:
                        return cls(**kwargs), ""
                    except Exception as exc:
                        err = cls.__name__ + ": " + type(exc).__name__ + ": " + str(exc)
                return None, err

            # The top-ranked candidate is the paper's model. If it cannot be
            # constructed, that IS the result -- do not quietly fall back to a
            # helper class that happens to build. A SimCLR file whose model
            # raised "degrees should be a sequence of length 2" was reported as
            # a stride error inside its Sobel filter, which pointed at the
            # wrong file entirely.
            model, ctor_error = try_construct(candidates[0])
            if model is None:
                check("model_instantiates", False, ctor_error)
                check("forward_pass", False, "model could not be constructed")
                check("backward_pass", False, "model could not be constructed")
                emit(); sys.exit(0)

            if model is None:
                check("model_instantiates", False, ctor_error)
                check("forward_pass", False, "no model instance", skipped=False)
                check("backward_pass", False, "no model instance", skipped=False)
                emit(); sys.exit(0)
            check("model_instantiates", True, type(model).__name__)

            # -- forward -------------------------------------------------
            if not INPUT_SHAPE:
                check("forward_pass", True, "no input_shape extracted", skipped=True)
                check("backward_pass", True, "no input_shape extracted", skipped=True)
                check("output_shape", True, "no input_shape extracted", skipped=True)
                emit(); sys.exit(0)

            def primary_tensor(value):
                """The main output of a forward pass.

                Returning several values is normal, not broken: a VAE hands
                back (recon, mu, logvar) and a distillation model hands back
                (logits, aux). Calling .shape on the container flagged those
                correct models as failures. Convention is that the primary
                output comes first, so unwrap to it.
                """
                if isinstance(value, torch.Tensor):
                    return value
                if isinstance(value, dict):
                    for key in ("logits", "out", "output", "prediction"):
                        if key in value and isinstance(value[key], torch.Tensor):
                            return value[key]
                    value = list(value.values())
                if isinstance(value, (tuple, list)) and value:
                    for item in value:
                        if isinstance(item, torch.Tensor):
                            return item
                return None

            out = raw_out = None
            try:
                model.eval()
                x = torch.randn(2, *INPUT_SHAPE)
                raw_out = model(x)
                out = primary_tensor(raw_out)
                if out is None:
                    check("forward_pass", False,
                          "forward returned no tensor: " + type(raw_out).__name__)
                else:
                    shape = str(tuple(out.shape))
                    extra = ("" if isinstance(raw_out, torch.Tensor)
                             else " (first of " + type(raw_out).__name__ + ")")
                    check("forward_pass", True, "output " + shape + extra)
            except Exception as exc:
                check("forward_pass", False,
                      type(exc).__name__ + ": " + str(exc).splitlines()[0][:200])

            # -- output shape --------------------------------------------
            if out is not None and NUM_CLASSES:
                if not isinstance(raw_out, torch.Tensor):
                    # Multi-output model: which element carries the class
                    # logits is not knowable here, and a VAE's first output is
                    # a reconstruction whose last dim is image width, not a
                    # class count. Asserting against it would invent a failure.
                    check("output_shape", True,
                          "multi-output model, head not identifiable", skipped=True)
                else:
                    try:
                        last = out.shape[-1]
                        check("output_shape", last == NUM_CLASSES,
                              "got " + str(last) + ", paper says " + str(NUM_CLASSES))
                    except Exception as exc:
                        check("output_shape", False, str(exc))
            elif out is not None:
                check("output_shape", True, "num_classes unknown", skipped=True)

            # -- backward ------------------------------------------------
            if out is not None:
                try:
                    model.train()
                    out2 = primary_tensor(model(torch.randn(2, *INPUT_SHAPE)))
                    if out2 is None:
                        raise RuntimeError("forward returned no tensor to backprop from")
                    loss = out2.float().pow(2).mean()
                    loss.backward()
                    has_grad = any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.parameters() if p.requires_grad
                    )
                    check("backward_pass", has_grad,
                          "gradients populated" if has_grad else "no gradients after backward()")
                except Exception as exc:
                    check("backward_pass", False,
                          type(exc).__name__ + ": " + str(exc).splitlines()[0][:200])
            else:
                check("backward_pass", False, "forward failed")

            emit()
        ''').strip() + "\n"
