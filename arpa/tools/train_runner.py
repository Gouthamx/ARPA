"""Train generated models on real data and compare against the paper's number.

This is the stage that makes "autoreproduce" mean something. Everything before
it verifies form: the code parses (syntax check) and executes (smoke check).
Neither says whether the model *learns*, which is the only claim a
reproduction can actually make.

What this measures, stated precisely
------------------------------------
A short training run -- minutes, not the paper's full schedule -- on the real
dataset, reporting:

    trains        the loop runs: loss computed, backward, optimizer steps
    learns        loss falls and test accuracy clears chance by a real margin
    accuracy      test accuracy after the capped run
    gap           that accuracy against the paper's reported_metric

The gap is diagnostic, not a verdict. A two-epoch run is not a reproduction of
a paper that trained for ninety, and reporting "reproduced: no" because a
model reached 88% against a published 93% would be dishonest -- the run was
never given the chance. `learns` is the honest pass/fail; `gap` is context.

Feasibility gate
----------------
Only datasets that can be fetched and trained in minutes are attempted --
the MNIST family and CIFAR. ImageNet-based papers (6 of the 10 benchmark
papers) are skipped rather than failed: at ~150GB behind manual registration
they cannot run unattended, and recording that as a failure would say
something about the dataset rather than about ARPA.
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

RESULT_MARKER = "ARPA_TRAIN_RESULT:"

# A capped run, not a schedule. Enough signal to tell learning from noise,
# short enough to sit inside a verification run.
DEFAULT_MAX_STEPS = 200
DEFAULT_EVAL_BATCHES = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_TIMEOUT_S = 1800  # a cold CIFAR-10 fetch dominates this; MNIST-family is minutes

# Datasets small enough to fetch and train inside the timeout, mapped from the
# names extraction produces. Anything absent here is skipped by design.
TORCHVISION_DATASETS = {
    "mnist": ("MNIST", 10, [1, 28, 28]),
    "fashionmnist": ("FashionMNIST", 10, [1, 28, 28]),
    "fashion-mnist": ("FashionMNIST", 10, [1, 28, 28]),
    "kmnist": ("KMNIST", 10, [1, 28, 28]),
    "kuzushiji": ("KMNIST", 10, [1, 28, 28]),
    "kuzushiji-mnist": ("KMNIST", 10, [1, 28, 28]),
    "emnist": ("EMNIST", 47, [1, 28, 28]),
    "cifar10": ("CIFAR10", 10, [3, 32, 32]),
    "cifar-10": ("CIFAR10", 10, [3, 32, 32]),
    "cifar100": ("CIFAR100", 100, [3, 32, 32]),
    "cifar-100": ("CIFAR100", 100, [3, 32, 32]),
}

# Clearing chance by this margin is what separates "learned something" from
# "initialised randomly and got lucky".
LEARNING_MARGIN = 1.5


def resolve_dataset(name: str | None) -> tuple[str, int, list[int]] | None:
    """Map an extracted dataset name onto a trainable torchvision dataset."""
    if not name:
        return None
    key = "".join(ch for ch in str(name).lower() if ch.isalnum() or ch == "-")
    if key in TORCHVISION_DATASETS:
        return TORCHVISION_DATASETS[key]
    # Papers name datasets loosely ("the Fashion-MNIST benchmark"). Longest
    # key first: "mnist" is a substring of "fashionmnist" and "kmnist", so
    # shortest-first matching would resolve every MNIST variant to plain MNIST.
    for candidate in sorted(TORCHVISION_DATASETS, key=len, reverse=True):
        if candidate in key:
            return TORCHVISION_DATASETS[candidate]
    return None


def normalize_metric(value: float | None) -> float | None:
    """Put a reported metric on the same 0-1 scale as measured accuracy.

    Papers state accuracy both ways -- "96.22" and "0.9622" -- and extraction
    passes through whichever the text used. Comparing a percentage against a
    fraction produced "paper 9622.0%, gap +9556.1%", which is worse than no
    comparison at all: it looks like a catastrophic failure to reproduce.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1.0:
        # No metric on this scale legitimately exceeds 1.0, so a larger value
        # is a percentage. Values above 100 are not a scale we recognise.
        return number / 100.0 if number <= 100.0 else None
    return number


@dataclass
class TrainingOutcome:
    attempted: bool = False
    trains: bool = False
    learns: bool = False
    skipped_reason: str | None = None
    dataset: str | None = None
    steps: int = 0
    initial_loss: float | None = None
    final_loss: float | None = None
    test_accuracy: float | None = None
    chance_accuracy: float | None = None
    reported_metric: float | None = None
    error: str | None = None
    raw_log: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def gap(self) -> float | None:
        """Points between the capped run and the paper's figure.

        Positive means the short run is behind the paper, which is the normal
        case and not by itself a failure.
        """
        if self.test_accuracy is None or self.reported_metric is None:
            return None
        return round(self.reported_metric - self.test_accuracy, 4)

    @property
    def summary(self) -> str:
        if self.skipped_reason:
            return f"skipped ({self.skipped_reason})"
        if not self.trains:
            return f"training failed ({self.error or 'unknown'})"
        parts = [f"acc {self.test_accuracy:.1%}" if self.test_accuracy is not None else "acc n/a"]
        if self.reported_metric is not None:
            parts.append(f"paper {self.reported_metric:.1%}")
            if self.gap is not None:
                parts.append(f"gap {self.gap:+.1%}")
        parts.append("learning" if self.learns else "NOT learning")
        return ", ".join(parts)


class TrainingRunner:
    """Trains a generated model briefly on real data and scores the result."""

    def __init__(
        self,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        eval_batches: int = DEFAULT_EVAL_BATCHES,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        data_root: str | Path = "data",
    ) -> None:
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.eval_batches = eval_batches
        self.timeout_s = timeout_s
        self.data_root = Path(data_root)

    def run(
        self,
        generated_dir: Path,
        dataset_name: str | None,
        *,
        reported_metric: float | None = None,
        num_classes: int | None = None,
    ) -> TrainingOutcome:
        generated_dir = Path(generated_dir)
        spec = resolve_dataset(dataset_name)
        if spec is None:
            return TrainingOutcome(
                skipped_reason=f"'{dataset_name}' is not a small trainable dataset",
                dataset=dataset_name,
                reported_metric=normalize_metric(reported_metric),
            )
        if not generated_dir.is_dir():
            return TrainingOutcome(
                skipped_reason="no generated code",
                reported_metric=normalize_metric(reported_metric),
            )

        ds_class, ds_classes, ds_shape = spec
        script = generated_dir / "_arpa_train.py"
        script.write_text(
            self._build_harness(
                dataset_class=ds_class,
                num_classes=num_classes or ds_classes,
                input_shape=ds_shape,
                data_root=str(self.data_root.resolve()).replace("\\", "/"),
            ),
            encoding="utf-8",
        )

        try:
            outcome = self._run_local(script)
        except subprocess.TimeoutExpired:
            outcome = TrainingOutcome(
                attempted=True,
                error=f"timed out after {self.timeout_s}s",
            )
        finally:
            script.unlink(missing_ok=True)

        outcome.dataset = ds_class
        outcome.reported_metric = normalize_metric(reported_metric)
        return outcome

    # -- internals ---------------------------------------------------------

    def _run_local(self, script: Path) -> TrainingOutcome:
        env = dict(os.environ)
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
        proc = subprocess.run(
            [sys.executable, script.name],
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            cwd=str(script.parent),
            env=env,
        )
        return self._parse(proc.stdout + proc.stderr, proc.returncode)

    def _parse(self, raw_log: str, exit_code: int) -> TrainingOutcome:
        for line in raw_log.splitlines():
            if not line.startswith(RESULT_MARKER):
                continue
            try:
                payload = json.loads(line[len(RESULT_MARKER):])
            except json.JSONDecodeError:
                break
            acc = payload.get("test_accuracy")
            chance = payload.get("chance_accuracy")
            initial, final = payload.get("initial_loss"), payload.get("final_loss")
            learns = bool(
                acc is not None
                and chance
                and acc > chance * LEARNING_MARGIN
                and initial is not None
                and final is not None
                and final < initial
            )
            return TrainingOutcome(
                attempted=True,
                trains=payload.get("trains", False),
                learns=learns,
                skipped_reason=payload.get("skipped_reason"),
                steps=payload.get("steps", 0),
                initial_loss=initial,
                final_loss=final,
                test_accuracy=acc,
                chance_accuracy=chance,
                error=payload.get("error"),
                raw_log=raw_log,
                notes=payload.get("notes", []),
            )

        tail = "\n".join(raw_log.strip().splitlines()[-6:])
        return TrainingOutcome(
            attempted=True,
            error=f"no result marker (exit={exit_code}): {tail[:300]}",
            raw_log=raw_log,
        )

    def _build_harness(
        self, *, dataset_class: str, num_classes: int, input_shape: list[int], data_root: str
    ) -> str:
        return textwrap.dedent(f'''
            """Generated by TrainingRunner. Safe to delete."""
            import json, sys, inspect, importlib, warnings
            warnings.filterwarnings("ignore")

            DATASET   = {dataset_class!r}
            NUM_CLASSES = {num_classes!r}
            SHAPE     = {input_shape!r}
            DATA_ROOT = {data_root!r}
            MAX_STEPS = {self.max_steps!r}
            BATCH     = {self.batch_size!r}
            EVAL_BATCHES = {self.eval_batches!r}

            out = {{"trains": False, "notes": []}}
            def emit():
                print("{RESULT_MARKER}" + json.dumps(out))

            try:
                import torch, torch.nn as nn
                from torch.utils.data import DataLoader
                from torchvision import datasets, transforms
            except Exception as exc:
                out["error"] = "torch/torchvision unavailable: " + str(exc)
                emit(); sys.exit(0)

            # -- the model under test --------------------------------------
            # Driven directly rather than through the generated train.py: that
            # script's CLI and schedule vary per paper, and the question here
            # is whether the architecture learns, not whether its argparse
            # works. A capped, identical loop keeps papers comparable.
            model = None
            for mod_name in ("model", "train"):
                try:
                    mod = importlib.import_module(mod_name)
                except Exception:
                    continue
                classes = [
                    obj for _, obj in inspect.getmembers(mod, inspect.isclass)
                    if issubclass(obj, nn.Module) and obj is not nn.Module
                    and obj.__module__ == mod_name
                ]
                classes.sort(key=lambda c: len(c.__mro__), reverse=True)
                for cls in classes:
                    for kwargs in ({{"num_classes": NUM_CLASSES}}, {{}}):
                        try:
                            model = cls(**kwargs); break
                        except Exception:
                            continue
                    if model is not None:
                        break
                if model is not None:
                    break

            if model is None:
                # Benchmark papers that compare sklearn estimators (Fashion-MNIST
                # ships LinearSVC/KNN results, not a network) have no torch model
                # to train. That is the paper's design, not a codegen failure.
                out["skipped_reason"] = "no torch model in the generated code (sklearn paper?)"
                emit(); sys.exit(0)

            # -- real data -------------------------------------------------
            tfs = [transforms.ToTensor()]
            if SHAPE[0] == 3:
                tfs.append(transforms.Normalize((0.5,)*3, (0.5,)*3))
            else:
                tfs.append(transforms.Normalize((0.5,), (0.5,)))
            tf = transforms.Compose(tfs)

            try:
                ds_cls = getattr(datasets, DATASET)
                kw = {{"root": DATA_ROOT, "download": True, "transform": tf}}
                if DATASET == "EMNIST":
                    kw["split"] = "balanced"
                train_ds = ds_cls(train=True, **kw)
                test_ds  = ds_cls(train=False, **kw)
            except Exception as exc:
                out["error"] = "dataset unavailable: " + type(exc).__name__ + ": " + str(exc)[:200]
                emit(); sys.exit(0)

            train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
            test_dl  = DataLoader(test_ds, batch_size=BATCH)

            # The generated model may expect a different channel count than the
            # dataset provides (a paper's model built for RGB, an MNIST loader
            # giving 1 channel). Adapt the input rather than call it a failure:
            # the mismatch is a codegen finding the smoke stage already reports.
            # What the MODEL wants, which is not always what the dataset gives.
            # Reading it off the first conv layer rather than assuming the
            # dataset's shape: a generated DenseNet for CIFAR-10 was built with
            # a 1-channel stem while CIFAR is RGB, and adapting toward the
            # dataset left the mismatch in place ("expected input to have 1
            # channels, but got 3"). The mismatch is a real codegen defect and
            # is recorded as a note -- but it must not stop us measuring
            # whether the model learns, which is the question this stage asks.
            model_in_channels = None
            for layer in model.modules():
                if isinstance(layer, nn.Conv2d):
                    model_in_channels = layer.in_channels
                    break
            if model_in_channels is None:
                for layer in model.modules():
                    if isinstance(layer, nn.Linear):
                        break

            if model_in_channels is not None and model_in_channels != SHAPE[0]:
                out["notes"].append(
                    "model expects " + str(model_in_channels) + " input channel(s), "
                    + DATASET + " provides " + str(SHAPE[0]) + "; adapted to measure learning"
                )

            def fit_input(x):
                want_c = model_in_channels or SHAPE[0]
                if x.shape[1] == want_c:
                    return x
                if want_c == 3 and x.shape[1] == 1:
                    return x.repeat(1, 3, 1, 1)
                if want_c == 1 and x.shape[1] == 3:
                    return x.mean(dim=1, keepdim=True)
                return x

            def primary(v):
                if isinstance(v, torch.Tensor):
                    return v
                if isinstance(v, dict):
                    v = list(v.values())
                if isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, torch.Tensor):
                            return item
                return None

            # -- is this even a classifier? --------------------------------
            # Accuracy is meaningless for a generative model: a VAE's output is
            # a reconstructed image, not class logits, and cross-entropy
            # against it fails on shape. Papers like that are skipped with a
            # reason rather than scored against a metric they never claimed.
            try:
                with torch.no_grad():
                    probe_x, _ = next(iter(train_dl))
                    probe = primary(model(fit_input(probe_x)))
                if probe is None:
                    out["error"] = "forward returned no tensor"
                    emit(); sys.exit(0)
                if probe.dim() != 2:
                    out["skipped_reason"] = (
                        "model output is " + str(tuple(probe.shape[1:]))
                        + ", not class logits (generative/non-classifier model)"
                    )
                    emit(); sys.exit(0)
            except Exception as exc:
                out["error"] = "probe forward failed: " + type(exc).__name__ + ": " + str(exc)[:200]
                emit(); sys.exit(0)

            # -- train -----------------------------------------------------
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            losses, steps = [], 0
            model.train()
            try:
                for x, y in train_dl:
                    logits = primary(model(fit_input(x)))
                    if logits is None:
                        out["error"] = "forward returned no tensor"
                        emit(); sys.exit(0)
                    if logits.shape[-1] != NUM_CLASSES:
                        out["notes"].append(
                            "head is " + str(logits.shape[-1]) + " wide, dataset has "
                            + str(NUM_CLASSES) + " classes; scored on the overlap"
                        )
                    loss = criterion(logits[:, :NUM_CLASSES] if logits.shape[-1] > NUM_CLASSES else logits,
                                     y.clamp(max=logits.shape[-1]-1))
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                    losses.append(float(loss.item())); steps += 1
                    if steps >= MAX_STEPS:
                        break
            except Exception as exc:
                out["error"] = "training failed: " + type(exc).__name__ + ": " + str(exc)[:200]
                out["steps"] = steps
                emit(); sys.exit(0)

            if not losses:
                out["error"] = "no training steps ran"
                emit(); sys.exit(0)

            out["trains"] = True
            out["steps"] = steps
            # Averaged over a window: single-batch losses are too noisy to say
            # anything about the direction of travel.
            head = sum(losses[:10]) / min(10, len(losses))
            tail = sum(losses[-10:]) / min(10, len(losses))
            out["initial_loss"] = round(head, 4)
            out["final_loss"] = round(tail, 4)

            # -- evaluate --------------------------------------------------
            model.eval()
            correct = total = 0
            try:
                with torch.no_grad():
                    for i, (x, y) in enumerate(test_dl):
                        if i >= EVAL_BATCHES:
                            break
                        logits = primary(model(fit_input(x)))
                        pred = logits[:, :NUM_CLASSES].argmax(dim=1)
                        correct += int((pred == y).sum()); total += int(y.numel())
                out["test_accuracy"] = round(correct / total, 4) if total else None
                out["chance_accuracy"] = round(1.0 / NUM_CLASSES, 4)
            except Exception as exc:
                out["error"] = "evaluation failed: " + type(exc).__name__ + ": " + str(exc)[:200]

            emit()
        ''').strip() + "\n"
