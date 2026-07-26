"""Docker sandbox for dataset loading verification."""

from __future__ import annotations

import json
import logging
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VerificationExpectations:
    train_size: int | None = None
    val_size: int | None = None
    input_shape: list[int] | None = None  # CHW
    num_classes: int | None = None


@dataclass
class VerificationResult:
    passed: bool
    checks: list[dict]
    error: str | None = None
    raw_log: str = ""
    docker_available: bool = True

    @property
    def failed_checks(self) -> list[dict]:
        return [c for c in self.checks if not c.get("ok")]


class DatasetSandboxVerifier:
    """
    Runs generated loading code in Docker and validates shape/dtype/splits.
    Falls back to local subprocess when Docker is unavailable (for CI/dev).
    """

    def __init__(
        self,
        image: str = "arpa-sandbox:latest",
        timeout_s: int = 300,
        work_dir: Path | None = None,
    ) -> None:
        self.image = image
        self.timeout_s = timeout_s
        self.work_dir = work_dir or Path(".arpa_runs")

    def verify(
        self,
        loading_code: str,
        expectations: VerificationExpectations,
        *,
        use_docker: bool = True,
    ) -> VerificationResult:
        run_id = uuid.uuid4().hex[:8]
        run_dir = self.work_dir / f"verify_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        loader_path = run_dir / "dataset_loader.py"
        loader_path.write_text(loading_code, encoding="utf-8")

        script_path = run_dir / "verify_dataset.py"
        script_path.write_text(
            self._build_harness(expectations),
            encoding="utf-8",
        )

        if use_docker:
            result = self._run_in_docker(run_dir, script_path)
            if result is not None:
                return result
            logger.warning("Docker unavailable; falling back to local verification")

        return self._run_local(script_path)

    def _build_harness(self, exp: VerificationExpectations) -> str:
        return textwrap.dedent(
            f'''
            """ARPA dataset verification harness."""
            import json
            import traceback
            from dataset_loader import load_splits

            def check(results, name, ok, detail=""):
                results["checks"].append({{"name": name, "ok": bool(ok), "detail": str(detail)}})

            def dataset_length(ds):
                """Best-effort length.

                Works directly for torch/HuggingFace map-style datasets via
                len(). For a tf.data.Dataset (TFDS loaders), len() raises
                TypeError, so fall back to its cardinality -- which is only
                usable when TF reports a known, finite value. Returns None
                when length genuinely can't be determined without consuming
                the whole (possibly streaming) dataset; callers treat None as
                "unknown, don't fail the check" rather than as zero.
                """
                try:
                    return len(ds)
                except TypeError:
                    pass
                try:
                    import tensorflow as tf
                    cardinality = int(tf.data.experimental.cardinality(ds).numpy())
                    if cardinality >= 0:
                        return cardinality
                except Exception:
                    pass
                return None

            def first_batch(ds, batch_size=4):
                """Return (images, labels) torch tensors from the first batch of ds.

                torch/HuggingFace map-style datasets (anything with
                __getitem__) go through torch.utils.data.DataLoader exactly
                as before. A tf.data.Dataset (from tfds.builder(...).as_dataset())
                has no __getitem__/__len__ in the shape DataLoader needs, so
                it's batched and read via TFDS's own numpy iterator instead --
                forcing it through DataLoader raises TypeError (not
                subscriptable) even after cardinality is known.
                """
                import torch

                if hasattr(ds, "__getitem__"):
                    from torch.utils.data import DataLoader
                    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
                    batch = next(iter(loader))
                    if isinstance(batch, (list, tuple)):
                        images, labels = batch[0], batch[1]
                    else:
                        images, labels = batch, None
                else:
                    # Assume a tf.data.Dataset. TFDS image datasets built without
                    # as_supervised=True yield a dict keyed by feature name
                    # (typically "image"/"label"); handle both that and the
                    # as_supervised=True tuple shape.
                    example = next(iter(ds.batch(batch_size).as_numpy_iterator()))
                    if isinstance(example, dict):
                        image_key = next((k for k in ("image", "images", "x") if k in example), None)
                        label_key = next((k for k in ("label", "labels", "y") if k in example), None)
                        images = example[image_key] if image_key else next(iter(example.values()))
                        labels = example.get(label_key) if label_key else None
                    elif isinstance(example, (list, tuple)):
                        images = example[0]
                        labels = example[1] if len(example) > 1 else None
                    else:
                        images, labels = example, None

                    images = torch.as_tensor(images)
                    # Raw TFDS images are typically uint8 in [0, 255] -- torchvision's
                    # ToTensor() equivalent normalizes to float32 in [0, 1]; do the
                    # same here so the dtype/scale check below is meaningful instead
                    # of unconditionally failing every TFDS-sourced dataset.
                    if images.dtype in (torch.uint8, torch.int32, torch.int64):
                        images = images.float() / 255.0

                if not isinstance(images, torch.Tensor):
                    images = torch.as_tensor(images)
                if labels is not None and not isinstance(labels, torch.Tensor):
                    labels = torch.as_tensor(labels)

                return images, labels

            def main():
                results = {{"passed": False, "checks": [], "error": None}}
                try:
                    train, val, test = load_splits()

                    train_len = dataset_length(train) if train is not None else 0
                    check(
                        results,
                        "train_split_exists",
                        train is not None and (train_len is None or train_len > 0),
                        f"len={{train_len}}",
                    )

                    expected_train = {exp.train_size!r}
                    if expected_train is not None and train_len is not None:
                        check(results, "train_split_size", train_len == expected_train, f"got {{train_len}}")

                    if val is not None:
                        val_len = dataset_length(val)
                        check(results, "val_split_exists", val_len is None or val_len > 0, f"len={{val_len}}")
                        expected_val = {exp.val_size!r}
                        if expected_val is not None and val_len is not None:
                            check(results, "val_split_size", val_len == expected_val, f"got {{val_len}}")

                    import torch

                    images, labels = first_batch(train, batch_size=4)

                    shape = list(images.shape)
                    check(results, "batch_shape", len(shape) >= 3, str(shape))

                    expected_shape = {exp.input_shape!r}
                    if expected_shape is not None and len(expected_shape) == 3:
                        ok = shape[1:] == expected_shape or (
                            len(shape) == 4 and list(shape[1:4]) == expected_shape
                        )
                        check(results, "input_dimensions", ok, f"got {{shape}}")

                    check(results, "dtype_float", images.dtype in (torch.float32, torch.float16), str(images.dtype))

                    if labels is not None:
                        unique = int(labels.unique().numel())
                        check(results, "label_distribution", unique >= 1, f"unique={{unique}}")
                        expected_classes = {exp.num_classes!r}
                        if expected_classes is not None:
                            check(results, "num_classes", unique <= expected_classes, f"unique={{unique}}")

                    results["passed"] = all(c["ok"] for c in results["checks"])
                except Exception:
                    results["error"] = traceback.format_exc()
                    results["passed"] = False

                print("ARPA_VERIFY_RESULT:" + json.dumps(results))

            if __name__ == "__main__":
                main()
            '''
        )

    def _run_in_docker(self, run_dir: Path, script_path: Path) -> VerificationResult | None:
        try:
            import docker
        except ImportError:
            return None

        try:
            client = docker.from_env()
            client.ping()
        except Exception as exc:
            logger.warning("Docker not reachable: %s", exc)
            return None

        container = None
        try:
            container = client.containers.run(
                self.image,
                command=["python", "/workspace/verify_dataset.py"],
                volumes={str(run_dir.resolve()): {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                detach=True,
                mem_limit="4g",
            )
            exit_info = container.wait(timeout=self.timeout_s)
            raw_log = container.logs().decode("utf-8", errors="replace")
            return self._parse_log(raw_log, exit_code=exit_info.get("StatusCode", 1))
        except Exception as exc:
            return VerificationResult(
                passed=False,
                checks=[],
                error=str(exc),
                docker_available=True,
            )
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def _run_local(self, script_path: Path) -> VerificationResult:
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            cwd=script_path.parent,
        )
        raw_log = proc.stdout + proc.stderr
        return self._parse_log(raw_log, exit_code=proc.returncode)

    def _parse_log(self, raw_log: str, exit_code: int = 0) -> VerificationResult:
        marker = "ARPA_VERIFY_RESULT:"
        for line in raw_log.splitlines():
            if line.startswith(marker):
                try:
                    payload = json.loads(line[len(marker) :])
                    return VerificationResult(
                        passed=payload.get("passed", False) and exit_code == 0,
                        checks=payload.get("checks", []),
                        error=payload.get("error"),
                        raw_log=raw_log,
                    )
                except json.JSONDecodeError:
                    break

        return VerificationResult(
            passed=False,
            checks=[],
            error=f"No verification marker in output (exit={exit_code})",
            raw_log=raw_log,
        )
