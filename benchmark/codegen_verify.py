"""Verification helpers for CodeGenAgent output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arpa.agents.codegen_agent import CodegenResult, GeneratedFile


_ML_IMPORT_RE = re.compile(
    r"\b(import torch|from torch|import sklearn|from sklearn|import nn)\b",
    re.IGNORECASE,
)
_ENTRYPOINT_RE = re.compile(
    r"(__name__\s*==\s*['\"]__main__['\"]|def main\s*\(|if __name__)",
    re.IGNORECASE,
)
_NN_MODULE_RE = re.compile(r"class\s+\w+\(.*Module.*\)", re.IGNORECASE)


@dataclass
class FileCheck:
    path: str
    exists: bool
    verified: bool
    nonempty: bool
    has_ml_imports: bool
    has_entrypoint: bool
    has_model_class: bool
    byte_size: int
    syntax_errors: list[str] = field(default_factory=list)


@dataclass
class CodegenVerification:
    paper_key: str
    success: bool
    escalated: bool
    escalation_reason: str | None
    files: list[FileCheck]
    output_dir: str | None = None
    elapsed_s: float = 0.0
    error: str | None = None
    dataset_name: str | None = None
    generation_log: list[str] = field(default_factory=list)

    @property
    def has_model_py(self) -> bool:
        return any(f.path == "model.py" and f.exists for f in self.files)

    @property
    def has_train_py(self) -> bool:
        return any(f.path == "train.py" and f.exists for f in self.files)

    @property
    def model_verified(self) -> bool:
        for f in self.files:
            if f.path == "model.py":
                return f.verified
        return False

    @property
    def train_verified(self) -> bool:
        for f in self.files:
            if f.path == "train.py":
                return f.verified
        return False

    @property
    def passed(self) -> bool:
        if self.error or self.escalated:
            return False
        if not self.success:
            return False
        if not self.has_model_py or not self.has_train_py:
            return False
        if not self.model_verified or not self.train_verified:
            return False
        for f in self.files:
            if f.path in ("model.py", "train.py") and not f.nonempty:
                return False
        return True

    @property
    def score(self) -> float:
        """0–100 verification score."""
        if self.error or self.escalated:
            return 0.0
        points = 0.0
        weights = {
            "success": 15,
            "model_exists": 20,
            "train_exists": 20,
            "model_verified": 20,
            "train_verified": 20,
            "model_imports": 2.5,
            "train_entrypoint": 2.5,
        }
        if self.success:
            points += weights["success"]
        if self.has_model_py:
            points += weights["model_exists"]
        if self.has_train_py:
            points += weights["train_exists"]
        if self.model_verified:
            points += weights["model_verified"]
        if self.train_verified:
            points += weights["train_verified"]
        model = _file_by_path(self.files, "model.py")
        train = _file_by_path(self.files, "train.py")
        if model and model.has_ml_imports:
            points += weights["model_imports"]
        if train and train.has_entrypoint:
            points += weights["train_entrypoint"]
        return points

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_key": self.paper_key,
            "passed": self.passed,
            "score": self.score,
            "success": self.success,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "error": self.error,
            "dataset_name": self.dataset_name,
            "elapsed_s": round(self.elapsed_s, 2),
            "output_dir": self.output_dir,
            "generation_log": self.generation_log,
            "files": [
                {
                    "path": f.path,
                    "exists": f.exists,
                    "verified": f.verified,
                    "nonempty": f.nonempty,
                    "has_ml_imports": f.has_ml_imports,
                    "has_entrypoint": f.has_entrypoint,
                    "has_model_class": f.has_model_class,
                    "byte_size": f.byte_size,
                    "syntax_errors": f.syntax_errors,
                }
                for f in self.files
            ],
        }


def _file_by_path(files: list[FileCheck], path: str) -> FileCheck | None:
    for f in files:
        if f.path == path:
            return f
    return None


def _check_file(gen_file: GeneratedFile) -> FileCheck:
    content = gen_file.content or ""
    return FileCheck(
        path=gen_file.path,
        exists=True,
        verified=gen_file.verified,
        nonempty=len(content.strip()) > 0,
        has_ml_imports=bool(_ML_IMPORT_RE.search(content)),
        has_entrypoint=bool(_ENTRYPOINT_RE.search(content)),
        has_model_class=bool(_NN_MODULE_RE.search(content)),
        byte_size=len(content.encode("utf-8")),
        syntax_errors=list(gen_file.syntax_errors),
    )


def verify_codegen_result(
    paper_key: str,
    result: CodegenResult,
    *,
    output_dir: str | Path | None = None,
    elapsed_s: float = 0.0,
    dataset_name: str | None = None,
    error: str | None = None,
) -> CodegenVerification:
    """Verify a CodegenResult against structural correctness criteria."""
    file_checks = [_check_file(f) for f in result.files]

    written_ok = True
    if output_dir:
        out = Path(output_dir)
        for expected in ("model.py", "train.py"):
            disk_path = out / expected
            if not disk_path.exists():
                written_ok = False
                if not any(f.path == expected for f in file_checks):
                    file_checks.append(
                        FileCheck(
                            path=expected,
                            exists=False,
                            verified=False,
                            nonempty=False,
                            has_ml_imports=False,
                            has_entrypoint=False,
                            has_model_class=False,
                            byte_size=0,
                        )
                    )

    verification = CodegenVerification(
        paper_key=paper_key,
        success=result.success and written_ok,
        escalated=result.escalated,
        escalation_reason=result.escalation_reason,
        files=file_checks,
        output_dir=str(output_dir) if output_dir else None,
        elapsed_s=elapsed_s,
        dataset_name=dataset_name,
        error=error,
        generation_log=list(result.generation_log),
    )
    return verification


def print_verification_report(results: list[CodegenVerification]) -> None:
    print("\n" + "=" * 78)
    print("ARPA CodeGenAgent VERIFICATION REPORT (10 papers)")
    print("=" * 78)

    passed = sum(1 for r in results if r.passed)
    print(f"\nPassed: {passed}/{len(results)}")

    for vr in results:
        status = "PASS" if vr.passed else "FAIL"
        print(f"\n[{status}] {vr.paper_key}  score={vr.score:.0f}/100  "
              f"time={vr.elapsed_s:.1f}s")
        if vr.dataset_name:
            print(f"  dataset: {vr.dataset_name}")
        if vr.error:
            print(f"  ERROR: {vr.error}")
        if vr.escalated:
            print(f"  escalated: {vr.escalation_reason}")
        if vr.output_dir:
            print(f"  output: {vr.output_dir}")
        for f in vr.files:
            mark = "ok" if f.verified and f.nonempty else "!!"
            print(
                f"    [{mark}] {f.path:12} verified={f.verified} "
                f"size={f.byte_size}B imports={f.has_ml_imports} "
                f"entry={f.has_entrypoint}"
            )
            if f.syntax_errors:
                print(f"         syntax: {f.syntax_errors[0][:80]}")

    total_score = sum(r.score for r in results)
    max_score = len(results) * 100
    overall = (total_score / max_score * 100) if max_score else 0
    print("\n" + "-" * 78)
    print(f"  OVERALL: {total_score:.0f}/{max_score:.0f} ({overall:.0f}%)")
    print("-" * 78)
