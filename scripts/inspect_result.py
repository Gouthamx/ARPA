"""Pretty-print key fields and the tail of the verification log from a result JSON.

Usage: python scripts/inspect_result.py .arpa_runs/<name>_result.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/inspect_result.py <result.json>")
        return 2
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    spec = data.get("spec") or {}
    print("dataset_name :", spec.get("dataset_name"))
    print("registry     :", spec.get("registry_source"), "/", spec.get("registry_id"))
    print("num_classes  :", spec.get("num_classes"))
    print("train/val/test:", spec.get("train_size"), spec.get("val_size"), spec.get("test_size"))
    print("input_shape  :", spec.get("input_shape"))
    print("verified     :", data.get("verified"))
    print("escalated    :", data.get("escalated"), "|", data.get("escalation_reason"))
    print("confidence   :", data.get("preprocess_confidence"))
    print("\n-- preprocess steps --")
    for s in spec.get("preprocess_steps", []):
        print(f"  [{s.get('confidence')}] {s.get('name')}")
    log = spec.get("verification_log") or ""
    print("\n-- verification log (tail) --")
    print(log[-1500:])
    # Surface ARPA_VERIFY_RESULT checks if present.
    marker = "ARPA_VERIFY_RESULT:"
    if marker in log:
        payload = log.split(marker)[-1].splitlines()[0]
        try:
            checks = json.loads(payload)
            print("\n-- checks --")
            for c in checks.get("checks", []):
                flag = "ok" if c.get("ok") else "FAIL"
                print(f"  [{flag}] {c.get('name')}: {c.get('detail')}")
            if checks.get("error"):
                print("  error:", checks["error"][:500])
        except json.JSONDecodeError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
