"""Temporary branch-only wiring probe for R1 implementation review.

This file is reverted before merge. It emits diagnostic evidence only and never
produces an R1 scientific verdict.
"""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

from . import qualification as q
from . import task_a_variance_r1 as r1

RESULT_COMMAND = "cb16.vs-c-vectorized-probe@v1"
EXPECTED_SPEC_SHA256 = "4150ca184e0ee0a4a308018d531ceab9e08b7205ff09a06ef6ed568c25b7b5d6"


def _result_dir() -> Path:
    raw = (os.environ.get("CB16_RESULT_DIR") or "").strip()
    if not raw:
        raise RuntimeError("CB16_RESULT_DIR is not set")
    return Path(raw)


def main() -> int:
    q.configure_deterministic_runtime()
    spec = r1.load_spec()
    r1.validate_spec(spec)
    spec_hash = r1.spec_sha256(spec)

    diagnostic = copy.deepcopy(spec)
    diagnostic["task_a"]["generations"] = 1
    diagnostic["task_a"]["arms"][r1.BASELINE_ARM] = {
        "batch_trajectories": 18,
        "positive_per_actual_account": 6,
        "control_per_actual_observed_pair": 2,
    }
    diagnostic["task_a"]["arms"][r1.HIGH_BATCH_ARM] = {
        "batch_trajectories": 90,
        "positive_per_actual_account": 30,
        "control_per_actual_observed_pair": 10,
    }

    started = time.perf_counter()
    record = r1.run_seed(diagnostic, 1201)
    elapsed = time.perf_counter() - started
    baseline = record["arms"][r1.BASELINE_ARM]
    high = record["arms"][r1.HIGH_BATCH_ARM]
    same_positive_pre = baseline["positive"]["pre"] == high["positive"]["pre"]
    same_control_pre = baseline["control"]["pre"] == high["control"]["pre"]
    passed = bool(
        spec_hash == EXPECTED_SPEC_SHA256
        and same_positive_pre
        and same_control_pre
        and baseline["positive"]["diagnostics"]["generations"] == 1
        and high["control"]["diagnostics"]["generations"] == 1
    )
    result = {
        "schema": "cb16.diagnostic.v1",
        "result_command": RESULT_COMMAND,
        "commit_sha": q.commit_sha(),
        "classification": "DIAGNOSTIC_PASS" if passed else "DIAGNOSTIC_FAIL",
        "scientific_verdict": "NOT_EVALUATED",
        "r1_scientific_result_produced": False,
        "r1_import_ok": True,
        "r1_spec_sha256": spec_hash,
        "r1_spec_hash_matches_preregistration": spec_hash == EXPECTED_SPEC_SHA256,
        "same_positive_pre": same_positive_pre,
        "same_control_pre": same_control_pre,
        "tiny_execution_seconds": elapsed,
        "tiny_batches": {"baseline": 18, "high": 90, "generations": 1},
    }
    out = _result_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out / "REPORT.md").write_text(
        "# R1 branch-only wiring probe\n\n"
        f"- classification: **{result['classification']}**\n"
        "- scientific verdict: **NOT_EVALUATED**\n"
        f"- spec hash matches preregistration: `{result['r1_spec_hash_matches_preregistration']}`\n"
        f"- B18/B90 positive PRE identical: `{same_positive_pre}`\n"
        f"- B18/B90 control PRE identical: `{same_control_pre}`\n"
        f"- tiny execution seconds: `{elapsed:.6f}`\n"
    )
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
