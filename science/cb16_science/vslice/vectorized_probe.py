"""Diagnostic-only probe for the vectorized VS-C execution engine.

It never emits a scientific verdict. It checks canonical-Physics equivalence,
runs short scalar/vectorized timing probes, and exercises the 4-worker seed
executor on a tiny copied diagnostic spec. The preregistered R0 evidence is not
re-run or reinterpreted.
"""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch

from . import controlled_tasks as tasks
from . import qualification as q
from . import qualification_vectorized as qv
from . import vectorized_tasks as vt
from .contracts import Direction, NominalAction
from .physics import execute_transition
from .policy import direction_index

RESULT_COMMAND = "cb16.vs-c-vectorized-probe@v1"
SMOKE_GENERATIONS = 4


def result_dir() -> Path:
    raw = (os.environ.get("CB16_RESULT_DIR") or "").strip()
    if not raw:
        raise RuntimeError("CB16_RESULT_DIR is not set")
    return Path(raw)


def _physics_equivalence(spec: Dict[str, Any]) -> Dict[str, Any]:
    env = tasks.task_a_environment(spec)
    # risk=1.0 forces the kappa=0.1 R0 Physics through cost-aware Permission
    # clipping/bisection, so this probe covers both interior and boundary cases.
    actions = (
        NominalAction(Direction.SHORT, 0.2),
        NominalAction(Direction.SHORT, 0.8),
        NominalAction(Direction.SHORT, 1.0),
        NominalAction(Direction.FLAT, 0.0),
        NominalAction(Direction.LONG, 0.2),
        NominalAction(Direction.LONG, 0.8),
        NominalAction(Direction.LONG, 1.0),
    )
    actual = []
    indices = []
    risks = []
    scalar = []
    for case_index, case in enumerate(env.cases):
        for action in actions:
            actual.append(case_index)
            indices.append(direction_index(action.direction))
            risks.append(action.requested_risk)
            scalar.append(
                execute_transition(
                    case.truth, action, env.open_next, env.close_next, env.config
                ).reward
            )
    vector = vt.task_a_reward_batch(
        env,
        torch.tensor(actual, dtype=torch.long),
        torch.tensor(indices, dtype=torch.long),
        # Fixed fixture actions are Python-float authority values. float64 here
        # makes this an equation-equivalence test rather than a float32 input
        # quantization test. Runtime Actor samples remain float32 by design.
        torch.tensor(risks, dtype=torch.float64),
    ).tolist()
    errors = [abs(a - b) for a, b in zip(scalar, vector)]
    return {
        "cases": len(errors),
        "max_abs_reward_error": max(errors, default=0.0),
        "passed": max(errors, default=0.0) <= 1e-12,
    }


def _timing_probe(spec: Dict[str, Any]) -> Dict[str, Any]:
    authority = q.parse_authority(spec)
    optimizer = q.parse_optimizer(spec)
    env_a = tasks.task_a_environment(spec)
    env_b = tasks.task_b_environment(spec)

    def timed(fn):
        start = time.perf_counter()
        fn()
        return time.perf_counter() - start

    torch.manual_seed(991)
    scalar_a = q.build_learner(authority, optimizer)
    scalar_a_seconds = timed(
        lambda: q.train_task_a(
            scalar_a,
            env_a,
            seed=991,
            mode=tasks.ARM_POSITIVE,
            generations=SMOKE_GENERATIONS,
        )
    )

    torch.manual_seed(991)
    vector_a = q.build_learner(authority, optimizer)
    vector_a_seconds = timed(
        lambda: qv._train_task_a(
            vector_a,
            env_a,
            seed=991,
            mode=tasks.ARM_POSITIVE,
            generations=SMOKE_GENERATIONS,
        )
    )

    torch.manual_seed(992)
    scalar_b = q.build_learner(authority, optimizer)
    scalar_b_seconds = timed(
        lambda: q.train_task_b(
            scalar_b,
            env_b,
            seed=992,
            mode=tasks.ARM_POSITIVE,
            generations=SMOKE_GENERATIONS,
        )
    )

    torch.manual_seed(992)
    vector_b = q.build_learner(authority, optimizer)
    vector_b_seconds = timed(
        lambda: qv._train_task_b(
            vector_b,
            env_b,
            seed=992,
            mode=tasks.ARM_POSITIVE,
            generations=SMOKE_GENERATIONS,
        )
    )

    return {
        "smoke_generations": SMOKE_GENERATIONS,
        "task_a": {
            "scalar_seconds": scalar_a_seconds,
            "vectorized_seconds": vector_a_seconds,
            "speedup": scalar_a_seconds / vector_a_seconds if vector_a_seconds else None,
        },
        "task_b": {
            "scalar_seconds": scalar_b_seconds,
            "vectorized_seconds": vector_b_seconds,
            "speedup": scalar_b_seconds / vector_b_seconds if vector_b_seconds else None,
        },
    }


def _worker_probe(spec: Dict[str, Any]) -> Dict[str, Any]:
    diagnostic = copy.deepcopy(spec)
    diagnostic["paired_seeds"] = [2201, 2202, 2203, 2204]
    diagnostic["task_a"]["generations"] = 2
    diagnostic["task_b"]["generations"] = 2
    start = time.perf_counter()
    records = qv.run_seed_set_vectorized(diagnostic, diagnostic["paired_seeds"], workers=4)
    elapsed = time.perf_counter() - start
    seeds = [record["seed"] for record in records]
    return {
        "workers": 4,
        "elapsed_seconds": elapsed,
        "returned_seed_order": seeds,
        "passed": seeds == diagnostic["paired_seeds"],
        "note": "diagnostic copied spec with 2 generations; no scientific gate is evaluated",
    }


def main() -> int:
    q.configure_deterministic_runtime()
    spec = q.load_spec()
    physics = _physics_equivalence(spec)
    timing = _timing_probe(spec)
    workers = _worker_probe(spec)
    passed = bool(physics["passed"] and workers["passed"])
    result = {
        "schema": "cb16.diagnostic.v1",
        "result_command": RESULT_COMMAND,
        "commit_sha": q.commit_sha(),
        "classification": "DIAGNOSTIC_PASS" if passed else "DIAGNOSTIC_FAIL",
        "scientific_verdict": "NOT_EVALUATED",
        "r0_evidence_reinterpreted": False,
        "engine": qv.ENGINE_ID,
        "max_seed_workers": qv.MAX_SEED_WORKERS,
        "physics_equivalence": physics,
        "timing": timing,
        "worker_probe": workers,
    }
    out = result_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = [
        "# VS-C Vectorized Execution Probe",
        "",
        f"- classification: **{result['classification']}**",
        "- scientific verdict: **NOT_EVALUATED**",
        "- R0 evidence is not re-run or reinterpreted.",
        f"- Task-A scalar/vectorized Physics max abs reward error: `{physics['max_abs_reward_error']}`",
        f"- Task-A {SMOKE_GENERATIONS}-generation speedup: `{timing['task_a']['speedup']:.3f}x`",
        f"- Task-B {SMOKE_GENERATIONS}-generation speedup: `{timing['task_b']['speedup']:.3f}x`",
        f"- 4-worker diagnostic seed order preserved: `{workers['passed']}`",
        "",
        "This probe is engineering evidence only. Any next scientific experiment must use a new run identity.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(report))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
