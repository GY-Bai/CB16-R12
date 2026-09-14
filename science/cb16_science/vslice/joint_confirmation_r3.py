"""R12 VS-C R3 fresh-seed joint learnability confirmation."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from . import controlled_tasks as tasks
from . import qualification as q
from . import qualification_vectorized as qv
from . import task_a_direction_entropy_r2 as r2
from .contracts import ContractError

RESULT_COMMAND = "cb16.vs-c-joint-confirmation-r3@v1"
EXPERIMENT_ID = "r12.vs_c.joint_confirmation.r3"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_c_joint_confirmation_r3.json"
SPEC_FILENAME = "experiment_spec.json"
RESULT_FILENAME = "RESULT.json"
REPORT_FILENAME = "REPORT.md"
DIRECTION_ENTROPY_COEFFICIENT = 0.005
EXIT_PASS, EXIT_SCIENTIFIC_FAIL, EXIT_EXECUTION_BLOCKED, EXIT_CONTRACT_MISMATCH = 0, 1, 3, 4


def load_spec() -> Dict[str, Any]:
    try:
        parsed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"cannot read preregistered spec: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"preregistered spec is invalid JSON: {exc}") from exc
    return dict(tasks.spec_mapping(parsed, "preregistered spec"))


def spec_sha256(spec: Mapping[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def paired_seeds(spec: Mapping[str, Any]) -> Tuple[int, ...]:
    seeds = tuple(tasks.spec_int(x, "paired_seeds entry", minimum=1) for x in tasks.spec_sequence(spec["paired_seeds"], "paired_seeds"))
    if seeds != tuple(range(2201, 2209)):
        raise ContractError(f"R3 freezes paired seeds 2201..2208, got {seeds!r}")
    if set(seeds) & set(range(1201, 1209)):
        raise ContractError("R3 fresh seeds overlap R0/R1/R2 seeds")
    return seeds


def parse_optimizer(spec: Mapping[str, Any]) -> q.OptimizerSettings:
    optimizer = tasks.spec_mapping(spec["optimizer"], "optimizer")
    expected = {
        "actor", "actor_lr", "critic", "critic_lr", "discount_factor", "replay",
        "direction_entropy_coefficient", "direction_entropy_target", "beta_risk_entropy_bonus",
    }
    if set(optimizer) != expected:
        raise ContractError("R3 optimizer key set mismatch")
    if tasks.spec_str(optimizer["actor"], "optimizer.actor") != "Adam" or tasks.spec_str(optimizer["critic"], "optimizer.critic") != "Adam":
        raise ContractError("R3 requires Adam for actor and critic")
    actor_lr = tasks.spec_float(optimizer["actor_lr"], "optimizer.actor_lr")
    critic_lr = tasks.spec_float(optimizer["critic_lr"], "optimizer.critic_lr")
    if actor_lr != 0.001 or critic_lr != 0.001:
        raise ContractError("R3 freezes actor_lr=critic_lr=0.001")
    if tasks.spec_float(optimizer["discount_factor"], "optimizer.discount_factor") != 1.0:
        raise ContractError("R3 uses undiscounted return-to-go")
    if tasks.spec_bool(optimizer["replay"], "optimizer.replay"):
        raise ContractError("R3 forbids replay")
    if tasks.spec_float(optimizer["direction_entropy_coefficient"], "optimizer.direction_entropy_coefficient") != DIRECTION_ENTROPY_COEFFICIENT:
        raise ContractError("R3 freezes direction entropy coefficient at 0.005")
    if tasks.spec_str(optimizer["direction_entropy_target"], "optimizer.direction_entropy_target") != "categorical_direction_only":
        raise ContractError("R3 entropy target must be categorical_direction_only")
    if tasks.spec_float(optimizer["beta_risk_entropy_bonus"], "optimizer.beta_risk_entropy_bonus") != 0.0:
        raise ContractError("R3 forbids Beta-risk entropy")
    return q.OptimizerSettings(actor_lr=actor_lr, critic_lr=critic_lr)


def validate_spec(spec: Mapping[str, Any]) -> None:
    if tasks.spec_str(spec["experiment_id"], "experiment_id") != EXPERIMENT_ID:
        raise ContractError(f"experiment_id must be {EXPERIMENT_ID!r}")
    r2.parse_authority(spec)
    parse_optimizer(spec)
    paired_seeds(spec)
    task_a = tasks.spec_mapping(spec["task_a"], "task_a")
    task_b = tasks.spec_mapping(spec["task_b"], "task_b")
    if (tasks.spec_int(task_a["batch_trajectories"], "task_a.batch_trajectories"), tasks.spec_int(task_a["generations"], "task_a.generations")) != (900, 256):
        raise ContractError("R3 freezes Task A at B900 x 256")
    if (tasks.spec_int(task_b["batch_trajectories"], "task_b.batch_trajectories"), tasks.spec_int(task_b["generations"], "task_b.generations")) != (128, 256):
        raise ContractError("R3 freezes Task B at B128 x 256")
    tasks.task_a_environment(spec)
    tasks.task_b_environment(spec)
    joint = tasks.spec_mapping(spec["joint_gate"], "joint_gate")
    for key in ("task_a", "task_b"):
        tasks.spec_mapping(joint[key], f"joint_gate.{key}")
    global_gate = tasks.spec_mapping(spec["global_gate"], "global_gate")
    for key in ("implementation_tests_must_pass", "fresh_seed_set_must_not_overlap_r0_r1_r2", "no_threshold_changes_after_results"):
        if not tasks.spec_bool(global_gate[key], f"global_gate.{key}"):
            raise ContractError(f"R3 requires global_gate.{key}=true")
    expected_classes = {
        "scientific_gate_miss_classification": "SCIENTIFIC_FAIL",
        "contract_violation_classification": "CONTRACT_MISMATCH",
        "runtime_or_environment_blocker_classification": "EXECUTION_BLOCKED",
    }
    for key, expected in expected_classes.items():
        if tasks.spec_str(global_gate[key], f"global_gate.{key}") != expected:
            raise ContractError(f"global_gate.{key} mismatch")
    required = {tasks.spec_str(x, "required_artifacts entry") for x in tasks.spec_sequence(spec["required_artifacts"], "required_artifacts")}
    if required != {SPEC_FILENAME, RESULT_FILENAME, REPORT_FILENAME}:
        raise ContractError("R3 required_artifacts mismatch")


def _seed_worker(payload: tuple[dict[str, Any], int]) -> Dict[str, Any]:
    spec, seed = payload
    q.configure_deterministic_runtime()
    authority = r2.parse_authority(spec)
    optimizer = parse_optimizer(spec)
    env_a = tasks.task_a_environment(spec)
    env_b = tasks.task_b_environment(spec)
    return qv.run_paired_seed_vectorized(
        spec, seed, authority, optimizer, env_a, env_b,
        direction_entropy_coefficient=DIRECTION_ENTROPY_COEFFICIENT,
    )


def run_seed_set(spec: Mapping[str, Any], seeds: Sequence[int]) -> list[Dict[str, Any]]:
    seed_list = tuple(int(seed) for seed in seeds)
    workers = min(qv.MAX_SEED_WORKERS, len(seed_list))
    if workers <= 1:
        return [_seed_worker((dict(spec), seed)) for seed in seed_list]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        return list(executor.map(_seed_worker, [(dict(spec), seed) for seed in seed_list]))


def joint_gate(spec: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gate = tasks.spec_mapping(spec["joint_gate"], "joint_gate")
    gate_a = q.task_a_aggregate_gate(records, tasks.spec_mapping(gate["task_a"], "joint_gate.task_a"))
    gate_b = q.task_b_aggregate_gate(records, tasks.spec_mapping(gate["task_b"], "joint_gate.task_b"))
    controls_valid = bool(gate_a["conditions"]["control_seed_pass_count"] and gate_b["conditions"]["control_seed_pass_count"])
    passed = bool(controls_valid and gate_a["passed"] and gate_b["passed"])
    outcome = "JOINT_LEARNABILITY_CONFIRMED" if passed else "CONTROL_INVALID" if not controls_valid else "JOINT_CONFIRMATION_FAILED"
    return {"task_a": gate_a, "task_b": gate_b, "controls_valid": controls_valid, "passed": passed, "outcome": outcome}


def runtime_record() -> Dict[str, Any]:
    record = q.runtime_record()
    record.update({"engine": qv.ENGINE_ID, "max_seed_workers": qv.MAX_SEED_WORKERS, "direction_entropy_coefficient": DIRECTION_ENTROPY_COEFFICIENT})
    return record


def run_experiment(spec: Mapping[str, Any], implementation_tests: Mapping[str, Any]) -> Dict[str, Any]:
    validate_spec(spec)
    test_gate = q.implementation_test_gate(implementation_tests, commit=q.commit_sha())
    if not test_gate["passed"]:
        raise ContractError("exact-commit implementation test gate is not green")
    records = run_seed_set(spec, paired_seeds(spec))
    gate = joint_gate(spec, records)
    classification = "PASS" if gate["passed"] else "SCIENTIFIC_FAIL"
    return {
        "schema": "cb16.result.v1", "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(), "classification": classification,
        "scientific_outcome": gate["outcome"], "runtime": runtime_record(),
        "implementation_tests": {**dict(implementation_tests), "gate": test_gate, "required_by_global_gate": True, "runner_executed_suite": True},
        "preregistered_spec": {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "canonical_sha256": spec_sha256(spec), "paired_seeds": list(paired_seeds(spec)), "declared_status": spec.get("status")},
        "seeds": records, "joint_gate": gate,
        "verdicts": {"task_a": "PASS" if gate["task_a"]["passed"] else "FAIL", "task_b": "PASS" if gate["task_b"]["passed"] else "FAIL", "global": "PASS" if gate["passed"] else "FAIL"},
    }


def render_report(result: Mapping[str, Any]) -> str:
    lines = ["# R12 VS-C Joint Learnability Confirmation R3 — REPORT", "", f"- classification: **{result['classification']}**", f"- scientific outcome: **{result['scientific_outcome']}**", f"- commit: `{result['commit_sha']}`", f"- fresh seeds: {result['preregistered_spec']['paired_seeds']}", "", "## Per-seed", "", "| seed | A pos/ctl | A POST MAE pos/ctl | B pos/ctl | B POST growth pos/ctl |", "| --- | --- | --- | --- | --- |"]
    for rec in result["seeds"]:
        a, b = rec["task_a"], rec["task_b"]
        lines.append(f"| {rec['seed']} | {'PASS' if a['positive']['seed_gate']['passed'] else 'FAIL'}/{'PASS' if a['control']['seed_gate']['passed'] else 'FAIL'} | {a['positive']['post']['target_exposure_mae']:.6f}/{a['control']['post']['target_exposure_mae']:.6f} | {'PASS' if b['positive']['seed_gate']['passed'] else 'FAIL'}/{'PASS' if b['control']['seed_gate']['passed'] else 'FAIL'} | {b['positive']['post']['mean_true_delayed_log_growth']:.6f}/{b['control']['post']['mean_true_delayed_log_growth']:.6f} |")
    lines += ["", "## Joint gate", ""]
    for task_name in ("task_a", "task_b"):
        block = result["joint_gate"][task_name]
        lines += [f"### {task_name}", ""]
        for name, value in block["components"].items():
            lines.append(f"- {name}: `{value}`")
        lines += [f"- passed: **{block['passed']}**", ""]
    lines += [f"Controls valid: **{result['joint_gate']['controls_valid']}**", "", f"Outcome: **{result['joint_gate']['outcome']}**", "", "## Limitations", "", "- Synthetic confirmation only; not a historical-profitability claim.", "- R3 does not promote lambda=0.005 to production authority.", "- No coefficient sweep, threshold rescue or extra generations are authorized under R3.", ""]
    return "\n".join(lines)


def write_artifacts(result_dir: Path, spec: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (result_dir / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (result_dir / REPORT_FILENAME).write_text(render_report(result))


def failure_result(spec: Mapping[str, Any] | None, classification: str, detail: str, implementation_tests: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    result = {"schema": "cb16.result.v1", "experiment_id": EXPERIMENT_ID, "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND), "commit_sha": q.commit_sha(), "classification": classification, "scientific_outcome": "NOT_EVALUATED", "runtime": runtime_record(), "seeds": [], "joint_gate": {}, "verdicts": {"task_a": "NOT_EVALUATED", "task_b": "NOT_EVALUATED", "global": classification}, "error": {"type": classification, "detail": detail[:2000]}}
    if spec is not None:
        result["preregistered_spec"] = {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "canonical_sha256": spec_sha256(spec), "paired_seeds": list(paired_seeds(spec)), "declared_status": spec.get("status")}
    if implementation_tests is not None:
        result["implementation_tests"] = dict(implementation_tests)
    return result


def write_failure(result_dir: Path, spec: Mapping[str, Any] | None, result: Mapping[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    if spec is not None:
        (result_dir / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (result_dir / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (result_dir / REPORT_FILENAME).write_text(f"# R12 VS-C Joint Confirmation R3 — REPORT\n\n- classification: **{result['classification']}**\n- scientific outcome: **NOT_EVALUATED**\n\n{result.get('error', {}).get('detail', '')}\n")


def run_qualification(result_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any], int]:
    spec = load_spec(); validate_spec(spec)
    implementation = q.run_implementation_test_suite()
    implementation_gate = q.implementation_test_gate(implementation, commit=q.commit_sha())
    if not implementation_gate["passed"]:
        classification, exit_code = q.implementation_failure_classification(implementation)
        detail = q.implementation_test_failure_detail(classification, implementation)
        evidence = {**dict(implementation), "gate": implementation_gate, "failure_classification": classification}
        result = failure_result(spec, classification, detail, evidence); write_failure(result_dir, spec, result)
        return spec, result, exit_code
    result = run_experiment(spec, implementation); write_artifacts(result_dir, spec, result)
    return spec, result, EXIT_PASS if result["classification"] == "PASS" else EXIT_SCIENTIFIC_FAIL


def main() -> int:
    try:
        raw = (os.environ.get("CB16_RESULT_DIR") or "").strip()
        if not raw or not (os.environ.get("CB16_COMMIT_SHA") or "").strip():
            raise ContractError("CB16_RESULT_DIR and CB16_COMMIT_SHA must be set")
        result_dir = Path(raw)
    except ContractError as exc:
        print(f"{RESULT_COMMAND}: EXECUTION_BLOCKED: {exc}", file=sys.stderr); return EXIT_EXECUTION_BLOCKED
    try:
        _spec, result, exit_code = run_qualification(result_dir)
    except ContractError as exc:
        try: spec = load_spec()
        except ContractError: spec = None
        result = failure_result(spec, "CONTRACT_MISMATCH", f"{type(exc).__name__}: {exc}"); write_failure(result_dir, spec, result)
        print(f"{RESULT_COMMAND}: CONTRACT_MISMATCH: {exc}", file=sys.stderr); return EXIT_CONTRACT_MISMATCH
    except Exception as exc:  # noqa: BLE001
        try: spec = load_spec()
        except ContractError: spec = None
        result = failure_result(spec, "EXECUTION_BLOCKED", f"{type(exc).__name__}: {exc}"); write_failure(result_dir, spec, result)
        print(f"{RESULT_COMMAND}: EXECUTION_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr); return EXIT_EXECUTION_BLOCKED
    print(f"{RESULT_COMMAND}: classification={result['classification']} outcome={result['scientific_outcome']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
