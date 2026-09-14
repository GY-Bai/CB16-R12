"""R12 VS-C Task-A categorical-direction entropy attribution R2.

R2 keeps the high-batch B900 Task-A surface frozen and changes one mechanism:
a small entropy bonus on the three-way categorical direction distribution.  The
conditional Beta risk distributions receive no entropy term.  This is an
attribution experiment, not a production hyperparameter search.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from . import controlled_tasks as tasks
from . import qualification as q
from . import qualification_vectorized as qv
from .contracts import ContractError

RESULT_COMMAND = "cb16.vs-c-task-a-direction-entropy-r2@v1"
EXPERIMENT_ID = "r12.vs_c.task_a_direction_entropy_attribution.r2"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_c_task_a_direction_entropy_r2.json"
SPEC_FILENAME = "experiment_spec.json"
RESULT_FILENAME = "RESULT.json"
REPORT_FILENAME = "REPORT.md"

EXIT_PASS = 0
EXIT_SCIENTIFIC_FAIL = 1
EXIT_EXECUTION_BLOCKED = 3
EXIT_CONTRACT_MISMATCH = 4

BASELINE_ARM = "baseline_no_entropy"
ENTROPY_ARM = "direction_entropy_005"
ARM_ORDER: Tuple[str, ...] = (BASELINE_ARM, ENTROPY_ARM)


@dataclass(frozen=True)
class ArmDefinition:
    name: str
    direction_entropy_coefficient: float


def load_spec() -> Dict[str, Any]:
    try:
        parsed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"cannot read preregistered spec {CONFIG_PATH}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"preregistered spec is not valid JSON: {exc}") from exc
    return dict(tasks.spec_mapping(parsed, "preregistered spec"))


def spec_sha256(spec: Mapping[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def paired_seeds(spec: Mapping[str, Any]) -> Tuple[int, ...]:
    seeds = tuple(
        tasks.spec_int(value, "paired_seeds entry", minimum=1)
        for value in tasks.spec_sequence(spec["paired_seeds"], "paired_seeds")
    )
    if seeds != tuple(range(1201, 1209)):
        raise ContractError(f"R2 freezes paired seeds 1201..1208, got {seeds!r}")
    return seeds


def parse_authority(spec: Mapping[str, Any]) -> q.Authority:
    authority = tasks.spec_mapping(spec["authority"], "authority")
    parsed = q.Authority(
        context_length=tasks.spec_int(authority["context_length"], "authority.context_length", minimum=2),
        sensory_seed=tasks.spec_int(authority["sensory_seed"], "authority.sensory_seed"),
        sensory_z_dim=tasks.spec_int(authority["sensory_z_dim"], "authority.sensory_z_dim", minimum=1),
        actor_hidden_width=tasks.spec_int(authority["actor_hidden_width"], "authority.actor_hidden_width", minimum=1),
        actor_hidden_layers=tasks.spec_int(authority["actor_hidden_layers"], "authority.actor_hidden_layers", minimum=1),
        device=tasks.spec_str(authority["device"], "authority.device"),
        torch_num_threads=tasks.spec_int(
            authority["torch_num_threads_per_worker"], "authority.torch_num_threads_per_worker", minimum=1
        ),
        deterministic_algorithms=tasks.spec_bool(
            authority["deterministic_algorithms"], "authority.deterministic_algorithms"
        ),
    )
    if parsed.device != "cpu" or parsed.torch_num_threads != 1 or not parsed.deterministic_algorithms:
        raise ContractError("R2 requires deterministic CPU execution with one Torch thread per worker")
    if tasks.spec_str(authority["normalization"], "authority.normalization") != "N0_ENDPOINT_ANCHORED_V1":
        raise ContractError("R2 requires N0 endpoint-anchored normalization")
    if tasks.spec_str(authority["execution_engine"], "authority.execution_engine") != qv.ENGINE_ID:
        raise ContractError(f"R2 requires execution engine {qv.ENGINE_ID!r}")
    max_workers = tasks.spec_int(authority["max_seed_workers"], "authority.max_seed_workers", minimum=1)
    if max_workers != qv.MAX_SEED_WORKERS:
        raise ContractError(f"R2 freezes max_seed_workers={qv.MAX_SEED_WORKERS}, got {max_workers}")
    return parsed


def parse_optimizer(spec: Mapping[str, Any]) -> q.OptimizerSettings:
    optimizer = tasks.spec_mapping(spec["optimizer"], "optimizer")
    expected_keys = {
        "actor", "actor_lr", "critic", "critic_lr", "discount_factor", "replay", "beta_risk_entropy_bonus"
    }
    if set(optimizer) != expected_keys:
        raise ContractError(f"R2 optimizer keys must be exactly {sorted(expected_keys)}, got {sorted(optimizer)}")
    if tasks.spec_str(optimizer["actor"], "optimizer.actor") != "Adam":
        raise ContractError("R2 actor optimizer must be Adam")
    if tasks.spec_str(optimizer["critic"], "optimizer.critic") != "Adam":
        raise ContractError("R2 critic optimizer must be Adam")
    actor_lr = tasks.spec_float(optimizer["actor_lr"], "optimizer.actor_lr")
    critic_lr = tasks.spec_float(optimizer["critic_lr"], "optimizer.critic_lr")
    if actor_lr != 0.001 or critic_lr != 0.001:
        raise ContractError(f"R2 freezes actor/critic lr at 0.001, got {actor_lr}/{critic_lr}")
    if tasks.spec_float(optimizer["discount_factor"], "optimizer.discount_factor") != 1.0:
        raise ContractError("R2 uses undiscounted return-to-go")
    if tasks.spec_bool(optimizer["replay"], "optimizer.replay"):
        raise ContractError("R2 forbids replay")
    if tasks.spec_float(optimizer["beta_risk_entropy_bonus"], "optimizer.beta_risk_entropy_bonus") != 0.0:
        raise ContractError("R2 forbids Beta-risk entropy")
    return q.OptimizerSettings(actor_lr=actor_lr, critic_lr=critic_lr)


def arm_definition(spec: Mapping[str, Any], arm_name: str) -> ArmDefinition:
    if arm_name not in ARM_ORDER:
        raise ContractError(f"unknown R2 arm {arm_name!r}")
    arms = tasks.spec_mapping(
        tasks.spec_mapping(spec["task_a"], "task_a")["arms"], "task_a.arms"
    )
    arm = tasks.spec_mapping(arms[arm_name], f"task_a.arms.{arm_name}")
    target = tasks.spec_str(arm["direction_entropy_target"], f"{arm_name}.direction_entropy_target")
    if target != "categorical_direction_only":
        raise ContractError(f"{arm_name} must target categorical_direction_only")
    coefficient = tasks.spec_float(
        arm["direction_entropy_coefficient"], f"{arm_name}.direction_entropy_coefficient"
    )
    expected = 0.0 if arm_name == BASELINE_ARM else 0.005
    if coefficient != expected:
        raise ContractError(f"{arm_name} coefficient must be {expected}, got {coefficient}")
    return ArmDefinition(arm_name, coefficient)


def task_a_seed_gate(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    return tasks.spec_mapping(
        tasks.spec_mapping(spec["task_a"], "task_a")["seed_pass_gate"], "task_a.seed_pass_gate"
    )


def validate_spec(spec: Mapping[str, Any]) -> None:
    if tasks.spec_str(spec["experiment_id"], "experiment_id") != EXPERIMENT_ID:
        raise ContractError(f"experiment_id must be {EXPERIMENT_ID!r}")
    parse_authority(spec)
    parse_optimizer(spec)
    paired_seeds(spec)
    task = tasks.spec_mapping(spec["task_a"], "task_a")
    if tasks.spec_int(task["generations"], "task_a.generations", minimum=1) != 256:
        raise ContractError("R2 freezes task_a.generations=256")
    if tasks.spec_int(task["trajectory_length"], "task_a.trajectory_length", minimum=1) != 1:
        raise ContractError("R2 Task A trajectories have length one")
    batch = tasks.spec_int(task["batch_trajectories"], "task_a.batch_trajectories", minimum=1)
    if batch != 900:
        raise ContractError(f"R2 freezes batch_trajectories=900, got {batch}")
    if tasks.spec_int(task["positive_per_actual_account"], "task_a.positive_per_actual_account") != 300:
        raise ContractError("R2 freezes positive_per_actual_account=300")
    if tasks.spec_int(task["control_per_actual_observed_pair"], "task_a.control_per_actual_observed_pair") != 100:
        raise ContractError("R2 freezes control_per_actual_observed_pair=100")
    if set(tasks.spec_mapping(task["arms"], "task_a.arms")) != set(ARM_ORDER):
        raise ContractError(f"task_a.arms must be exactly {ARM_ORDER}")
    for arm in ARM_ORDER:
        arm_definition(spec, arm)
    # Build once during validation so AccountTruth, Physics and balanced 900 schedules fail closed.
    env = tasks.task_a_environment(spec)
    if len(env.positive_schedule) != 900 or len(env.control_schedule) != 900:
        raise ContractError("R2 Task-A schedules must each contain exactly 900 trajectories")
    gate = task_a_seed_gate(spec)
    if tasks.spec_int(gate["direction_correct_count_min"], "seed gate direction") != 3:
        raise ContractError("R2 freezes direction_correct_count_min=3")
    if tasks.spec_float(gate["target_exposure_mae_max"], "seed gate mae") != 0.15:
        raise ContractError("R2 freezes target_exposure_mae_max=0.15")
    attribution = tasks.spec_mapping(spec["attribution_gate"], "attribution_gate")
    expected_gate = {
        "baseline_control_seed_pass_count_max": 2,
        "entropy_positive_seed_pass_count_min": 6,
        "entropy_control_seed_pass_count_max": 2,
        "entropy_minus_baseline_positive_pass_count_min": 3,
    }
    for key, expected in expected_gate.items():
        if tasks.spec_int(attribution[key], f"attribution_gate.{key}") != expected:
            raise ContractError(f"R2 freezes attribution_gate.{key}={expected}")
    global_gate = tasks.spec_mapping(spec["global_gate"], "global_gate")
    if not tasks.spec_bool(global_gate["implementation_tests_must_pass"], "implementation_tests_must_pass"):
        raise ContractError("implementation tests must pass")
    if not tasks.spec_bool(global_gate["no_threshold_changes_after_results"], "no_threshold_changes_after_results"):
        raise ContractError("R2 forbids threshold changes after results")
    expected_classes = {
        "scientific_gate_miss_classification": "SCIENTIFIC_FAIL",
        "contract_violation_classification": "CONTRACT_MISMATCH",
        "runtime_or_environment_blocker_classification": "EXECUTION_BLOCKED",
    }
    for key, expected in expected_classes.items():
        if tasks.spec_str(global_gate[key], f"global_gate.{key}") != expected:
            raise ContractError(f"global_gate.{key} must be {expected!r}")
    required = {
        tasks.spec_str(item, "required_artifacts entry")
        for item in tasks.spec_sequence(spec["required_artifacts"], "required_artifacts")
    }
    if required != {SPEC_FILENAME, RESULT_FILENAME, REPORT_FILENAME}:
        raise ContractError("required_artifacts must be experiment_spec.json, RESULT.json, REPORT.md")


def _all_flat(metrics: Mapping[str, Any]) -> bool:
    cases = tasks.spec_sequence(metrics["cases"], "evaluation cases")
    return bool(cases) and all(case["direction"] == "FLAT" for case in cases)


def build_seed_learners(
    spec: Mapping[str, Any], seed: int
) -> Dict[str, Tuple[Any, Any]]:
    authority = parse_authority(spec)
    optimizer = parse_optimizer(spec)
    learners: Dict[str, Tuple[Any, Any]] = {}
    for arm_name in ARM_ORDER:
        coefficient = arm_definition(spec, arm_name).direction_entropy_coefficient
        torch.manual_seed(seed)
        positive = q.build_learner(
            authority, optimizer, direction_entropy_coefficient=coefficient
        )
        torch.manual_seed(seed)
        control = q.build_learner(
            authority, optimizer, direction_entropy_coefficient=coefficient
        )
        q.require_identical_parameters(positive, control)
        learners[arm_name] = (positive, control)
    q.require_identical_parameters(learners[BASELINE_ARM][0], learners[ENTROPY_ARM][0])
    q.require_identical_parameters(learners[BASELINE_ARM][1], learners[ENTROPY_ARM][1])
    if not torch.equal(
        learners[BASELINE_ARM][0].sensory.projection,
        learners[ENTROPY_ARM][0].sensory.projection,
    ):
        raise ContractError("R2 paired arms do not share bitwise-identical frozen sensory")
    return learners


def _arm_result(
    spec: Mapping[str, Any], arm_name: str, positive, control, *, seed: int
) -> Dict[str, Any]:
    env = tasks.task_a_environment(spec)
    generations = tasks.spec_int(spec["task_a"]["generations"], "task_a.generations", minimum=1)
    gate = task_a_seed_gate(spec)
    pre_positive = tasks.evaluate_task_a(positive, env)
    pre_control = tasks.evaluate_task_a(control, env)
    diagnostics_positive = qv._train_task_a(
        positive, env, seed=seed, mode=tasks.ARM_POSITIVE, generations=generations
    )
    diagnostics_control = qv._train_task_a(
        control, env, seed=seed, mode=tasks.ARM_CONTROL, generations=generations
    )
    post_positive = tasks.evaluate_task_a(positive, env)
    post_control = tasks.evaluate_task_a(control, env)
    return {
        "direction_entropy_coefficient": arm_definition(spec, arm_name).direction_entropy_coefficient,
        "positive": {
            "pre": pre_positive,
            "post": post_positive,
            "seed_gate": q.task_a_seed_gate(post_positive, gate),
            "all_flat_post": _all_flat(post_positive),
            "diagnostics": diagnostics_positive,
        },
        "control": {
            "pre": pre_control,
            "post": post_control,
            "seed_gate": q.task_a_seed_gate(post_control, gate),
            "all_flat_post": _all_flat(post_control),
            "diagnostics": diagnostics_control,
        },
        "paired": {
            "positive_target_mae_improvement": pre_positive["target_exposure_mae"] - post_positive["target_exposure_mae"],
            "control_target_mae_improvement": pre_control["target_exposure_mae"] - post_control["target_exposure_mae"],
            "positive_post_target_mae_strictly_better_than_control": bool(
                post_positive["target_exposure_mae"] < post_control["target_exposure_mae"]
            ),
        },
    }


def run_seed(spec: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    learners = build_seed_learners(spec, seed)
    baseline = _arm_result(spec, BASELINE_ARM, *learners[BASELINE_ARM], seed=seed)
    entropy = _arm_result(spec, ENTROPY_ARM, *learners[ENTROPY_ARM], seed=seed)
    if baseline["positive"]["pre"] != entropy["positive"]["pre"]:
        raise ContractError("baseline and entropy positive learners did not start identically")
    if baseline["control"]["pre"] != entropy["control"]["pre"]:
        raise ContractError("baseline and entropy control learners did not start identically")
    return {"seed": seed, "arms": {BASELINE_ARM: baseline, ENTROPY_ARM: entropy}}


def _seed_worker(payload: Tuple[Dict[str, Any], int]) -> Dict[str, Any]:
    spec, seed = payload
    q.configure_deterministic_runtime()
    return run_seed(spec, seed)


def run_seed_set(spec: Mapping[str, Any], seeds: Sequence[int]) -> list[Dict[str, Any]]:
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ContractError("seed set must not be empty")
    workers = min(qv.MAX_SEED_WORKERS, len(seed_list))
    if workers == 1:
        return [_seed_worker((dict(spec), seed)) for seed in seed_list]
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        return list(executor.map(_seed_worker, [(dict(spec), seed) for seed in seed_list]))


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def arm_aggregate(seed_records: Sequence[Mapping[str, Any]], arm_name: str) -> Dict[str, Any]:
    positive = [record["arms"][arm_name]["positive"] for record in seed_records]
    control = [record["arms"][arm_name]["control"] for record in seed_records]
    return {
        "positive_seed_pass_count": sum(bool(item["seed_gate"]["passed"]) for item in positive),
        "control_seed_pass_count": sum(bool(item["seed_gate"]["passed"]) for item in control),
        "positive_all_flat_post_count": sum(bool(item["all_flat_post"]) for item in positive),
        "control_all_flat_post_count": sum(bool(item["all_flat_post"]) for item in control),
        "positive_median_post_target_mae": _median(
            [float(item["post"]["target_exposure_mae"]) for item in positive]
        ),
        "control_median_post_target_mae": _median(
            [float(item["post"]["target_exposure_mae"]) for item in control]
        ),
        "positive_median_pre_to_post_target_mae_improvement": _median(
            [float(record["arms"][arm_name]["paired"]["positive_target_mae_improvement"]) for record in seed_records]
        ),
    }


def attribution_gate(
    spec: Mapping[str, Any], aggregates: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    gate = tasks.spec_mapping(spec["attribution_gate"], "attribution_gate")
    baseline = aggregates[BASELINE_ARM]
    entropy = aggregates[ENTROPY_ARM]
    components = {
        "baseline_control_seed_pass_count": baseline["control_seed_pass_count"],
        "entropy_positive_seed_pass_count": entropy["positive_seed_pass_count"],
        "entropy_control_seed_pass_count": entropy["control_seed_pass_count"],
        "entropy_minus_baseline_positive_pass_count": (
            entropy["positive_seed_pass_count"] - baseline["positive_seed_pass_count"]
        ),
    }
    conditions = {
        "baseline_control_seed_pass_count": components["baseline_control_seed_pass_count"]
        <= tasks.spec_int(gate["baseline_control_seed_pass_count_max"], "baseline control max"),
        "entropy_positive_seed_pass_count": components["entropy_positive_seed_pass_count"]
        >= tasks.spec_int(gate["entropy_positive_seed_pass_count_min"], "entropy positive min"),
        "entropy_control_seed_pass_count": components["entropy_control_seed_pass_count"]
        <= tasks.spec_int(gate["entropy_control_seed_pass_count_max"], "entropy control max"),
        "entropy_minus_baseline_positive_pass_count": components["entropy_minus_baseline_positive_pass_count"]
        >= tasks.spec_int(gate["entropy_minus_baseline_positive_pass_count_min"], "entropy-baseline min"),
    }
    control_valid = bool(
        conditions["baseline_control_seed_pass_count"] and conditions["entropy_control_seed_pass_count"]
    )
    supported = bool(control_valid and all(conditions.values()))
    outcome = (
        "DIRECTION_EXPLORATION_SUPPORTED"
        if supported
        else "CONTROL_INVALID"
        if not control_valid
        else "DIRECTION_ENTROPY_NOT_SUFFICIENT"
    )
    return {
        "preregistered": dict(gate),
        "components": components,
        "conditions": conditions,
        "control_valid": control_valid,
        "passed": supported,
        "outcome": outcome,
    }


def runtime_record() -> Dict[str, Any]:
    record = q.runtime_record()
    record.update({"engine": qv.ENGINE_ID, "max_seed_workers": qv.MAX_SEED_WORKERS})
    return record


def run_experiment(
    spec: Mapping[str, Any], *, implementation_tests: Mapping[str, Any]
) -> Dict[str, Any]:
    q.configure_deterministic_runtime()
    validate_spec(spec)
    gate = q.implementation_test_gate(implementation_tests, commit=q.commit_sha())
    if not gate["passed"]:
        raise ContractError("exact-commit implementation test gate is not green")
    seeds = paired_seeds(spec)
    seed_records = run_seed_set(spec, seeds)
    aggregates = {arm: arm_aggregate(seed_records, arm) for arm in ARM_ORDER}
    attribution = attribution_gate(spec, aggregates)
    classification = "PASS" if attribution["passed"] else "SCIENTIFIC_FAIL"
    return {
        "schema": "cb16.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(),
        "classification": classification,
        "scientific_outcome": attribution["outcome"],
        "runtime": runtime_record(),
        "implementation_tests": {
            **dict(implementation_tests),
            "gate": gate,
            "required_by_global_gate": True,
            "runner_executed_suite": True,
        },
        "preregistered_spec": {
            "path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "canonical_sha256": spec_sha256(spec),
            "paired_seeds": list(seeds),
            "declared_status": spec.get("status"),
        },
        "seeds": seed_records,
        "aggregates": aggregates,
        "attribution_gate": attribution,
        "verdicts": {
            "direction_exploration_hypothesis": "PASS" if attribution["passed"] else "FAIL",
            "global": "PASS" if attribution["passed"] else "FAIL",
        },
    }


def _direction_triplet(metrics: Mapping[str, Any]) -> str:
    return "/".join(str(case["direction"]) for case in metrics["cases"])


def render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# R12 VS-C Task-A Direction-Entropy Attribution R2 — REPORT",
        "",
        f"- classification: **{result['classification']}**",
        f"- scientific outcome: **{result['scientific_outcome']}**",
        f"- commit: `{result['commit_sha']}`",
        f"- engine: `{result['runtime']['engine']}`",
        f"- paired seeds: {result['preregistered_spec']['paired_seeds']}",
        "",
        "## Per-seed behavior",
        "",
        "| seed | arm | lambda_dir | positive POST dirs | positive POST MAE | pos gate | control POST MAE | ctl gate | all-flat positive |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | --- | --- |",
    ]
    for record in result["seeds"]:
        for arm in ARM_ORDER:
            block = record["arms"][arm]
            lines.append(
                f"| {record['seed']} | {arm} | {block['direction_entropy_coefficient']:.3f} | "
                f"{_direction_triplet(block['positive']['post'])} | "
                f"{block['positive']['post']['target_exposure_mae']:.6f} | "
                f"{'PASS' if block['positive']['seed_gate']['passed'] else 'FAIL'} | "
                f"{block['control']['post']['target_exposure_mae']:.6f} | "
                f"{'PASS' if block['control']['seed_gate']['passed'] else 'FAIL'} | "
                f"{block['positive']['all_flat_post']} |"
            )
    lines.extend(["", "## Aggregate arms", ""])
    for arm in ARM_ORDER:
        aggregate = result["aggregates"][arm]
        lines.extend([
            f"### {arm}", "",
            f"- positive seed passes: `{aggregate['positive_seed_pass_count']}`",
            f"- control seed passes: `{aggregate['control_seed_pass_count']}`",
            f"- positive all-FLAT POST count: `{aggregate['positive_all_flat_post_count']}`",
            f"- positive median POST MAE: `{aggregate['positive_median_post_target_mae']:.6f}`",
            f"- positive median PRE→POST MAE improvement: `{aggregate['positive_median_pre_to_post_target_mae_improvement']:.6f}`",
            "",
        ])
    gate = result["attribution_gate"]
    lines.extend(["## Attribution gate", "", "| component | value | condition |", "| --- | ---: | --- |"])
    for name, value in gate["components"].items():
        lines.append(f"| {name} | {value} | {'ok' if gate['conditions'][name] else 'not met'} |")
    lines.extend([
        "", f"Outcome: **{gate['outcome']}**", "",
        "Only categorical direction entropy differs between arms; Beta-risk entropy remains zero.",
        "`all_flat_post` is diagnostic only and is not part of the gate.", "",
        "## Limitations", "",
        "- This is a controlled synthetic mechanism experiment, not a historical-profitability claim.",
        "- A supported attribution would not promote lambda=0.005 to production authority.",
        "- A gate miss does not authorize an entropy sweep under this experiment identity.", "",
    ])
    return "\n".join(lines)


def write_artifacts(result_dir: Path, spec: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    directory = Path(result_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / REPORT_FILENAME).write_text(render_report(result), encoding="utf-8")


def failure_result(
    spec: Mapping[str, Any] | None,
    *,
    classification: str,
    detail: str,
    implementation_tests: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": "cb16.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(),
        "classification": classification,
        "scientific_outcome": "NOT_EVALUATED",
        "runtime": runtime_record(),
        "seeds": [],
        "aggregates": {},
        "attribution_gate": {},
        "verdicts": {"direction_exploration_hypothesis": "NOT_EVALUATED", "global": classification},
        "error": {"type": classification, "detail": detail[:2000]},
    }
    if spec is not None:
        result["preregistered_spec"] = {
            "path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "canonical_sha256": spec_sha256(spec),
            "paired_seeds": list(paired_seeds(spec)),
            "declared_status": spec.get("status"),
        }
    if implementation_tests is not None:
        result["implementation_tests"] = dict(implementation_tests)
    return result


def render_failure_report(result: Mapping[str, Any]) -> str:
    error = result.get("error", {})
    return "\n".join([
        "# R12 VS-C Task-A Direction-Entropy Attribution R2 — REPORT", "",
        f"- classification: **{result['classification']}**",
        f"- commit: `{result['commit_sha']}`", "",
        "No scientific attribution was produced.", "",
        f"{error.get('type')}: {error.get('detail')}", "",
    ])


def _write_failure_artifacts(
    result_dir: Path, spec: Mapping[str, Any] | None, result: Mapping[str, Any]
) -> None:
    directory = Path(result_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if spec is not None:
        (directory / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / REPORT_FILENAME).write_text(render_failure_report(result), encoding="utf-8")


def result_dir_from_environment() -> Path:
    raw = (os.environ.get("CB16_RESULT_DIR") or "").strip()
    if not raw:
        raise ContractError("CB16_RESULT_DIR is not set")
    return Path(raw)


def run_qualification(result_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    spec = load_spec()
    validate_spec(spec)
    implementation_tests = q.run_implementation_test_suite()
    implementation_gate = q.implementation_test_gate(implementation_tests, commit=q.commit_sha())
    if not implementation_gate["passed"]:
        classification, exit_code = q.implementation_failure_classification(implementation_tests)
        detail = q.implementation_test_failure_detail(classification, implementation_tests)
        evidence = {**dict(implementation_tests), "gate": implementation_gate, "failure_classification": classification}
        result = failure_result(
            spec, classification=classification, detail=detail, implementation_tests=evidence
        )
        _write_failure_artifacts(result_dir, spec, result)
        return spec, result, exit_code
    result = run_experiment(spec, implementation_tests=implementation_tests)
    write_artifacts(result_dir, spec, result)
    exit_code = EXIT_PASS if result["classification"] == "PASS" else EXIT_SCIENTIFIC_FAIL
    return spec, result, exit_code


def main() -> int:
    try:
        result_dir = result_dir_from_environment()
        if not (os.environ.get("CB16_COMMIT_SHA") or "").strip():
            raise ContractError("CB16_COMMIT_SHA is not set")
    except ContractError as exc:
        print(f"{RESULT_COMMAND}: EXECUTION_BLOCKED: {exc}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED
    try:
        _spec, result, exit_code = run_qualification(result_dir)
    except ContractError as exc:
        try:
            spec = load_spec()
        except ContractError:
            spec = None
        result = failure_result(spec, classification="CONTRACT_MISMATCH", detail=f"{type(exc).__name__}: {exc}")
        _write_failure_artifacts(result_dir, spec, result)
        print(f"{RESULT_COMMAND}: CONTRACT_MISMATCH: {exc}", file=sys.stderr)
        return EXIT_CONTRACT_MISMATCH
    except Exception as exc:  # noqa: BLE001
        try:
            spec = load_spec()
        except ContractError:
            spec = None
        result = failure_result(spec, classification="EXECUTION_BLOCKED", detail=f"{type(exc).__name__}: {exc}")
        _write_failure_artifacts(result_dir, spec, result)
        print(f"{RESULT_COMMAND}: EXECUTION_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED
    print(f"{RESULT_COMMAND}: classification={result['classification']} outcome={result['scientific_outcome']}")
    print(
        f"BUILD_REPORT: {RESULT_COMMAND} classification={result['classification']}; "
        f"artifacts {SPEC_FILENAME}, {RESULT_FILENAME}, {REPORT_FILENAME} written; "
        "R2 thresholds and entropy coefficient were not changed after viewing results"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_ORDER", "BASELINE_ARM", "CONFIG_PATH", "ENTROPY_ARM", "EXPERIMENT_ID",
    "RESULT_COMMAND", "arm_aggregate", "arm_definition", "attribution_gate",
    "build_seed_learners", "load_spec", "main", "paired_seeds", "parse_authority",
    "parse_optimizer", "run_experiment", "run_seed", "run_seed_set", "spec_sha256",
    "validate_spec",
]
