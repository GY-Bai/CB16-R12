"""R12 VS-E R0 single-checkpoint controlled dual-task composition."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from ..evidence_scope import EvidenceScopeError, validate_experiment_spec, validate_result_against_spec
from . import controlled_tasks as tasks
from . import joint_confirmation_r3 as r3
from . import qualification as q
from . import task_a_direction_entropy_r2 as r2
from . import vectorized_tasks as vt
from .contracts import ContractError
from .trajectory import Trajectory, TrajectoryStep

RESULT_COMMAND = "cb16.vs-e-single-checkpoint-composition-r0@v1"
EXPERIMENT_ID = "r12.vs_e.single_checkpoint_composition.r0"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_e_single_checkpoint_composition_r0.json"
PARENT_SPEC_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_c_joint_confirmation_r3.json"
SPEC_FILE_SHA256 = "6ca58656d2c29c976db762f1eb3eb6259208017ac0cae5060e5d02012c446fc5"
PARENT_RAW_SHA256 = "e08a8c10d417d34600c97599b8f149bf4092cbe683da7edfa5ad32b7740cb56d"
SPEC_FILENAME, RESULT_FILENAME, REPORT_FILENAME = "experiment_spec.json", "RESULT.json", "REPORT.md"
EXIT_PASS, EXIT_SCIENTIFIC_FAIL, EXIT_EXECUTION_BLOCKED, EXIT_CONTRACT_MISMATCH = 0, 1, 3, 4
MAX_SEED_WORKERS = 4


def load_spec() -> Dict[str, Any]:
    raw = CONFIG_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SPEC_FILE_SHA256:
        raise ContractError(f"VS-E R0 spec byte SHA mismatch: {digest}")
    try:
        return dict(tasks.spec_mapping(json.loads(raw), "VS-E R0 spec"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"VS-E R0 spec invalid JSON: {exc}") from exc


def _parent_spec() -> Mapping[str, Any]:
    raw = PARENT_SPEC_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PARENT_RAW_SHA256:
        raise ContractError(f"VS-C R3 parent spec byte SHA mismatch: {digest}")
    return tasks.spec_mapping(json.loads(raw), "VS-C R3 parent spec")


def paired_seeds(spec: Mapping[str, Any]) -> Tuple[int, ...]:
    seeds = tuple(tasks.spec_int(x, "paired_seeds entry", minimum=1) for x in tasks.spec_sequence(spec["paired_seeds"], "paired_seeds"))
    if seeds != tuple(range(2501, 2509)):
        raise ContractError(f"VS-E R0 freezes paired seeds 2501..2508, got {seeds!r}")
    old = set(range(1201, 1209)) | set(range(2201, 2209))
    if set(seeds) & old:
        raise ContractError("VS-E R0 fresh seeds overlap prior VS-C controlled seed sets")
    return seeds


def validate_spec(spec: Mapping[str, Any]) -> None:
    try:
        validate_experiment_spec(spec)
    except EvidenceScopeError as exc:
        raise ContractError(f"VS-E R0 claim-authority spec invalid: {exc}") from exc
    if spec.get("experiment_id") != EXPERIMENT_ID or spec.get("experiment_role") != "QUALIFICATION":
        raise ContractError("VS-E R0 identity/role mismatch")
    parent = _parent_spec()
    for key in ("authority", "optimizer", "task_a", "task_b"):
        if spec.get(key) != parent.get(key):
            raise ContractError(f"VS-E R0 must inherit VS-C R3 {key} exactly")
    gates = tasks.spec_mapping(spec["reused_r3_final_gates"], "reused_r3_final_gates")
    if gates.get("task_a") != parent["joint_gate"]["task_a"] or gates.get("task_b") != parent["joint_gate"]["task_b"]:
        raise ContractError("VS-E R0 reused R3 aggregate gates differ from parent")
    r2.parse_authority(spec)
    r3.parse_optimizer(spec)
    tasks.task_a_environment(spec)
    tasks.task_b_environment(spec)
    paired_seeds(spec)
    schedule = tasks.spec_mapping(spec["training_schedule"], "training_schedule")
    if tasks.spec_int(schedule["cycles"], "training_schedule.cycles") != 256:
        raise ContractError("VS-E R0 requires 256 cycles")
    if list(schedule["updates_per_cycle"]) != ["TASK_A", "TASK_B"]:
        raise ContractError("VS-E R0 update order must be TASK_A then TASK_B")
    if (int(schedule["total_updates"]), int(schedule["task_a_updates"]), int(schedule["task_b_updates"])) != (512, 256, 256):
        raise ContractError("VS-E R0 update counts mismatch")
    if bool(schedule["parameter_reset_between_updates"]) or bool(schedule["optimizer_reset_between_updates"]):
        raise ContractError("VS-E R0 forbids parameter/optimizer reset")
    if list(schedule["evaluation_checkpoints"]) != ["PRE", "AFTER_FINAL_TASK_A_UPDATE", "AFTER_FINAL_TASK_B_UPDATE"]:
        raise ContractError("VS-E R0 evaluation checkpoint set mismatch")
    if "global monotonic 0..511" not in str(schedule["learner_generation_id"]):
        raise ContractError("VS-E R0 global learner generation semantics missing")
    if "cycle index 0..255" not in str(schedule["task_local_generation_index"]):
        raise ContractError("VS-E R0 task-local RNG generation semantics missing")


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(v) for v in value)
    return value


def _tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys() and all(_tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def _evaluation_snapshot(learner: Any) -> Dict[str, Any]:
    return {
        "generation": learner.generation_id,
        "actor": _clone_tree(learner.actor.state_dict()),
        "critic": _clone_tree(learner.critic.state_dict()),
        "actor_optimizer": _clone_tree(learner.actor_optimizer.state_dict()),
        "critic_optimizer": _clone_tree(learner.critic_optimizer.state_dict()),
    }


def _assert_same_snapshot(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    if before["generation"] != after["generation"]:
        raise ContractError("evaluation changed learner generation")
    for key in ("actor", "critic", "actor_optimizer", "critic_optimizer"):
        if not _tree_equal(before[key], after[key]):
            raise ContractError(f"evaluation mutated {key}")


def evaluate_both(learner: Any, env_a: tasks.TaskAEnvironment, env_b: tasks.TaskBEnvironment) -> Dict[str, Any]:
    before = _evaluation_snapshot(learner)
    metrics = {"task_a": tasks.evaluate_task_a(learner, env_a), "task_b": tasks.evaluate_task_b(learner, env_b)}
    after = _evaluation_snapshot(learner)
    _assert_same_snapshot(before, after)
    return metrics


def build_task_b_interleaved_generation(
    learner: Any,
    env: tasks.TaskBEnvironment,
    *,
    global_generation_id: int,
    local_cycle: int,
    mode: str,
    seed: int,
) -> Tuple[Trajectory, ...]:
    """Parent Task-B semantics with separate learner/global and task-local RNG generation indices."""
    if learner.generation_id != global_generation_id:
        raise ContractError("VS-E Task-B global generation does not match learner generation")
    states0 = vt._task_b_step0_states(learner, env)
    dir0, risk0 = vt.sample_action_batch(learner, states0)
    target0, delayed = vt._task_b_delayed_reward(env, dir0, risk0)
    z_neutral = learner.encode(torch.from_numpy(env.neutral_market)).to(torch.float32)
    exposure = target0.to(torch.float32)
    capacity = torch.clamp(1.0 - torch.abs(exposure) / float(env.config.hard_exposure_limit), min=0.0)
    account1 = torch.stack((exposure, torch.ones_like(exposure), capacity), dim=-1)
    states1 = torch.cat((z_neutral.expand(states0.shape[0], z_neutral.shape[-1]), account1), dim=-1)
    dir1, risk1 = vt.sample_action_batch(learner, states1)
    if mode == tasks.ARM_POSITIVE:
        assigned = delayed
    elif mode == tasks.ARM_CONTROL:
        order = tasks.no_fixed_point_permutation(delayed.numel(), seed=seed, generation=local_cycle)
        assigned = delayed.index_select(0, torch.tensor(order, dtype=torch.long))
    else:
        raise ContractError(f"unknown Task-B arm {mode!r}")
    directions0 = vt._direction_values(dir0)
    directions1 = vt._direction_values(dir1)
    return tuple(
        Trajectory(
            steps=(
                TrajectoryStep(state=states0[i], direction=directions0[i], requested_risk=float(risk0[i]), reward=0.0, terminal=False, truncated=False, generation_id=global_generation_id),
                TrajectoryStep(state=states1[i], direction=directions1[i], requested_risk=float(risk1[i]), reward=float(assigned[i]), terminal=False, truncated=False, generation_id=global_generation_id),
            ),
            generation_id=global_generation_id,
            complete=True,
        )
        for i in range(states0.shape[0])
    )


def _task_a_update(learner: Any, env: tasks.TaskAEnvironment, *, seed: int, mode: str, cycle: int) -> Mapping[str, float]:
    torch.manual_seed(q._collection_seed(seed, "task_a", mode, cycle))
    generation = learner.generation_id
    batch = vt.build_task_a_one_step_batch(learner, env, generation, mode=mode)
    report = learner.update_one_step_batch(batch)
    if report.generation_id != generation or report.next_generation_id != generation + 1:
        raise ContractError("VS-E Task-A update violated global generation sequence")
    return {"actor_loss": report.actor_loss, "critic_loss": report.critic_loss}


def _task_b_update(learner: Any, env: tasks.TaskBEnvironment, *, seed: int, mode: str, cycle: int) -> Mapping[str, float]:
    torch.manual_seed(q._collection_seed(seed, "task_b", mode, cycle))
    generation = learner.generation_id
    batch = build_task_b_interleaved_generation(learner, env, global_generation_id=generation, local_cycle=cycle, mode=mode, seed=seed)
    report = learner.update(batch)
    if report.generation_id != generation or report.next_generation_id != generation + 1:
        raise ContractError("VS-E Task-B update violated global generation sequence")
    return {"actor_loss": report.actor_loss, "critic_loss": report.critic_loss}


def _loss_summary(records: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    if not records:
        return {"updates": 0}
    return {
        "updates": len(records),
        "first_actor_loss": float(records[0]["actor_loss"]),
        "last_actor_loss": float(records[-1]["actor_loss"]),
        "first_critic_loss": float(records[0]["critic_loss"]),
        "last_critic_loss": float(records[-1]["critic_loss"]),
        "note": "loss diagnostics only; no scientific gate is derived from them",
    }


def _optimizer_steps(learner: Any) -> Tuple[int, ...]:
    values = []
    for optimizer in (learner.actor_optimizer, learner.critic_optimizer):
        for state in optimizer.state.values():
            step = state.get("step")
            if step is not None:
                values.append(int(step.item() if isinstance(step, torch.Tensor) else step))
    return tuple(values)


def _checkpoint_record(metrics: Mapping[str, Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
    a_gate = q.task_a_seed_gate(metrics["task_a"], tasks.spec_mapping(spec["task_a"]["seed_pass_gate"], "task_a.seed_pass_gate"))
    b_gate = q.task_b_seed_gate(metrics["task_b"], tasks.spec_mapping(spec["task_b"]["seed_pass_gate"], "task_b.seed_pass_gate"))
    return {"task_a": {"metrics": metrics["task_a"], "seed_gate": a_gate}, "task_b": {"metrics": metrics["task_b"], "seed_gate": b_gate}, "joint_pass": bool(a_gate["passed"] and b_gate["passed"])}


def run_seed(spec: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    q.configure_deterministic_runtime()
    authority = r2.parse_authority(spec)
    optimizer = r3.parse_optimizer(spec)
    env_a = tasks.task_a_environment(spec)
    env_b = tasks.task_b_environment(spec)
    torch.manual_seed(seed)
    positive = q.build_learner(authority, optimizer, direction_entropy_coefficient=r3.DIRECTION_ENTROPY_COEFFICIENT)
    torch.manual_seed(seed)
    control = q.build_learner(authority, optimizer, direction_entropy_coefficient=r3.DIRECTION_ENTROPY_COEFFICIENT)
    q.require_identical_parameters(positive, control)
    pos_actor_opt, pos_critic_opt = id(positive.actor_optimizer), id(positive.critic_optimizer)
    ctl_actor_opt, ctl_critic_opt = id(control.actor_optimizer), id(control.critic_optimizer)
    pre_pos, pre_ctl = evaluate_both(positive, env_a, env_b), evaluate_both(control, env_a, env_b)
    pos_a_diag: list[Mapping[str, float]] = []; pos_b_diag: list[Mapping[str, float]] = []
    ctl_a_diag: list[Mapping[str, float]] = []; ctl_b_diag: list[Mapping[str, float]] = []
    after_a_pos = after_a_ctl = after_b_pos = after_b_ctl = None
    for cycle in range(256):
        if positive.generation_id != 2 * cycle or control.generation_id != 2 * cycle:
            raise ContractError("VS-E learner generation drift before Task A")
        pos_a_diag.append(_task_a_update(positive, env_a, seed=seed, mode=tasks.ARM_POSITIVE, cycle=cycle))
        ctl_a_diag.append(_task_a_update(control, env_a, seed=seed, mode=tasks.ARM_CONTROL, cycle=cycle))
        if cycle == 255:
            after_a_pos, after_a_ctl = evaluate_both(positive, env_a, env_b), evaluate_both(control, env_a, env_b)
        if positive.generation_id != 2 * cycle + 1 or control.generation_id != 2 * cycle + 1:
            raise ContractError("VS-E learner generation drift before Task B")
        pos_b_diag.append(_task_b_update(positive, env_b, seed=seed, mode=tasks.ARM_POSITIVE, cycle=cycle))
        ctl_b_diag.append(_task_b_update(control, env_b, seed=seed, mode=tasks.ARM_CONTROL, cycle=cycle))
        if cycle == 255:
            after_b_pos, after_b_ctl = evaluate_both(positive, env_a, env_b), evaluate_both(control, env_a, env_b)
    if positive.generation_id != 512 or control.generation_id != 512:
        raise ContractError("VS-E learners must finish at global generation 512")
    if (id(positive.actor_optimizer), id(positive.critic_optimizer)) != (pos_actor_opt, pos_critic_opt) or (id(control.actor_optimizer), id(control.critic_optimizer)) != (ctl_actor_opt, ctl_critic_opt):
        raise ContractError("VS-E optimizer identity changed during interleaved training")
    for name, learner in (("positive", positive), ("control", control)):
        steps = _optimizer_steps(learner)
        if not steps or any(step != 512 for step in steps):
            raise ContractError(f"VS-E {name} optimizer state did not persist for all 512 updates")
    assert after_a_pos is not None and after_a_ctl is not None and after_b_pos is not None and after_b_ctl is not None
    checkpoints = {
        "PRE": {"positive": _checkpoint_record(pre_pos, spec), "control": _checkpoint_record(pre_ctl, spec)},
        "AFTER_FINAL_TASK_A_UPDATE": {"positive": _checkpoint_record(after_a_pos, spec), "control": _checkpoint_record(after_a_ctl, spec)},
        "AFTER_FINAL_TASK_B_UPDATE": {"positive": _checkpoint_record(after_b_pos, spec), "control": _checkpoint_record(after_b_ctl, spec)},
    }
    final_a_pos, final_a_ctl = after_b_pos["task_a"], after_b_ctl["task_a"]
    final_b_pos, final_b_ctl = after_b_pos["task_b"], after_b_ctl["task_b"]
    return {
        "seed": seed,
        "final_generation": 512,
        "checkpoints": checkpoints,
        "task_a": {
            "positive": {"pre": pre_pos["task_a"], "post": final_a_pos, "seed_gate": checkpoints["AFTER_FINAL_TASK_B_UPDATE"]["positive"]["task_a"]["seed_gate"], "diagnostics": _loss_summary(pos_a_diag)},
            "control": {"pre": pre_ctl["task_a"], "post": final_a_ctl, "seed_gate": checkpoints["AFTER_FINAL_TASK_B_UPDATE"]["control"]["task_a"]["seed_gate"], "diagnostics": _loss_summary(ctl_a_diag)},
            "paired": {
                "positive_target_mae_improvement": pre_pos["task_a"]["target_exposure_mae"] - final_a_pos["target_exposure_mae"],
                "control_target_mae_improvement": pre_ctl["task_a"]["target_exposure_mae"] - final_a_ctl["target_exposure_mae"],
                "positive_post_target_mae_strictly_better_than_control": bool(final_a_pos["target_exposure_mae"] < final_a_ctl["target_exposure_mae"]),
            },
        },
        "task_b": {
            "positive": {"pre": pre_pos["task_b"], "post": final_b_pos, "seed_gate": checkpoints["AFTER_FINAL_TASK_B_UPDATE"]["positive"]["task_b"]["seed_gate"], "diagnostics": _loss_summary(pos_b_diag)},
            "control": {"pre": pre_ctl["task_b"], "post": final_b_ctl, "seed_gate": checkpoints["AFTER_FINAL_TASK_B_UPDATE"]["control"]["task_b"]["seed_gate"], "diagnostics": _loss_summary(ctl_b_diag)},
            "paired": {
                "positive_true_delayed_log_growth_improvement": final_b_pos["mean_true_delayed_log_growth"] - pre_pos["task_b"]["mean_true_delayed_log_growth"],
                "control_true_delayed_log_growth_improvement": final_b_ctl["mean_true_delayed_log_growth"] - pre_ctl["task_b"]["mean_true_delayed_log_growth"],
                "positive_minus_control_true_delayed_log_growth": final_b_pos["mean_true_delayed_log_growth"] - final_b_ctl["mean_true_delayed_log_growth"],
            },
        },
    }


def _seed_worker(payload: tuple[dict[str, Any], int]) -> Dict[str, Any]:
    spec, seed = payload
    return run_seed(spec, seed)


def run_seed_set(spec: Mapping[str, Any]) -> list[Dict[str, Any]]:
    seeds = paired_seeds(spec)
    workers = min(MAX_SEED_WORKERS, len(seeds))
    if workers <= 1:
        return [run_seed(spec, seed) for seed in seeds]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        return list(executor.map(_seed_worker, [(dict(spec), seed) for seed in seeds]))


def composition_gate(spec: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gates = tasks.spec_mapping(spec["reused_r3_final_gates"], "reused_r3_final_gates")
    task_a = q.task_a_aggregate_gate(records, tasks.spec_mapping(gates["task_a"], "reused_r3_final_gates.task_a"))
    task_b = q.task_b_aggregate_gate(records, tasks.spec_mapping(gates["task_b"], "reused_r3_final_gates.task_b"))
    cg = tasks.spec_mapping(spec["composition_gate"], "composition_gate")
    def count(checkpoint: str, arm: str) -> int:
        return sum(bool(r["checkpoints"][checkpoint][arm]["joint_pass"]) for r in records)
    counts = {
        "positive_joint_seed_pass_count_after_final_A": count("AFTER_FINAL_TASK_A_UPDATE", "positive"),
        "positive_joint_seed_pass_count_after_final_B": count("AFTER_FINAL_TASK_B_UPDATE", "positive"),
        "control_joint_seed_pass_count_after_final_A": count("AFTER_FINAL_TASK_A_UPDATE", "control"),
        "control_joint_seed_pass_count_after_final_B": count("AFTER_FINAL_TASK_B_UPDATE", "control"),
    }
    conditions = {
        "positive_joint_after_final_A": counts["positive_joint_seed_pass_count_after_final_A"] >= int(cg["positive_joint_seed_pass_count_after_final_A_min"]),
        "positive_joint_after_final_B": counts["positive_joint_seed_pass_count_after_final_B"] >= int(cg["positive_joint_seed_pass_count_after_final_B_min"]),
        "control_joint_after_final_A": counts["control_joint_seed_pass_count_after_final_A"] <= int(cg["control_joint_seed_pass_count_after_final_A_max"]),
        "control_joint_after_final_B": counts["control_joint_seed_pass_count_after_final_B"] <= int(cg["control_joint_seed_pass_count_after_final_B_max"]),
    }
    coexistence_pass = all(conditions.values())
    return {"task_a": task_a, "task_b": task_b, "coexistence": {"counts": counts, "conditions": conditions, "passed": coexistence_pass}, "passed": bool(task_a["passed"] and task_b["passed"] and coexistence_pass)}


def run_experiment(spec: Mapping[str, Any], implementation: Mapping[str, Any]) -> Dict[str, Any]:
    validate_spec(spec)
    implementation_gate = q.implementation_test_gate(implementation, commit=q.commit_sha())
    if not implementation_gate["passed"]:
        raise ContractError("VS-E R0 exact-commit implementation test gate is not green")
    records = run_seed_set(spec)
    gate = composition_gate(spec, records)
    supported = bool(gate["passed"])
    result = {
        "schema": "cb16.result.v2",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(),
        "classification": "PASS" if supported else "SCIENTIFIC_FAIL",
        "execution_status": "EXECUTED",
        "validity_status": "VALID",
        "scientific_outcomes": {"single_checkpoint_dual_task_composition": "QUALIFIED" if supported else "NOT_QUALIFIED"},
        "implementation_tests": {**dict(implementation), "gate": implementation_gate, "required_by_global_gate": True, "runner_executed_suite": True},
        "preregistered_spec": {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "file_sha256": SPEC_FILE_SHA256, "paired_seeds": list(paired_seeds(spec)), "declared_status": spec.get("status")},
        "seeds": records,
        "composition_gate": gate,
        "gate_results": [
            {"gate_id": "G_FINAL_TASK_A_R3_COMPATIBLE", "passed": bool(gate["task_a"]["passed"]), "metrics": gate["task_a"]["components"]},
            {"gate_id": "G_FINAL_TASK_B_R3_COMPATIBLE", "passed": bool(gate["task_b"]["passed"]), "metrics": gate["task_b"]["components"]},
            {"gate_id": "G_SINGLE_CHECKPOINT_SWITCH_COEXISTENCE", "passed": bool(gate["coexistence"]["passed"]), "metrics": gate["coexistence"]["counts"]},
        ],
        "claim_assessments": [{"claim_id": "C_SINGLE_CHECKPOINT_DUAL_TASK_COMPOSITION", "qualification_status": "QUALIFIED" if supported else "NOT_QUALIFIED", "inference_status": "SUPPORTED" if supported else "GATE_NOT_MET", "attribution_status": "UNRESOLVED"}],
        "prior_evidence_assessment": [{"claim_ref": "r12.vs_c.joint_confirmation.r3", "status": "UNAFFECTED_IN_ORIGINAL_SCOPE"}],
        "promotion_decision": {"decision": "PROMOTE_SINGLE_CHECKPOINT_DUAL_TASK_COMPOSITION" if supported else "NO_PROMOTION", "reason": "VS-E R0 can only promote exact synthetic single-checkpoint dual-task coexistence under the frozen interleaved schedule"},
        "provenance": {"commit_sha": q.commit_sha(), "spec_file_sha256": SPEC_FILE_SHA256, "parent_vs_c_r3_run_id": 34908126391, "parent_vs_c_r3_commit": "465c8c916d0f67472b1575e177e8e24b1e29a4cf", "parent_vs_c_r3_artifact_digest": "sha256:7b10221a4dea2cd245e2a00a947efb97a4894f3d6b1ee1586c7fda31cde6a276", "historical_market_data_read": False, "final_holdout_accessed": False},
    }
    try:
        validate_result_against_spec(result, spec)
    except EvidenceScopeError as exc:
        raise ContractError(f"VS-E R0 result violates claim authority: {exc}") from exc
    return result


def render_report(result: Mapping[str, Any]) -> str:
    gate = result.get("composition_gate", {})
    co = gate.get("coexistence", {})
    lines = [
        "# R12 VS-E Single-Checkpoint Composition R0 — REPORT", "",
        f"- classification: **{result['classification']}**",
        f"- outcome: **{result.get('scientific_outcomes', {}).get('single_checkpoint_dual_task_composition', 'NOT_EVALUATED')}**",
        f"- promotion: **{result.get('promotion_decision', {}).get('decision', 'NO_PROMOTION')}**",
        "- scope: **synthetic single-checkpoint Task A + Task B composition only**", "",
        "## Final gates", "",
        f"- Task A R3-compatible final gate: **{gate.get('task_a', {}).get('passed', False)}**",
        f"- Task B R3-compatible final gate: **{gate.get('task_b', {}).get('passed', False)}**",
        f"- switch coexistence gate: **{co.get('passed', False)}**",
    ]
    for key, value in co.get("counts", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Authority boundary", "", "- PASS does not establish historical-market information, profitability, continuing-account composition, or general multitask capability.", "- FAIL does not invalidate VS-C R3 separate-task confirmation and does not identify a unique interference/capacity mechanism.", "- No historical market data or final holdout are read by VS-E R0.", ""]
    return "\n".join(lines)


def failure_result(spec: Mapping[str, Any] | None, classification: str, detail: str, implementation: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    result = {
        "schema": "cb16.result.v2", "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND), "commit_sha": q.commit_sha(),
        "classification": classification, "execution_status": "NOT_EXECUTED", "validity_status": "INVALID",
        "scientific_outcomes": {}, "gate_results": [],
        "claim_assessments": [{"claim_id": "C_SINGLE_CHECKPOINT_DUAL_TASK_COMPOSITION", "qualification_status": "NOT_QUALIFIED", "inference_status": "INVALID_FOR_CLAIM", "attribution_status": "NOT_APPLICABLE"}],
        "prior_evidence_assessment": [{"claim_ref": "r12.vs_c.joint_confirmation.r3", "status": "UNAFFECTED_IN_ORIGINAL_SCOPE"}],
        "promotion_decision": {"decision": "NO_PROMOTION", "reason": "formal run did not reach valid VS-E R0 adjudication"},
        "provenance": {"commit_sha": q.commit_sha(), "spec_file_sha256": SPEC_FILE_SHA256, "historical_market_data_read": False, "final_holdout_accessed": False},
        "error": {"type": classification, "detail": detail[:2000]},
    }
    if implementation is not None:
        result["implementation_tests"] = dict(implementation)
    if spec is not None:
        try:
            validate_result_against_spec(result, spec)
        except EvidenceScopeError as exc:
            raise ContractError(f"VS-E R0 failure result violates claim authority: {exc}") from exc
    return result


def write_artifacts(result_dir: Path, spec: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (result_dir / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (result_dir / REPORT_FILENAME).write_text(render_report(result))


def run_qualification(result_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any], int]:
    spec = load_spec(); validate_spec(spec)
    implementation = q.run_implementation_test_suite(); gate = q.implementation_test_gate(implementation, commit=q.commit_sha())
    if not gate["passed"]:
        classification, code = q.implementation_failure_classification(implementation)
        result = failure_result(spec, classification, q.implementation_test_failure_detail(classification, implementation), {**dict(implementation), "gate": gate})
        write_artifacts(result_dir, spec, result); return spec, result, code
    result = run_experiment(spec, implementation); write_artifacts(result_dir, spec, result)
    return spec, result, EXIT_PASS if result["classification"] == "PASS" else EXIT_SCIENTIFIC_FAIL


def main() -> int:
    raw = (os.environ.get("CB16_RESULT_DIR") or "").strip()
    if not raw or not (os.environ.get("CB16_COMMIT_SHA") or "").strip():
        return EXIT_EXECUTION_BLOCKED
    result_dir = Path(raw)
    try:
        _spec, result, code = run_qualification(result_dir)
    except ContractError as exc:
        try: spec = load_spec()
        except Exception: spec = None
        result = failure_result(spec, "CONTRACT_MISMATCH", f"{type(exc).__name__}: {exc}"); write_artifacts(result_dir, spec or {}, result); return EXIT_CONTRACT_MISMATCH
    except Exception as exc:
        try: spec = load_spec()
        except Exception: spec = None
        result = failure_result(spec, "EXECUTION_BLOCKED", f"{type(exc).__name__}: {exc}"); write_artifacts(result_dir, spec or {}, result); return EXIT_EXECUTION_BLOCKED
    print(f"{RESULT_COMMAND}: classification={result['classification']} outcome={result['scientific_outcomes']['single_checkpoint_dual_task_composition']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
