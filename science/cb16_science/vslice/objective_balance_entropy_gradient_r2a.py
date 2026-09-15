"""R12 VS-D R2A training-only objective-balance diagnostic.

Measures the reward-driven categorical-direction policy-gradient and the
lambda-weighted categorical entropy gradient on the exact same theta_g and
sampled Jan-Feb training batch before each unchanged parent-R0 update.
March and final holdout are outside this runner's data authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from ..evidence_scope import validate_experiment_spec, validate_result_against_spec
from ..normalization.transforms import n0_endpoint_log_ratios
from . import controlled_tasks as tasks
from . import historical_market_canary_r0 as r0
from . import qualification as q
from . import vectorized_tasks as vt
from .contracts import ContractError
from .learner import OnPolicyLearner, OnPolicyOneStepBatch
RESULT_COMMAND = "cb16.vs-d-objective-balance-r2a@v1"
EXPERIMENT_ID = "r12.vs_d.objective_balance_entropy_gradient_diagnostic.r2a"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_d_objective_balance_entropy_gradient_r2a.json"
PARENT_R0_CONFIG = REPO_ROOT / "config" / "experiments" / "r12_vs_d_historical_market_canary_r0.json"
SPEC_FILENAME = "experiment_spec.json"
RESULT_FILENAME = "RESULT.json"
REPORT_FILENAME = "REPORT.md"

EXIT_PASS = 0
EXIT_SCIENTIFIC_FAIL = 1
EXIT_EXECUTION_BLOCKED = 3
EXIT_CONTRACT_MISMATCH = 4

ARM_POSITIVE_ID = r0.ARM_POSITIVE_ID
ARM_CONTROL_ID = r0.ARM_CONTROL_ID
RATIO_EPSILON = 1e-12
Q25_INDEX = 63


@dataclass(frozen=True)
class TrainingDataset:
    train_market: np.ndarray
    train_open: np.ndarray
    train_close: np.ndarray
    evidence: Mapping[str, Any]


def load_spec() -> Dict[str, Any]:
    try:
        return dict(tasks.spec_mapping(json.loads(CONFIG_PATH.read_text()), "R2A spec"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read R2A preregistered spec: {exc}") from exc

def spec_sha256(spec: Mapping[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _parent_spec() -> Mapping[str, Any]:
    try:
        return tasks.spec_mapping(json.loads(PARENT_R0_CONFIG.read_text()), "parent R0 spec")
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read parent R0 spec: {exc}") from exc


def validate_spec(spec: Mapping[str, Any]) -> None:
    validate_experiment_spec(spec)
    if spec.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("R2A experiment_id mismatch")
    scope = tasks.spec_mapping(spec["claim_scope"], "claim_scope")
    if not tasks.spec_bool(scope["march_read_forbidden"], "march_read_forbidden"):
        raise ContractError("R2A must forbid March reads")
    if tasks.spec_bool(scope["validation_data_read"], "validation_data_read"):
        raise ContractError("R2A is training-only")

    data = tasks.spec_mapping(spec["data"], "data")
    names = [tasks.spec_str(x["name"], "archive.name") for x in data["allowed_archives"]]
    if names != ["BTCUSDT-1m-2020-01.zip", "BTCUSDT-1m-2020-02.zip"]:
        raise ContractError("R2A allowed archives must be exactly Jan-Feb")
    hourly = tasks.spec_mapping(data["hourly_aggregation"], "hourly_aggregation")
    if tasks.spec_int(hourly["expected_hourly_bars"], "expected_hourly_bars") != 1440:
        raise ContractError("R2A Jan-Feb hourly count must be 1440")
    parent = _parent_spec()
    for key in ("authority", "learner", "model_seeds", "training_positive"):
        if spec[key] != parent[key]:
            raise ContractError(f"R2A must preserve parent R0 {key}")
    parent_control = {k: v for k, v in parent["training_control"].items() if k != "evaluation"}
    r2a_control = {k: v for k, v in spec["training_control"].items() if k != "r2a_evaluation"}
    if r2a_control != parent_control:
        raise ContractError("R2A must preserve parent R0 shuffled-control training semantics")
    if spec["training_control"].get("r2a_evaluation") != "NOT_APPLICABLE_TRAINING_ONLY_DIAGNOSTIC":
        raise ContractError("R2A control evaluation must be not applicable")

    learner = tasks.spec_mapping(spec["learner"], "learner")
    if tasks.spec_float(learner["direction_entropy_coefficient"], "lambda") != 0.005:
        raise ContractError("R2A freezes lambda_dir=0.005")
    if tasks.spec_int(learner["generations"], "generations") != 256:
        raise ContractError("R2A freezes 256 generations")
    if tuple(int(x) for x in spec["model_seeds"]) != tuple(range(3101, 3109)):
        raise ContractError("R2A freezes seeds 3101..3108")
    required = set(tasks.spec_sequence(spec["required_artifacts"], "required_artifacts"))
    if required != {SPEC_FILENAME, RESULT_FILENAME, REPORT_FILENAME}:
        raise ContractError("R2A required artifacts mismatch")


def load_training_dataset(spec: Mapping[str, Any]) -> TrainingDataset:
    times, minutes, evidence = r0._read_allowed_minutes(spec)
    hourly_times, hourly = r0.aggregate_utc_hourly(times, minutes)
    if len(hourly_times) != 1440:
        raise ContractError(f"R2A expected 1440 Jan-Feb hourly bars, got {len(hourly_times)}")
    data = tasks.spec_mapping(spec["data"], "data")
    context = tasks.spec_mapping(data["context"], "data.context")
    train = tasks.spec_mapping(data["train"], "data.train")
    raw_window_hours = tasks.spec_int(context["raw_window_hours"], "raw_window_hours")
    train_idx = r0._consequence_indices(
        hourly_times,
        r0._parse_utc_ms(train["consequence_hour_start_utc"], "train.start"),
        r0._parse_utc_ms(train["consequence_hour_end_utc"], "train.end"),
        tasks.spec_int(train["expected_decisions"], "train.expected_decisions"),
    )
    windows = hourly[r0._window_indices(train_idx, raw_window_hours)]
    normalized = n0_endpoint_log_ratios(windows).values
    if normalized.shape != (1375, 64, 5):
        raise ContractError(f"R2A training normalized shape mismatch: {normalized.shape!r}")
    if not np.isfinite(normalized).all():
        raise ContractError("R2A normalized training contexts must be finite")
    details = dict(evidence)
    details.update({
        "hourly_bars": 1440,
        "train_decisions": int(len(train_idx)),
        "march_opened": False,
        "causal_window_relation": "context_hours_strictly_before_consequence_hour",
    })
    return TrainingDataset(
        train_market=np.asarray(normalized, dtype=np.float64),
        train_open=np.asarray(hourly[train_idx, 0], dtype=np.float64),
        train_close=np.asarray(hourly[train_idx, 3], dtype=np.float64),
        evidence=details,
    )

def _grad_stats(grads: Sequence[torch.Tensor]) -> tuple[float, torch.Tensor]:
    flat = torch.cat([g.detach().reshape(-1).to(torch.float64) for g in grads])
    norm = float(torch.linalg.vector_norm(flat))
    if not math.isfinite(norm):
        raise ContractError("R2A diagnostic gradient norm must be finite")
    return norm, flat


def _snapshot_grads(module: torch.nn.Module) -> tuple[torch.Tensor | None, ...]:
    return tuple(None if p.grad is None else p.grad.detach().clone() for p in module.parameters())


def _require_same_grads(before: Sequence[torch.Tensor | None], module: torch.nn.Module) -> None:
    after = tuple(p.grad for p in module.parameters())
    if len(before) != len(after):
        raise ContractError("R2A diagnostic changed actor parameter count")
    for old, new in zip(before, after):
        if old is None:
            if new is not None:
                raise ContractError("R2A diagnostic populated persistent actor .grad")
        elif new is None or not torch.equal(old, new.detach()):
            raise ContractError("R2A diagnostic mutated persistent actor .grad")


def measure_objective_balance(
    learner: OnPolicyLearner,
    states: torch.Tensor,
    directions: torch.Tensor,
    risks: torch.Tensor,
    rewards: torch.Tensor,
) -> Dict[str, float]:
    """Observe objective components without changing RNG, params, grads or optimizer state."""

    rng_before = torch.random.get_rng_state().clone()
    grad_before = _snapshot_grads(learner.actor)
    params = tuple(learner.actor.parameters())
    output = learner.actor.forward(states)
    categorical = Categorical(logits=output.direction_logits)
    values = learner.critic(states)
    advantages = rewards.to(torch.float64) - values.detach().to(torch.float64)
    cat_log_prob = categorical.log_prob(directions.to(torch.long))
    direction_pg_loss = -(cat_log_prob.to(torch.float64) * advantages).mean()
    full_log_prob = learner.actor.log_prob_batch(states, directions, risks)
    full_pg_loss = -(full_log_prob.to(torch.float64) * advantages).mean()
    mean_entropy = categorical.entropy().mean()
    coefficient = learner.direction_entropy_coefficient
    weighted_entropy_loss = -coefficient * mean_entropy.to(torch.float64)
    combined_actor_loss = full_pg_loss + weighted_entropy_loss

    g_direction = torch.autograd.grad(direction_pg_loss, params, retain_graph=True)
    g_entropy = torch.autograd.grad(weighted_entropy_loss, params, retain_graph=True)
    g_full = torch.autograd.grad(full_pg_loss, params)
    direction_norm, direction_flat = _grad_stats(g_direction)
    entropy_norm, entropy_flat = _grad_stats(g_entropy)
    full_norm, _ = _grad_stats(g_full)
    ratio = entropy_norm / max(direction_norm, RATIO_EPSILON)
    log_ratio = math.log10(max(ratio, 1e-300))
    denominator = direction_norm * entropy_norm
    cosine = float(torch.dot(direction_flat, entropy_flat) / denominator) if denominator > 0.0 else 0.0

    if not torch.equal(rng_before, torch.random.get_rng_state()):
        raise ContractError("R2A diagnostic consumed torch RNG")
    _require_same_grads(grad_before, learner.actor)
    metrics = {
        "direction_pg_grad_l2": direction_norm,
        "weighted_entropy_grad_l2": entropy_norm,
        "full_pg_grad_l2": full_norm,
        "entropy_to_direction_pg_grad_ratio": ratio,
        "log10_entropy_to_direction_pg_grad_ratio": log_ratio,
        "direction_pg_entropy_cosine": cosine,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std(unbiased=False)),
        "mean_advantage": float(advantages.mean()),
        "std_advantage": float(advantages.std(unbiased=False)),
        "mean_direction_entropy": float(mean_entropy.detach()),
        "direction_pg_loss": float(direction_pg_loss.detach()),
        "full_pg_loss": float(full_pg_loss.detach()),
        "weighted_entropy_loss": float(weighted_entropy_loss.detach()),
        "combined_actor_loss": float(combined_actor_loss.detach()),
    }
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ContractError("R2A diagnostic metrics must all be finite")
    return metrics

def _summarize_generations(records: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    if len(records) != 256:
        raise ContractError(f"R2A requires 256 generation diagnostics, got {len(records)}")
    ratios = sorted(float(x["log10_entropy_to_direction_pg_grad_ratio"]) for x in records)
    q25 = ratios[Q25_INDEX]
    return {
        "generation_count": len(records),
        "discrete_q25_log10_ratio_index63": q25,
        "median_log10_ratio": statistics.median(ratios),
        "fraction_generations_entropy_grad_gt_direction_pg": sum(x > 0.0 for x in ratios) / len(ratios),
        "mean_direction_entropy": statistics.mean(float(x["mean_direction_entropy"]) for x in records),
        "mean_reward_std": statistics.mean(float(x["std_reward"]) for x in records),
    }


def train_arm_with_diagnostics(
    learner: OnPolicyLearner,
    states: torch.Tensor,
    open_next: np.ndarray,
    close_next: np.ndarray,
    *,
    seed: int,
    arm_id: int,
    generations: int,
    nominal_exposure_budget: float,
) -> Dict[str, Any]:
    records: list[Dict[str, float]] = []
    count = int(states.shape[0])
    for generation in range(generations):
        if learner.generation_id != generation:
            raise ContractError("R2A learner generation boundary mismatch")
        torch.manual_seed(r0._collection_seed(seed, arm_id, generation))
        directions, risks = vt.sample_action_batch(learner, states)
        consequence_open, consequence_close = open_next, close_next
        if arm_id == ARM_CONTROL_ID:
            order = tasks.no_fixed_point_permutation(count, seed=seed, generation=generation)
            permutation = np.asarray(order, dtype=np.int64)
            consequence_open, consequence_close = open_next[permutation], close_next[permutation]
        elif arm_id != ARM_POSITIVE_ID:
            raise ContractError(f"unknown R2A arm id {arm_id}")
        rewards = r0.historical_reward_batch(
            directions,
            risks,
            consequence_open,
            consequence_close,
            nominal_exposure_budget=nominal_exposure_budget,
        )
        metrics = measure_objective_balance(learner, states, directions, risks, rewards)
        batch = OnPolicyOneStepBatch(
            states=states,
            direction_indices=directions,
            requested_risks=risks,
            rewards=rewards,
            generation_id=generation,
        )
        update = learner.update_one_step_batch(batch)
        if not math.isclose(update.actor_loss, metrics["combined_actor_loss"], rel_tol=1e-6, abs_tol=1e-9):
            raise ContractError("R2A diagnostic decomposition does not match unchanged learner actor loss")
        records.append({"generation": generation, **metrics})
    return {"summary": _summarize_generations(records), "generations": records}


def _seed_worker(payload: Tuple[Dict[str, Any], TrainingDataset, int]) -> Dict[str, Any]:
    spec, dataset, seed = payload
    q.configure_deterministic_runtime()
    authority = r0.parse_authority(spec)
    optimizer = r0.parse_optimizer(spec)
    entropy = tasks.spec_float(spec["learner"]["direction_entropy_coefficient"], "lambda")
    generations = tasks.spec_int(spec["learner"]["generations"], "generations", minimum=1)
    budget = tasks.spec_float(spec["authority"]["nominal_exposure_budget"], "budget")
    account_row = r0._flat_account_row(spec)

    torch.manual_seed(seed)
    positive = q.build_learner(authority, optimizer, direction_entropy_coefficient=entropy)
    torch.manual_seed(seed)
    control = q.build_learner(authority, optimizer, direction_entropy_coefficient=entropy)
    q.require_identical_parameters(positive, control)
    pos_states = r0._historical_states(positive, dataset.train_market, account_row)
    ctl_states = r0._historical_states(control, dataset.train_market, account_row)
    if not torch.equal(pos_states, ctl_states):
        raise ContractError("R2A paired arms must start from identical states")
    positive_record = train_arm_with_diagnostics(
        positive, pos_states, dataset.train_open, dataset.train_close,
        seed=seed, arm_id=ARM_POSITIVE_ID, generations=generations,
        nominal_exposure_budget=budget,
    )
    control_record = train_arm_with_diagnostics(
        control, ctl_states, dataset.train_open, dataset.train_close,
        seed=seed, arm_id=ARM_CONTROL_ID, generations=generations,
        nominal_exposure_budget=budget,
    )
    threshold = tasks.spec_float(
        spec["seed_gate"]["positive_arm_discrete_q25_log10_entropy_to_direction_pg_grad_ratio_strictly_gt"],
        "seed_gate.threshold",
    )
    q25 = float(positive_record["summary"]["discrete_q25_log10_ratio_index63"])
    return {
        "seed": seed,
        "positive": positive_record,
        "control": control_record,
        "seed_gate": {
            "threshold": threshold,
            "passed": q25 > threshold,
            "positive_discrete_q25_log10_ratio_index63": q25,
        },
    }


def run_seed_set(spec: Mapping[str, Any], dataset: TrainingDataset) -> list[Dict[str, Any]]:
    seeds = tuple(int(x) for x in spec["model_seeds"])
    workers = min(tasks.spec_int(spec["authority"]["max_seed_workers"], "max_seed_workers", minimum=1), len(seeds))
    payloads = [(dict(spec), dataset, seed) for seed in seeds]
    if workers <= 1:
        return [_seed_worker(x) for x in payloads]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        return list(executor.map(_seed_worker, payloads))


def aggregate_gate(spec: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values = [float(x["positive"]["summary"]["discrete_q25_log10_ratio_index63"]) for x in records]
    pass_count = sum(bool(x["seed_gate"]["passed"]) for x in records)
    gate = tasks.spec_mapping(spec["aggregate_gate"], "aggregate_gate")
    min_count = tasks.spec_int(gate["seed_pass_count_min"], "aggregate_gate.seed_pass_count_min")
    median_threshold = tasks.spec_float(
        gate["median_across_8_seeds_of_discrete_q25_log10_ratio_strictly_gt"],
        "aggregate_gate.median_threshold",
    )
    median_q25 = statistics.median(values)
    conditions = {
        "seed_pass_count": pass_count >= min_count,
        "median_seed_q25_log10_ratio": median_q25 > median_threshold,
    }
    return {
        "seed_pass_count": pass_count,
        "median_seed_q25_log10_ratio": median_q25,
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def runtime_record() -> Dict[str, Any]:
    record = q.runtime_record()
    record.update({
        "max_seed_workers": 4,
        "historical_interval": "BTCUSDT_2020-01_2020-02_training_only",
        "march_read_forbidden": True,
    })
    return record


def _prior_evidence() -> list[Dict[str, Any]]:
    return [
        {"claim_ref": "r12.vs_c.joint_confirmation.r3", "status": "UNAFFECTED_IN_ORIGINAL_SCOPE", "transfer": "OBJECTIVE_SCALE_TRANSFER_UNRESOLVED"},
        {"claim_ref": "r12.vs_d.historical_market_information_canary.r0", "status": "UNAFFECTED_IN_ORIGINAL_SCOPE", "current_scoped_assessment": "TESTED_CONFIGURATION_DEV_DECISION_GATE_NOT_MET"},
        {"claim_ref": "r12.vs_d.evaluation_adapter_transfer_diagnostic.r1", "status": "UNAFFECTED_IN_ORIGINAL_SCOPE", "current_scoped_assessment": "ADAPTER_AND_DISTRIBUTION_DIAGNOSTIC_GATES_NOT_MET"},
    ]

def run_experiment(
    spec: Mapping[str, Any], dataset: TrainingDataset, implementation_tests: Mapping[str, Any]
) -> Dict[str, Any]:
    validate_spec(spec)
    q.configure_deterministic_runtime()
    test_gate = q.implementation_test_gate(implementation_tests, commit=q.commit_sha())
    if not test_gate["passed"]:
        raise ContractError("R2A exact-commit implementation test gate is not green")
    records = run_seed_set(spec, dataset)
    r0._load_manifest(spec)
    gate = aggregate_gate(spec, records)
    passed = bool(gate["passed"])
    promotion = "PROMOTE_ENTROPY_SCALE_INTERVENTION_HYPOTHESIS" if passed else "NO_PROMOTION"
    result = {
        "schema": "cb16.result.v2",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(),
        "classification": "PASS" if passed else "SCIENTIFIC_FAIL",
        "execution_status": "EXECUTED",
        "validity_status": "VALID",
        "scientific_outcome": "DIRECTION_ENTROPY_GRADIENT_DOMINANCE_QUALIFIED" if passed else "DIRECTION_ENTROPY_GRADIENT_DOMINANCE_NOT_QUALIFIED",
        "runtime": runtime_record(),
        "implementation_tests": {**dict(implementation_tests), "gate": test_gate, "required_by_global_gate": True, "runner_executed_suite": True},
        "preregistered_spec": {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "canonical_sha256": spec_sha256(spec), "model_seeds": list(spec["model_seeds"]), "declared_status": spec.get("status")},
        "data_evidence": {**dict(dataset.evidence), "manifest_final_holdout_accessed_post_run": False},
        "seeds": records,
        "aggregate_gate": gate,
    }
    result.update({
        "gate_results": [
            {"gate_id": "G_OBJECTIVE_BALANCE_SEED", "passed": gate["seed_pass_count"] >= tasks.spec_int(spec["aggregate_gate"]["seed_pass_count_min"], "aggregate seed minimum"), "metrics": {"seed_pass_count": gate["seed_pass_count"]}},
            {"gate_id": "G_OBJECTIVE_BALANCE_AGGREGATE", "passed": passed, "metrics": {"seed_pass_count": gate["seed_pass_count"], "median_seed_q25_log10_ratio": gate["median_seed_q25_log10_ratio"]}, "conditions": gate["conditions"]},
        ],
        "claim_assessments": [{
            "claim_id": "C_DIRECTION_ENTROPY_GRADIENT_DOMINANCE",
            "qualification_status": "QUALIFIED" if passed else "NOT_QUALIFIED",
            "inference_status": "SUPPORTED" if passed else "GATE_NOT_MET",
            "attribution_status": "PARTIAL" if passed else "UNRESOLVED",
        }],
        "prior_evidence_assessment": _prior_evidence(),
        "promotion_decision": {
            "decision": promotion,
            "reason": "R2A can only promote a new objective-scale intervention hypothesis; causal attribution to parent gate misses remains unresolved",
        },
        "provenance": {
            "commit_sha": q.commit_sha(),
            "spec_file_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            "parent_r0_run_id": 34911876483,
            "parent_r1_run_id": 34919793143,
            "parent_r1_artifact_digest": "sha256:5c092c0a81b40db7be6f03898a8c01785f10ffbd73bae7a60e4210532ff3c67e",
            "march_read": False,
            "final_holdout_accessed": False,
            "freshness": "DIAGNOSTIC_REUSED_JAN_FEB_TRAINING_NOT_CONFIRMATION",
        },
    })
    validate_result_against_spec(result, spec)
    return result


def render_report(result: Mapping[str, Any]) -> str:
    gate = result["aggregate_gate"]
    lines = [
        "# R12 VS-D Objective-Balance / Entropy-Gradient Diagnostic R2A — REPORT", "",
        f"- classification: **{result['classification']}**",
        f"- scientific outcome: **{result['scientific_outcome']}**",
        f"- promotion: **{result['promotion_decision']['decision']}**",
        "- data: **Jan-Feb training only; March not read**", "",
        "## Per-seed summary", "",
        "| seed | pass | positive q25 log10 ratio | positive frac entropy>PG | control q25 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for record in result["seeds"]:
        pos = record["positive"]["summary"]
        ctl = record["control"]["summary"]
        lines.append(
            f"| {record['seed']} | {'PASS' if record['seed_gate']['passed'] else 'FAIL'} | "
            f"{pos['discrete_q25_log10_ratio_index63']:.6f} | "
            f"{pos['fraction_generations_entropy_grad_gt_direction_pg']:.4f} | "
            f"{ctl['discrete_q25_log10_ratio_index63']:.6f} |"
        )
    lines += [
        "", "## Aggregate gate", "",
        f"- seed pass count: `{gate['seed_pass_count']}/8`",
        f"- median seed q25 log10 ratio: `{gate['median_seed_q25_log10_ratio']:.12g}`",
        f"- seed-count condition: **{gate['conditions']['seed_pass_count']}**",
        f"- median-q25 condition: **{gate['conditions']['median_seed_q25_log10_ratio']}**",
        "", "## Authority boundary", "",
        "- PASS would only justify a new preregistered entropy/objective-scale intervention.",
        "- PASS would not prove entropy caused the parent R0/R1 gate misses.",
        "- FAIL would not falsify representation, horizon, learner family, or full CB16.",
        "- No March development validation or final holdout was opened by this experiment.", "",
    ]
    return "\n".join(lines)


def write_artifacts(result_dir: Path, spec: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (result_dir / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (result_dir / REPORT_FILENAME).write_text(render_report(result))

def failure_result(spec: Mapping[str, Any] | None, classification: str, detail: str,
                   implementation_tests: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": "cb16.result.v2",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(),
        "classification": classification,
        "execution_status": classification,
        "validity_status": "INVALID_OR_NOT_EVALUATED",
        "scientific_outcome": "NOT_EVALUATED",
        "runtime": runtime_record(),
        "seeds": [],
        "aggregate_gate": {},
        "gate_results": [],
        "claim_assessments": [{
            "claim_id": "C_DIRECTION_ENTROPY_GRADIENT_DOMINANCE",
            "qualification_status": "NOT_APPLICABLE",
            "inference_status": "NOT_TESTED",
            "attribution_status": "UNRESOLVED",
        }],
    }
    result.update({
        "prior_evidence_assessment": _prior_evidence(),
        "promotion_decision": {
            "decision": "NO_PROMOTION",
            "reason": "scientific claim was not validly evaluated",
        },
        "provenance": {
            "commit_sha": q.commit_sha(),
            "march_read": False,
            "final_holdout_accessed": False,
        },
        "error": {"type": classification, "detail": detail[:2000]},
    })
    if spec is not None:
        result["preregistered_spec"] = {
            "path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "canonical_sha256": spec_sha256(spec),
            "declared_status": spec.get("status"),
        }
        validate_result_against_spec(result, spec)
    if implementation_tests is not None:
        result["implementation_tests"] = dict(implementation_tests)
    return result

def write_failure(result_dir: Path, spec: Mapping[str, Any] | None, result: Mapping[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    if spec is not None:
        (result_dir / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (result_dir / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (result_dir / REPORT_FILENAME).write_text(
        "# R12 VS-D Objective-Balance Diagnostic R2A — REPORT\n\n"
        f"- classification: **{result['classification']}**\n"
        "- scientific outcome: **NOT_EVALUATED**\n"
        "- March was not read by this runner.\n\n"
        f"{result.get('error', {}).get('detail', '')}\n"
    )


def run_qualification(result_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    spec = load_spec()
    validate_spec(spec)
    implementation = q.run_implementation_test_suite()
    implementation_gate = q.implementation_test_gate(implementation, commit=q.commit_sha())
    if not implementation_gate["passed"]:
        classification, exit_code = q.implementation_failure_classification(implementation)
        detail = q.implementation_test_failure_detail(classification, implementation)
        evidence = {**dict(implementation), "gate": implementation_gate, "failure_classification": classification}
        result = failure_result(spec, classification, detail, evidence)
        write_failure(result_dir, spec, result)
        return spec, result, exit_code
    dataset = load_training_dataset(spec)
    result = run_experiment(spec, dataset, implementation)
    write_artifacts(result_dir, spec, result)
    return spec, result, EXIT_PASS if result["classification"] == "PASS" else EXIT_SCIENTIFIC_FAIL

def main() -> int:
    raw = (os.environ.get("CB16_RESULT_DIR") or "").strip()
    if not raw or not (os.environ.get("CB16_COMMIT_SHA") or "").strip():
        print(f"{RESULT_COMMAND}: EXECUTION_BLOCKED: CB16_RESULT_DIR and CB16_COMMIT_SHA must be set", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED
    result_dir = Path(raw)
    try:
        _spec, result, exit_code = run_qualification(result_dir)
    except r0.HistoricalDataUnavailable as exc:
        try:
            spec = load_spec()
        except ContractError:
            spec = None
        result = failure_result(spec, "EXECUTION_BLOCKED", f"{type(exc).__name__}: {exc}")
        write_failure(result_dir, spec, result)
        return EXIT_EXECUTION_BLOCKED
    except ContractError as exc:
        try:
            spec = load_spec()
        except ContractError:
            spec = None
        result = failure_result(spec, "CONTRACT_MISMATCH", f"{type(exc).__name__}: {exc}")
        write_failure(result_dir, spec, result)
        return EXIT_CONTRACT_MISMATCH
    except Exception as exc:  # noqa: BLE001
        try:
            spec = load_spec()
        except ContractError:
            spec = None
        result = failure_result(spec, "EXECUTION_BLOCKED", f"{type(exc).__name__}: {exc}")
        write_failure(result_dir, spec, result)
        return EXIT_EXECUTION_BLOCKED
    print(f"{RESULT_COMMAND}: classification={result['classification']} seed_pass_count={result['aggregate_gate']['seed_pass_count']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
