"""R12 VS-D R1 evaluation-adapter transfer diagnostic.

The scientific method is frozen in
``config/experiments/r12_vs_d_evaluation_adapter_transfer_r1.json``.  Parent
R0 training/data semantics are reused unchanged.  The only new scientific axis
is deterministic evaluation of expected one-step log-growth under the complete
stochastic policy distribution.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from ..evidence_scope import EvidenceScopeError, validate_experiment_spec, validate_result_against_spec
from . import controlled_tasks as tasks
from . import historical_market_canary_r0 as r0
from . import qualification as q
from .learner import OnPolicyLearner
from .policy import DIRECTION_INDEX_FLAT, DIRECTION_INDEX_LONG, DIRECTION_INDEX_SHORT
from .contracts import ContractError

RESULT_COMMAND = "cb16.vs-d-evaluation-adapter-transfer-r1@v1"
EXPERIMENT_ID = "r12.vs_d.evaluation_adapter_transfer_diagnostic.r1"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_d_evaluation_adapter_transfer_r1.json"
PARENT_CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_d_historical_market_canary_r0.json"
SPEC_FILENAME = "experiment_spec.json"
RESULT_FILENAME = "RESULT.json"
REPORT_FILENAME = "REPORT.md"

FROZEN_SPEC_FILE_SHA256 = "767e70bc90003bddf60fa085d15cf6bcbf00f27db1ccadeaeb615504fe3bb29a"
FROZEN_PARENT_SPEC_FILE_SHA256 = "af589855ff534995725f40f41cf5ad02c7f0773aef71d4ace3da49affb297cd2"

EXIT_PASS = 0
EXIT_SCIENTIFIC_FAIL = 1
EXIT_EXECUTION_BLOCKED = 3
EXIT_CONTRACT_MISMATCH = 4

BOOTSTRAP_DOMAIN = 0x41445231
ADAPTER_DELTA_CONTROL = 10
ADAPTER_DELTA_PRE = 11
EXPECTED_POS_CONTROL = 20
EXPECTED_POS_PRE = 21
EXPECTED_POS_ABSOLUTE = 22

def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_spec() -> Dict[str, Any]:
    try:
        spec = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read R1 preregistered spec: {exc}") from exc
    if _file_sha256(CONFIG_PATH) != FROZEN_SPEC_FILE_SHA256:
        raise ContractError("R1 preregistered spec bytes differ from the frozen implementation authority")
    try:
        validate_experiment_spec(spec)
    except EvidenceScopeError as exc:
        raise ContractError(f"R1 claim-authority spec mismatch: {exc}") from exc
    return dict(spec)


def load_parent_spec() -> Dict[str, Any]:
    try:
        parent = json.loads(PARENT_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read parent R0 spec: {exc}") from exc
    if _file_sha256(PARENT_CONFIG_PATH) != FROZEN_PARENT_SPEC_FILE_SHA256:
        raise ContractError("parent R0 spec bytes changed; R1 comparability is invalid")
    r0.validate_spec(parent)
    return dict(parent)


def validate_r1_against_parent(spec: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    training = tasks.spec_mapping(spec["training"], "training")
    parent_learner = tasks.spec_mapping(parent["learner"], "parent.learner")
    expected_pairs = {
        "generations": parent_learner["generations"],
        "actor_lr": parent_learner["actor_lr"],
        "critic_lr": parent_learner["critic_lr"],
        "direction_entropy_coefficient": parent_learner["direction_entropy_coefficient"],
        "beta_risk_entropy_bonus": parent_learner["beta_risk_entropy_bonus"],
        "replay": parent_learner["replay"],
    }
    for key, expected in expected_pairs.items():
        if training.get(key) != expected:
            raise ContractError(f"R1 training.{key} must exactly match parent R0")
    if tuple(training["model_seeds"]) != tuple(parent["model_seeds"]):
        raise ContractError("R1 model seeds must exactly match parent R0")

    data = tasks.spec_mapping(spec["data"], "data")
    parent_data = tasks.spec_mapping(parent["data"], "parent.data")
    parent_archives = tasks.spec_sequence(parent_data["allowed_archives"], "parent.data.allowed_archives")
    if list(data["allowed_archives"]) != list(parent_archives):
        raise ContractError("R1 allowed archives must exactly match parent R0")
    if data["expected_hourly_bars"] != parent_data["hourly_aggregation"]["expected_hourly_bars"]:
        raise ContractError("R1 hourly bar count must match parent R0")
    if data["train_decisions"] != parent_data["train"]["expected_decisions"]:
        raise ContractError("R1 training decision count must match parent R0")
    if data["validation_decisions"] != parent_data["validation"]["expected_decisions"]:
        raise ContractError("R1 validation decision count must match parent R0")
    if data["validation_day_blocks"] != parent_data["validation"]["expected_utc_day_blocks"]:
        raise ContractError("R1 validation day blocks must match parent R0")
    if spec["claim_scope"]["changed_axis"] != "evaluation_adapter_only":
        raise ContractError("R1 changed axis must remain evaluation_adapter_only")
    if spec["research_series_policy"]["freshness_status"] != "NOT_FRESH_CONFIRMATION":
        raise ContractError("R1 must record March as reused development, not fresh confirmation")
    evaluation = tasks.spec_mapping(spec["evaluation"], "evaluation")
    quad = tasks.spec_mapping(evaluation["beta_expectation"], "evaluation.beta_expectation")
    if evaluation["baseline_adapter"] != "deterministic_argmax_direction_plus_beta_mean":
        raise ContractError("R1 baseline adapter mismatch")
    if evaluation["diagnostic_adapter"] != "deterministic_expected_stochastic_policy_log_growth":
        raise ContractError("R1 diagnostic adapter mismatch")
    if quad["method"] != "Gauss_Legendre_quadrature_on_[0,1]" or quad["nodes"] != 64:
        raise ContractError("R1 quadrature must be frozen to 64-node Gauss-Legendre")
    if quad["dtype"] != "float64" or quad["random_sampling"] is not False:
        raise ContractError("R1 quadrature must be deterministic float64 without sampling")

def _quadrature(nodes: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    x, w = np.polynomial.legendre.leggauss(nodes)
    risk = torch.from_numpy((x + 1.0) * 0.5).to(torch.float64)
    weight = torch.from_numpy(w * 0.5).to(torch.float64)
    return risk, weight


def _beta_expectation(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    rewards_by_risk: torch.Tensor,
    risk: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a = alpha.to(torch.float64).unsqueeze(1)
    b = beta.to(torch.float64).unsqueeze(1)
    x = risk.unsqueeze(0)
    log_norm = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    log_pdf = (a - 1.0) * torch.log(x) + (b - 1.0) * torch.log1p(-x) - log_norm
    weighted = torch.exp(log_pdf) * weight.unsqueeze(0)
    mass = weighted.sum(dim=1)
    if not bool(torch.isfinite(mass).all()) or not bool((mass > 0.0).all()):
        raise ContractError("R1 Beta quadrature mass must be finite and positive")
    expected = (weighted * rewards_by_risk).sum(dim=1) / mass
    return expected, mass


def evaluate_policy_distribution(
    learner: OnPolicyLearner,
    states: torch.Tensor,
    open_next: np.ndarray,
    close_next: np.ndarray,
    *,
    nodes: int = 64,
) -> Dict[str, Any]:
    learner._require_generation_snapshot()
    open_t = torch.as_tensor(open_next, dtype=torch.float64)
    close_t = torch.as_tensor(close_next, dtype=torch.float64)
    if open_t.shape != close_t.shape or open_t.ndim != 1 or open_t.numel() != states.shape[0]:
        raise ContractError("R1 consequence arrays must align one-to-one with validation states")
    ratio = close_t / open_t - 1.0
    if not bool(torch.isfinite(ratio).all()) or not bool(((ratio > -1.0) & (ratio < 1.0)).all()):
        raise ContractError("R1 full-exposure one-step return must lie strictly inside (-1, 1)")
    risk, weight = _quadrature(nodes)
    rr = ratio.unsqueeze(1) * risk.unsqueeze(0)
    short_rewards = torch.log1p(-rr)
    long_rewards = torch.log1p(rr)

    with torch.no_grad():
        output = learner.actor.forward(states)
        probs = torch.softmax(output.direction_logits.to(torch.float64), dim=-1)
        short_expected, short_mass = _beta_expectation(
            output.short_alpha, output.short_beta, short_rewards, risk, weight
        )
        long_expected, long_mass = _beta_expectation(
            output.long_alpha, output.long_beta, long_rewards, risk, weight
        )
        expected = (
            probs[:, DIRECTION_INDEX_SHORT] * short_expected
            + probs[:, DIRECTION_INDEX_LONG] * long_expected
        )
        entropy = -(probs * torch.log(probs.clamp_min(torch.finfo(torch.float64).tiny))).sum(dim=1)
        short_mean = output.short_alpha.to(torch.float64) / (
            output.short_alpha.to(torch.float64) + output.short_beta.to(torch.float64)
        )
        long_mean = output.long_alpha.to(torch.float64) / (
            output.long_alpha.to(torch.float64) + output.long_beta.to(torch.float64)
        )
    if not bool(torch.isfinite(expected).all()):
        raise ContractError("R1 expected policy log-growth must be finite")
    return {
        "mean_expected_one_step_log_growth": float(expected.mean()),
        "hourly_expected_log_growth": [float(x) for x in expected.tolist()],
        "diagnostics": {
            "mean_direction_entropy": float(entropy.mean()),
            "mean_short_probability": float(probs[:, DIRECTION_INDEX_SHORT].mean()),
            "mean_flat_probability": float(probs[:, DIRECTION_INDEX_FLAT].mean()),
            "mean_long_probability": float(probs[:, DIRECTION_INDEX_LONG].mean()),
            "mean_expected_active_probability": float((1.0 - probs[:, DIRECTION_INDEX_FLAT]).mean()),
            "mean_short_beta_mean": float(short_mean.mean()),
            "mean_long_beta_mean": float(long_mean.mean()),
            "max_abs_short_quadrature_mass_error": float((short_mass - 1.0).abs().max()),
            "max_abs_long_quadrature_mass_error": float((long_mass - 1.0).abs().max()),
        },
    }

def bootstrap_lcb(values: Sequence[float], day_index: Sequence[int], *, seed: int, comparison_id: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    days = np.asarray(day_index, dtype=np.int64)
    if array.shape != (744,) or days.shape != (744,):
        raise ContractError("R1 bootstrap requires exactly 744 validation observations")
    blocks = [array[days == day] for day in range(31)]
    if any(block.shape != (24,) for block in blocks):
        raise ContractError("R1 bootstrap requires 31 UTC day blocks of 24 hours")
    rng = random.Random(tasks.derived_stream_seed(BOOTSTRAP_DOMAIN, seed, comparison_id))
    estimates = []
    for _ in range(4096):
        picked = [rng.randrange(31) for _ in range(31)]
        estimates.append(float(np.mean([blocks[index].mean() for index in picked])))
    estimates.sort()
    return estimates[204]


def _seed_worker(payload: Tuple[Dict[str, Any], Dict[str, Any], r0.HistoricalDataset, int]) -> Dict[str, Any]:
    spec, parent, dataset, seed = payload
    q.configure_deterministic_runtime()
    authority = r0.parse_authority(parent)
    optimizer = r0.parse_optimizer(parent)
    account_row = r0._flat_account_row(parent)
    parent_authority = tasks.spec_mapping(parent["authority"], "parent.authority")
    parent_learner = tasks.spec_mapping(parent["learner"], "parent.learner")
    budget = tasks.spec_float(parent_authority["nominal_exposure_budget"], "parent.authority.nominal_exposure_budget")
    entropy = tasks.spec_float(parent_learner["direction_entropy_coefficient"], "parent.learner.direction_entropy_coefficient")
    generations = tasks.spec_int(parent_learner["generations"], "parent.learner.generations", minimum=1)

    torch.manual_seed(seed)
    positive = q.build_learner(authority, optimizer, direction_entropy_coefficient=entropy)
    torch.manual_seed(seed)
    control = q.build_learner(authority, optimizer, direction_entropy_coefficient=entropy)
    q.require_identical_parameters(positive, control)

    train_states_pos = r0._historical_states(positive, dataset.train_market, account_row)
    train_states_ctl = r0._historical_states(control, dataset.train_market, account_row)
    val_states_pos = r0._historical_states(positive, dataset.validation_market, account_row)
    val_states_ctl = r0._historical_states(control, dataset.validation_market, account_row)
    if not torch.equal(train_states_pos, train_states_ctl) or not torch.equal(val_states_pos, val_states_ctl):
        raise ContractError("R1 paired positive/control state tensors must be bitwise identical")

    pre_argmax = r0.evaluate_validation(
        positive, val_states_pos, dataset.validation_open, dataset.validation_close,
        nominal_exposure_budget=budget,
    )
    pre_expected = evaluate_policy_distribution(
        positive, val_states_pos, dataset.validation_open, dataset.validation_close
    )
    train_positive = r0._train_learner(
        positive, train_states_pos, dataset.train_open, dataset.train_close,
        seed=seed, arm_id=r0.ARM_POSITIVE_ID, generations=generations,
        nominal_exposure_budget=budget,
    )
    train_control = r0._train_learner(
        control, train_states_ctl, dataset.train_open, dataset.train_close,
        seed=seed, arm_id=r0.ARM_CONTROL_ID, generations=generations,
        nominal_exposure_budget=budget,
    )
    post_pos_argmax = r0.evaluate_validation(
        positive, val_states_pos, dataset.validation_open, dataset.validation_close,
        nominal_exposure_budget=budget,
    )
    post_ctl_argmax = r0.evaluate_validation(
        control, val_states_ctl, dataset.validation_open, dataset.validation_close,
        nominal_exposure_budget=budget,
    )
    post_pos_expected = evaluate_policy_distribution(
        positive, val_states_pos, dataset.validation_open, dataset.validation_close
    )
    post_ctl_expected = evaluate_policy_distribution(
        control, val_states_ctl, dataset.validation_open, dataset.validation_close
    )

    arg_pre = np.asarray(pre_argmax["hourly_log_growth"], dtype=np.float64)
    arg_pos = np.asarray(post_pos_argmax["hourly_log_growth"], dtype=np.float64)
    arg_ctl = np.asarray(post_ctl_argmax["hourly_log_growth"], dtype=np.float64)
    exp_pre = np.asarray(pre_expected["hourly_expected_log_growth"], dtype=np.float64)
    exp_pos = np.asarray(post_pos_expected["hourly_expected_log_growth"], dtype=np.float64)
    exp_ctl = np.asarray(post_ctl_expected["hourly_expected_log_growth"], dtype=np.float64)
    arg_pc = arg_pos - arg_ctl
    arg_pp = arg_pos - arg_pre
    exp_pc = exp_pos - exp_ctl
    exp_pp = exp_pos - exp_pre
    delta_pc = exp_pc - arg_pc
    delta_pp = exp_pp - arg_pp

    lcb = {
        "adapter_delta_true_minus_control": bootstrap_lcb(
            delta_pc, dataset.validation_day_index, seed=seed, comparison_id=ADAPTER_DELTA_CONTROL
        ),
        "adapter_delta_true_minus_pre": bootstrap_lcb(
            delta_pp, dataset.validation_day_index, seed=seed, comparison_id=ADAPTER_DELTA_PRE
        ),
        "expected_positive_minus_control": bootstrap_lcb(
            exp_pc, dataset.validation_day_index, seed=seed, comparison_id=EXPECTED_POS_CONTROL
        ),
        "expected_positive_minus_pre": bootstrap_lcb(
            exp_pp, dataset.validation_day_index, seed=seed, comparison_id=EXPECTED_POS_PRE
        ),
        "expected_positive_mean_log_growth": bootstrap_lcb(
            exp_pos, dataset.validation_day_index, seed=seed, comparison_id=EXPECTED_POS_ABSOLUTE
        ),
    }
    points = {
        "argmax_positive_mean_log_growth": float(arg_pos.mean()),
        "argmax_control_mean_log_growth": float(arg_ctl.mean()),
        "argmax_pre_mean_log_growth": float(arg_pre.mean()),
        "argmax_positive_minus_control": float(arg_pc.mean()),
        "argmax_positive_minus_pre": float(arg_pp.mean()),
        "expected_positive_mean_log_growth": float(exp_pos.mean()),
        "expected_control_mean_log_growth": float(exp_ctl.mean()),
        "expected_pre_mean_log_growth": float(exp_pre.mean()),
        "expected_positive_minus_control": float(exp_pc.mean()),
        "expected_positive_minus_pre": float(exp_pp.mean()),
        "adapter_delta_true_minus_control": float(delta_pc.mean()),
        "adapter_delta_true_minus_pre": float(delta_pp.mean()),
    }
    adapter_seed_pass = (
        lcb["adapter_delta_true_minus_control"] > 0.0
        and lcb["adapter_delta_true_minus_pre"] > 0.0
    )
    distribution_seed_pass = (
        lcb["expected_positive_minus_control"] > 0.0
        and lcb["expected_positive_minus_pre"] > 0.0
    )

    def compact_expected(block: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "mean_expected_one_step_log_growth": block["mean_expected_one_step_log_growth"],
            "diagnostics": dict(block["diagnostics"]),
        }

    return {
        "seed": seed,
        "point_estimates": points,
        "bootstrap_lcb": lcb,
        "seed_gates": {
            "adapter_transfer": adapter_seed_pass,
            "distribution_level_relation": distribution_seed_pass,
        },
        "argmax": {
            "pre": {k: v for k, v in pre_argmax.items() if k != "hourly_log_growth"},
            "post_positive": {k: v for k, v in post_pos_argmax.items() if k != "hourly_log_growth"},
            "post_control": {k: v for k, v in post_ctl_argmax.items() if k != "hourly_log_growth"},
        },
        "policy_distribution": {
            "pre": compact_expected(pre_expected),
            "post_positive": compact_expected(post_pos_expected),
            "post_control": compact_expected(post_ctl_expected),
        },
        "training_diagnostics": {"positive": train_positive, "control": train_control},
    }


def run_seed_set(
    spec: Mapping[str, Any], parent: Mapping[str, Any], dataset: r0.HistoricalDataset
) -> list[Dict[str, Any]]:
    seeds = tuple(int(seed) for seed in spec["training"]["model_seeds"])
    workers = min(4, len(seeds))
    payloads = [(dict(spec), dict(parent), dataset, seed) for seed in seeds]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        return list(executor.map(_seed_worker, payloads))

def aggregate_gates(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    adapter_count = sum(bool(record["seed_gates"]["adapter_transfer"]) for record in records)
    distribution_count = sum(
        bool(record["seed_gates"]["distribution_level_relation"]) for record in records
    )
    median = lambda key: statistics.median(
        float(record["point_estimates"][key]) for record in records
    )
    adapter_metrics = {
        "seed_pass_count": adapter_count,
        "median_delta_true_minus_control": median("adapter_delta_true_minus_control"),
        "median_delta_true_minus_pre": median("adapter_delta_true_minus_pre"),
    }
    adapter_conditions = {
        "seed_pass_count_min_6": adapter_count >= 6,
        "median_delta_true_minus_control_gt_0": adapter_metrics["median_delta_true_minus_control"] > 0.0,
        "median_delta_true_minus_pre_gt_0": adapter_metrics["median_delta_true_minus_pre"] > 0.0,
    }
    distribution_metrics = {
        "seed_pass_count": distribution_count,
        "median_expected_positive_minus_control": median("expected_positive_minus_control"),
        "median_expected_positive_minus_pre": median("expected_positive_minus_pre"),
        "median_expected_positive_mean_log_growth": median("expected_positive_mean_log_growth"),
    }
    distribution_conditions = {
        "seed_pass_count_min_6": distribution_count >= 6,
        "median_expected_positive_minus_control_gt_0": distribution_metrics["median_expected_positive_minus_control"] > 0.0,
        "median_expected_positive_minus_pre_gt_0": distribution_metrics["median_expected_positive_minus_pre"] > 0.0,
        "median_expected_positive_mean_log_growth_gt_0": distribution_metrics["median_expected_positive_mean_log_growth"] > 0.0,
    }
    return {
        "adapter": {
            "metrics": adapter_metrics,
            "conditions": adapter_conditions,
            "passed": all(adapter_conditions.values()),
        },
        "distribution": {
            "metrics": distribution_metrics,
            "conditions": distribution_conditions,
            "passed": all(distribution_conditions.values()),
        },
    }

def _promotion(adapter_pass: bool, distribution_pass: bool) -> str:
    if adapter_pass and distribution_pass:
        return "PROMOTE_DISTRIBUTION_LEVEL_RELATION_TO_FRESH_VALIDATION"
    if adapter_pass:
        return "PROMOTE_DISTRIBUTION_ALIGNED_EVALUATION_HYPOTHESIS"
    return "NO_PROMOTION"


def build_result(
    spec: Mapping[str, Any],
    parent: Mapping[str, Any],
    dataset: r0.HistoricalDataset,
    records: Sequence[Mapping[str, Any]],
    implementation_tests: Mapping[str, Any],
) -> Dict[str, Any]:
    gates = aggregate_gates(records)
    adapter_pass = bool(gates["adapter"]["passed"])
    distribution_pass = bool(gates["distribution"]["passed"])
    classification = "PASS" if adapter_pass else "SCIENTIFIC_FAIL"
    result = {
        "schema": "cb16.result.v2",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "classification": classification,
        "execution_status": "EXECUTED",
        "validity_status": "VALID",
        "scientific_outcomes": {
            "adapter_transfer": spec["outcomes"]["adapter_gate_pass" if adapter_pass else "adapter_gate_fail"],
            "distribution_level_relation": spec["outcomes"]["distribution_gate_pass" if distribution_pass else "distribution_gate_fail"],
        },
        "gate_results": [
            {
                "gate_id": "G_ADAPTER_SEED",
                "passed": gates["adapter"]["metrics"]["seed_pass_count"] >= 6,
                "metrics": {"seed_pass_count": gates["adapter"]["metrics"]["seed_pass_count"]},
            },
            {
                "gate_id": "G_ADAPTER_AGGREGATE",
                "passed": adapter_pass,
                "metrics": gates["adapter"]["metrics"],
                "conditions": gates["adapter"]["conditions"],
            },
            {
                "gate_id": "G_DISTRIBUTION_SEED",
                "passed": gates["distribution"]["metrics"]["seed_pass_count"] >= 6,
                "metrics": {"seed_pass_count": gates["distribution"]["metrics"]["seed_pass_count"]},
            },
            {
                "gate_id": "G_DISTRIBUTION_AGGREGATE",
                "passed": distribution_pass,
                "metrics": gates["distribution"]["metrics"],
                "conditions": gates["distribution"]["conditions"],
            },
        ],
        "claim_assessments": [
            {
                "claim_id": "C_ADAPTER_TRANSFER",
                "qualification_status": "QUALIFIED" if adapter_pass else "NOT_QUALIFIED",
                "inference_status": "SUPPORTED" if adapter_pass else "GATE_NOT_MET",
                "attribution_status": "PARTIAL" if adapter_pass else "UNRESOLVED",
            },
            {
                "claim_id": "C_DISTRIBUTION_LEVEL_RELATION",
                "qualification_status": "QUALIFIED" if distribution_pass else "NOT_QUALIFIED",
                "inference_status": "SUPPORTED" if distribution_pass else "GATE_NOT_MET",
                "attribution_status": "UNRESOLVED",
            },
        ],
        "prior_evidence_assessment": [
            {
                "claim_ref": "r12.vs_c.joint_confirmation.r3",
                "status": "UNAFFECTED_IN_ORIGINAL_SCOPE",
                "transfer": "TRANSFER_APPLICABILITY_UNRESOLVED",
            },
            {
                "claim_ref": "r12.vs_d.historical_market_information_canary.r0",
                "status": "UNAFFECTED_IN_ORIGINAL_SCOPE",
                "current_scoped_assessment": "TESTED_CONFIGURATION_DEV_DECISION_GATE_NOT_MET",
            },
        ],        "promotion_decision": {
            "decision": _promotion(adapter_pass, distribution_pass),
            "reason": "promotion is limited to the preregistered diagnostic scope on reused March development",
        },
        "provenance": {
            "commit_sha": q.commit_sha(),
            "spec_file_sha256": _file_sha256(CONFIG_PATH),
            "parent_spec_file_sha256": _file_sha256(PARENT_CONFIG_PATH),
            "parent_run_id": spec["parent_evidence"]["run_id"],
            "parent_artifact_sha256": spec["parent_evidence"]["artifact_sha256"],
            "validation_freshness": spec["claim_scope"]["validation_freshness"],
            "final_holdout_accessed": False,
        },
        "implementation_tests": dict(implementation_tests),
        "data_evidence": {
            **dict(dataset.evidence),
            "manifest_final_holdout_accessed_post_run": False,
        },
        "seeds": list(records),
        "aggregate_gates": gates,
    }
    try:
        validate_result_against_spec(result, spec)
    except EvidenceScopeError as exc:
        raise ContractError(f"R1 result violates claim authority: {exc}") from exc
    return result

def failure_result(
    spec: Mapping[str, Any] | None,
    classification: str,
    detail: str,
    implementation_tests: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    claim_ids = ["C_ADAPTER_TRANSFER", "C_DISTRIBUTION_LEVEL_RELATION"]
    result = {
        "schema": "cb16.result.v2",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "classification": classification,
        "execution_status": "BLOCKED" if classification != "PASS" else "EXECUTED",
        "validity_status": "INVALID_FOR_CLAIM",
        "gate_results": [],
        "claim_assessments": [
            {
                "claim_id": claim_id,
                "qualification_status": "NOT_APPLICABLE",
                "inference_status": "INVALID_FOR_CLAIM",
                "attribution_status": "NOT_APPLICABLE",
            }
            for claim_id in claim_ids
        ],
        "prior_evidence_assessment": [],
        "promotion_decision": {"decision": "NO_PROMOTION", "reason": "formal run did not reach valid scientific adjudication"},
        "provenance": {"commit_sha": q.commit_sha()},
        "error": {"type": classification, "detail": detail[:2000]},
        "seeds": [],
    }
    if implementation_tests is not None:
        result["implementation_tests"] = dict(implementation_tests)
    if spec is not None:
        try:
            validate_result_against_spec(result, spec)
        except EvidenceScopeError as exc:
            raise ContractError(f"failure result violates claim authority: {exc}") from exc
    return result

def render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# R12 VS-D Evaluation Adapter Transfer Diagnostic R1 — REPORT",
        "",
        f"- classification: **{result['classification']}**",
        f"- adapter outcome: **{result.get('scientific_outcomes', {}).get('adapter_transfer', 'NOT_EVALUATED')}**",
        f"- distribution outcome: **{result.get('scientific_outcomes', {}).get('distribution_level_relation', 'NOT_EVALUATED')}**",
        f"- promotion: **{result['promotion_decision']['decision']}**",
        "- March is reused development material, not fresh confirmation.",
        "",
    ]
    if not result.get("seeds"):
        lines += ["## Blocker", "", result.get("error", {}).get("detail", "unknown blocker"), ""]
        return "\n".join(lines)
    lines += [
        "## Per-seed diagnostic",
        "",
        "| seed | adapter seed | distribution seed | argmax pos-ctl | expected pos-ctl | adapter delta ctl | adapter delta PRE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in result["seeds"]:
        p = record["point_estimates"]
        lines.append(
            f"| {record['seed']} | {'PASS' if record['seed_gates']['adapter_transfer'] else 'FAIL'} | "
            f"{'PASS' if record['seed_gates']['distribution_level_relation'] else 'FAIL'} | "
            f"{p['argmax_positive_minus_control']:.9f} | {p['expected_positive_minus_control']:.9f} | "
            f"{p['adapter_delta_true_minus_control']:.9f} | {p['adapter_delta_true_minus_pre']:.9f} |"
        )
    lines += ["", "## Aggregate gates", ""]
    for gate_name in ("adapter", "distribution"):
        gate = result["aggregate_gates"][gate_name]
        lines.append(f"### {gate_name}")
        lines.append("")
        lines.append(f"- passed: **{gate['passed']}**")
        for key, value in gate["metrics"].items():
            lines.append(f"- {key}: `{value}`")
        for key, value in gate["conditions"].items():
            lines.append(f"- {key}: **{value}**")
        lines.append("")
    lines += [
        "## Authority boundary",
        "",
        "- This run changes only evaluation of the frozen parent-R0 policy distribution.",
        "- PASS does not qualify historical market information because March was already inspected.",
        "- FAIL does not falsify representation, objective, horizon, learner family, or full CB16.",
        "- Final holdout remains unopened.",
        "",
    ]
    return "\n".join(lines)


def write_artifacts(result_dir: Path, spec: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (result_dir / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (result_dir / REPORT_FILENAME).write_text(render_report(result), encoding="utf-8")

def run_qualification(result_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any], int]:
    spec = load_spec()
    parent = load_parent_spec()
    validate_r1_against_parent(spec, parent)
    implementation = q.run_implementation_test_suite()
    implementation_gate = q.implementation_test_gate(implementation, commit=q.commit_sha())
    if not implementation_gate["passed"]:
        classification, exit_code = q.implementation_failure_classification(implementation)
        detail = q.implementation_test_failure_detail(classification, implementation)
        evidence = {
            **dict(implementation),
            "gate": implementation_gate,
            "failure_classification": classification,
        }
        result = failure_result(spec, classification, detail, evidence)
        write_artifacts(result_dir, spec, result)
        return spec, result, exit_code

    dataset = r0.load_historical_dataset(parent)
    records = run_seed_set(spec, parent, dataset)
    r0._load_manifest(parent)
    result = build_result(spec, parent, dataset, records, implementation)
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
        if spec is not None:
            write_artifacts(result_dir, spec, result)
        return EXIT_EXECUTION_BLOCKED
    except ContractError as exc:
        try:
            spec = load_spec()
        except ContractError:
            spec = None
        result = failure_result(spec, "CONTRACT_MISMATCH", f"{type(exc).__name__}: {exc}")
        if spec is not None:
            write_artifacts(result_dir, spec, result)
        print(f"{RESULT_COMMAND}: CONTRACT_MISMATCH: {exc}", file=sys.stderr)
        return EXIT_CONTRACT_MISMATCH
    except Exception as exc:  # noqa: BLE001
        try:
            spec = load_spec()
        except ContractError:
            spec = None
        result = failure_result(spec, "EXECUTION_BLOCKED", f"{type(exc).__name__}: {exc}")
        if spec is not None:
            write_artifacts(result_dir, spec, result)
        print(f"{RESULT_COMMAND}: EXECUTION_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED
    print(
        f"{RESULT_COMMAND}: classification={result['classification']} "
        f"adapter={result['scientific_outcomes']['adapter_transfer']} "
        f"distribution={result['scientific_outcomes']['distribution_level_relation']}"
    )
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
