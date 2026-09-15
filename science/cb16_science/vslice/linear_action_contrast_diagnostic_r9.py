"""R12 VS-D R9 linear action-contrast diagnostic."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..evidence_scope import EvidenceScopeError, validate_experiment_spec, validate_result_against_spec
from ..normalization.transforms import n0_endpoint_log_ratios
from . import bounded_horizon_accessibility_screen_r5 as r5
from . import controlled_tasks as tasks
from . import historical_market_canary_r0 as r0
from . import qualification as q
from . import representation_accessibility_r3 as r3
from .contracts import ContractError

RESULT_COMMAND = "cb16.vs-d-linear-action-contrast-diagnostic-r9@v1"
EXPERIMENT_ID = "r12.vs_d.linear_action_contrast_diagnostic.r9"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config/experiments/r12_vs_d_linear_action_contrast_diagnostic_r9.json"
SPEC_FILE_SHA256 = "0da7b886a53fcab1b4f0c3b80dfe66d3f9705d550b80d2bdbbb478ecc678a2ae"
SPEC_FILENAME, RESULT_FILENAME, REPORT_FILENAME = "experiment_spec.json", "RESULT.json", "REPORT.md"
EXIT_PASS, EXIT_SCIENTIFIC_FAIL, EXIT_EXECUTION_BLOCKED, EXIT_CONTRACT_MISMATCH = 0, 1, 3, 4


@dataclass(frozen=True)
class ActionContrastDataset:
    fit_x: np.ndarray
    evaluation_x: np.ndarray
    fit_target: np.ndarray
    evaluation_target: np.ndarray
    evidence: Mapping[str, Any]


def load_spec() -> Dict[str, Any]:
    raw = CONFIG_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SPEC_FILE_SHA256:
        raise ContractError(f"R9 spec byte SHA mismatch: {digest}")
    try:
        return dict(tasks.spec_mapping(json.loads(raw), "R9 spec"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"R9 spec invalid JSON: {exc}") from exc


def validate_spec(spec: Mapping[str, Any]) -> None:
    try:
        validate_experiment_spec(spec)
    except EvidenceScopeError as exc:
        raise ContractError(f"R9 claim-authority spec invalid: {exc}") from exc
    if spec.get("experiment_id") != EXPERIMENT_ID or spec.get("experiment_role") != "DIAGNOSTIC":
        raise ContractError("R9 identity/role mismatch")
    data = tasks.spec_mapping(spec["data"], "data")
    names = [tasks.spec_str(x["name"], "archive.name") for x in data["allowed_archives"]]
    if names != ["BTCUSDT-1m-2020-01.zip", "BTCUSDT-1m-2020-02.zip"]:
        raise ContractError("R9 allowed archives must be Jan-Feb only")
    if int(data["hourly_aggregation"]["expected_hourly_bars"]) != 1440:
        raise ContractError("R9 hourly count mismatch")
    if int(data["fit"]["expected_decisions"]) != 615 or int(data["evaluation"]["expected_decisions"]) != 672:
        raise ContractError("R9 fit/evaluation counts mismatch")
    context = data["context"]
    if (int(context["represented_hours"]), int(context["raw_window_hours"]), int(context["feature_width"])) != (64, 65, 320):
        raise ContractError("R9 context mismatch")
    probe = tasks.spec_mapping(spec["probe"], "probe")
    if float(probe["ridge_lambda"]) != 1.0:
        raise ContractError("R9 ridge lambda must remain 1.0")
    if tuple(int(x) for x in probe["control_seeds"]) != tuple(range(23001, 23009)):
        raise ContractError("R9 control seeds mismatch")
    if probe["action_mapping"] != "numpy_sign_of_score; exact zero maps FLAT":
        raise ContractError("R9 action mapping mismatch")
    if probe["per_step_contrast"] != "direction_times_realized_arithmetic_return":
        raise ContractError("R9 contrast definition mismatch")
    uncertainty = tasks.spec_mapping(spec["uncertainty_protocol"], "uncertainty_protocol")
    frozen = (
        int(uncertainty["bootstrap_replicates"]),
        int(uncertainty["bootstrap_seed"]),
        int(uncertainty["block_length_hours"]),
        int(uncertainty["sample_length_hours"]),
        int(uncertainty["lower_bound_order_index_zero_based"]),
    )
    if frozen != (4096, 24001, 48, 672, 204):
        raise ContractError("R9 bootstrap contract mismatch")


def load_dataset(spec: Mapping[str, Any]) -> ActionContrastDataset:
    times, minutes, evidence = r0._read_allowed_minutes(spec)
    hourly_times, hourly = r0.aggregate_utc_hourly(times, minutes)
    if len(hourly_times) != 1440:
        raise ContractError("R9 expected 1440 Jan-Feb hourly bars")
    data = spec["data"]
    fit, evaluation = data["fit"], data["evaluation"]
    raw_window = int(data["context"]["raw_window_hours"])
    fit_idx = r0._consequence_indices(
        hourly_times,
        r0._parse_utc_ms(fit["consequence_hour_start_utc"], "fit.start"),
        r0._parse_utc_ms(fit["consequence_hour_end_utc"], "fit.end"),
        int(fit["expected_decisions"]),
    )
    evaluation_idx = r0._consequence_indices(
        hourly_times,
        r0._parse_utc_ms(evaluation["consequence_hour_start_utc"], "evaluation.start"),
        r0._parse_utc_ms(evaluation["consequence_hour_end_utc"], "evaluation.end"),
        int(evaluation["expected_decisions"]),
    )
    if np.intersect1d(fit_idx, evaluation_idx).size:
        raise ContractError("R9 fit/evaluation consequences overlap")
    fit_n0 = n0_endpoint_log_ratios(hourly[r0._window_indices(fit_idx, raw_window)]).values
    evaluation_n0 = n0_endpoint_log_ratios(hourly[r0._window_indices(evaluation_idx, raw_window)]).values
    if fit_n0.shape != (615, 64, 5) or evaluation_n0.shape != (672, 64, 5):
        raise ContractError("R9 N0 shapes mismatch")
    fit_x = np.asarray(fit_n0, dtype=np.float64).reshape(615, 320)
    evaluation_x = np.asarray(evaluation_n0, dtype=np.float64).reshape(672, 320)
    fit_target = r3._target(hourly[fit_idx, 0], hourly[fit_idx, 3])
    evaluation_target = r3._target(hourly[evaluation_idx, 0], hourly[evaluation_idx, 3])
    if not np.isfinite(fit_x).all() or not np.isfinite(evaluation_x).all():
        raise ContractError("R9 N0 surfaces must be finite")
    details = dict(evidence)
    details.update({
        "hourly_bars": 1440,
        "fit_decisions": 615,
        "evaluation_decisions": 672,
        "march_opened": False,
        "causal_window_relation": "64h_N0_strictly_before_one_hour_consequence",
    })
    return ActionContrastDataset(fit_x, evaluation_x, fit_target, evaluation_target, details)


def control_orders(length: int, seeds: Sequence[int]) -> Mapping[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    identity = np.arange(length)
    for seed in seeds:
        order = np.asarray(tasks.no_fixed_point_permutation(length, seed=int(seed), generation=0), dtype=np.int64)
        if order.shape != (length,) or np.any(order == identity):
            raise ContractError("R9 control permutation invalid")
        out[int(seed)] = order
    return out


def fit_scores(dataset: ActionContrastDataset, spec: Mapping[str, Any]) -> Dict[str, Any]:
    probe = spec["probe"]
    seeds = [int(x) for x in probe["control_seeds"]]
    orders = control_orders(len(dataset.fit_target), seeds)
    scaler = r3.fit_standardizer(dataset.fit_x)
    true_probe = r3.fit_ridge(dataset.fit_x, dataset.fit_target, ridge_lambda=1.0, standardizer=scaler)
    true_score = true_probe.predict(dataset.evaluation_x)
    controls = []
    for seed in seeds:
        shuffled = np.asarray(dataset.fit_target)[orders[seed]]
        if not np.array_equal(np.sort(shuffled), np.sort(dataset.fit_target)):
            raise ContractError("R9 shuffled control must preserve target multiset")
        controls.append(r3.fit_ridge(dataset.fit_x, shuffled, ridge_lambda=1.0, standardizer=scaler).predict(dataset.evaluation_x))
    return {
        "true_score": np.asarray(true_score, dtype=np.float64),
        "control_scores": tuple(np.asarray(x, dtype=np.float64) for x in controls),
        "active_feature_count": int(scaler.active.sum()),
        "input_feature_count": int(scaler.active.size),
    }


def direction_from_score(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ContractError("R9 score must be a finite 1D vector")
    return np.sign(values).astype(np.int8)


def per_step_contrast(score: np.ndarray, realized_return: np.ndarray) -> np.ndarray:
    direction = direction_from_score(score).astype(np.float64)
    target = np.asarray(realized_return, dtype=np.float64)
    if target.shape != direction.shape or not np.isfinite(target).all():
        raise ContractError("R9 realized return shape/finite mismatch")
    contrast = direction * target
    if not np.isfinite(contrast).all():
        raise ContractError("R9 action contrast must be finite")
    return contrast


def _lcb(values: Sequence[float], index: int) -> float:
    ordered = sorted(float(x) for x in values)
    if not 0 <= index < len(ordered):
        raise ContractError("R9 LCB index invalid")
    return ordered[index]


def evaluate_action_contrast(dataset: ActionContrastDataset, scores: Mapping[str, Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
    target = np.asarray(dataset.evaluation_target, dtype=np.float64)
    true_score = np.asarray(scores["true_score"], dtype=np.float64)
    control_scores = [np.asarray(x, dtype=np.float64) for x in scores["control_scores"]]
    true_contrast = per_step_contrast(true_score, target)
    control_contrasts = [per_step_contrast(score, target) for score in control_scores]
    true_mean = float(np.mean(true_contrast))
    control_means = [float(np.mean(x)) for x in control_contrasts]
    median_control = float(np.median(control_means))
    uncertainty = spec["uncertainty_protocol"]
    samples = r5.circular_block_samples(
        672,
        int(uncertainty["block_length_hours"]),
        int(uncertainty["bootstrap_replicates"]),
        int(uncertainty["bootstrap_seed"]),
    )
    true_draws = []
    delta_draws = []
    for idx in samples:
        true_block = float(np.mean(true_contrast[idx]))
        control_block = [float(np.mean(x[idx])) for x in control_contrasts]
        true_draws.append(true_block)
        delta_draws.append(true_block - float(np.median(control_block)))
    oi = int(uncertainty["lower_bound_order_index_zero_based"])
    true_lcb = _lcb(true_draws, oi)
    delta_lcb = _lcb(delta_draws, oi)
    direction = direction_from_score(true_score)
    return {
        "point": {
            "true_mean_unit_direction_return_contrast": true_mean,
            "median_shuffle_mean_unit_direction_return_contrast": median_control,
            "true_minus_median_shuffle_mean_direction_return_contrast": true_mean - median_control,
        },
        "bootstrap": {
            "true_mean_unit_direction_return_contrast_lcb": true_lcb,
            "true_minus_median_shuffle_mean_direction_return_contrast_lcb": delta_lcb,
        },
        "direction_counts": {
            "SHORT": int(np.sum(direction < 0)),
            "FLAT_EXACT_ZERO": int(np.sum(direction == 0)),
            "LONG": int(np.sum(direction > 0)),
        },
        "control_point_means": control_means,
        "active_feature_count": int(scores["active_feature_count"]),
        "input_feature_count": int(scores["input_feature_count"]),
    }


def run_experiment(spec: Mapping[str, Any], dataset: ActionContrastDataset, implementation: Mapping[str, Any]) -> Dict[str, Any]:
    validate_spec(spec)
    gate = q.implementation_test_gate(implementation, commit=q.commit_sha())
    if not gate["passed"]:
        raise ContractError("R9 exact-commit implementation test gate is not green")
    diagnostic = evaluate_action_contrast(dataset, fit_scores(dataset, spec), spec)
    boot = diagnostic["bootstrap"]
    supported = (
        boot["true_mean_unit_direction_return_contrast_lcb"] > 0.0
        and boot["true_minus_median_shuffle_mean_direction_return_contrast_lcb"] > 0.0
    )
    promotion = "PROMOTE_LINEAR_ACTION_CONTRAST_CANDIDATE_HYPOTHESIS" if supported else "NO_PROMOTION"
    result = {
        "schema": "cb16.result.v2",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(),
        "classification": "PASS" if supported else "SCIENTIFIC_FAIL",
        "execution_status": "EXECUTED",
        "validity_status": "VALID",
        "scientific_outcomes": {"linear_action_contrast_accessibility": "SUPPORTED" if supported else "NOT_QUALIFIED"},
        "implementation_tests": {**dict(implementation), "gate": gate, "required_by_global_gate": True, "runner_executed_suite": True},
        "preregistered_spec": {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "file_sha256": SPEC_FILE_SHA256, "declared_status": spec.get("status")},
        "data_evidence": {**dict(dataset.evidence), "manifest_final_holdout_accessed_post_run": False},
        "action_contrast": diagnostic,
        "gate_results": [{
            "gate_id": "G_LINEAR_ACTION_CONTRAST",
            "passed": supported,
            "metrics": {
                "true_mean_contrast_lcb": boot["true_mean_unit_direction_return_contrast_lcb"],
                "true_minus_median_shuffle_contrast_lcb": boot["true_minus_median_shuffle_mean_direction_return_contrast_lcb"],
            },
        }],
        "claim_assessments": [{
            "claim_id": "C_LINEAR_ACTION_CONTRAST_ACCESSIBILITY",
            "qualification_status": "QUALIFIED" if supported else "NOT_QUALIFIED",
            "inference_status": "SUPPORTED" if supported else "GATE_NOT_MET",
            "attribution_status": "UNRESOLVED",
        }],
        "prior_evidence_assessment": [
            {"claim_ref": "r12.vs_d.bounded_ridge_regularization_accessibility_screen.r8", "status": "UNAFFECTED_IN_ORIGINAL_SCOPE"},
            {"claim_ref": "r12.vs_d.march_temporal_robustness_of_64h_linear_hint.r7", "status": "UNAFFECTED_IN_ORIGINAL_SCOPE"},
        ],
        "promotion_decision": {
            "decision": promotion,
            "reason": "R9 is an adaptively reused, zero-cost unit-direction diagnostic; any positive result requires a separately preregistered decision/economic experiment",
        },
        "provenance": {
            "commit_sha": q.commit_sha(),
            "spec_file_sha256": SPEC_FILE_SHA256,
            "parent_r8_run_id": 34930296695,
            "parent_r8_artifact_digest": "sha256:95f48f74874734344f25b9cc284afcc29480d20c59b7ef82cf4bf829c35c8a46",
            "freshness": "DIAGNOSTIC_ADAPTIVELY_REUSED_JAN_FEB_AFTER_R8",
            "march_read": False,
            "final_holdout_accessed": False,
        },
    }
    try:
        validate_result_against_spec(result, spec)
    except EvidenceScopeError as exc:
        raise ContractError(f"R9 result violates claim authority: {exc}") from exc
    return result


def render_report(result: Mapping[str, Any]) -> str:
    a = result.get("action_contrast", {})
    p = a.get("point", {})
    b = a.get("bootstrap", {})
    d = a.get("direction_counts", {})
    lines = [
        "# R12 VS-D Linear Action Contrast Diagnostic R9 — REPORT",
        "",
        f"- classification: **{result['classification']}**",
        f"- outcome: **{result.get('scientific_outcomes', {}).get('linear_action_contrast_accessibility')}**",
        f"- promotion: **{result.get('promotion_decision', {}).get('decision', 'NO_PROMOTION')}**",
        "- role: **DIAGNOSTIC — adaptively reused February, not qualification of profitability**",
        "",
        "## Action contrast",
        "",
        f"- true mean unit-direction contrast: `{p.get('true_mean_unit_direction_return_contrast')}`",
        f"- median shuffled-control mean contrast: `{p.get('median_shuffle_mean_unit_direction_return_contrast')}`",
        f"- true minus median shuffle: `{p.get('true_minus_median_shuffle_mean_direction_return_contrast')}`",
        f"- true mean contrast LCB: `{b.get('true_mean_unit_direction_return_contrast_lcb')}`",
        f"- true minus shuffle contrast LCB: `{b.get('true_minus_median_shuffle_mean_direction_return_contrast_lcb')}`",
        f"- direction counts SHORT/FLAT/LONG: `{d.get('SHORT')}/{d.get('FLAT_EXACT_ZERO')}/{d.get('LONG')}`",
        "",
        "## Authority boundary",
        "",
        "- This is zero-cost unit-direction action contrast, not profitability or account economics.",
        "- February is adaptively reused and not confirmatory.",
        "- PASS may only promote a new decision-contrast candidate hypothesis.",
        "- FAIL does not exclude nonlinear, state-dependent, cost-aware, or account-aware action contrasts.",
        "- March and final holdout remain unopened.",
        "",
    ]
    return "\n".join(lines)


def failure_result(spec: Mapping[str, Any] | None, classification: str, detail: str, implementation: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    result = {
        "schema": "cb16.result.v2",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(),
        "classification": classification,
        "execution_status": "NOT_EXECUTED",
        "validity_status": "INVALID",
        "scientific_outcomes": {},
        "gate_results": [],
        "claim_assessments": [{
            "claim_id": "C_LINEAR_ACTION_CONTRAST_ACCESSIBILITY",
            "qualification_status": "NOT_APPLICABLE",
            "inference_status": "INVALID_FOR_CLAIM",
            "attribution_status": "NOT_APPLICABLE",
        }],
        "prior_evidence_assessment": [],
        "promotion_decision": {"decision": "NO_PROMOTION", "reason": "formal run did not reach valid R9 adjudication"},
        "provenance": {"commit_sha": q.commit_sha(), "spec_file_sha256": SPEC_FILE_SHA256},
        "error": {"type": classification, "detail": detail[:2000]},
    }
    if implementation is not None:
        result["implementation_tests"] = dict(implementation)
    if spec is not None:
        try:
            validate_result_against_spec(result, spec)
        except EvidenceScopeError as exc:
            raise ContractError(f"R9 failure result violates claim authority: {exc}") from exc
    return result


def write_artifacts(result_dir: Path, spec: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (result_dir / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (result_dir / REPORT_FILENAME).write_text(render_report(result))


def run_qualification(result_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    spec = load_spec()
    validate_spec(spec)
    implementation = q.run_implementation_test_suite()
    gate = q.implementation_test_gate(implementation, commit=q.commit_sha())
    if not gate["passed"]:
        classification, code = q.implementation_failure_classification(implementation)
        result = failure_result(spec, classification, q.implementation_test_failure_detail(classification, implementation), {**dict(implementation), "gate": gate})
        write_artifacts(result_dir, spec, result)
        return spec, result, code
    dataset = load_dataset(spec)
    result = run_experiment(spec, dataset, implementation)
    write_artifacts(result_dir, spec, result)
    return spec, result, EXIT_PASS if result["classification"] == "PASS" else EXIT_SCIENTIFIC_FAIL


def main() -> int:
    raw = (os.environ.get("CB16_RESULT_DIR") or "").strip()
    if not raw or not (os.environ.get("CB16_COMMIT_SHA") or "").strip():
        return EXIT_EXECUTION_BLOCKED
    result_dir = Path(raw)
    try:
        _spec, result, code = run_qualification(result_dir)
    except r0.HistoricalDataUnavailable as exc:
        try:
            spec = load_spec()
        except Exception:
            spec = None
        result = failure_result(spec, "EXECUTION_BLOCKED", f"{type(exc).__name__}: {exc}")
        write_artifacts(result_dir, spec or {}, result)
        return EXIT_EXECUTION_BLOCKED
    except ContractError as exc:
        try:
            spec = load_spec()
        except Exception:
            spec = None
        result = failure_result(spec, "CONTRACT_MISMATCH", f"{type(exc).__name__}: {exc}")
        write_artifacts(result_dir, spec or {}, result)
        return EXIT_CONTRACT_MISMATCH
    except Exception as exc:
        try:
            spec = load_spec()
        except Exception:
            spec = None
        result = failure_result(spec, "EXECUTION_BLOCKED", f"{type(exc).__name__}: {exc}")
        write_artifacts(result_dir, spec or {}, result)
        return EXIT_EXECUTION_BLOCKED
    print(f"{RESULT_COMMAND}: classification={result['classification']} promotion={result['promotion_decision']['decision']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
