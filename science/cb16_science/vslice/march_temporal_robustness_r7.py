"""R12 VS-D R7 March temporal robustness diagnostic for the 64h linear hint."""
from __future__ import annotations

import hashlib, json, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ..evidence_scope import EvidenceScopeError, validate_experiment_spec, validate_result_against_spec
from ..normalization.transforms import n0_endpoint_log_ratios
from . import controlled_tasks as tasks
from . import historical_market_canary_r0 as r0
from . import qualification as q
from . import representation_accessibility_r3 as r3
from .contracts import ContractError

RESULT_COMMAND = "cb16.vs-d-march-temporal-robustness-r7@v1"
EXPERIMENT_ID = "r12.vs_d.march_temporal_robustness_of_64h_linear_hint.r7"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config/experiments/r12_vs_d_march_temporal_robustness_r7.json"
SPEC_FILE_SHA256 = "31862bd6a50fcaa0ab9dc4379073d10e5ec55245a928fea7481d3e62bc4c8416"
SPEC_FILENAME, RESULT_FILENAME, REPORT_FILENAME = "experiment_spec.json", "RESULT.json", "REPORT.md"
EXIT_PASS, EXIT_SCIENTIFIC_FAIL, EXIT_EXECUTION_BLOCKED, EXIT_CONTRACT_MISMATCH = 0, 1, 3, 4


@dataclass(frozen=True)
class RobustnessDataset:
    fit_x: np.ndarray
    march_x: np.ndarray
    fit_target: np.ndarray
    march_target: np.ndarray
    march_day_index: np.ndarray
    evidence: Mapping[str, Any]


def load_spec() -> Dict[str, Any]:
    raw = CONFIG_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SPEC_FILE_SHA256:
        raise ContractError(f"R7 spec byte SHA mismatch: {digest}")
    try:
        return dict(tasks.spec_mapping(json.loads(raw), "R7 spec"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"R7 spec invalid JSON: {exc}") from exc


def validate_spec(spec: Mapping[str, Any]) -> None:
    try:
        validate_experiment_spec(spec)
    except EvidenceScopeError as exc:
        raise ContractError(f"R7 claim-authority spec invalid: {exc}") from exc
    if spec.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("R7 experiment_id mismatch")
    if spec.get("experiment_role") != "ROBUSTNESS":
        raise ContractError("R7 must remain ROBUSTNESS")
    data = tasks.spec_mapping(spec["data"], "data")
    names = [tasks.spec_str(x["name"], "archive.name") for x in data["allowed_archives"]]
    if names != ["BTCUSDT-1m-2020-01.zip", "BTCUSDT-1m-2020-02.zip", "BTCUSDT-1m-2020-03.zip"]:
        raise ContractError("R7 allowed archives must be exactly Jan-Mar")
    if int(data["hourly_aggregation"]["expected_hourly_bars"]) != 2184:
        raise ContractError("R7 hourly count mismatch")
    context = tasks.spec_mapping(data["context"], "context")
    if (int(context["represented_hours"]), int(context["raw_window_hours"])) != (64, 65):
        raise ContractError("R7 context must remain N0 64h from a 65h raw window")
    fit = tasks.spec_mapping(data["fit"], "fit")
    evaluation = tasks.spec_mapping(data["evaluation"], "evaluation")
    if int(fit["expected_decisions"]) != 615 or int(evaluation["expected_decisions"]) != 744:
        raise ContractError("R7 fit/evaluation counts mismatch")
    if int(evaluation["expected_utc_day_blocks"]) != 31 or int(evaluation["decisions_per_day"]) != 24:
        raise ContractError("R7 March day-block contract mismatch")
    if not bool(evaluation["target_fitting_forbidden"]):
        raise ContractError("R7 must forbid target fitting on March")
    probe = tasks.spec_mapping(spec["probe"], "probe")
    if float(probe["ridge_lambda"]) != 1.0:
        raise ContractError("R7 ridge lambda mismatch")
    if tuple(int(x) for x in probe["control_seeds"]) != tuple(range(18001, 18009)):
        raise ContractError("R7 control seeds mismatch")
    u = tasks.spec_mapping(spec["uncertainty_protocol"], "uncertainty_protocol")
    if (int(u["bootstrap_replicates"]), int(u["bootstrap_seed"]), int(u["day_blocks"]), int(u["hours_per_day"]), int(u["lower_bound_order_index_zero_based"])) != (4096, 20001, 31, 24, 204):
        raise ContractError("R7 bootstrap contract mismatch")


def load_dataset(spec: Mapping[str, Any]) -> RobustnessDataset:
    times, minutes, evidence = r0._read_allowed_minutes(spec)
    hourly_times, hourly = r0.aggregate_utc_hourly(times, minutes)
    if len(hourly_times) != 2184:
        raise ContractError(f"R7 expected 2184 Jan-Mar hourly bars, got {len(hourly_times)}")
    data = spec["data"]
    fit = data["fit"]
    evaluation = data["evaluation"]
    raw_window = int(data["context"]["raw_window_hours"])
    fit_idx = r0._consequence_indices(
        hourly_times,
        r0._parse_utc_ms(fit["consequence_hour_start_utc"], "fit.start"),
        r0._parse_utc_ms(fit["consequence_hour_end_utc"], "fit.end"),
        int(fit["expected_decisions"]),
    )
    march_idx = r0._consequence_indices(
        hourly_times,
        r0._parse_utc_ms(evaluation["consequence_hour_start_utc"], "evaluation.start"),
        r0._parse_utc_ms(evaluation["consequence_hour_end_utc"], "evaluation.end"),
        int(evaluation["expected_decisions"]),
    )
    if np.intersect1d(fit_idx, march_idx).size:
        raise ContractError("R7 January fit and March evaluation consequences overlap")
    fit_n0 = n0_endpoint_log_ratios(hourly[r0._window_indices(fit_idx, raw_window)]).values
    march_n0 = n0_endpoint_log_ratios(hourly[r0._window_indices(march_idx, raw_window)]).values
    if fit_n0.shape != (615, 64, 5) or march_n0.shape != (744, 64, 5):
        raise ContractError("R7 N0 shapes do not match frozen chronology")
    fit_x = np.asarray(fit_n0, dtype=np.float64).reshape(615, 320)
    march_x = np.asarray(march_n0, dtype=np.float64).reshape(744, 320)
    if not np.isfinite(fit_x).all() or not np.isfinite(march_x).all():
        raise ContractError("R7 N0 surfaces must be finite")
    fit_target = r3._target(hourly[fit_idx, 0], hourly[fit_idx, 3])
    march_target = r3._target(hourly[march_idx, 0], hourly[march_idx, 3])
    march_start = hourly_times[march_idx][0]
    day_index = ((hourly_times[march_idx] - march_start) // (24 * r0.MILLISECONDS_PER_HOUR)).astype(np.int64)
    counts = np.bincount(day_index, minlength=31)
    if day_index.min() != 0 or day_index.max() != 30 or not np.array_equal(counts, np.full(31, 24)):
        raise ContractError("R7 March evaluation must contain 31 UTC blocks of 24 hours")
    details = dict(evidence)
    details.update({
        "hourly_bars": 2184,
        "fit_decisions": 615,
        "march_evaluation_decisions": 744,
        "march_day_blocks": 31,
        "march_opened": True,
        "march_freshness": "ADAPTIVELY_REUSED_NOT_CONFIRMATION",
        "causal_window_relation": "each_64h_N0_state_strictly_before_one_hour_consequence",
        "february_role": "causal_context_carryover_only_for_early_March_states",
    })
    return RobustnessDataset(fit_x, march_x, fit_target, march_target, day_index, details)


def control_orders(length: int, seeds: Sequence[int]) -> Mapping[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for seed in seeds:
        order = np.asarray(tasks.no_fixed_point_permutation(length, seed=int(seed), generation=0), dtype=np.int64)
        if order.shape != (length,) or np.any(order == np.arange(length)):
            raise ContractError("R7 control permutation must be a no-fixed-point permutation")
        out[int(seed)] = order
    return out


def fit_predictions(dataset: RobustnessDataset, spec: Mapping[str, Any]) -> Dict[str, Any]:
    probe = spec["probe"]
    ridge = float(probe["ridge_lambda"])
    seeds = [int(x) for x in probe["control_seeds"]]
    scaler = r3.fit_standardizer(dataset.fit_x)
    true_probe = r3.fit_ridge(dataset.fit_x, dataset.fit_target, ridge_lambda=ridge, standardizer=scaler)
    true_prediction = true_probe.predict(dataset.march_x)
    orders = control_orders(len(dataset.fit_target), seeds)
    controls = []
    for seed in seeds:
        shuffled = np.asarray(dataset.fit_target)[orders[seed]]
        if not np.array_equal(np.sort(shuffled), np.sort(dataset.fit_target)):
            raise ContractError("R7 shuffled target must preserve January target multiset")
        control_probe = r3.fit_ridge(dataset.fit_x, shuffled, ridge_lambda=ridge, standardizer=scaler)
        controls.append(control_probe.predict(dataset.march_x))
    return {
        "true_prediction": true_prediction,
        "control_predictions": tuple(controls),
        "active_feature_count": int(scaler.active.sum()),
        "input_feature_count": int(scaler.active.size),
    }


def march_bootstrap_indices(day_index: np.ndarray, *, replicates: int, seed: int) -> Tuple[np.ndarray, ...]:
    days = np.asarray(day_index, dtype=np.int64)
    if days.shape != (744,):
        raise ContractError("R7 bootstrap requires exactly 744 March observations")
    blocks = tuple(np.flatnonzero(days == day) for day in range(31))
    if any(block.shape != (24,) for block in blocks):
        raise ContractError("R7 bootstrap requires 31 day blocks of 24 observations")
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        picked = [rng.randrange(31) for _ in range(31)]
        samples.append(np.concatenate([blocks[day] for day in picked]))
    return tuple(samples)


def _lcb(values: Sequence[float], index: int) -> float:
    ordered = sorted(float(v) for v in values)
    if not 0 <= index < len(ordered):
        raise ContractError("R7 LCB index invalid")
    return ordered[index]


def evaluate_robustness(dataset: RobustnessDataset, predictions: Mapping[str, Any], spec: Mapping[str, Any]) -> Dict[str, Any]:
    true = np.asarray(predictions["true_prediction"])
    controls = [np.asarray(x) for x in predictions["control_predictions"]]
    point_true = r3.pearson_corr(true, dataset.march_target)
    control_corrs = [r3.pearson_corr(x, dataset.march_target) for x in controls]
    point_control = float(np.median(control_corrs))
    u = spec["uncertainty_protocol"]
    samples = march_bootstrap_indices(dataset.march_day_index, replicates=int(u["bootstrap_replicates"]), seed=int(u["bootstrap_seed"]))
    true_bs, delta_bs = [], []
    for idx in samples:
        y = dataset.march_target[idx]
        tb = r3.pearson_corr(true[idx], y)
        cb = [r3.pearson_corr(x[idx], y) for x in controls]
        true_bs.append(tb)
        delta_bs.append(tb - float(np.median(cb)))
    order_index = int(u["lower_bound_order_index_zero_based"])
    return {
        "point": {
            "true_corr": point_true,
            "median_shuffle_corr": point_control,
            "true_minus_median_shuffle_corr": point_true - point_control,
        },
        "bootstrap": {
            "true_corr_lcb": _lcb(true_bs, order_index),
            "true_minus_median_shuffle_corr_lcb": _lcb(delta_bs, order_index),
        },
        "control_point_corrs": control_corrs,
        "active_feature_count": int(predictions["active_feature_count"]),
        "input_feature_count": int(predictions["input_feature_count"]),
    }


def run_experiment(spec: Mapping[str, Any], dataset: RobustnessDataset, implementation: Mapping[str, Any]) -> Dict[str, Any]:
    validate_spec(spec)
    test_gate = q.implementation_test_gate(implementation, commit=q.commit_sha())
    if not test_gate["passed"]:
        raise ContractError("R7 exact-commit implementation test gate is not green")
    evaluated = evaluate_robustness(dataset, fit_predictions(dataset, spec), spec)
    b = evaluated["bootstrap"]
    supported = b["true_corr_lcb"] > 0 and b["true_minus_median_shuffle_corr_lcb"] > 0
    promotion = "PROMOTE_TEMPORAL_ROBUSTNESS_CANDIDATE_HYPOTHESIS" if supported else "NO_PROMOTION"
    result = {
        "schema": "cb16.result.v2",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(),
        "classification": "PASS" if supported else "SCIENTIFIC_FAIL",
        "execution_status": "EXECUTED",
        "validity_status": "VALID",
        "scientific_outcomes": {"march_temporal_robustness_of_64h_linear_hint": "SUPPORTED" if supported else "NOT_SUPPORTED"},
        "implementation_tests": {**dict(implementation), "gate": test_gate, "required_by_global_gate": True, "runner_executed_suite": True},
        "preregistered_spec": {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "file_sha256": SPEC_FILE_SHA256, "declared_status": spec.get("status")},
        "data_evidence": {**dict(dataset.evidence), "manifest_final_holdout_accessed_post_run": False},
        "robustness": evaluated,
        "gate_results": [{"gate_id": "G_MARCH_ROBUSTNESS", "passed": supported, "metrics": dict(evaluated["bootstrap"])}],
        "claim_assessments": [{"claim_id": "C_MARCH_TEMPORAL_ROBUSTNESS_OF_64H_LINEAR_HINT", "qualification_status": "NOT_APPLICABLE", "inference_status": "SUPPORTED" if supported else "GATE_NOT_MET", "attribution_status": "UNRESOLVED"}],
        "prior_evidence_assessment": [
            {"claim_ref": "r12.vs_d.bounded_context_length_accessibility_screen.r6", "status": "UNAFFECTED_IN_ORIGINAL_SCOPE"},
            {"claim_ref": "r12.vs_d.historical_market_information_canary.r0", "status": "UNAFFECTED_IN_ORIGINAL_SCOPE"},
        ],
        "promotion_decision": {"decision": promotion, "reason": "R7 is adaptively reused March robustness evidence, not confirmation or qualification"},
        "provenance": {
            "commit_sha": q.commit_sha(),
            "spec_file_sha256": SPEC_FILE_SHA256,
            "parent_r6_run_id": 34927748495,
            "parent_r6_artifact_digest": "sha256:fc47019f45edaa2745bb7a62ff91685be4c85a888ceaf7cec97e5bedbd4d592f",
            "freshness": "ROBUSTNESS_ADAPTIVELY_REUSED_MARCH_AFTER_R0_R1_AND_R3_R6",
            "march_read": True,
            "march_fresh_confirmation": False,
            "final_holdout_accessed": False,
        },
    }
    try:
        validate_result_against_spec(result, spec)
    except EvidenceScopeError as exc:
        raise ContractError(f"R7 result violates claim authority: {exc}") from exc
    return result


def render_report(result: Mapping[str, Any]) -> str:
    r = result.get("robustness", {})
    p, b = r.get("point", {}), r.get("bootstrap", {})
    return "\n".join([
        "# R12 VS-D March Temporal Robustness R7 — REPORT",
        "",
        f"- classification: **{result['classification']}**",
        f"- robustness: **{result.get('scientific_outcomes',{}).get('march_temporal_robustness_of_64h_linear_hint','NOT_EVALUATED')}**",
        f"- promotion: **{result.get('promotion_decision',{}).get('decision','NO_PROMOTION')}**",
        "- role: **ROBUSTNESS — adaptively reused March, not confirmation**",
        "",
        "## March metrics",
        "",
        f"- true_corr: `{p.get('true_corr')}`",
        f"- median_shuffle_corr: `{p.get('median_shuffle_corr')}`",
        f"- true_minus_median_shuffle_corr: `{p.get('true_minus_median_shuffle_corr')}`",
        f"- true_corr_lcb: `{b.get('true_corr_lcb')}`",
        f"- true_minus_median_shuffle_corr_lcb: `{b.get('true_minus_median_shuffle_corr_lcb')}`",
        "",
        "## Authority boundary",
        "",
        "- March was already inspected in R0/R1 and is ADAPTIVELY_REUSED, not fresh confirmation.",
        "- PASS may only promote a temporal-robustness candidate hypothesis for a new experiment.",
        "- FAIL does not prove regime shift, absence of market information, or a unique root cause.",
        "- Final holdout remains unopened.",
        "",
    ])


def failure_result(spec: Mapping[str, Any] | None, classification: str, detail: str, implementation: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    result = {
        "schema": "cb16.result.v2", "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND), "commit_sha": q.commit_sha(),
        "classification": classification, "execution_status": "NOT_EXECUTED", "validity_status": "INVALID",
        "scientific_outcomes": {}, "gate_results": [],
        "claim_assessments": [{"claim_id": "C_MARCH_TEMPORAL_ROBUSTNESS_OF_64H_LINEAR_HINT", "qualification_status": "NOT_APPLICABLE", "inference_status": "INVALID_FOR_CLAIM", "attribution_status": "NOT_APPLICABLE"}],
        "prior_evidence_assessment": [],
        "promotion_decision": {"decision": "NO_PROMOTION", "reason": "formal run did not reach valid robustness adjudication"},
        "provenance": {"commit_sha": q.commit_sha(), "spec_file_sha256": SPEC_FILE_SHA256},
        "error": {"type": classification, "detail": detail[:2000]},
    }
    if implementation is not None:
        result["implementation_tests"] = dict(implementation)
    if spec is not None:
        try:
            validate_result_against_spec(result, spec)
        except EvidenceScopeError as exc:
            raise ContractError(f"R7 failure result violates claim authority: {exc}") from exc
    return result


def write_artifacts(result_dir: Path, spec: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (result_dir / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (result_dir / REPORT_FILENAME).write_text(render_report(result))


def run_qualification(result_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    spec = load_spec(); validate_spec(spec)
    implementation = q.run_implementation_test_suite(); gate = q.implementation_test_gate(implementation, commit=q.commit_sha())
    if not gate["passed"]:
        classification, code = q.implementation_failure_classification(implementation)
        result = failure_result(spec, classification, q.implementation_test_failure_detail(classification, implementation), {**dict(implementation), "gate": gate})
        write_artifacts(result_dir, spec, result); return spec, result, code
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
        try: spec = load_spec()
        except Exception: spec = None
        result = failure_result(spec, "EXECUTION_BLOCKED", f"{type(exc).__name__}: {exc}"); write_artifacts(result_dir, spec or {}, result); return EXIT_EXECUTION_BLOCKED
    except ContractError as exc:
        try: spec = load_spec()
        except Exception: spec = None
        result = failure_result(spec, "CONTRACT_MISMATCH", f"{type(exc).__name__}: {exc}"); write_artifacts(result_dir, spec or {}, result); return EXIT_CONTRACT_MISMATCH
    except Exception as exc:
        try: spec = load_spec()
        except Exception: spec = None
        result = failure_result(spec, "EXECUTION_BLOCKED", f"{type(exc).__name__}: {exc}"); write_artifacts(result_dir, spec or {}, result); return EXIT_EXECUTION_BLOCKED
    print(f"{RESULT_COMMAND}: classification={result['classification']} promotion={result['promotion_decision']['decision']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
