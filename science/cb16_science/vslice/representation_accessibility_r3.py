"""R12 VS-D R3 representation-accessibility diagnostic.

This runner fits one frozen closed-form ridge probe family on January 2020 and
measures out-of-sample accessibility on February 2020 from two existing
surfaces: flattened N0 and FrozenSensory z=32. March and final holdout are never
read. The experiment is diagnostic and emits claim-local ``cb16.result.v2``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from ..evidence_scope import EvidenceScopeError, validate_experiment_spec, validate_result_against_spec
from ..normalization.transforms import n0_endpoint_log_ratios
from . import controlled_tasks as tasks
from . import historical_market_canary_r0 as r0
from . import qualification as q
from .contracts import ContractError
from .sensory import FrozenSensory

RESULT_COMMAND = "cb16.vs-d-representation-accessibility-r3@v1"
EXPERIMENT_ID = "r12.vs_d.representation_accessibility_diagnostic.r3"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_d_representation_accessibility_r3.json"
SPEC_FILE_SHA256 = "04a2151aa822c685abbf744c854cd0ed8a87df48b633367f2043e7ed217f4111"
SPEC_FILENAME = "experiment_spec.json"
RESULT_FILENAME = "RESULT.json"
REPORT_FILENAME = "REPORT.md"

EXIT_PASS = 0
EXIT_SCIENTIFIC_FAIL = 1
EXIT_EXECUTION_BLOCKED = 3
EXIT_CONTRACT_MISMATCH = 4
STD_FLOOR = 1e-12


class HistoricalDataUnavailable(RuntimeError):
    """Required frozen historical data is unavailable to the formal runner."""


@dataclass(frozen=True)
class ProbeDataset:
    fit_market: np.ndarray
    fit_target: np.ndarray
    validation_market: np.ndarray
    validation_target: np.ndarray
    validation_day_index: np.ndarray
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    active: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.mean.size:
            raise ContractError("R3 feature width mismatch")
        selected = array[:, self.active]
        out = (selected - self.mean[self.active]) / self.std[self.active]
        if not np.isfinite(out).all():
            raise ContractError("R3 standardized features must be finite")
        return out


@dataclass(frozen=True)
class RidgeProbe:
    standardizer: Standardizer
    target_mean: float
    weights: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        x = self.standardizer.transform(values)
        prediction = self.target_mean + x @ self.weights
        if not np.isfinite(prediction).all():
            raise ContractError("R3 probe predictions must be finite")
        return np.asarray(prediction, dtype=np.float64)


def load_spec() -> Dict[str, Any]:
    try:
        raw = CONFIG_PATH.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read R3 spec: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SPEC_FILE_SHA256:
        raise ContractError(f"R3 spec byte SHA mismatch: {digest}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"R3 spec is invalid JSON: {exc}") from exc
    return dict(tasks.spec_mapping(parsed, "R3 spec"))


def validate_spec(spec: Mapping[str, Any]) -> None:
    try:
        validate_experiment_spec(spec)
    except EvidenceScopeError as exc:
        raise ContractError(f"R3 claim-authority spec invalid: {exc}") from exc
    if spec.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("R3 experiment_id mismatch")
    scope = tasks.spec_mapping(spec["claim_scope"], "claim_scope")
    if not tasks.spec_bool(scope["march_read_forbidden"], "march_read_forbidden"):
        raise ContractError("R3 must forbid March reads")
    data = tasks.spec_mapping(spec["data"], "data")
    names = [tasks.spec_str(x["name"], "archive.name") for x in data["allowed_archives"]]
    if names != ["BTCUSDT-1m-2020-01.zip", "BTCUSDT-1m-2020-02.zip"]:
        raise ContractError("R3 allowed archives must be exactly Jan-Feb")

    hourly = tasks.spec_mapping(data["hourly_aggregation"], "hourly_aggregation")
    if tasks.spec_int(hourly["expected_hourly_bars"], "expected_hourly_bars") != 1440:
        raise ContractError("R3 Jan-Feb hourly count must be 1440")
    fit = tasks.spec_mapping(data["probe_fit"], "probe_fit")
    validation = tasks.spec_mapping(data["probe_validation"], "probe_validation")
    if tasks.spec_int(fit["expected_decisions"], "fit.expected_decisions") != 679:
        raise ContractError("R3 January fit count must be 679")
    if tasks.spec_int(validation["expected_decisions"], "validation.expected_decisions") != 696:
        raise ContractError("R3 February validation count must be 696")
    if tasks.spec_int(validation["expected_day_blocks"], "validation.expected_day_blocks") != 29:
        raise ContractError("R3 February validation must have 29 day blocks")

    probe = tasks.spec_mapping(spec["probe"], "probe")
    if tasks.spec_float(probe["ridge_lambda"], "ridge_lambda") != 1.0:
        raise ContractError("R3 freezes ridge lambda=1.0")
    controls = tasks.spec_mapping(spec["negative_controls"], "negative_controls")
    if tuple(int(x) for x in controls["control_seeds"]) != tuple(range(4401, 4409)):
        raise ContractError("R3 freezes control seeds 4401..4408")
    uncertainty = tasks.spec_mapping(spec["uncertainty_protocol"], "uncertainty_protocol")
    if tasks.spec_int(uncertainty["bootstrap_replicates"], "bootstrap_replicates") != 4096:
        raise ContractError("R3 freezes 4096 bootstrap replicates")
    if tasks.spec_int(uncertainty["lower_bound_order_index_zero_based"], "lcb_index") != 204:
        raise ContractError("R3 freezes bootstrap order index 204")


def _target(open_price: np.ndarray, close_price: np.ndarray) -> np.ndarray:
    open_array = np.asarray(open_price, dtype=np.float64)
    close_array = np.asarray(close_price, dtype=np.float64)
    if open_array.shape != close_array.shape or open_array.ndim != 1:
        raise ContractError("R3 consequence prices must be matching 1D arrays")
    if not np.isfinite(open_array).all() or not np.isfinite(close_array).all():
        raise ContractError("R3 consequence prices must be finite")
    if not bool((open_array > 0.0).all() and (close_array > 0.0).all()):
        raise ContractError("R3 consequence prices must be strictly positive")
    values = close_array / open_array - 1.0
    if not np.isfinite(values).all():
        raise ContractError("R3 target must be finite")
    return values


def load_probe_dataset(spec: Mapping[str, Any]) -> ProbeDataset:
    times, minutes, evidence = r0._read_allowed_minutes(spec)
    hourly_times, hourly = r0.aggregate_utc_hourly(times, minutes)
    if len(hourly_times) != 1440:
        raise ContractError(f"R3 expected 1440 Jan-Feb hourly bars, got {len(hourly_times)}")

    data = tasks.spec_mapping(spec["data"], "data")
    context = tasks.spec_mapping(data["context"], "data.context")
    fit = tasks.spec_mapping(data["probe_fit"], "data.probe_fit")
    validation = tasks.spec_mapping(data["probe_validation"], "data.probe_validation")
    raw_window_hours = tasks.spec_int(context["raw_window_hours"], "raw_window_hours")

    fit_idx = r0._consequence_indices(
        hourly_times,
        r0._parse_utc_ms(fit["consequence_hour_start_utc"], "probe_fit.start"),
        r0._parse_utc_ms(fit["consequence_hour_end_utc"], "probe_fit.end"),
        tasks.spec_int(fit["expected_decisions"], "probe_fit.expected_decisions"),
    )
    validation_idx = r0._consequence_indices(
        hourly_times,
        r0._parse_utc_ms(validation["consequence_hour_start_utc"], "probe_validation.start"),
        r0._parse_utc_ms(validation["consequence_hour_end_utc"], "probe_validation.end"),
        tasks.spec_int(validation["expected_decisions"], "probe_validation.expected_decisions"),
    )
    if np.intersect1d(fit_idx, validation_idx).size:
        raise ContractError("R3 January fit and February validation consequences overlap")
    fit_windows = hourly[r0._window_indices(fit_idx, raw_window_hours)]
    validation_windows = hourly[r0._window_indices(validation_idx, raw_window_hours)]
    fit_market = n0_endpoint_log_ratios(fit_windows).values
    validation_market = n0_endpoint_log_ratios(validation_windows).values
    if fit_market.shape != (679, 64, 5) or validation_market.shape != (696, 64, 5):
        raise ContractError("R3 normalized N0 shapes do not match frozen chronology")
    if not np.isfinite(fit_market).all() or not np.isfinite(validation_market).all():
        raise ContractError("R3 N0 contexts must be finite")

    validation_start = hourly_times[validation_idx][0]
    day_index = ((hourly_times[validation_idx] - validation_start) // (24 * r0.MILLISECONDS_PER_HOUR)).astype(np.int64)
    counts = np.bincount(day_index, minlength=29)
    if day_index.min() != 0 or day_index.max() != 28 or not np.array_equal(counts, np.full(29, 24)):
        raise ContractError("R3 February validation must contain 29 UTC blocks of 24 hours")

    fit_target = _target(hourly[fit_idx, 0], hourly[fit_idx, 3])
    validation_target = _target(hourly[validation_idx, 0], hourly[validation_idx, 3])
    details = dict(evidence)
    details.update({
        "hourly_bars": 1440,
        "probe_fit_decisions": int(fit_idx.size),
        "probe_validation_decisions": int(validation_idx.size),
        "validation_day_blocks": 29,
        "march_opened": False,
        "causal_window_relation": "context_hours_strictly_before_consequence_hour",
    })
    return ProbeDataset(
        fit_market=np.asarray(fit_market, dtype=np.float64),
        fit_target=fit_target,
        validation_market=np.asarray(validation_market, dtype=np.float64),
        validation_target=validation_target,
        validation_day_index=day_index,
        evidence=details,
    )


def representation_surfaces(dataset: ProbeDataset, spec: Mapping[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    n0_fit = dataset.fit_market.reshape(dataset.fit_market.shape[0], -1)
    n0_validation = dataset.validation_market.reshape(dataset.validation_market.shape[0], -1)
    representation = tasks.spec_mapping(spec["representation"], "representation")
    sensory_spec = tasks.spec_mapping(representation["frozen_sensory"], "frozen_sensory")
    sensory = FrozenSensory(
        z_dim=tasks.spec_int(sensory_spec["z_dim"], "z_dim", minimum=1),
        context_length=64,
        seed=tasks.spec_int(sensory_spec["seed"], "sensory_seed"),
        dtype=torch.float32,
    )
    if list(sensory.parameters()):
        raise ContractError("R3 FrozenSensory must expose zero trainable parameters")
    with torch.no_grad():
        z_fit = sensory(torch.from_numpy(dataset.fit_market)).detach().cpu().numpy().astype(np.float64)
        z_validation = sensory(torch.from_numpy(dataset.validation_market)).detach().cpu().numpy().astype(np.float64)
    if n0_fit.shape != (679, 320) or n0_validation.shape != (696, 320):
        raise ContractError("R3 N0 flattened width mismatch")
    if z_fit.shape != (679, 32) or z_validation.shape != (696, 32):
        raise ContractError("R3 FrozenSensory width mismatch")
    return {
        "N0_FLATTENED_64x5": (n0_fit, n0_validation),
        "FROZEN_SENSORY_Z32_SEED12012": (z_fit, z_validation),
    }


def fit_standardizer(values: np.ndarray) -> Standardizer:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ContractError("R3 fit features must be a finite 2D matrix")
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    active = std > STD_FLOOR
    if not bool(active.any()):
        raise ContractError("R3 probe has no non-constant training features")
    return Standardizer(mean=mean, std=std, active=active)


def fit_ridge(values: np.ndarray, target: np.ndarray, *, ridge_lambda: float, standardizer: Standardizer | None = None) -> RidgeProbe:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size:
        raise ContractError("R3 ridge fit shapes are incompatible")
    if not np.isfinite(y).all() or not math.isfinite(ridge_lambda) or ridge_lambda <= 0.0:
        raise ContractError("R3 ridge target/lambda contract failed")
    scaler = standardizer or fit_standardizer(x)
    xs = scaler.transform(x)
    target_mean = float(y.mean())
    yc = y - target_mean
    gram = xs.T @ xs + ridge_lambda * np.eye(xs.shape[1], dtype=np.float64)
    rhs = xs.T @ yc
    try:
        weights = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise ContractError(f"R3 ridge solve failed: {exc}") from exc
    if not np.isfinite(weights).all():
        raise ContractError("R3 ridge weights must be finite")
    return RidgeProbe(standardizer=scaler, target_mean=target_mean, weights=weights)


def pearson_corr(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ContractError("R3 correlation inputs must be matching 1D arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ContractError("R3 correlation inputs must be finite")
    ac = a - a.mean()
    bc = b - b.mean()
    denominator = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if not denominator > 0.0:
        raise ContractError("R3 correlation is undefined for zero-variance input")
    value = float(np.dot(ac, bc) / denominator)
    if not math.isfinite(value):
        raise ContractError("R3 correlation must be finite")
    return max(-1.0, min(1.0, value))


def _sign_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    mask = y != 0.0
    if not bool(mask.any()):
        raise ContractError("R3 sign accuracy has no nonzero targets")
    return float((np.sign(p[mask]) == np.sign(y[mask])).mean())


def point_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "pearson_corr": pearson_corr(prediction, target),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "sign_accuracy": _sign_accuracy(prediction, target),
    }


def fit_surface_predictions(
    fit_x: np.ndarray,
    validation_x: np.ndarray,
    fit_target: np.ndarray,
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    probe_spec = tasks.spec_mapping(spec["probe"], "probe")
    ridge_lambda = tasks.spec_float(probe_spec["ridge_lambda"], "ridge_lambda")
    controls = tasks.spec_mapping(spec["negative_controls"], "negative_controls")
    seeds = tuple(int(x) for x in controls["control_seeds"])
    generation = tasks.spec_int(controls["generation_argument"], "generation_argument")

    scaler = fit_standardizer(fit_x)
    true_probe = fit_ridge(fit_x, fit_target, ridge_lambda=ridge_lambda, standardizer=scaler)
    true_prediction = true_probe.predict(validation_x)
    control_predictions = []
    for seed in seeds:
        order = np.asarray(tasks.no_fixed_point_permutation(len(fit_target), seed=seed, generation=generation), dtype=np.int64)
        shuffled = np.asarray(fit_target, dtype=np.float64)[order]
        if not np.array_equal(np.sort(shuffled), np.sort(fit_target)):
            raise ContractError("R3 shuffled control must preserve the January target multiset")
        control_probe = fit_ridge(fit_x, shuffled, ridge_lambda=ridge_lambda, standardizer=scaler)
        control_predictions.append(control_probe.predict(validation_x))
    return {
        "true_prediction": true_prediction,
        "control_predictions": tuple(control_predictions),
        "active_feature_count": int(scaler.active.sum()),
        "input_feature_count": int(scaler.active.size),
    }


def bootstrap_indices(day_index: np.ndarray, *, replicates: int, seed: int) -> Tuple[np.ndarray, ...]:
    days = np.asarray(day_index, dtype=np.int64)
    if days.shape != (696,):
        raise ContractError("R3 bootstrap requires exactly 696 February observations")
    blocks = tuple(np.flatnonzero(days == day) for day in range(29))
    if any(block.shape != (24,) for block in blocks):
        raise ContractError("R3 bootstrap requires 29 day blocks of 24 observations")
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        picked = [rng.randrange(29) for _ in range(29)]
        samples.append(np.concatenate([blocks[day] for day in picked]))
    return tuple(samples)


def _lcb(values: Sequence[float], order_index: int) -> float:
    ordered = sorted(float(x) for x in values)
    if not 0 <= order_index < len(ordered):
        raise ContractError("R3 bootstrap order index is out of range")
    return ordered[order_index]


def bootstrap_surface(
    target: np.ndarray,
    true_prediction: np.ndarray,
    controls: Sequence[np.ndarray],
    samples: Sequence[np.ndarray],
    *,
    order_index: int,
) -> Dict[str, Any]:
    true_values = []
    delta_values = []
    for indices in samples:
        y = target[indices]
        true_corr = pearson_corr(true_prediction[indices], y)
        control_corrs = [pearson_corr(prediction[indices], y) for prediction in controls]
        true_values.append(true_corr)
        delta_values.append(true_corr - float(np.median(control_corrs)))
    return {
        "true_corr_lcb": _lcb(true_values, order_index),
        "true_minus_median_shuffle_corr_lcb": _lcb(delta_values, order_index),
    }


def bootstrap_gap(
    target: np.ndarray,
    n0_prediction: np.ndarray,
    z_prediction: np.ndarray,
    samples: Sequence[np.ndarray],
    *,
    order_index: int,
) -> float:
    values = []
    for indices in samples:
        y = target[indices]
        values.append(pearson_corr(n0_prediction[indices], y) - pearson_corr(z_prediction[indices], y))
    return _lcb(values, order_index)


def evaluate_surface(
    name: str,
    fit_x: np.ndarray,
    validation_x: np.ndarray,
    dataset: ProbeDataset,
    spec: Mapping[str, Any],
    samples: Sequence[np.ndarray],
    order_index: int,
) -> Dict[str, Any]:
    fitted = fit_surface_predictions(fit_x, validation_x, dataset.fit_target, spec)
    true_prediction = fitted["true_prediction"]
    controls = fitted["control_predictions"]
    control_metrics = [point_metrics(prediction, dataset.validation_target) for prediction in controls]
    point = point_metrics(true_prediction, dataset.validation_target)
    point["median_shuffle_pearson_corr"] = float(np.median([x["pearson_corr"] for x in control_metrics]))
    point["true_minus_median_shuffle_corr"] = point["pearson_corr"] - point["median_shuffle_pearson_corr"]
    bootstrap = bootstrap_surface(
        dataset.validation_target, true_prediction, controls, samples, order_index=order_index
    )
    return {
        "surface": name,
        "point": point,
        "bootstrap": bootstrap,
        "active_feature_count": fitted["active_feature_count"],
        "input_feature_count": fitted["input_feature_count"],
        "control_point_metrics": control_metrics,
        "true_prediction": true_prediction,
    }


def surface_gate(record: Mapping[str, Any]) -> Dict[str, Any]:
    bootstrap = tasks.spec_mapping(record["bootstrap"], "surface.bootstrap")
    conditions = {
        "true_corr_lcb_gt_0": float(bootstrap["true_corr_lcb"]) > 0.0,
        "true_minus_median_shuffle_corr_lcb_gt_0": float(bootstrap["true_minus_median_shuffle_corr_lcb"]) > 0.0,
    }
    return {"conditions": conditions, "passed": all(conditions.values())}


def runtime_record() -> Dict[str, Any]:
    record = q.runtime_record()
    record.update({"experiment_runtime":"R3_FIXED_RIDGE_CPU","historical_interval":"BTCUSDT_2020-01_2020-02"})
    return record


def _clean_surface(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if key != "true_prediction"}


def run_experiment(
    spec: Mapping[str, Any],
    dataset: ProbeDataset,
    implementation_tests: Mapping[str, Any],
) -> Dict[str, Any]:
    validate_spec(spec)
    test_gate = q.implementation_test_gate(implementation_tests, commit=q.commit_sha())
    if not test_gate["passed"]:
        raise ContractError("R3 exact-commit implementation test gate is not green")
    surfaces = representation_surfaces(dataset, spec)
    uncertainty = tasks.spec_mapping(spec["uncertainty_protocol"], "uncertainty_protocol")
    replicates = tasks.spec_int(uncertainty["bootstrap_replicates"], "bootstrap_replicates")
    bootstrap_seed = tasks.spec_int(uncertainty["bootstrap_seed"], "bootstrap_seed")
    order_index = tasks.spec_int(uncertainty["lower_bound_order_index_zero_based"], "lcb_index")
    samples = bootstrap_indices(dataset.validation_day_index, replicates=replicates, seed=bootstrap_seed)

    n0_fit, n0_validation = surfaces["N0_FLATTENED_64x5"]
    z_fit, z_validation = surfaces["FROZEN_SENSORY_Z32_SEED12012"]
    n0 = evaluate_surface(
        "N0_FLATTENED_64x5", n0_fit, n0_validation, dataset, spec, samples, order_index
    )
    z = evaluate_surface(
        "FROZEN_SENSORY_Z32_SEED12012", z_fit, z_validation, dataset, spec, samples, order_index
    )
    n0_gate = surface_gate(n0)
    z_gate = surface_gate(z)
    gap_lcb = bootstrap_gap(
        dataset.validation_target,
        n0["true_prediction"],
        z["true_prediction"],
        samples,
        order_index=order_index,
    )
    gap_conditions = {
        "C_N0_LINEAR_ACCESSIBILITY_qualified": n0_gate["passed"],
        "N0_minus_FrozenSensory_true_corr_lcb_gt_0": gap_lcb > 0.0,
    }
    gap_gate = {"conditions": gap_conditions, "passed": all(gap_conditions.values()), "n0_minus_z_true_corr_lcb": gap_lcb}

    if z_gate["passed"]:
        promotion = "PROMOTE_POST_SENSORY_DIAGNOSTIC_ROUTE"
    elif n0_gate["passed"] and gap_gate["passed"]:
        promotion = "PROMOTE_SENSORY_RETENTION_GAP_HYPOTHESIS"
    elif n0_gate["passed"]:
        promotion = "PROMOTE_N0_ACCESSIBILITY_DIAGNOSTIC_ROUTE"
    else:
        promotion = "NO_PROMOTION"
    any_qualified = n0_gate["passed"] or z_gate["passed"] or gap_gate["passed"]

    classification = "PASS" if any_qualified else "SCIENTIFIC_FAIL"
    result = {
        "schema": "cb16.result.v2",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(),
        "classification": classification,
        "execution_status": "EXECUTED",
        "validity_status": "VALID",
        "scientific_outcomes": {
            "n0_linear_accessibility": "QUALIFIED" if n0_gate["passed"] else "NOT_QUALIFIED",
            "frozen_sensory_linear_accessibility": "QUALIFIED" if z_gate["passed"] else "NOT_QUALIFIED",
            "sensory_retention_gap": "QUALIFIED" if gap_gate["passed"] else "NOT_QUALIFIED",
        },
        "runtime": runtime_record(),
        "implementation_tests": {
            **dict(implementation_tests),
            "gate": test_gate,
            "required_by_global_gate": True,
            "runner_executed_suite": True,
        },
        "preregistered_spec": {
            "path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "file_sha256": SPEC_FILE_SHA256,
            "declared_status": spec.get("status"),
        },

        "data_evidence": {
            **dict(dataset.evidence),
            "manifest_final_holdout_accessed_post_run": False,
        },
        "target_diagnostics": {
            "January_mean": float(dataset.fit_target.mean()),
            "January_std": float(dataset.fit_target.std(ddof=0)),
            "February_mean": float(dataset.validation_target.mean()),
            "February_std": float(dataset.validation_target.std(ddof=0)),
        },
        "surfaces": {
            "N0_FLATTENED_64x5": _clean_surface(n0),
            "FROZEN_SENSORY_Z32_SEED12012": _clean_surface(z),
        },
        "gate_results": [
            {
                "gate_id": "G_N0_ACCESSIBILITY",
                "passed": n0_gate["passed"],
                "conditions": n0_gate["conditions"],
                "metrics": dict(n0["bootstrap"]),
            },
            {
                "gate_id": "G_FROZEN_SENSORY_ACCESSIBILITY",
                "passed": z_gate["passed"],
                "conditions": z_gate["conditions"],
                "metrics": dict(z["bootstrap"]),
            },
            {
                "gate_id": "G_SENSORY_RETENTION_GAP",
                "passed": gap_gate["passed"],
                "conditions": gap_gate["conditions"],
                "metrics": {"n0_minus_z_true_corr_lcb": gap_lcb},
            },
        ],
        "claim_assessments": [
            {
                "claim_id": "C_N0_LINEAR_ACCESSIBILITY",
                "qualification_status": "QUALIFIED" if n0_gate["passed"] else "NOT_QUALIFIED",
                "inference_status": "SUPPORTED" if n0_gate["passed"] else "GATE_NOT_MET",
                "attribution_status": "UNRESOLVED",
            },
            {
                "claim_id": "C_FROZEN_SENSORY_LINEAR_ACCESSIBILITY",
                "qualification_status": "QUALIFIED" if z_gate["passed"] else "NOT_QUALIFIED",
                "inference_status": "SUPPORTED" if z_gate["passed"] else "GATE_NOT_MET",
                "attribution_status": "UNRESOLVED",
            },
            {
                "claim_id": "C_SENSORY_RETENTION_GAP",
                "qualification_status": "QUALIFIED" if gap_gate["passed"] else "NOT_QUALIFIED",
                "inference_status": "SUPPORTED" if gap_gate["passed"] else "GATE_NOT_MET",
                "attribution_status": "UNRESOLVED",
            },
        ],
        "prior_evidence_assessment": [
            {
                "claim_ref": "r12.vs_d.historical_market_information_canary.r0",
                "status": "UNAFFECTED_IN_ORIGINAL_SCOPE",
                "current_scoped_assessment": "TESTED_CONFIGURATION_DEV_DECISION_GATE_NOT_MET",
            },
            {
                "claim_ref": "r12.vs_d.evaluation_adapter_transfer_diagnostic.r1",
                "status": "UNAFFECTED_IN_ORIGINAL_SCOPE",
            },
            {
                "claim_ref": "r12.vs_d.objective_balance_entropy_gradient_diagnostic.r2a",
                "status": "UNAFFECTED_IN_ORIGINAL_SCOPE",
            },
        ],
        "promotion_decision": {
            "decision": promotion,
            "reason": "promotion is limited to the preregistered reused-development diagnostic route",
        },
        "provenance": {
            "commit_sha": q.commit_sha(),
            "spec_file_sha256": SPEC_FILE_SHA256,
            "parent_r0_run_id": 34911876483,
            "parent_r1_run_id": 34919793143,
            "parent_r2a_run_id": 34922256352,
            "parent_r2a_artifact_digest": "sha256:9e899032c2e5e3639eaab897d1889edc3b46267454448a1bf5003a652c437e09",
            "freshness": "DIAGNOSTIC_REUSED_JAN_FEB_NOT_CONFIRMATION",
            "march_read": False,
            "final_holdout_accessed": False,
        },
    }
    try:
        validate_result_against_spec(result, spec)
    except EvidenceScopeError as exc:
        raise ContractError(f"R3 result violates claim authority: {exc}") from exc
    return result


def render_report(result: Mapping[str, Any]) -> str:
    outcomes = result["scientific_outcomes"]
    lines = [
        "# R12 VS-D Representation Accessibility Diagnostic R3 — REPORT",
        "",
        f"- classification: **{result['classification']}**",
        f"- N0 accessibility: **{outcomes['n0_linear_accessibility']}**",
        f"- FrozenSensory accessibility: **{outcomes['frozen_sensory_linear_accessibility']}**",
        f"- sensory retention gap: **{outcomes['sensory_retention_gap']}**",
        f"- promotion: **{result['promotion_decision']['decision']}**",
        "- data: **January fit / February reused-development validation; March not read**",
        "",
        "## Surface diagnostics",
        "",
        "| surface | corr | shuffle median | delta | corr LCB | delta LCB | active/features |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, record in result["surfaces"].items():
        point = record["point"]
        bootstrap = record["bootstrap"]
        lines.append(
            f"| {name} | {point['pearson_corr']:.9f} | "
            f"{point['median_shuffle_pearson_corr']:.9f} | "
            f"{point['true_minus_median_shuffle_corr']:.9f} | "
            f"{bootstrap['true_corr_lcb']:.9f} | "
            f"{bootstrap['true_minus_median_shuffle_corr_lcb']:.9f} | "
            f"{record['active_feature_count']}/{record['input_feature_count']} |"
        )
    gap = next(x for x in result["gate_results"] if x["gate_id"] == "G_SENSORY_RETENTION_GAP")
    lines += [
        "",
        "## Retention-gap diagnostic",
        "",
        f"- paired N0 - FrozenSensory correlation LCB: `{gap['metrics']['n0_minus_z_true_corr_lcb']:.12g}`",
        f"- gate: **{gap['passed']}**",
        "",
        "## Authority boundary",
        "",
        "- This is reused Jan-Feb development, not fresh confirmation.",
        "- A surface PASS establishes only fixed-probe accessibility in this scope.",
        "- A retention-gap PASS does not uniquely attribute loss to random projection or tanh.",
        "- A gate miss does not imply market information is absent or nonlinearly inaccessible.",
        "- March and final holdout remain unopened.",
        "",
    ]
    return "\n".join(lines)


def write_artifacts(result_dir: Path, spec: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (result_dir / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (result_dir / REPORT_FILENAME).write_text(render_report(result))


def failure_result(
    spec: Mapping[str, Any] | None,
    classification: str,
    detail: str,
    implementation_tests: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    claim_ids = (
        [str(x["claim_id"]) for x in spec["claims"]]
        if spec is not None
        else [
            "C_N0_LINEAR_ACCESSIBILITY",
            "C_FROZEN_SENSORY_LINEAR_ACCESSIBILITY",
            "C_SENSORY_RETENTION_GAP",
        ]
    )
    result: Dict[str, Any] = {
        "schema": "cb16.result.v2",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(),
        "classification": classification,
        "execution_status": "NOT_EXECUTED",
        "validity_status": "INVALID",
        "scientific_outcomes": {},
        "runtime": runtime_record(),
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
        "promotion_decision": {
            "decision": "NO_PROMOTION",
            "reason": "formal run did not reach valid scientific adjudication",
        },
        "provenance": {"commit_sha": q.commit_sha(), "spec_file_sha256": SPEC_FILE_SHA256},
        "error": {"type": classification, "detail": detail[:2000]},
    }
    if implementation_tests is not None:
        result["implementation_tests"] = dict(implementation_tests)
    if spec is not None:
        try:
            validate_result_against_spec(result, spec)
        except EvidenceScopeError as exc:
            raise ContractError(f"R3 failure result violates claim authority: {exc}") from exc
    return result


def write_failure(result_dir: Path, spec: Mapping[str, Any] | None, result: Mapping[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    if spec is not None:
        (result_dir / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (result_dir / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (result_dir / REPORT_FILENAME).write_text(
        "# R12 VS-D Representation Accessibility Diagnostic R3 — REPORT\n\n"
        f"- classification: **{result['classification']}**\n"
        "- scientific outcome: **NOT_EVALUATED**\n\n"
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
        evidence = {
            **dict(implementation),
            "gate": implementation_gate,
            "failure_classification": classification,
        }
        result = failure_result(spec, classification, detail, evidence)
        write_failure(result_dir, spec, result)
        return spec, result, exit_code
    dataset = load_probe_dataset(spec)
    result = run_experiment(spec, dataset, implementation)
    write_artifacts(result_dir, spec, result)
    exit_code = EXIT_PASS if result["classification"] == "PASS" else EXIT_SCIENTIFIC_FAIL
    return spec, result, exit_code


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
        print(f"{RESULT_COMMAND}: EXECUTION_BLOCKED: {exc}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED
    except ContractError as exc:
        try:
            spec = load_spec()
        except ContractError:
            spec = None
        result = failure_result(spec, "CONTRACT_MISMATCH", f"{type(exc).__name__}: {exc}")
        write_failure(result_dir, spec, result)
        print(f"{RESULT_COMMAND}: CONTRACT_MISMATCH: {exc}", file=sys.stderr)
        return EXIT_CONTRACT_MISMATCH
    except Exception as exc:  # noqa: BLE001
        try:
            spec = load_spec()
        except ContractError:
            spec = None
        result = failure_result(spec, "EXECUTION_BLOCKED", f"{type(exc).__name__}: {exc}")
        write_failure(result_dir, spec, result)
        print(f"{RESULT_COMMAND}: EXECUTION_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_EXECUTION_BLOCKED
    print(
        f"{RESULT_COMMAND}: classification={result['classification']} "
        f"n0={result['scientific_outcomes']['n0_linear_accessibility']} "
        f"z={result['scientific_outcomes']['frozen_sensory_linear_accessibility']} "
        f"gap={result['scientific_outcomes']['sensory_retention_gap']}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
