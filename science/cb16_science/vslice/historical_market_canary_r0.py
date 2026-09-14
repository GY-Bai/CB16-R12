"""R12 VS-D historical market-information canary R0.

Formal runner for the preregistered BTCUSDT 2020-01..03 development slice.
The experiment reads only the three frozen archives, aggregates them to UTC 1h,
trains positive and shuffled-consequence learners on Jan-Feb, and evaluates
PRE/POST policies on untouched March development validation.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from ..normalization import data as norm_data
from ..normalization.transforms import n0_endpoint_log_ratios
from . import controlled_tasks as tasks
from . import qualification as q
from . import vectorized_tasks as vt
from .contracts import (
    AccountTruth,
    ContractError,
    PERMISSION_BISECTION_ITERATIONS,
    PhysicsConfig,
    account_state_from_truth,
)
from .learner import OnPolicyLearner, OnPolicyOneStepBatch
from .policy import DIRECTION_INDEX_FLAT, DIRECTION_INDEX_SHORT, account_state_vector

RESULT_COMMAND = "cb16.vs-d-historical-market-canary-r0@v1"
EXPERIMENT_ID = "r12.vs_d.historical_market_information_canary.r0"
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_d_historical_market_canary_r0.json"
SPEC_FILENAME = "experiment_spec.json"
RESULT_FILENAME = "RESULT.json"
REPORT_FILENAME = "REPORT.md"

EXIT_PASS = 0
EXIT_SCIENTIFIC_FAIL = 1
EXIT_EXECUTION_BLOCKED = 3
EXIT_CONTRACT_MISMATCH = 4

MILLISECONDS_PER_HOUR = 3_600_000
MINUTES_PER_HOUR = 60
BOOTSTRAP_DOMAIN = 0x42535452
COLLECTION_DOMAIN = 0x48495354
ARM_POSITIVE_ID = 0
ARM_CONTROL_ID = 1
COMPARISON_POSITIVE = 0
COMPARISON_CONTROL = 1
COMPARISON_PRE = 2


@dataclass(frozen=True)
class HistoricalDataset:
    train_market: np.ndarray
    train_open: np.ndarray
    train_close: np.ndarray
    validation_market: np.ndarray
    validation_open: np.ndarray
    validation_close: np.ndarray
    validation_day_index: np.ndarray
    evidence: Mapping[str, Any]


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


def _parse_utc_ms(value: str, name: str) -> int:
    text = tasks.spec_str(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{name} is not an ISO-8601 timestamp: {text!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ContractError(f"{name} must be UTC with a Z/+00:00 offset")
    return int(parsed.timestamp() * 1000)


def parse_authority(spec: Mapping[str, Any]) -> q.Authority:
    authority = tasks.spec_mapping(spec["authority"], "authority")
    parsed = q.Authority(
        context_length=tasks.spec_int(authority["context_length"], "authority.context_length", minimum=2),
        sensory_seed=tasks.spec_int(authority["sensory_seed"], "authority.sensory_seed"),
        sensory_z_dim=tasks.spec_int(authority["sensory_z_dim"], "authority.sensory_z_dim", minimum=1),
        actor_hidden_width=tasks.spec_int(authority["actor_hidden_width"], "authority.actor_hidden_width", minimum=1),
        actor_hidden_layers=tasks.spec_int(authority["actor_hidden_layers"], "authority.actor_hidden_layers", minimum=1),
        device=tasks.spec_str(authority["device"], "authority.device"),
        torch_num_threads=tasks.spec_int(authority["torch_num_threads_per_worker"], "authority.torch_num_threads_per_worker", minimum=1),
        deterministic_algorithms=tasks.spec_bool(authority["deterministic_algorithms"], "authority.deterministic_algorithms"),
    )
    if parsed.device != "cpu" or parsed.torch_num_threads != 1 or not parsed.deterministic_algorithms:
        raise ContractError("VS-D requires deterministic CPU execution with one Torch thread per worker")
    if tasks.spec_float(authority["proportional_friction_kappa"], "authority.proportional_friction_kappa") != 0.0:
        raise ContractError("VS-D R0 freezes proportional friction to zero")
    if tasks.spec_str(authority["funding"], "authority.funding") != "OFF":
        raise ContractError("VS-D R0 freezes funding OFF")
    if tasks.spec_bool(authority["symbol_identity_input"], "authority.symbol_identity_input"):
        raise ContractError("VS-D R0 forbids symbol identity input")
    if tasks.spec_float(authority["initial_equity"], "authority.initial_equity") != 1.0:
        raise ContractError("VS-D R0 freezes initial equity to 1")
    if tasks.spec_float(authority["initial_quantity"], "authority.initial_quantity") != 0.0:
        raise ContractError("VS-D R0 freezes initial quantity to 0")
    return parsed


def parse_optimizer(spec: Mapping[str, Any]) -> q.OptimizerSettings:
    learner = tasks.spec_mapping(spec["learner"], "learner")
    if tasks.spec_str(learner["actor_optimizer"], "learner.actor_optimizer") != "Adam":
        raise ContractError("VS-D R0 requires Adam actor optimizer")
    if tasks.spec_str(learner["critic_optimizer"], "learner.critic_optimizer") != "Adam":
        raise ContractError("VS-D R0 requires Adam critic optimizer")
    actor_lr = tasks.spec_float(learner["actor_lr"], "learner.actor_lr")
    critic_lr = tasks.spec_float(learner["critic_lr"], "learner.critic_lr")
    if actor_lr != 0.001 or critic_lr != 0.001:
        raise ContractError("VS-D R0 freezes actor_lr=critic_lr=0.001")
    if tasks.spec_float(learner["discount_factor"], "learner.discount_factor") != 1.0:
        raise ContractError("VS-D R0 uses undiscounted one-step returns")
    if tasks.spec_bool(learner["replay"], "learner.replay"):
        raise ContractError("VS-D R0 forbids replay")
    if tasks.spec_float(learner["direction_entropy_coefficient"], "learner.direction_entropy_coefficient") != 0.005:
        raise ContractError("VS-D R0 freezes direction entropy coefficient at 0.005")
    if tasks.spec_str(learner["direction_entropy_target"], "learner.direction_entropy_target") != "categorical_direction_only":
        raise ContractError("VS-D R0 entropy target must be categorical_direction_only")
    if tasks.spec_float(learner["beta_risk_entropy_bonus"], "learner.beta_risk_entropy_bonus") != 0.0:
        raise ContractError("VS-D R0 freezes Beta-risk entropy to zero")
    if tasks.spec_int(learner["generations"], "learner.generations", minimum=1) != 256:
        raise ContractError("VS-D R0 freezes generations=256")
    if tasks.spec_str(learner["training_batch"], "learner.training_batch") != "all_1375_training_windows_each_generation":
        raise ContractError("VS-D R0 training batch contract mismatch")
    return q.OptimizerSettings(actor_lr=actor_lr, critic_lr=critic_lr)


def model_seeds(spec: Mapping[str, Any]) -> Tuple[int, ...]:
    seeds = tuple(tasks.spec_int(value, "model_seeds entry", minimum=1) for value in tasks.spec_sequence(spec["model_seeds"], "model_seeds"))
    if seeds != tuple(range(3101, 3109)):
        raise ContractError(f"VS-D R0 freezes model seeds 3101..3108, got {seeds!r}")
    return seeds


def validate_spec(spec: Mapping[str, Any]) -> None:
    if tasks.spec_str(spec["experiment_id"], "experiment_id") != EXPERIMENT_ID:
        raise ContractError(f"experiment_id must be {EXPERIMENT_ID!r}")
    if tasks.spec_str(spec["claim_level"], "claim_level") != "HISTORICAL_MARKET_INFORMATION_CANARY_ONLY":
        raise ContractError("VS-D R0 claim_level mismatch")
    authority = parse_authority(spec)
    parse_optimizer(spec)
    model_seeds(spec)

    data = tasks.spec_mapping(spec["data"], "data")
    if tasks.spec_str(data["symbol"], "data.symbol") != "BTCUSDT":
        raise ContractError("VS-D R0 freezes symbol BTCUSDT")
    if tasks.spec_str(data["source_interval"], "data.source_interval") != "1m":
        raise ContractError("VS-D R0 source interval must be 1m")
    if tasks.spec_str(data["decision_interval"], "data.decision_interval") != "1h_UTC":
        raise ContractError("VS-D R0 decision interval must be UTC 1h")
    if tasks.spec_str(data["source_root"], "data.source_root") != "/cb16/raw/klines_1m/BTCUSDT":
        raise ContractError("VS-D R0 source_root mismatch")
    if tasks.spec_str(data["manifest_path"], "data.manifest_path") != "/cb16/raw/DOWNLOAD_MANIFEST.json":
        raise ContractError("VS-D R0 manifest_path mismatch")
    if not tasks.spec_bool(data["require_manifest_final_holdout_accessed_false"], "data.require_manifest_final_holdout_accessed_false"):
        raise ContractError("VS-D R0 requires final_holdout_accessed=false")

    archives = tasks.spec_sequence(data["allowed_archives"], "data.allowed_archives")
    expected_names = [f"BTCUSDT-1m-2020-0{month}.zip" for month in (1, 2, 3)]
    names = [tasks.spec_str(tasks.spec_mapping(x, "archive")["name"], "archive.name") for x in archives]
    if names != expected_names:
        raise ContractError(f"VS-D R0 allowed archive names mismatch: {names!r}")
    for raw_entry in archives:
        entry = tasks.spec_mapping(raw_entry, "archive")
        digest = tasks.spec_str(entry["sha256"], "archive.sha256")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ContractError("archive.sha256 must be 64 lowercase hex characters")
        tasks.spec_int(entry["expected_rows"], "archive.expected_rows", minimum=1)

    hourly = tasks.spec_mapping(data["hourly_aggregation"], "data.hourly_aggregation")
    if tasks.spec_str(hourly["timezone"], "data.hourly_aggregation.timezone") != "UTC":
        raise ContractError("VS-D R0 hourly aggregation timezone must be UTC")
    if not tasks.spec_bool(hourly["require_exactly_60_contiguous_1m_rows_per_hour"], "data.hourly_aggregation.require_exactly_60_contiguous_1m_rows_per_hour"):
        raise ContractError("VS-D R0 requires exactly 60 contiguous 1m rows per hour")
    if tasks.spec_int(hourly["expected_hourly_bars"], "data.hourly_aggregation.expected_hourly_bars") != 2184:
        raise ContractError("VS-D R0 expects exactly 2184 hourly bars")

    context = tasks.spec_mapping(data["context"], "data.context")
    if tasks.spec_str(context["normalization"], "data.context.normalization") != "N0_ENDPOINT_ANCHORED_V1":
        raise ContractError("VS-D R0 requires N0 endpoint-anchored normalization")
    if tasks.spec_int(context["represented_hours"], "data.context.represented_hours") != authority.context_length:
        raise ContractError("VS-D R0 represented_hours must equal authority.context_length")
    if tasks.spec_int(context["retained_predecessor_hours"], "data.context.retained_predecessor_hours") != 1:
        raise ContractError("VS-D R0 requires one retained predecessor hour")
    if tasks.spec_int(context["raw_window_hours"], "data.context.raw_window_hours") != authority.context_length + 1:
        raise ContractError("VS-D R0 raw_window_hours mismatch")

    train = tasks.spec_mapping(data["train"], "data.train")
    validation = tasks.spec_mapping(data["validation"], "data.validation")
    if tasks.spec_int(train["expected_decisions"], "data.train.expected_decisions") != 1375:
        raise ContractError("VS-D R0 freezes 1375 training decisions")
    if tasks.spec_int(validation["expected_decisions"], "data.validation.expected_decisions") != 744:
        raise ContractError("VS-D R0 freezes 744 validation decisions")
    if tasks.spec_int(validation["expected_utc_day_blocks"], "data.validation.expected_utc_day_blocks") != 31:
        raise ContractError("VS-D R0 freezes 31 validation day blocks")
    if tasks.spec_int(validation["decisions_per_day"], "data.validation.decisions_per_day") != 24:
        raise ContractError("VS-D R0 freezes 24 decisions per validation day")
    if not tasks.spec_bool(validation["training_updates_forbidden"], "data.validation.training_updates_forbidden"):
        raise ContractError("VS-D R0 forbids validation updates")
    if tasks.spec_bool(validation["march_consequence_rows_used_for_training"], "data.validation.march_consequence_rows_used_for_training"):
        raise ContractError("March consequence rows must not be used for training")

    stats = tasks.spec_mapping(spec["validation_statistics"], "validation_statistics")
    if tasks.spec_int(stats["day_blocks"], "validation_statistics.day_blocks") != 31:
        raise ContractError("VS-D R0 bootstrap day_blocks mismatch")
    if tasks.spec_int(stats["hours_per_block"], "validation_statistics.hours_per_block") != 24:
        raise ContractError("VS-D R0 bootstrap hours_per_block mismatch")
    if tasks.spec_int(stats["bootstrap_replicates"], "validation_statistics.bootstrap_replicates") != 4096:
        raise ContractError("VS-D R0 freezes 4096 bootstrap replicates")
    if tasks.spec_str(stats["lower_bound_order_statistic"], "validation_statistics.lower_bound_order_statistic") != "sort_4096_replicates_and_take_index_204_zero_based":
        raise ContractError("VS-D R0 bootstrap lower-bound order statistic mismatch")

    aggregate = tasks.spec_mapping(spec["aggregate_gate"], "aggregate_gate")
    if tasks.spec_int(aggregate["seed_pass_count_min"], "aggregate_gate.seed_pass_count_min") != 6:
        raise ContractError("VS-D R0 freezes aggregate seed pass minimum at 6")

    global_gate = tasks.spec_mapping(spec["global_gate"], "global_gate")
    for key in (
        "implementation_tests_must_pass",
        "validation_updates_forbidden",
        "only_allowed_archives_may_be_opened",
        "final_holdout_must_remain_unaccessed",
        "no_threshold_or_training_changes_after_results",
    ):
        if not tasks.spec_bool(global_gate[key], f"global_gate.{key}"):
            raise ContractError(f"VS-D R0 requires global_gate.{key}=true")
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
        raise ContractError("VS-D R0 required_artifacts mismatch")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"cannot read allowed archive {path.name}: {exc}") from exc
    return digest.hexdigest()


def _load_manifest(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    data = tasks.spec_mapping(spec["data"], "data")
    path = Path(tasks.spec_str(data["manifest_path"], "data.manifest_path"))
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"cannot read frozen data manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ContractError(f"frozen data manifest is invalid JSON: {exc}") from exc
    manifest = tasks.spec_mapping(parsed, "data manifest")
    constraints = tasks.spec_mapping(manifest.get("constraints"), "data manifest.constraints")
    if constraints.get("final_holdout_accessed") is not False:
        raise ContractError("data manifest must state final_holdout_accessed=false")
    return manifest


def _read_allowed_minutes(spec: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    _load_manifest(spec)
    data = tasks.spec_mapping(spec["data"], "data")
    source_dir = Path(tasks.spec_str(data["source_root"], "data.source_root"))
    entries = tasks.spec_sequence(data["allowed_archives"], "data.allowed_archives")
    all_times = []
    all_values = []
    archive_evidence = []
    for raw_entry in entries:
        entry = tasks.spec_mapping(raw_entry, "archive")
        name = tasks.spec_str(entry["name"], "archive.name")
        path = source_dir / name
        if not path.is_file():
            raise ContractError(f"required allowed archive is missing: {name}")
        digest = _sha256_file(path)
        expected_digest = tasks.spec_str(entry["sha256"], "archive.sha256")
        if digest != expected_digest:
            raise ContractError(f"archive SHA256 mismatch for {name}")
        try:
            times, values, info = norm_data.read_archive(path)
        except norm_data.DataUnavailable as exc:
            raise ContractError(f"archive contract failed for {name}: {exc}") from exc
        expected_rows = tasks.spec_int(entry["expected_rows"], "archive.expected_rows", minimum=1)
        if len(times) != expected_rows or info.row_count != expected_rows:
            raise ContractError(f"archive row count mismatch for {name}")
        if len(times) > 1 and not np.all(np.diff(times) == norm_data.MILLISECONDS_PER_MINUTE):
            raise ContractError(f"archive 1m continuity mismatch for {name}")
        valid, reasons = norm_data.validate_ohlcv(values)
        if not bool(valid.all()):
            raise ContractError(f"archive {name} contains invalid OHLCV rows: {reasons!r}")
        all_times.append(times)
        all_values.append(values)
        archive_evidence.append({"name": name, "sha256": digest, "rows": int(len(times)), "member_name": info.member_name})
    times = np.concatenate(all_times)
    values = np.concatenate(all_values, axis=0)
    if len(times) > 1 and not np.all(np.diff(times) == norm_data.MILLISECONDS_PER_MINUTE):
        raise ContractError("allowed archives are not contiguous across month boundaries")
    return times, values, {"manifest_final_holdout_accessed": False, "archives": archive_evidence}


def aggregate_utc_hourly(times: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(times, dtype=np.int64)
    ohlcv = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or ohlcv.shape != (timestamps.size, 5):
        raise ContractError("minute inputs must have shapes [N] and [N,5]")
    if timestamps.size == 0 or timestamps.size % MINUTES_PER_HOUR:
        raise ContractError("minute count must be a non-empty multiple of 60")
    if timestamps[0] % MILLISECONDS_PER_HOUR != 0:
        raise ContractError("first minute must be aligned to a UTC hour")
    if not np.all(np.diff(timestamps) == norm_data.MILLISECONDS_PER_MINUTE):
        raise ContractError("minute series must be exactly 1m contiguous")
    grouped_times = timestamps.reshape(-1, MINUTES_PER_HOUR)
    expected_offsets = np.arange(MINUTES_PER_HOUR, dtype=np.int64) * norm_data.MILLISECONDS_PER_MINUTE
    if not np.all(grouped_times - grouped_times[:, :1] == expected_offsets):
        raise ContractError("every UTC hour must contain exactly 60 contiguous minute rows")
    if not np.all(grouped_times[:, 0] % MILLISECONDS_PER_HOUR == 0):
        raise ContractError("hour groups must begin at exact UTC hour boundaries")
    grouped = ohlcv.reshape(-1, MINUTES_PER_HOUR, 5)
    hourly = np.column_stack((
        grouped[:, 0, 0],
        grouped[:, :, 1].max(axis=1),
        grouped[:, :, 2].min(axis=1),
        grouped[:, -1, 3],
        grouped[:, :, 4].sum(axis=1),
    ))
    if not np.all(np.isfinite(hourly)):
        raise ContractError("hourly OHLCV must be finite")
    return grouped_times[:, 0].copy(), hourly


def _window_indices(consequence_indices: np.ndarray, raw_window_hours: int) -> np.ndarray:
    consequence = np.asarray(consequence_indices, dtype=np.int64)
    offsets = np.arange(-raw_window_hours, 0, dtype=np.int64)
    indices = consequence[:, None] + offsets[None, :]
    if indices.size and int(indices.min()) < 0:
        raise ContractError("historical context would reach before the allowed data slice")
    return indices


def _consequence_indices(hourly_times: np.ndarray, start_ms: int, end_ms: int, expected_count: int) -> np.ndarray:
    mask = (hourly_times >= start_ms) & (hourly_times <= end_ms)
    indices = np.flatnonzero(mask)
    if indices.size != expected_count:
        raise ContractError(f"consequence selection expected {expected_count} hours, got {indices.size}")
    expected = np.arange(start_ms, end_ms + MILLISECONDS_PER_HOUR, MILLISECONDS_PER_HOUR)
    if not np.array_equal(hourly_times[indices], expected):
        raise ContractError("selected consequence hours are not the exact frozen UTC chronology")
    return indices


def build_dataset_from_hourly(
    spec: Mapping[str, Any], hourly_times: np.ndarray, hourly: np.ndarray, *, evidence: Mapping[str, Any] | None = None
) -> HistoricalDataset:
    data = tasks.spec_mapping(spec["data"], "data")
    context = tasks.spec_mapping(data["context"], "data.context")
    raw_window_hours = tasks.spec_int(context["raw_window_hours"], "data.context.raw_window_hours")
    train = tasks.spec_mapping(data["train"], "data.train")
    validation = tasks.spec_mapping(data["validation"], "data.validation")
    train_idx = _consequence_indices(
        hourly_times,
        _parse_utc_ms(train["consequence_hour_start_utc"], "data.train.consequence_hour_start_utc"),
        _parse_utc_ms(train["consequence_hour_end_utc"], "data.train.consequence_hour_end_utc"),
        tasks.spec_int(train["expected_decisions"], "data.train.expected_decisions"),
    )
    validation_idx = _consequence_indices(
        hourly_times,
        _parse_utc_ms(validation["consequence_hour_start_utc"], "data.validation.consequence_hour_start_utc"),
        _parse_utc_ms(validation["consequence_hour_end_utc"], "data.validation.consequence_hour_end_utc"),
        tasks.spec_int(validation["expected_decisions"], "data.validation.expected_decisions"),
    )
    if np.intersect1d(train_idx, validation_idx).size:
        raise ContractError("training and validation consequence hours overlap")
    march_start = _parse_utc_ms(validation["consequence_hour_start_utc"], "data.validation.consequence_hour_start_utc")
    if bool((hourly_times[train_idx] >= march_start).any()):
        raise ContractError("March consequence hours entered the training set")

    train_windows = hourly[_window_indices(train_idx, raw_window_hours)]
    validation_windows = hourly[_window_indices(validation_idx, raw_window_hours)]
    train_normalized = n0_endpoint_log_ratios(train_windows).values
    validation_normalized = n0_endpoint_log_ratios(validation_windows).values
    if train_normalized.shape != (1375, 64, 5):
        raise ContractError(f"training normalized shape mismatch: {train_normalized.shape!r}")
    if validation_normalized.shape != (744, 64, 5):
        raise ContractError(f"validation normalized shape mismatch: {validation_normalized.shape!r}")
    if not np.isfinite(train_normalized).all() or not np.isfinite(validation_normalized).all():
        raise ContractError("N0 historical contexts must be finite")

    day_index = ((hourly_times[validation_idx] - hourly_times[validation_idx][0]) // (24 * MILLISECONDS_PER_HOUR)).astype(np.int64)
    counts = np.bincount(day_index, minlength=31)
    if day_index.min() != 0 or day_index.max() != 30 or not np.array_equal(counts, np.full(31, 24)):
        raise ContractError("March validation must contain exactly 31 UTC day blocks of 24 hours")

    details = dict(evidence or {})
    details.update({
        "hourly_bars": int(len(hourly_times)),
        "train_decisions": int(len(train_idx)),
        "validation_decisions": int(len(validation_idx)),
        "validation_day_blocks": 31,
        "causal_window_relation": "context_hours_strictly_before_consequence_hour",
    })
    return HistoricalDataset(
        train_market=np.asarray(train_normalized, dtype=np.float64),
        train_open=np.asarray(hourly[train_idx, 0], dtype=np.float64),
        train_close=np.asarray(hourly[train_idx, 3], dtype=np.float64),
        validation_market=np.asarray(validation_normalized, dtype=np.float64),
        validation_open=np.asarray(hourly[validation_idx, 0], dtype=np.float64),
        validation_close=np.asarray(hourly[validation_idx, 3], dtype=np.float64),
        validation_day_index=day_index,
        evidence=details,
    )


def load_historical_dataset(spec: Mapping[str, Any]) -> HistoricalDataset:
    minute_times, minute_values, evidence = _read_allowed_minutes(spec)
    hourly_times, hourly = aggregate_utc_hourly(minute_times, minute_values)
    expected = tasks.spec_int(
        tasks.spec_mapping(tasks.spec_mapping(spec["data"], "data")["hourly_aggregation"], "data.hourly_aggregation")["expected_hourly_bars"],
        "data.hourly_aggregation.expected_hourly_bars",
    )
    if len(hourly_times) != expected:
        raise ContractError(f"hourly aggregation expected {expected} bars, got {len(hourly_times)}")
    return build_dataset_from_hourly(spec, hourly_times, hourly, evidence=evidence)


def _physics_config(spec: Mapping[str, Any]) -> PhysicsConfig:
    authority = tasks.spec_mapping(spec["authority"], "authority")
    return PhysicsConfig(
        context_length=tasks.spec_int(authority["context_length"], "authority.context_length", minimum=2),
        nominal_exposure_budget=tasks.spec_float(authority["nominal_exposure_budget"], "authority.nominal_exposure_budget"),
        hard_exposure_limit=tasks.spec_float(authority["hard_exposure_limit"], "authority.hard_exposure_limit"),
        proportional_friction_kappa=0.0,
        permission_bisection_iterations=PERMISSION_BISECTION_ITERATIONS,
    )


def _flat_account_row(spec: Mapping[str, Any]) -> torch.Tensor:
    config = _physics_config(spec)
    truth = AccountTruth(equity=1.0, quantity=0.0, mark_price=1.0)
    row = account_state_vector(account_state_from_truth(truth, config))
    expected = torch.tensor([0.0, 1.0, 1.0], dtype=row.dtype)
    if not torch.equal(row, expected):
        raise ContractError(f"flat historical account row mismatch: {row.tolist()!r}")
    return row


def _historical_states(learner: OnPolicyLearner, normalized_market: np.ndarray, account_row: torch.Tensor) -> torch.Tensor:
    market = torch.from_numpy(np.asarray(normalized_market, dtype=np.float64))
    z = learner.encode(market).to(torch.float32)
    return torch.cat((z, account_row.expand(z.shape[0], account_row.shape[-1])), dim=-1)


def historical_reward_batch(
    direction_indices: torch.Tensor,
    requested_risks: torch.Tensor,
    open_next: np.ndarray | torch.Tensor,
    close_next: np.ndarray | torch.Tensor,
    *,
    nominal_exposure_budget: float = 1.0,
) -> torch.Tensor:
    if direction_indices.ndim != 1 or requested_risks.ndim != 1:
        raise ContractError("historical action tensors must be one-dimensional")
    if direction_indices.numel() != requested_risks.numel():
        raise ContractError("historical direction/risk tensors must have equal length")
    open_tensor = torch.as_tensor(open_next, dtype=torch.float64)
    close_tensor = torch.as_tensor(close_next, dtype=torch.float64)
    if open_tensor.ndim != 1 or close_tensor.ndim != 1:
        raise ContractError("historical consequence prices must be one-dimensional")
    if open_tensor.numel() != direction_indices.numel() or close_tensor.numel() != direction_indices.numel():
        raise ContractError("historical consequence price counts must match actions")
    if not bool(torch.isfinite(open_tensor).all() and torch.isfinite(close_tensor).all()):
        raise ContractError("historical consequence prices must be finite")
    if not bool((open_tensor > 0.0).all() and (close_tensor > 0.0).all()):
        raise ContractError("historical consequence prices must be strictly positive")
    signed = direction_indices.to(torch.float64) - 1.0
    targets = signed * requested_risks.to(torch.float64) * float(nominal_exposure_budget)
    growth = 1.0 + targets * (close_tensor / open_tensor - 1.0)
    if not bool((growth > 0.0).all()):
        raise ContractError("historical one-step equity must stay strictly positive")
    rewards = torch.log(growth)
    if not bool(torch.isfinite(rewards).all()):
        raise ContractError("historical one-step log-growth must be finite")
    return rewards


def _collection_seed(seed: int, arm_id: int, generation: int) -> int:
    return tasks.derived_stream_seed(COLLECTION_DOMAIN, seed, arm_id, generation)


def _train_learner(
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
    records = []
    count = states.shape[0]
    for generation in range(generations):
        if learner.generation_id != generation:
            raise ContractError("historical learner generation boundary mismatch")
        torch.manual_seed(_collection_seed(seed, arm_id, generation))
        directions, risks = vt.sample_action_batch(learner, states)
        consequence_open = open_next
        consequence_close = close_next
        if arm_id == ARM_CONTROL_ID:
            order = tasks.no_fixed_point_permutation(count, seed=seed, generation=generation)
            permutation = np.asarray(order, dtype=np.int64)
            consequence_open = open_next[permutation]
            consequence_close = close_next[permutation]
        elif arm_id != ARM_POSITIVE_ID:
            raise ContractError(f"unknown historical training arm id {arm_id}")
        rewards = historical_reward_batch(
            directions,
            risks,
            consequence_open,
            consequence_close,
            nominal_exposure_budget=nominal_exposure_budget,
        )
        batch = OnPolicyOneStepBatch(
            states=states,
            direction_indices=directions,
            requested_risks=risks,
            rewards=rewards,
            generation_id=generation,
        )
        report = learner.update_one_step_batch(batch)
        records.append((report.actor_loss, report.critic_loss))
    return {
        "generations": generations,
        "trajectory_count_per_generation": int(count),
        "mean_actor_loss": float(np.mean([x[0] for x in records])) if records else 0.0,
        "mean_critic_loss": float(np.mean([x[1] for x in records])) if records else 0.0,
    }


def _deterministic_actions(learner: OnPolicyLearner, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    learner._require_generation_snapshot()
    with torch.no_grad():
        output = learner.actor.forward(states)
        indices = torch.argmax(output.direction_logits, dim=-1).to(torch.long)
        risks = torch.zeros(indices.shape[0], dtype=learner.actor.dtype)
        nonflat = indices != DIRECTION_INDEX_FLAT
        if bool(nonflat.any()):
            short = indices == DIRECTION_INDEX_SHORT
            alpha = torch.where(short, output.short_alpha, output.long_alpha)
            beta = torch.where(short, output.short_beta, output.long_beta)
            means = alpha.double() / (alpha.double() + beta.double())
            risks[nonflat] = means[nonflat].to(learner.actor.dtype)
    return indices, risks


def evaluate_validation(
    learner: OnPolicyLearner,
    states: torch.Tensor,
    open_next: np.ndarray,
    close_next: np.ndarray,
    *,
    nominal_exposure_budget: float,
) -> Dict[str, Any]:
    directions, risks = _deterministic_actions(learner, states)
    growth = historical_reward_batch(directions, risks, open_next, close_next, nominal_exposure_budget=nominal_exposure_budget)
    return {
        "mean_one_step_log_growth": float(growth.mean()),
        "active_decision_fraction": float((directions != DIRECTION_INDEX_FLAT).to(torch.float64).mean()),
        "hourly_log_growth": [float(x) for x in growth.tolist()],
    }


def bootstrap_lcb(
    values: Sequence[float],
    day_index: Sequence[int],
    *,
    seed: int,
    comparison_id: int,
    replicates: int = 4096,
    order_index: int = 204,
) -> float:
    array = np.asarray(values, dtype=np.float64)
    days = np.asarray(day_index, dtype=np.int64)
    if array.shape != (744,) or days.shape != (744,):
        raise ContractError("VS-D R0 bootstrap requires exactly 744 validation observations")
    if not np.isfinite(array).all():
        raise ContractError("bootstrap values must be finite")
    blocks = [array[days == day] for day in range(31)]
    if any(block.shape != (24,) for block in blocks):
        raise ContractError("bootstrap requires 31 day blocks of exactly 24 observations")
    rng = random.Random(tasks.derived_stream_seed(BOOTSTRAP_DOMAIN, seed, comparison_id))
    estimates = []
    for _ in range(replicates):
        picked = [rng.randrange(31) for _ in range(31)]
        estimates.append(float(np.mean([blocks[index].mean() for index in picked])))
    estimates.sort()
    if not 0 <= order_index < len(estimates):
        raise ContractError("bootstrap order statistic index is out of range")
    return estimates[order_index]


def seed_gate(spec: Mapping[str, Any], record: Mapping[str, Any]) -> Dict[str, Any]:
    gate = tasks.spec_mapping(spec["seed_pass_gate"], "seed_pass_gate")
    values = (
        float(record["bootstrap_lcb"]["post_positive_mean_one_step_log_growth"]),
        float(record["bootstrap_lcb"]["post_positive_minus_control_mean_one_step_log_growth"]),
        float(record["bootstrap_lcb"]["post_positive_minus_pre_mean_one_step_log_growth"]),
    )
    thresholds = (
        tasks.spec_float(gate["post_positive_mean_log_growth_lcb_strictly_gt"], "seed gate positive"),
        tasks.spec_float(gate["post_positive_minus_control_mean_log_growth_lcb_strictly_gt"], "seed gate positive-control"),
        tasks.spec_float(gate["post_positive_minus_pre_mean_log_growth_lcb_strictly_gt"], "seed gate positive-pre"),
    )
    conditions = {
        "post_positive_mean_log_growth_lcb": values[0] > thresholds[0],
        "post_positive_minus_control_mean_log_growth_lcb": values[1] > thresholds[1],
        "post_positive_minus_pre_mean_log_growth_lcb": values[2] > thresholds[2],
    }
    return {"conditions": conditions, "passed": all(conditions.values())}


def _seed_worker(payload: Tuple[Dict[str, Any], HistoricalDataset, int]) -> Dict[str, Any]:
    spec, dataset, seed = payload
    q.configure_deterministic_runtime()
    authority = parse_authority(spec)
    optimizer = parse_optimizer(spec)
    account_row = _flat_account_row(spec)
    budget = tasks.spec_float(tasks.spec_mapping(spec["authority"], "authority")["nominal_exposure_budget"], "authority.nominal_exposure_budget")
    entropy = tasks.spec_float(tasks.spec_mapping(spec["learner"], "learner")["direction_entropy_coefficient"], "learner.direction_entropy_coefficient")
    generations = tasks.spec_int(tasks.spec_mapping(spec["learner"], "learner")["generations"], "learner.generations", minimum=1)

    torch.manual_seed(seed)
    positive = q.build_learner(authority, optimizer, direction_entropy_coefficient=entropy)
    torch.manual_seed(seed)
    control = q.build_learner(authority, optimizer, direction_entropy_coefficient=entropy)
    q.require_identical_parameters(positive, control)

    positive_train_states = _historical_states(positive, dataset.train_market, account_row)
    control_train_states = _historical_states(control, dataset.train_market, account_row)
    positive_validation_states = _historical_states(positive, dataset.validation_market, account_row)
    control_validation_states = _historical_states(control, dataset.validation_market, account_row)
    if not torch.equal(positive_train_states, control_train_states):
        raise ContractError("paired positive/control historical states must be bitwise identical")
    if not torch.equal(positive_validation_states, control_validation_states):
        raise ContractError("paired positive/control validation states must be bitwise identical")

    pre = evaluate_validation(positive, positive_validation_states, dataset.validation_open, dataset.validation_close, nominal_exposure_budget=budget)
    diagnostics_positive = _train_learner(
        positive, positive_train_states, dataset.train_open, dataset.train_close,
        seed=seed, arm_id=ARM_POSITIVE_ID, generations=generations, nominal_exposure_budget=budget,
    )
    diagnostics_control = _train_learner(
        control, control_train_states, dataset.train_open, dataset.train_close,
        seed=seed, arm_id=ARM_CONTROL_ID, generations=generations, nominal_exposure_budget=budget,
    )
    post_positive = evaluate_validation(positive, positive_validation_states, dataset.validation_open, dataset.validation_close, nominal_exposure_budget=budget)
    post_control = evaluate_validation(control, control_validation_states, dataset.validation_open, dataset.validation_close, nominal_exposure_budget=budget)

    pos = np.asarray(post_positive["hourly_log_growth"], dtype=np.float64)
    ctl = np.asarray(post_control["hourly_log_growth"], dtype=np.float64)
    pre_values = np.asarray(pre["hourly_log_growth"], dtype=np.float64)
    bootstrap = {
        "post_positive_mean_one_step_log_growth": bootstrap_lcb(pos, dataset.validation_day_index, seed=seed, comparison_id=COMPARISON_POSITIVE),
        "post_positive_minus_control_mean_one_step_log_growth": bootstrap_lcb(pos - ctl, dataset.validation_day_index, seed=seed, comparison_id=COMPARISON_CONTROL),
        "post_positive_minus_pre_mean_one_step_log_growth": bootstrap_lcb(pos - pre_values, dataset.validation_day_index, seed=seed, comparison_id=COMPARISON_PRE),
    }
    point = {
        "post_positive_mean_one_step_log_growth": float(pos.mean()),
        "post_control_mean_one_step_log_growth": float(ctl.mean()),
        "pre_untrained_mean_one_step_log_growth": float(pre_values.mean()),
        "post_positive_minus_control_mean_one_step_log_growth": float((pos - ctl).mean()),
        "post_positive_minus_pre_mean_one_step_log_growth": float((pos - pre_values).mean()),
        "post_positive_active_decision_fraction": post_positive["active_decision_fraction"],
        "post_control_active_decision_fraction": post_control["active_decision_fraction"],
    }
    record = {
        "seed": seed,
        "pre": {k: v for k, v in pre.items() if k != "hourly_log_growth"},
        "post_positive": {k: v for k, v in post_positive.items() if k != "hourly_log_growth"},
        "post_control": {k: v for k, v in post_control.items() if k != "hourly_log_growth"},
        "point_estimates": point,
        "bootstrap_lcb": bootstrap,
        "diagnostics": {"positive": diagnostics_positive, "control": diagnostics_control},
    }
    record["seed_gate"] = seed_gate(spec, record)
    return record


def run_seed_set(spec: Mapping[str, Any], dataset: HistoricalDataset, seeds: Sequence[int]) -> list[Dict[str, Any]]:
    seed_list = tuple(int(seed) for seed in seeds)
    if not seed_list:
        raise ContractError("model seed set must be non-empty")
    max_workers = tasks.spec_int(tasks.spec_mapping(spec["authority"], "authority")["max_seed_workers"], "authority.max_seed_workers", minimum=1)
    workers = min(max_workers, len(seed_list))
    payloads = [(dict(spec), dataset, seed) for seed in seed_list]
    if workers <= 1:
        return [_seed_worker(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        return list(executor.map(_seed_worker, payloads))


def aggregate_gate(spec: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gate = tasks.spec_mapping(spec["aggregate_gate"], "aggregate_gate")
    pass_count = sum(1 for record in records if record["seed_gate"]["passed"])
    metrics = {
        "median_post_positive_mean_log_growth": statistics.median(float(record["point_estimates"]["post_positive_mean_one_step_log_growth"]) for record in records),
        "median_post_positive_minus_control_mean_log_growth": statistics.median(float(record["point_estimates"]["post_positive_minus_control_mean_one_step_log_growth"]) for record in records),
        "median_post_positive_minus_pre_mean_log_growth": statistics.median(float(record["point_estimates"]["post_positive_minus_pre_mean_one_step_log_growth"]) for record in records),
    }
    conditions = {
        "seed_pass_count": pass_count >= tasks.spec_int(gate["seed_pass_count_min"], "aggregate_gate.seed_pass_count_min"),
        "median_post_positive_mean_log_growth": metrics["median_post_positive_mean_log_growth"] > tasks.spec_float(gate["median_post_positive_mean_log_growth_strictly_gt"], "aggregate_gate.median_post_positive_mean_log_growth_strictly_gt"),
        "median_post_positive_minus_control_mean_log_growth": metrics["median_post_positive_minus_control_mean_log_growth"] > tasks.spec_float(gate["median_post_positive_minus_control_mean_log_growth_strictly_gt"], "aggregate_gate.median_post_positive_minus_control_mean_log_growth_strictly_gt"),
        "median_post_positive_minus_pre_mean_log_growth": metrics["median_post_positive_minus_pre_mean_log_growth"] > tasks.spec_float(gate["median_post_positive_minus_pre_mean_log_growth_strictly_gt"], "aggregate_gate.median_post_positive_minus_pre_mean_log_growth_strictly_gt"),
    }
    return {"seed_pass_count": pass_count, "metrics": metrics, "conditions": conditions, "passed": all(conditions.values())}


def runtime_record() -> Dict[str, Any]:
    record = q.runtime_record()
    record.update({"max_seed_workers": 4, "historical_interval": "BTCUSDT_2020-01_2020-03"})
    return record


def run_experiment(spec: Mapping[str, Any], dataset: HistoricalDataset, implementation_tests: Mapping[str, Any]) -> Dict[str, Any]:
    validate_spec(spec)
    q.configure_deterministic_runtime()
    test_gate = q.implementation_test_gate(implementation_tests, commit=q.commit_sha())
    if not test_gate["passed"]:
        raise ContractError("exact-commit implementation test gate is not green")
    records = run_seed_set(spec, dataset, model_seeds(spec))
    gate = aggregate_gate(spec, records)
    classification = "PASS" if gate["passed"] else "SCIENTIFIC_FAIL"
    outcome = "HISTORICAL_MARKET_INFORMATION_QUALIFIED" if gate["passed"] else "HISTORICAL_MARKET_INFORMATION_NOT_QUALIFIED"
    return {
        "schema": "cb16.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(),
        "classification": classification,
        "scientific_outcome": outcome,
        "runtime": runtime_record(),
        "implementation_tests": {**dict(implementation_tests), "gate": test_gate, "required_by_global_gate": True, "runner_executed_suite": True},
        "preregistered_spec": {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "canonical_sha256": spec_sha256(spec), "model_seeds": list(model_seeds(spec)), "declared_status": spec.get("status")},
        "data_evidence": dict(dataset.evidence),
        "seeds": records,
        "aggregate_gate": gate,
        "verdicts": {"global": "PASS" if gate["passed"] else "FAIL"},
    }


def render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# R12 VS-D Historical Market-Information Canary R0 — REPORT", "",
        f"- classification: **{result['classification']}**",
        f"- scientific outcome: **{result['scientific_outcome']}**",
        f"- commit: `{result['commit_sha']}`",
        f"- seeds: {result['preregistered_spec']['model_seeds']}", "",
        "## Per-seed", "",
        "| seed | pass | positive | control | PRE | pos-control | pos-PRE | three LCBs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in result["seeds"]:
        point = record["point_estimates"]
        lcb = record["bootstrap_lcb"]
        lines.append(
            f"| {record['seed']} | {'PASS' if record['seed_gate']['passed'] else 'FAIL'} | "
            f"{point['post_positive_mean_one_step_log_growth']:.9f} | "
            f"{point['post_control_mean_one_step_log_growth']:.9f} | "
            f"{point['pre_untrained_mean_one_step_log_growth']:.9f} | "
            f"{point['post_positive_minus_control_mean_one_step_log_growth']:.9f} | "
            f"{point['post_positive_minus_pre_mean_one_step_log_growth']:.9f} | "
            f"{lcb['post_positive_mean_one_step_log_growth']:.9f}, "
            f"{lcb['post_positive_minus_control_mean_one_step_log_growth']:.9f}, "
            f"{lcb['post_positive_minus_pre_mean_one_step_log_growth']:.9f} |"
        )
    gate = result["aggregate_gate"]
    lines += ["", "## Aggregate gate", "", f"- seed pass count: `{gate['seed_pass_count']}/8`"]
    for name, value in gate["metrics"].items():
        lines.append(f"- {name}: `{value:.12g}`")
    for name, value in gate["conditions"].items():
        lines.append(f"- {name}: **{value}**")
    lines += [
        "", f"Outcome: **{result['scientific_outcome']}**", "", "## Boundary", "",
        "- March is development validation, not final holdout.",
        "- This is a zero-friction, funding-off raw market-information canary.",
        "- It is not a continuing-account, transaction-cost, multi-symbol or production-authority result.",
        "- No rescue, extra data, threshold change, extra generations or coefficient sweep is authorized.", "",
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
    result = {
        "schema": "cb16.result.v1", "experiment_id": EXPERIMENT_ID,
        "result_command": os.environ.get("CB16_RESULT_COMMAND", RESULT_COMMAND),
        "commit_sha": q.commit_sha(), "classification": classification,
        "scientific_outcome": "NOT_EVALUATED", "runtime": runtime_record(), "seeds": [],
        "aggregate_gate": {}, "verdicts": {"global": classification},
        "error": {"type": classification, "detail": detail[:2000]},
    }
    if spec is not None:
        result["preregistered_spec"] = {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "canonical_sha256": spec_sha256(spec), "model_seeds": list(model_seeds(spec)), "declared_status": spec.get("status")}
    if implementation_tests is not None:
        result["implementation_tests"] = dict(implementation_tests)
    return result


def write_failure(result_dir: Path, spec: Mapping[str, Any] | None, result: Mapping[str, Any]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    if spec is not None:
        (result_dir / SPEC_FILENAME).write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    (result_dir / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (result_dir / REPORT_FILENAME).write_text(
        "# R12 VS-D Historical Market-Information Canary R0 — REPORT\n\n"
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
        evidence = {**dict(implementation), "gate": implementation_gate, "failure_classification": classification}
        result = failure_result(spec, classification, detail, evidence)
        write_failure(result_dir, spec, result)
        return spec, result, exit_code
    dataset = load_historical_dataset(spec)
    result = run_experiment(spec, dataset, implementation)
    write_artifacts(result_dir, spec, result)
    return spec, result, EXIT_PASS if result["classification"] == "PASS" else EXIT_SCIENTIFIC_FAIL


def main() -> int:
    try:
        raw = (os.environ.get("CB16_RESULT_DIR") or "").strip()
        if not raw or not (os.environ.get("CB16_COMMIT_SHA") or "").strip():
            raise ContractError("CB16_RESULT_DIR and CB16_COMMIT_SHA must be set")
        result_dir = Path(raw)
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
    print(f"{RESULT_COMMAND}: classification={result['classification']} outcome={result['scientific_outcome']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
