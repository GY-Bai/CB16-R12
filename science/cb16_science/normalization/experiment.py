"""Science entrypoint: R12 normalization ablation R0.

Runs five fixed K-line normalization families (N0..N4) on identical BTCUSDT 1m
windows, measures the contract's invariant probes, projects every candidate
through one frozen linear projection and writes exactly:

* ``experiment_spec.json``
* ``RESULT.json``
* ``REPORT.md``

into ``$CB16_RESULT_DIR``.  It trains nothing, scores nothing and promotes no
winner.  It reads the canonical read-only market-data tree and never writes to
it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .contract import (
    CANDIDATES,
    CONTEXT_LENGTH,
    CANDIDATE_STATUS_INVALID,
    CANDIDATE_STATUS_OK,
    CANDIDATE_STATUS_PARTIAL,
    EPS_STD,
    EXIT_EVIDENCE_INSUFFICIENT,
    EXIT_INPUT_UNAVAILABLE,
    EXIT_OK,
    FREQUENCY,
    INTERVAL_MONTHS,
    LOGICAL_DATA_NAME,
    MAX_EVALUATION_WINDOWS,
    OUTPUT_CHANNELS,
    PREDECESSOR_BARS,
    PRICE_SCALE_FACTOR,
    PROJECTION_CHECKSUM_ALGORITHM,
    RESULT_COMMAND,
    SCALE_INVARIANCE_ATOL,
    STATUS_EVIDENCE_INSUFFICIENT,
    STATUS_INPUT_UNAVAILABLE,
    STATUS_OK,
    SYMBOL,
    VOLUME_SCALE_FACTOR,
    WINDOW_SELECTION_METHOD,
)
from . import diagnostics
from .data import (
    DataUnavailable,
    LoadedBars,
    describe_root,
    load_bars,
    month_bounds_ms,
    resolve_klines_root,
    select_evenly_spaced,
    selected_endpoint_sha256,
    valid_endpoints,
    window_matrix,
)
from .reporting import (
    REPORT_FILENAME,
    RESULT_FILENAME,
    SPEC_FILENAME,
    base_result,
    build_experiment_spec,
    render_report,
    write_artifacts,
)
from .transforms import TransformOutput, transform_for


@dataclass(frozen=True)
class ExperimentOutcome:
    spec: Dict[str, Any]
    result: Dict[str, Any]
    report: str
    exit_code: int
    summary_line: str


# ---------------------------------------------------------------------------
# Dataset description
# ---------------------------------------------------------------------------


def _iso_utc(milliseconds: int) -> str:
    return (
        datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def _dataset_description(
    loaded: LoadedBars, months: Sequence[str], symbol: str
) -> Dict[str, Any]:
    return {
        "logical_name": LOGICAL_DATA_NAME,
        "symbol": symbol,
        "frequency": FREQUENCY,
        "interval_start_utc": _iso_utc(month_bounds_ms(months[0])[0]),
        "interval_end_utc": _iso_utc(
            month_bounds_ms(months[-1])[1] - 60_000
        ),
        "root_source": loaded.root_source,
        "archive_count": len(loaded.archives),
        "archives": loaded.archive_basenames,
        "archive_details": [
            {
                "month": info.month,
                "basename": info.basename,
                "member_name": info.member_name,
                "header_detected": info.header_detected,
                "row_count": info.row_count,
                "first_open_time_ms": info.first_open_time_ms,
                "last_open_time_ms": info.last_open_time_ms,
                "rows_match_calendar_month": info.rows_match_calendar_month,
            }
            for info in loaded.archives
        ],
        "csv_header_detected": any(info.header_detected for info in loaded.archives),
        "loaded_bar_count": loaded.count,
        "raw_valid_bar_count": loaded.valid_count,
        "raw_invalid_bar_count": loaded.count - loaded.valid_count,
        "raw_invalid_reasons": dict(loaded.invalid_reasons),
        "continuous_60s_spacing": loaded.continuous_60s_spacing,
        "first_open_time_ms": int(loaded.open_time_ms[0]) if loaded.count else None,
        "last_open_time_ms": int(loaded.open_time_ms[-1]) if loaded.count else None,
    }


def _window_description(
    loaded: LoadedBars, endpoints: np.ndarray, selected: np.ndarray, context_length: int
) -> Dict[str, Any]:
    possible = max(0, loaded.count - context_length)
    return {
        "context_length": context_length,
        "retained_predecessor_bars": PREDECESSOR_BARS,
        "raw_window_bars": context_length + PREDECESSOR_BARS,
        "candidate_valid_endpoints": int(endpoints.shape[0]),
        "excluded_endpoints": int(possible - endpoints.shape[0]),
        "max_evaluation_windows": int(MAX_EVALUATION_WINDOWS),
        "selected_window_count": int(selected.shape[0]),
        "method": WINDOW_SELECTION_METHOD,
        "endpoint_sha256": selected_endpoint_sha256(selected, context_length),
        "same_endpoints_for_all_candidates": True,
    }


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------


def _candidate_status(invalid_windows: int, windows: int) -> str:
    if windows == 0 or invalid_windows >= windows:
        return CANDIDATE_STATUS_INVALID
    if invalid_windows > 0:
        return CANDIDATE_STATUS_PARTIAL
    return CANDIDATE_STATUS_OK


def evaluate_candidate(
    candidate: Mapping[str, Any],
    raw_windows: np.ndarray,
    ohlcv: np.ndarray,
    endpoints: np.ndarray,
    context_length: int,
    matrix: np.ndarray,
    *,
    atol: float = SCALE_INVARIANCE_ATOL,
) -> Dict[str, Any]:
    """Measure one candidate on the shared window set.  Ranks nothing."""

    candidate_id = candidate["candidate_id"]
    transform = transform_for(candidate_id)
    output: TransformOutput = transform(raw_windows)
    invalid_windows = int(np.count_nonzero(output.invalid_price_sigma))
    windows = int(output.values.shape[0])

    finite = diagnostics.finite_output_rate(output.values)
    causality = diagnostics.causality_probe(
        transform, ohlcv, endpoints, context_length
    )
    price_scale = diagnostics.price_scale_invariance_probe(transform, raw_windows, atol=atol)
    volume_unit = diagnostics.volume_unit_invariance_probe(transform, raw_windows, atol=atol)
    ordering = diagnostics.ohlc_ordering_summary(output.values)
    candle_range = diagnostics.candle_range_correlation(output.values, raw_windows)
    magnitude = diagnostics.magnitude_summary(output.values)
    runtime = diagnostics.runtime_probe(transform, raw_windows)

    flattened = diagnostics.flatten_windows(output.values)
    z = diagnostics.project(matrix, flattened)
    z_price = diagnostics.project(
        matrix,
        diagnostics.flatten_windows(
            transform(
                diagnostics.rescale_windows(raw_windows, price_factor=PRICE_SCALE_FACTOR)
            ).values
        ),
    )
    z_volume = diagnostics.project(
        matrix,
        diagnostics.flatten_windows(
            transform(
                diagnostics.rescale_windows(raw_windows, volume_factor=VOLUME_SCALE_FACTOR)
            ).values
        ),
    )
    projection = diagnostics.projection_summary(
        z, price_scale_z=z_price, volume_scale_z=z_volume
    )
    projection["checksum"] = diagnostics.projection_checksum(matrix)
    projection["checksum_algorithm"] = PROJECTION_CHECKSUM_ALGORITHM

    return {
        "candidate_id": candidate_id,
        "name": candidate["name"],
        "role": candidate["role"],
        "description": candidate["description"],
        "status": _candidate_status(invalid_windows, windows),
        "output_shape": [context_length, len(OUTPUT_CHANNELS)],
        "channels": list(OUTPUT_CHANNELS),
        "valid_window_count": windows - invalid_windows,
        "invalid_window_count": invalid_windows,
        "invalid_window_reason": (
            f"price standard deviation below eps_std={EPS_STD:g}" if invalid_windows else None
        ),
        "invariants": {
            "finite_output": finite,
            "no_lookahead": causality,
            "price_scale_invariance": price_scale,
            "volume_unit_invariance": volume_unit,
            "ohlc_ordering": ordering,
            "candle_range_correlation": candle_range,
            "numerical_magnitude": magnitude,
            "runtime": runtime,
        },
        "sensory_projection": projection,
        "interpretation": {
            "preserves_nominal_scale_invariance": bool(price_scale["within_tolerance"]),
            "preserves_ohlc_ordering": bool(ordering["preserves_ohlc_ordering"]),
            "preserves_relative_volatility_amplitude": bool(
                candidate["preserves_relative_volatility_amplitude"]
            ),
            "invertible_to_relative_price_path_up_to_common_scale": bool(
                candidate["invertible_to_relative_price_path_up_to_common_scale"]
            ),
            "known_information_removed": list(candidate["known_information_removed"]),
        },
    }


def _unevaluated_candidate(candidate: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "name": candidate["name"],
        "role": candidate["role"],
        "description": candidate["description"],
        "status": CANDIDATE_STATUS_INVALID,
        "output_shape": [CONTEXT_LENGTH, len(OUTPUT_CHANNELS)],
        "channels": list(OUTPUT_CHANNELS),
        "valid_window_count": 0,
        "invalid_window_count": 0,
        "invalid_window_reason": None,
        "unevaluated_reason": reason,
        "invariants": {},
        "sensory_projection": {},
        "interpretation": {
            "preserves_nominal_scale_invariance": None,
            "preserves_ohlc_ordering": None,
            "preserves_relative_volatility_amplitude": bool(
                candidate["preserves_relative_volatility_amplitude"]
            ),
            "invertible_to_relative_price_path_up_to_common_scale": bool(
                candidate["invertible_to_relative_price_path_up_to_common_scale"]
            ),
            "known_information_removed": list(candidate["known_information_removed"]),
        },
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    result_dir: Path,
    klines_root: Optional[Path] = None,
    symbol: str = SYMBOL,
    months: Sequence[str] = INTERVAL_MONTHS,
    max_windows: int = MAX_EVALUATION_WINDOWS,
    env: Optional[Mapping[str, str]] = None,
    write: bool = True,
) -> ExperimentOutcome:
    """Run the frozen experiment once and (by default) write its artifacts.

    The context length is frozen at ``CONTEXT_LENGTH`` and the frozen projection
    width follows from it (``CONTEXT_LENGTH * 5 == 320``).  ``months`` and
    ``max_windows`` exist so tests can run against small fixtures; the
    allowlisted entrypoint always calls this function with the contract values.
    """

    context_length = CONTEXT_LENGTH
    environment = dict(env if env is not None else os.environ)
    commit_sha = (environment.get("CB16_COMMIT_SHA") or "").strip() or "not-supplied"
    root, root_source = resolve_klines_root(symbol, explicit=klines_root, env=environment)
    matrix = diagnostics.projection_matrix()
    projection = diagnostics.projection_config(matrix)

    try:
        loaded = load_bars(root, symbol, months, root_source=root_source)
    except DataUnavailable as exc:
        dataset = {
            "logical_name": LOGICAL_DATA_NAME,
            "symbol": symbol,
            "frequency": FREQUENCY,
            "interval_start_utc": _iso_utc(month_bounds_ms(months[0])[0]),
            "interval_end_utc": _iso_utc(month_bounds_ms(months[-1])[1] - 60_000),
            "root": describe_root(root),
            "root_source": root_source,
            "archive_count": 0,
            "archives": [],
            "error": str(exc),
        }
        selection = {
            "context_length": context_length,
            "retained_predecessor_bars": PREDECESSOR_BARS,
            "max_evaluation_windows": int(max_windows),
            "selected_window_count": 0,
            "method": WINDOW_SELECTION_METHOD,
            "same_endpoints_for_all_candidates": True,
        }
        result = base_result(
            commit_sha=commit_sha,
            status=STATUS_INPUT_UNAVAILABLE,
            projection=projection,
            dataset=dataset,
            window_selection=selection,
            warnings=[f"canonical input unavailable: {exc}"],
        )
        for candidate in CANDIDATES:
            result["candidates"][candidate["candidate_id"]] = _unevaluated_candidate(
                candidate, "canonical input unavailable"
            )
        spec = build_experiment_spec(
            commit_sha=commit_sha,
            projection=projection,
            resolved_inputs=dataset,
            window_selection=selection,
        )
        report = render_report(result)
        if write:
            write_artifacts(result_dir, spec=spec, result=result, report=report)
        return ExperimentOutcome(
            spec=spec,
            result=result,
            report=report,
            exit_code=EXIT_INPUT_UNAVAILABLE,
            summary_line=(
                f"{RESULT_COMMAND}: status={result['status']} "
                f"archives=0 candidates=0"
            ),
        )

    dataset = _dataset_description(loaded, months, symbol)
    dataset["root"] = describe_root(root)
    endpoints = valid_endpoints(loaded.valid, context_length)
    selected = select_evenly_spaced(endpoints, max_windows)
    selection = _window_description(loaded, endpoints, selected, context_length)

    warnings = []
    if loaded.count - loaded.valid_count:
        warnings.append(
            f"{loaded.count - loaded.valid_count} raw bars failed validation and were "
            "excluded from every candidate's window set"
        )
    for info in loaded.archives:
        if not info.rows_match_calendar_month:
            warnings.append(
                f"archive {info.basename} holds {info.row_count} rows, not the "
                "calendar-month minute count"
            )

    if selected.shape[0] == 0:
        result = base_result(
            commit_sha=commit_sha,
            status=STATUS_EVIDENCE_INSUFFICIENT,
            projection=projection,
            dataset=dataset,
            window_selection=selection,
            warnings=warnings + ["no valid window endpoint in the frozen interval"],
        )
        for candidate in CANDIDATES:
            result["candidates"][candidate["candidate_id"]] = _unevaluated_candidate(
                candidate, "no valid window endpoint"
            )
        spec = build_experiment_spec(
            commit_sha=commit_sha,
            projection=projection,
            resolved_inputs=dataset,
            window_selection=selection,
        )
        report = render_report(result)
        if write:
            write_artifacts(result_dir, spec=spec, result=result, report=report)
        return ExperimentOutcome(
            spec=spec,
            result=result,
            report=report,
            exit_code=EXIT_EVIDENCE_INSUFFICIENT,
            summary_line=f"{RESULT_COMMAND}: status={result['status']} windows=0",
        )

    raw_windows = window_matrix(loaded.ohlcv, selected, context_length)
    result = base_result(
        commit_sha=commit_sha,
        status=STATUS_OK,
        projection=projection,
        dataset=dataset,
        window_selection=selection,
        warnings=warnings,
    )
    for candidate in CANDIDATES:
        result["candidates"][candidate["candidate_id"]] = evaluate_candidate(
            candidate,
            raw_windows,
            loaded.ohlcv,
            selected,
            context_length,
            matrix,
        )

    spec = build_experiment_spec(
        commit_sha=commit_sha,
        projection=projection,
        resolved_inputs=dataset,
        window_selection=selection,
    )
    report = render_report(result)
    if write:
        write_artifacts(result_dir, spec=spec, result=result, report=report)

    return ExperimentOutcome(
        spec=spec,
        result=result,
        report=report,
        exit_code=EXIT_OK,
        summary_line=(
            f"{RESULT_COMMAND}: status={result['status']} "
            f"windows={selection['selected_window_count']} "
            f"candidates={','.join(result['candidates'].keys())} "
            f"endpoints_sha256={selection['endpoint_sha256'][:12]} "
            f"projection_sha256={projection['checksum'][:12]}"
        ),
    )


def main() -> int:
    """Science-lane entrypoint: write the three declared artifacts."""

    result_dir = Path(os.environ["CB16_RESULT_DIR"])
    outcome = run_experiment(result_dir=result_dir)
    print(outcome.summary_line)
    print(
        f"BUILD_REPORT: {RESULT_COMMAND} status={outcome.result['status']}; artifacts "
        f"{SPEC_FILENAME}, {RESULT_FILENAME}, {REPORT_FILENAME} written; no winner promoted"
    )
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
