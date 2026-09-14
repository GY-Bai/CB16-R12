"""Output formatting for the R12 normalization ablation.

This module turns the frozen contract plus measured evidence into the three
declared artifacts (``experiment_spec.json``, ``RESULT.json``, ``REPORT.md``).
It performs no measurement and no transform math, and it never ranks or
promotes a candidate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .contract import (
    CANDIDATES,
    CAUSALITY_PRICE_MULTIPLIER,
    CAUSALITY_PRICE_OFFSET,
    CAUSALITY_PROBE_SAMPLE,
    CAUSALITY_VOLUME_MULTIPLIER,
    CAUSALITY_VOLUME_OFFSET,
    CANDLE_RANGE_REFERENCE,
    CONTEXT_LENGTH,
    CONTRACT_FILE,
    EPS_STD,
    EPS_V,
    FLOAT_DTYPE,
    FREQUENCY,
    INTERPRETATION_LABELS,
    INTERVAL_END_UTC,
    INTERVAL_MONTHS,
    INTERVAL_START_UTC,
    LOGICAL_DATA_NAME,
    MAX_EVALUATION_WINDOWS,
    NO_WINNER_STATEMENT,
    OUTPUT_CHANNELS,
    PREDECESSOR_BARS,
    PRICE_SCALE_FACTOR,
    RESULT_COMMAND,
    RUNTIME_REPEATS,
    RUNTIME_WARMUPS,
    SCALE_INVARIANCE_ATOL,
    SCHEMA_RESULT,
    SCHEMA_SPEC,
    SCOPE_NOTE,
    STATUS_EVIDENCE_INSUFFICIENT,
    STATUS_INPUT_UNAVAILABLE,
    STATUS_OK,
    SYMBOL,
    VALIDATION_RULES,
    VOLUME_SCALE_FACTOR,
    WINDOW_SELECTION_METHOD,
)

SPEC_FILENAME = "experiment_spec.json"
RESULT_FILENAME = "RESULT.json"
REPORT_FILENAME = "REPORT.md"
DECLARED_ARTIFACTS = (SPEC_FILENAME, RESULT_FILENAME, REPORT_FILENAME)


# ---------------------------------------------------------------------------
# experiment_spec.json
# ---------------------------------------------------------------------------


def build_experiment_spec(
    *,
    commit_sha: str,
    projection: Mapping[str, Any],
    resolved_inputs: Mapping[str, Any],
    window_selection: Mapping[str, Any],
) -> Dict[str, Any]:
    """Freeze every contract value plus resolved inputs and the commit SHA."""

    candidate_freeze = [
        {
            "candidate_id": candidate["candidate_id"],
            "name": candidate["name"],
            "role": candidate["role"],
            "description": candidate["description"],
            "output_shape": [CONTEXT_LENGTH, len(OUTPUT_CHANNELS)],
            "channels": list(OUTPUT_CHANNELS),
        }
        for candidate in CANDIDATES
    ]
    return {
        "schema": SCHEMA_SPEC,
        "command": RESULT_COMMAND,
        "contract_file": CONTRACT_FILE,
        "commit_sha": commit_sha,
        "no_winner_promoted": True,
        "scope_note": SCOPE_NOTE,
        "frozen_contract": {
            "fixed_data_scope": {
                "logical_data_name": LOGICAL_DATA_NAME,
                "symbol": SYMBOL,
                "frequency": FREQUENCY,
                "interval_months": list(INTERVAL_MONTHS),
                "interval_start_utc": INTERVAL_START_UTC,
                "interval_end_utc": INTERVAL_END_UTC,
                "context_length": CONTEXT_LENGTH,
                "retained_predecessor_bars": PREDECESSOR_BARS,
                "max_evaluation_windows": MAX_EVALUATION_WINDOWS,
                "window_selection": WINDOW_SELECTION_METHOD,
            },
            "numeric_contract": {
                "float_dtype": FLOAT_DTYPE,
                "eps_v": EPS_V,
                "eps_std": EPS_STD,
            },
            "raw_window_validation": list(VALIDATION_RULES),
            "invariant_probes": {
                "price_scale_multiplier": PRICE_SCALE_FACTOR,
                "volume_unit_multiplier": VOLUME_SCALE_FACTOR,
                "scale_invariance_atol": SCALE_INVARIANCE_ATOL,
                "candle_range_reference": CANDLE_RANGE_REFERENCE,
                "causality_sample_size": CAUSALITY_PROBE_SAMPLE,
                "causality_price_multiplier": CAUSALITY_PRICE_MULTIPLIER,
                "causality_price_offset": CAUSALITY_PRICE_OFFSET,
                "causality_volume_multiplier": CAUSALITY_VOLUME_MULTIPLIER,
                "causality_volume_offset": CAUSALITY_VOLUME_OFFSET,
                "runtime_warmups": RUNTIME_WARMUPS,
                "runtime_repeats": RUNTIME_REPEATS,
            },
            "candidates": candidate_freeze,
            "interpretation_labels": dict(INTERPRETATION_LABELS),
            "declared_outputs": list(DECLARED_ARTIFACTS),
        },
        "frozen_sensory_projection": dict(projection),
        "resolved_inputs": dict(resolved_inputs),
        "resolved_window_selection": dict(window_selection),
    }


# ---------------------------------------------------------------------------
# RESULT.json helpers
# ---------------------------------------------------------------------------


def base_result(
    *,
    commit_sha: str,
    status: str,
    projection: Mapping[str, Any],
    dataset: Mapping[str, Any],
    window_selection: Mapping[str, Any],
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_RESULT,
        "command": RESULT_COMMAND,
        "status": status,
        "commit_sha": commit_sha,
        "contract_file": CONTRACT_FILE,
        "scope_note": SCOPE_NOTE,
        "no_winner_promoted": True,
        "projection_config": dict(projection),
        "dataset": dict(dataset),
        "window_selection": dict(window_selection),
        "interpretation_labels": dict(INTERPRETATION_LABELS),
        "candidates": {},
        "warnings": list(warnings or []),
    }


# ---------------------------------------------------------------------------
# REPORT.md
# ---------------------------------------------------------------------------


def _flag(value: Any) -> str:
    if value is None:
        return "n/a"
    return "true" if value else "false"


def _number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _candidate_table(result: Mapping[str, Any]) -> List[str]:
    lines = [
        "| candidate | status | valid windows | finite entry rate | no lookahead | price x100 max diff | volume x1000 max diff | OHLC order fraction | log-range r | median runtime (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate_id, candidate in result.get("candidates", {}).items():
        invariants = candidate.get("invariants", {})
        lines.append(
            "| `{cid}` | `{status}` | {valid} | {finite} | {lookahead} | {price} | {volume} | {order} | {r} | {runtime} |".format(
                cid=candidate_id,
                status=candidate.get("status"),
                valid=candidate.get("valid_window_count"),
                finite=_number(invariants.get("finite_output", {}).get("finite_entry_rate")),
                lookahead=_flag(invariants.get("no_lookahead", {}).get("no_lookahead")),
                price=_number(
                    invariants.get("price_scale_invariance", {}).get("max_abs_difference")
                ),
                volume=_number(
                    invariants.get("volume_unit_invariance", {}).get("max_abs_difference")
                ),
                order=_number(invariants.get("ohlc_ordering", {}).get("fraction")),
                r=_number(
                    invariants.get("candle_range_correlation", {}).get("pearson_r")
                ),
                runtime=_number(invariants.get("runtime", {}).get("median_seconds")),
            )
        )
    if len(lines) == 2:
        lines.append("| (no candidate was evaluated) | | | | | | | | | |")
    return lines


def _interpretation_table(result: Mapping[str, Any]) -> List[str]:
    lines = [
        "| candidate | scale invariant | OHLC order | relative vol amplitude | invertible up to common scale | known information removed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for candidate_id, candidate in result.get("candidates", {}).items():
        interpretation = candidate.get("interpretation", {})
        removed = interpretation.get("known_information_removed") or []
        lines.append(
            "| `{cid}` | {scale} | {order} | {amplitude} | {invertible} | {removed} |".format(
                cid=candidate_id,
                scale=_flag(interpretation.get("preserves_nominal_scale_invariance")),
                order=_flag(interpretation.get("preserves_ohlc_ordering")),
                amplitude=_flag(interpretation.get("preserves_relative_volatility_amplitude")),
                invertible=_flag(
                    interpretation.get("invertible_to_relative_price_path_up_to_common_scale")
                ),
                removed=", ".join(f"`{item}`" for item in removed) if removed else "(none declared)",
            )
        )
    if len(lines) == 2:
        lines.append("| (no candidate was evaluated) | | | | | |")
    return lines


def _projection_table(result: Mapping[str, Any]) -> List[str]:
    lines = [
        "| candidate | finite Z rows | per-dim variance min/median/max | covariance effective rank | norm p01/p50/p99 | price x100 repr. error | volume x1000 repr. error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate_id, candidate in result.get("candidates", {}).items():
        projection = candidate.get("sensory_projection", {})
        variance = projection.get("per_dimension_variance") or {}
        norm = projection.get("representation_norm") or {}
        lines.append(
            "| `{cid}` | {finite} | {vmin}/{vmed}/{vmax} | {rank} | {p01}/{p50}/{p99} | {price} | {volume} |".format(
                cid=candidate_id,
                finite=_number(projection.get("finite_z_row_fraction")),
                vmin=_number(variance.get("min")),
                vmed=_number(variance.get("median")),
                vmax=_number(variance.get("max")),
                rank=_number(projection.get("covariance_effective_rank")),
                p01=_number(norm.get("p01")),
                p50=_number(norm.get("p50")),
                p99=_number(norm.get("p99")),
                price=_number(projection.get("price_scale_representation_max_abs_error")),
                volume=_number(projection.get("volume_unit_representation_max_abs_error")),
            )
        )
    if len(lines) == 2:
        lines.append("| (no candidate was evaluated) | | | | | | |")
    return lines


def _status_section(result: Mapping[str, Any]) -> List[str]:
    status = result.get("status")
    lines = [f"- formal run status: `{status}`"]
    if status == STATUS_OK:
        lines.append(
            "- the experiment executed and emitted complete evidence; individual "
            "candidates may carry invalid windows without turning this into a "
            "scientific PASS/FAIL claim"
        )
    elif status == STATUS_INPUT_UNAVAILABLE:
        lines.append(
            "- the canonical read-only input could not supply the frozen scope; "
            "no candidate was evaluated"
        )
    elif status == STATUS_EVIDENCE_INSUFFICIENT:
        lines.append(
            "- the frozen scope yielded no valid window endpoint; no candidate "
            "was evaluated"
        )
    return lines


def render_report(result: Mapping[str, Any]) -> str:
    """Compact human-readable evidence summary.  Never promotes a candidate."""

    dataset = result.get("dataset", {})
    selection = result.get("window_selection", {})
    projection = result.get("projection_config", {})

    lines: List[str] = [
        "# R12 Normalization Ablation R0 — REPORT",
        "",
        f"**{NO_WINNER_STATEMENT}**",
        "",
        "Exploratory normalization material only.  This interval is not a "
        "profitability result and not a holdout result, and this report ranks "
        "nothing.",
        "",
        "## Run",
        "",
        f"- command: `{result.get('command')}`",
        f"- schema: `{result.get('schema')}`",
        f"- contract: `{result.get('contract_file')}`",
        f"- commit: `{result.get('commit_sha')}`",
        f"- candidate set: `{', '.join(result.get('candidates', {}).keys()) or '(none evaluated)'}`",
    ]
    lines += _status_section(result)
    lines += [
        "",
        "## Data scope",
        "",
        f"- logical input: `{dataset.get('logical_name')}`",
        f"- symbol / frequency: `{dataset.get('symbol')}` / `{dataset.get('frequency')}`",
        f"- interval (UTC): `{dataset.get('interval_start_utc')}` .. `{dataset.get('interval_end_utc')}`",
        f"- archives ({_number(dataset.get('archive_count'))}): "
        + ", ".join(f"`{name}`" for name in dataset.get("archives", []) or ["(none)"]),
        f"- loaded bars: `{_number(dataset.get('loaded_bar_count'))}` "
        f"(raw-valid `{_number(dataset.get('raw_valid_bar_count'))}`, raw-invalid "
        f"`{_number(dataset.get('raw_invalid_bar_count'))}`)",
        f"- CSV header detected: `{_flag(dataset.get('csv_header_detected'))}`; "
        f"1m-continuous spacing: `{_flag(dataset.get('continuous_60s_spacing'))}`",
        "",
        "## Windows",
        "",
        f"- context length: `{selection.get('context_length')}` represented bars plus "
        f"`{selection.get('retained_predecessor_bars')}` retained predecessor bar",
        f"- candidate valid endpoints: `{_number(selection.get('candidate_valid_endpoints'))}`; "
        f"maximum windows: `{_number(selection.get('max_evaluation_windows'))}`; "
        f"selected: `{_number(selection.get('selected_window_count'))}`",
        f"- selection: {selection.get('method')}",
        f"- endpoint digest (`sha256`): `{selection.get('endpoint_sha256') or 'n/a'}`",
        f"- identical endpoint set for every candidate: "
        f"`{_flag(selection.get('same_endpoints_for_all_candidates'))}`",
        "",
        "## Invariant probes",
        "",
    ]
    lines += _candidate_table(result)
    lines += [
        "",
        "Ordering and invariance columns are measured, not assumed: a scale "
        "invariance difference is the maximum absolute change in transformed "
        f"coordinates after multiplying the raw source window by {PRICE_SCALE_FACTOR:g} "
        f"(price) or {VOLUME_SCALE_FACTOR:g} (volume); the boolean label uses an "
        f"absolute tolerance of {SCALE_INVARIANCE_ATOL:g}.",
        "",
        "## Frozen sensory projection",
        "",
        f"- fixed Gaussian matrix: `{projection.get('shape')}` from "
        f"`{projection.get('bit_generator')}` seed `{projection.get('seed')}`, "
        f"rows {projection.get('row_normalization')}, trained/mutated: "
        f"`{_flag(projection.get('trained_or_mutated'))}`",
        f"- configuration checksum (`{projection.get('checksum_algorithm')}`): "
        f"`{projection.get('checksum')}`",
        "",
    ]
    lines += _projection_table(result)
    lines += [
        "",
        "This projection is one deterministic common information/stability probe; "
        "it is not a trained model and no model winner is derived from it.",
        "",
        "## Interpretation fields (not a winner)",
        "",
    ]
    lines += _interpretation_table(result)
    lines += [
        "",
        "`preserves_nominal_scale_invariance` and `preserves_ohlc_ordering` are "
        "measured by the probes above.  The remaining fields are declared from "
        "the fixed formula and are defined in `interpretation_labels` in "
        "`RESULT.json`; they are labels, not scores.",
        "",
        "## Limits",
        "",
        "- One symbol (BTCUSDT), one exploratory 2020-01..2020-03 interval, one "
        "bar frequency (1m), no Central Brain training, no Account Physics and "
        "no profitability claim.",
        "- Runtime values are measured wall-clock seconds and are the only "
        "run-variant fields in this package; every other reported number is a "
        "deterministic function of the frozen inputs and the commit.",
        f"- **{NO_WINNER_STATEMENT}**: this report deliberately contains no "
        "ranking, no score and no recommendation.",
        "",
    ]
    if result.get("warnings"):
        lines += ["## Warnings", ""]
        lines += [f"- {warning}" for warning in result["warnings"]]
        lines += [""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_artifacts(
    result_dir: Path,
    *,
    spec: Mapping[str, Any],
    result: Mapping[str, Any],
    report: Optional[str] = None,
) -> List[str]:
    """Write exactly the three declared artifacts (plus the declared report)."""

    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / SPEC_FILENAME).write_text(dumps(spec), encoding="utf-8")
    (result_dir / RESULT_FILENAME).write_text(dumps(result), encoding="utf-8")
    (result_dir / REPORT_FILENAME).write_text(
        report if report is not None else render_report(result), encoding="utf-8"
    )
    return list(DECLARED_ARTIFACTS)
