"""Invariant probes and the frozen sensory projection.

This module measures properties of already-computed transforms.  It contains no
transform formulas, no dispatcher/GitHub/OCI knowledge and no model training:
the only "model" here is the contract's frozen Gaussian linear projection, which
is constructed once from ``Generator(PCG64(seed=24680))``, row-normalized to
unit L2 norm and never mutated.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .contract import (
    CAUSALITY_PRICE_MULTIPLIER,
    CAUSALITY_PRICE_OFFSET,
    CAUSALITY_PROBE_SAMPLE,
    CAUSALITY_VOLUME_MULTIPLIER,
    CAUSALITY_VOLUME_OFFSET,
    CANDLE_RANGE_REFERENCE,
    FLOAT_DTYPE,
    PRICE_SCALE_FACTOR,
    PROJECTION_BIT_GENERATOR,
    PROJECTION_CHECKSUM_ALGORITHM,
    PROJECTION_ROW_NORMALIZATION,
    PROJECTION_SEED,
    PROJECTION_SHAPE,
    RUNTIME_REPEATS,
    RUNTIME_WARMUPS,
    SCALE_INVARIANCE_ATOL,
    VOLUME_SCALE_FACTOR,
)
from .data import select_evenly_spaced
from .transforms import TransformOutput

Transform = Callable[[np.ndarray], TransformOutput]


# ---------------------------------------------------------------------------
# Generic numeric helpers
# ---------------------------------------------------------------------------


def _nan_safe_max_abs(a: np.ndarray, b: np.ndarray) -> float:
    with np.errstate(invalid="ignore"):
        difference = np.abs(a - b)
    finite = np.isfinite(difference)
    if not finite.any():
        return 0.0
    return float(difference[finite].max())


def compare_outputs(a: TransformOutput, b: TransformOutput, atol: float) -> Dict[str, Any]:
    """Exact-shape comparison of two transform outputs.

    ``NaN`` in the same position counts as equal; any other difference counts as
    a mismatch.  ``max_abs_difference`` is reported over finite pairs so a
    NaN-vs-finite mismatch is visible through ``mismatch_count`` instead of
    being hidden by the tolerance.
    """

    if a.values.shape != b.values.shape:
        raise ValueError("transform outputs to compare must share a shape")
    close = np.isclose(a.values, b.values, rtol=0.0, atol=atol, equal_nan=True)
    valid_match = bool(np.array_equal(a.invalid_price_sigma, b.invalid_price_sigma))
    mismatches = int((~close).sum())
    return {
        "max_abs_difference": _nan_safe_max_abs(a.values, b.values),
        "mismatch_count": mismatches,
        "compared_entries": int(a.values.size),
        "price_validity_mask_match": valid_match,
        "within_tolerance": bool(mismatches == 0 and valid_match),
    }


# ---------------------------------------------------------------------------
# Probe 1: finite-output rate
# ---------------------------------------------------------------------------


def finite_output_rate(values: np.ndarray) -> Dict[str, Any]:
    entries = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(entries)
    fully_finite_windows = finite.all(axis=tuple(range(1, finite.ndim)))
    return {
        "finite_entry_rate": float(finite.mean()),
        "fully_finite_window_rate": float(fully_finite_windows.mean()),
        "finite_entries": int(finite.sum()),
        "total_entries": int(finite.size),
        "windows": int(finite.shape[0]),
    }


# ---------------------------------------------------------------------------
# Probe 2: causality / no-lookahead
# ---------------------------------------------------------------------------


def perturb_future_bars(ohlcv: np.ndarray, endpoint: int) -> Tuple[np.ndarray, int]:
    """Deterministically perturb every raw bar strictly after ``endpoint``."""

    modified = np.array(ohlcv, dtype=np.float64, copy=True)
    changed = int(modified.shape[0] - (endpoint + 1))
    if changed <= 0:
        return modified, 0
    modified[endpoint + 1 :, :4] = (
        modified[endpoint + 1 :, :4] * CAUSALITY_PRICE_MULTIPLIER + CAUSALITY_PRICE_OFFSET
    )
    modified[endpoint + 1 :, 4] = (
        modified[endpoint + 1 :, 4] * CAUSALITY_VOLUME_MULTIPLIER + CAUSALITY_VOLUME_OFFSET
    )
    return modified, changed


def causality_probe(
    transform: Transform,
    ohlcv: np.ndarray,
    endpoints: np.ndarray,
    context_length: int,
    *,
    sample_size: int = CAUSALITY_PROBE_SAMPLE,
    atol: float = 0.0,
) -> Dict[str, Any]:
    """Bounded deterministic no-lookahead probe.

    For a bounded, evenly spaced sample of endpoints the raw bars strictly after
    the endpoint are rewritten and the state at the endpoint is recomputed from
    the modified series.  A transform that reads future bars would change its
    answer.  The normal transform code never reads future values; this probe
    measures that claim instead of asserting it.
    """

    values = np.asarray(ohlcv, dtype=np.float64)
    candidates = np.asarray(endpoints, dtype=np.int64)
    candidates = candidates[
        (candidates >= context_length) & (candidates < values.shape[0] - 1)
    ]
    if candidates.shape[0] == 0:
        return {
            "probe_status": "NOT_APPLICABLE",
            "reason": (
                "no selected endpoint has both a full source window and a raw bar "
                "strictly after it"
            ),
            "no_lookahead": None,
        }
    sample = select_evenly_spaced(candidates, min(sample_size, candidates.shape[0]))

    worst_difference = 0.0
    mismatches = 0
    perturbed_total = 0
    identical_sources = True
    for endpoint in sample:
        start = int(endpoint) - context_length
        source = values[start : int(endpoint) + 1]
        modified, changed = perturb_future_bars(values, int(endpoint))
        modified_source = modified[start : int(endpoint) + 1]
        if not np.array_equal(source, modified_source, equal_nan=True):
            identical_sources = False
        perturbed_total += changed
        baseline = transform(source[None, :, :])
        after = transform(modified_source[None, :, :])
        comparison = compare_outputs(baseline, after, atol)
        worst_difference = max(worst_difference, comparison["max_abs_difference"])
        mismatches += comparison["mismatch_count"]

    return {
        "probe_status": "MEASURED",
        "sample_size": int(sample.shape[0]),
        "perturbed_future_bars": perturbed_total,
        "source_window_identical_after_perturbation": bool(identical_sources),
        "max_abs_difference": worst_difference,
        "mismatch_count": mismatches,
        "no_lookahead": bool(identical_sources and mismatches == 0),
        "method": (
            "raw bars strictly after each sampled endpoint are multiplied by "
            f"{CAUSALITY_PRICE_MULTIPLIER:g} and shifted by {CAUSALITY_PRICE_OFFSET:g} "
            "(OHLC) and rescaled for volume, then the endpoint state is recomputed"
        ),
    }


# ---------------------------------------------------------------------------
# Probe 3 + 4: price-scale and volume-unit invariance
# ---------------------------------------------------------------------------


def rescale_windows(
    raw: np.ndarray,
    *,
    price_factor: Optional[float] = None,
    volume_factor: Optional[float] = None,
) -> np.ndarray:
    """Scale every raw OHLC (or every raw V) in the whole retained source window."""

    scaled = np.array(raw, dtype=np.float64, copy=True)
    if price_factor is not None:
        scaled[..., :4] = scaled[..., :4] * price_factor
    if volume_factor is not None:
        scaled[..., 4] = scaled[..., 4] * volume_factor
    return scaled


def invariance_probe(
    transform: Transform,
    raw_windows: np.ndarray,
    *,
    price_factor: Optional[float] = None,
    volume_factor: Optional[float] = None,
    atol: float = SCALE_INVARIANCE_ATOL,
) -> Dict[str, Any]:
    """Measure the transformed difference caused by a raw-unit rescaling."""

    scaled = rescale_windows(
        raw_windows, price_factor=price_factor, volume_factor=volume_factor
    )
    comparison = compare_outputs(transform(raw_windows), transform(scaled), atol)
    comparison["probe"] = {
        "price_multiplier": price_factor,
        "volume_multiplier": volume_factor,
        "source_window": "all retained raw bars of the window, including the predecessor",
        "float_dtype": FLOAT_DTYPE,
    }
    comparison["atol"] = atol
    return comparison


def price_scale_invariance_probe(
    transform: Transform, raw_windows: np.ndarray, *, atol: float = SCALE_INVARIANCE_ATOL
) -> Dict[str, Any]:
    """Probe 3: multiply every raw OHLC in the source window by 100."""

    return invariance_probe(
        transform, raw_windows, price_factor=PRICE_SCALE_FACTOR, atol=atol
    )


def volume_unit_invariance_probe(
    transform: Transform, raw_windows: np.ndarray, *, atol: float = SCALE_INVARIANCE_ATOL
) -> Dict[str, Any]:
    """Probe 4: multiply every raw V in the source window by 1000."""

    return invariance_probe(
        transform, raw_windows, volume_factor=VOLUME_SCALE_FACTOR, atol=atol
    )


# ---------------------------------------------------------------------------
# Probe 5: OHLC ordering in transformed coordinates
# ---------------------------------------------------------------------------


def ohlc_ordering_summary(values: np.ndarray) -> Dict[str, Any]:
    """Fraction of represented bars whose transformed OHLC order is well formed.

    ``NaN`` entries never satisfy the comparison and are therefore counted as
    violations; ordering is measured, never repaired.
    """

    entries = np.asarray(values, dtype=np.float64)
    open_, high, low, close = entries[..., 0], entries[..., 1], entries[..., 2], entries[..., 3]
    high_ok = high >= np.maximum(open_, close)
    low_ok = low <= np.minimum(open_, close)
    satisfying = high_ok & low_ok
    total = int(satisfying.size)
    satisfied = int(satisfying.sum())
    return {
        "represented_bars": total,
        "satisfying_bars": satisfied,
        "violating_bars": total - satisfied,
        "fraction": float(satisfied / total) if total else 0.0,
        "preserves_ohlc_ordering": bool(total > 0 and satisfied == total),
    }


# ---------------------------------------------------------------------------
# Probe 6: candle-range information
# ---------------------------------------------------------------------------


def candle_range_correlation(values: np.ndarray, raw_windows: np.ndarray) -> Dict[str, Any]:
    """Pearson correlation between transformed ``(H - L)`` and raw ``log(H/L)``."""

    entries = np.asarray(values, dtype=np.float64)
    raw = np.asarray(raw_windows, dtype=np.float64)[..., 1:, :]
    transformed_range = (entries[..., 1] - entries[..., 2]).reshape(-1)
    raw_range = np.log(raw[..., 1] / raw[..., 2]).reshape(-1)
    paired = np.isfinite(transformed_range) & np.isfinite(raw_range)
    x = transformed_range[paired]
    y = raw_range[paired]
    result: Dict[str, Any] = {
        "reference": CANDLE_RANGE_REFERENCE,
        "paired_bars": int(paired.sum()),
    }
    if x.shape[0] < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        result.update(
            {
                "pearson_r": None,
                "note": "correlation undefined: fewer than two paired bars or a zero-variance side",
            }
        )
        return result
    result["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
    return result


# ---------------------------------------------------------------------------
# Probe 7: numerical magnitude
# ---------------------------------------------------------------------------


def magnitude_summary(values: np.ndarray) -> Dict[str, Any]:
    entries = np.asarray(values, dtype=np.float64)
    flattened = entries.reshape(-1, entries.shape[-1])
    finite = np.isfinite(flattened)
    per_channel_mean = []
    per_channel_std = []
    for channel in range(flattened.shape[1]):
        column = flattened[:, channel]
        column = column[np.isfinite(column)]
        if column.size:
            per_channel_mean.append(float(column.mean()))
            per_channel_std.append(float(column.std()))
        else:
            per_channel_mean.append(None)
            per_channel_std.append(None)
    all_finite = flattened[np.isfinite(flattened)]
    if all_finite.size:
        p01, p50, p99 = (float(value) for value in np.percentile(all_finite, [1.0, 50.0, 99.0]))
        max_abs = float(np.abs(all_finite).max())
    else:
        p01 = p50 = p99 = max_abs = None
    return {
        "per_channel_mean": per_channel_mean,
        "per_channel_std": per_channel_std,
        "global_p01": p01,
        "global_p50": p50,
        "global_p99": p99,
        "global_max_abs": max_abs,
    }


# ---------------------------------------------------------------------------
# Probe 8: runtime
# ---------------------------------------------------------------------------


def runtime_probe(
    transform: Transform,
    raw_windows: np.ndarray,
    *,
    warmups: int = RUNTIME_WARMUPS,
    repeats: int = RUNTIME_REPEATS,
) -> Dict[str, Any]:
    """Wall-clock seconds over the identical window set, one warm-up then 3 runs."""

    for _ in range(max(0, warmups)):
        transform(raw_windows)
    samples: List[float] = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        transform(raw_windows)
        samples.append(time.perf_counter() - started)
    return {
        "warmups": int(warmups),
        "repeats": int(repeats),
        "samples_seconds": samples,
        "median_seconds": float(np.median(samples)),
        "note": (
            "measured wall-clock values; the only run-variant fields in the "
            "result package, required by the task contract"
        ),
    }


# ---------------------------------------------------------------------------
# Frozen sensory projection
# ---------------------------------------------------------------------------


def projection_matrix(
    seed: int = PROJECTION_SEED, shape: Tuple[int, int] = PROJECTION_SHAPE
) -> np.ndarray:
    """Fixed Gaussian projection, row-normalized to unit L2 norm, never mutated."""

    generator = np.random.Generator(np.random.PCG64(seed))
    matrix = generator.standard_normal(shape)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


def projection_checksum(
    matrix: np.ndarray,
    seed: int = PROJECTION_SEED,
    shape: Tuple[int, int] = PROJECTION_SHAPE,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        (
            f"cb16.projection.v1|bit_generator={PROJECTION_BIT_GENERATOR}|seed={seed}"
            f"|shape={shape[0]}x{shape[1]}|row_normalization={PROJECTION_ROW_NORMALIZATION}"
            f"|dtype={FLOAT_DTYPE}|"
        ).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(matrix, dtype=np.float64).tobytes())
    return digest.hexdigest()


def flatten_windows(values: np.ndarray) -> np.ndarray:
    entries = np.asarray(values, dtype=np.float64)
    return entries.reshape(entries.shape[0], -1)


def project(matrix: np.ndarray, flattened: np.ndarray) -> np.ndarray:
    return np.asarray(flattened, dtype=np.float64) @ np.asarray(matrix, dtype=np.float64).T


def projection_summary(
    z: np.ndarray,
    *,
    price_scale_z: Optional[np.ndarray] = None,
    volume_scale_z: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Stability/information summary of one candidate's projected states."""

    entries = np.asarray(z, dtype=np.float64)
    finite_rows = np.isfinite(entries).all(axis=1)
    usable = entries[finite_rows]
    summary: Dict[str, Any] = {
        "finite_z_row_fraction": float(finite_rows.mean()) if entries.shape[0] else 0.0,
        "windows": int(entries.shape[0]),
        "dimensions": int(entries.shape[1]) if entries.ndim == 2 else 0,
    }

    if usable.shape[0] >= 2:
        per_dimension_variance = np.var(usable, axis=0)
        summary["per_dimension_variance"] = {
            "min": float(np.min(per_dimension_variance)),
            "median": float(np.median(per_dimension_variance)),
            "max": float(np.max(per_dimension_variance)),
        }
        summary["covariance_effective_rank"] = _effective_rank(usable)
        norms = np.linalg.norm(usable, axis=1)
        summary["representation_norm"] = {
            "p01": float(np.percentile(norms, 1.0)),
            "p50": float(np.percentile(norms, 50.0)),
            "p99": float(np.percentile(norms, 99.0)),
        }
    else:
        summary["per_dimension_variance"] = {"min": None, "median": None, "max": None}
        summary["covariance_effective_rank"] = None
        summary["representation_norm"] = {"p01": None, "p50": None, "p99": None}

    for label, other in (("price_scale", price_scale_z), ("volume_unit", volume_scale_z)):
        if other is None:
            summary[f"{label}_representation_max_abs_error"] = None
            continue
        summary[f"{label}_representation_max_abs_error"] = _nan_safe_max_abs(entries, other)
    return summary


def _effective_rank(usable: np.ndarray) -> Optional[float]:
    """``exp(-sum(p_i log p_i))`` over normalized non-negative eigenvalues."""

    if usable.shape[0] < 2:
        return None
    covariance = np.cov(usable, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    non_negative = eigenvalues[eigenvalues > 0.0]
    total = float(non_negative.sum())
    if non_negative.size == 0 or total <= 0.0:
        return None
    probabilities = non_negative / total
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def projection_config(matrix: np.ndarray) -> Dict[str, Any]:
    return {
        "seed": PROJECTION_SEED,
        "bit_generator": PROJECTION_BIT_GENERATOR,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "row_normalization": PROJECTION_ROW_NORMALIZATION,
        "checksum_algorithm": PROJECTION_CHECKSUM_ALGORITHM,
        "checksum": projection_checksum(matrix),
        "trained_or_mutated": False,
    }
