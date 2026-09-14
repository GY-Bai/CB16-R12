"""Pure normalization transforms N0..N4.

Contract surface
----------------

Every transform takes a raw window tensor of shape ``(..., L + 1, 5)`` whose
columns are ``open, high, low, close, volume`` in ``float64``, where index
``-1`` is the represented endpoint bar ``t`` and index ``-2`` is ``t - 1``.
The extra leading bar is the retained predecessor; it is only used by the
return-based transforms (N1, N4).

Every transform returns :class:`TransformOutput` with ``values`` of shape
``(..., L, 5)`` and channel order ``(pO, pH, pL, pC, v)`` plus a per-window
boolean flag marking windows whose price channels were invalidated by the
contract's standard-deviation floor (``sd < eps_std``).  Invalid price channels
are reported as ``NaN``; they are never repaired.

The functions are pure: they import nothing from the dispatcher, GitHub, OCI,
account state or Central Brain code, and they hold no state.  They assume
raw-bar validation (finite, positive, correctly ordered OHLC) has already run
in the data layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from .contract import EPS_STD, EPS_V

OHLCV_COLUMNS = 5
PRICE_COLUMNS = 4


@dataclass(frozen=True)
class TransformOutput:
    """Transformed windows plus the per-window price-channel validity flag."""

    values: np.ndarray
    invalid_price_sigma: np.ndarray


def _as_batch(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != OHLCV_COLUMNS:
        raise ValueError("raw window must have shape (..., L + 1, 5)")
    if values.shape[-2] < 2:
        raise ValueError("raw window must retain at least one predecessor bar")
    return values


def _represented(raw: np.ndarray) -> np.ndarray:
    """The ``L`` represented bars, dropping the retained predecessor."""

    return raw[..., 1:, :]


def _previous_close(raw: np.ndarray) -> np.ndarray:
    """``C_{tau-1}`` for every represented bar, shape ``(..., L, 1)``."""

    return raw[..., :-1, 3:4]


def _relative_volume(volume: np.ndarray) -> np.ndarray:
    """``log(max(V_tau / median(V_window), eps_v))`` over the represented bars."""

    median = np.median(volume, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.maximum(volume / median[..., None], EPS_V))


def _pack(price: np.ndarray, volume: np.ndarray) -> np.ndarray:
    return np.concatenate([price, volume[..., None]], axis=-1)


def _no_invalid(raw: np.ndarray) -> np.ndarray:
    return np.zeros(raw.shape[:-2], dtype=bool)


def _mask_price(price: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    return np.where(invalid[..., None, None], np.nan, price)


# ---------------------------------------------------------------------------
# N0 - endpoint-anchored log ratios
# ---------------------------------------------------------------------------


def n0_endpoint_log_ratios(raw: np.ndarray) -> TransformOutput:
    values = _as_batch(raw)
    window = _represented(values)
    anchor = values[..., -1:, 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        price = np.log(window[..., :PRICE_COLUMNS] / anchor)
        volume = _relative_volume(window[..., 4])
    return TransformOutput(_pack(price, volume), _no_invalid(values))


# ---------------------------------------------------------------------------
# N1 - per-bar log-return geometry
# ---------------------------------------------------------------------------


def n1_per_bar_log_returns(raw: np.ndarray) -> TransformOutput:
    values = _as_batch(raw)
    window = _represented(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        price = np.log(window[..., :PRICE_COLUMNS] / _previous_close(values))
        volume = _relative_volume(window[..., 4])
    return TransformOutput(_pack(price, volume), _no_invalid(values))


# ---------------------------------------------------------------------------
# N2 - causal window z-score with shared price statistics
# ---------------------------------------------------------------------------


def n2_causal_window_zscore(raw: np.ndarray) -> TransformOutput:
    values = _as_batch(raw)
    window = _represented(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_window = np.log(window[..., :PRICE_COLUMNS])
        close = log_window[..., 3]
        mu_close = close.mean(axis=-1)
        sd_close = close.std(axis=-1)
        # NaN sd (impossible after validation) also fails the floor comparison.
        invalid = ~(sd_close >= EPS_STD)
        price = (log_window - mu_close[..., None, None]) / sd_close[..., None, None]
        price = _mask_price(price, invalid)

        log_volume = np.log(np.maximum(window[..., 4], EPS_V))
        mu_volume = log_volume.mean(axis=-1)
        sd_volume = log_volume.std(axis=-1)
        volume = (log_volume - mu_volume[..., None]) / np.maximum(sd_volume, EPS_STD)[..., None]
    return TransformOutput(_pack(price, volume), invalid)


# ---------------------------------------------------------------------------
# N3 - RevIN-style per-channel instance normalization
# ---------------------------------------------------------------------------


def n3_revin_per_channel(raw: np.ndarray) -> TransformOutput:
    values = _as_batch(raw)
    window = _represented(values)
    channels = np.empty_like(window)
    with np.errstate(divide="ignore", invalid="ignore"):
        channels[..., :PRICE_COLUMNS] = np.log(window[..., :PRICE_COLUMNS])
        channels[..., 4] = np.log(np.maximum(window[..., 4], EPS_V))
    mean = channels.mean(axis=-2, keepdims=True)
    std = channels.std(axis=-2, keepdims=True)
    normalized = (channels - mean) / np.maximum(std, EPS_STD)
    return TransformOutput(normalized, _no_invalid(values))


# ---------------------------------------------------------------------------
# N4 - volatility-normalized return geometry
# ---------------------------------------------------------------------------


def n4_volatility_normalized_returns(raw: np.ndarray) -> TransformOutput:
    values = _as_batch(raw)
    window = _represented(values)
    previous_close = _previous_close(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        price_returns = np.log(window[..., :PRICE_COLUMNS] / previous_close)
        close_returns = np.log(window[..., 3] / values[..., :-1, 3])
        sigma = close_returns.std(axis=-1)
        invalid = ~(sigma >= EPS_STD)
        price = _mask_price(price_returns / sigma[..., None, None], invalid)
        volume = _relative_volume(window[..., 4])
    return TransformOutput(_pack(price, volume), invalid)


TRANSFORMS: Dict[str, Callable[[np.ndarray], TransformOutput]] = {
    "N0": n0_endpoint_log_ratios,
    "N1": n1_per_bar_log_returns,
    "N2": n2_causal_window_zscore,
    "N3": n3_revin_per_channel,
    "N4": n4_volatility_normalized_returns,
}


def transform_for(candidate_id: str) -> Callable[[np.ndarray], TransformOutput]:
    try:
        return TRANSFORMS[candidate_id]
    except KeyError as exc:
        raise KeyError(f"unknown candidate id: {candidate_id!r}") from exc
