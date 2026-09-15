"""R12 VS-A market input: one canonical N0 window normalizer.

The N0 transform itself is the already-reviewed
:func:`cb16_science.normalization.transforms.n0_endpoint_log_ratios`.  This
module only adapts the caller-facing contract and never re-implements,
duplicates or alters the N0 math.

Contract (``docs/experiments/frozen/R12_VS_A_PHYSICS_R0.md`` section 2):

* input contains exactly ``L`` represented bars plus the one retained
  predecessor used by the common normalization package;
* output is the represented ``[L, 5]`` N0 tensor;
* the predecessor is not part of the returned tensor;
* no raw OHLCV is returned in parallel;
* normalization errors remain explicit.

This function is the canonical VS-A raw-market boundary, so it fails closed
before and after N0.  The window is validated first (finite values, strictly
positive OHLC, valid per-bar ordering, finite non-negative volume and a strictly
positive causal represented-window median volume) and the normalized tensor is
required to be finite.  Nothing is repaired, imputed or replaced by a global
statistic, and the N0 formulas are untouched.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..normalization.transforms import OHLCV_COLUMNS, n0_endpoint_log_ratios
from .contracts import ContractError, PhysicsConfig

#: N0 output channel order, unchanged from the normalization package.
N0_CHANNELS = ("pO", "pH", "pL", "pC", "v")

_OPEN, _HIGH, _LOW, _CLOSE, _VOLUME = range(OHLCV_COLUMNS)


def _validate_raw_window(values: np.ndarray) -> None:
    """Fail closed on a non-canonical raw window before N0 sees it.

    Every rule uses only the supplied window: the retained predecessor plus the
    represented bars.  The median volume of the causal represented window is
    the same statistic N0 uses, so a zero median is rejected here instead of
    being repaired with a global or future statistic.
    """

    if not np.all(np.isfinite(values)):
        raise ContractError("raw_window must contain only finite values")
    if not np.all(values[:, :4] > 0.0):
        raise ContractError("raw_window OHLC prices must be strictly positive")

    open_ = values[:, _OPEN]
    high = values[:, _HIGH]
    low = values[:, _LOW]
    close = values[:, _CLOSE]
    if not np.all(low <= np.minimum(open_, close)):
        raise ContractError("raw_window requires low <= min(open, close) on every bar")
    if not np.all(np.maximum(open_, close) <= high):
        raise ContractError("raw_window requires max(open, close) <= high on every bar")

    volume = values[:, _VOLUME]
    if not np.all(volume >= 0.0):
        raise ContractError("raw_window volume must be non-negative")
    represented_median = np.median(volume[1:])
    if not represented_median > 0.0:
        raise ContractError(
            "raw_window represented-window median volume must be strictly positive"
        )


def normalize_market_window(
    raw_window: np.ndarray, config: Optional[PhysicsConfig] = None
) -> np.ndarray:
    """Normalize one raw OHLCV window to the represented ``[L, 5]`` N0 tensor.

    ``raw_window`` has shape ``(L + 1, 5)``: one retained predecessor bar
    followed by the ``L`` represented bars, columns ``(open, high, low, close,
    volume)`` in ``float64``.  The returned tensor has shape ``(L, 5)`` and
    drops the predecessor; no raw OHLCV is returned alongside it.

    ``config`` is optional: when omitted, the frozen controlled-slice default
    :class:`PhysicsConfig` applies, so ``L`` must equal ``context_length=64``.
    A small hand-test window must pass an explicit
    ``PhysicsConfig(context_length=L)``.  The N0 formula itself does not depend
    on the configuration.
    """

    try:
        values = np.asarray(raw_window, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError("raw_window must be a numeric array of shape (L + 1, 5)") from exc
    if values.ndim != 2 or values.shape[-1] != OHLCV_COLUMNS:
        raise ContractError("raw_window must have shape (L + 1, 5)")
    represented = values.shape[0] - 1
    if represented < 1:
        raise ContractError("raw_window must retain one predecessor and at least one represented bar")
    if config is None:
        config = PhysicsConfig()
    elif not isinstance(config, PhysicsConfig):
        raise ContractError(f"config must be a PhysicsConfig, got {type(config).__name__}")
    if represented != config.context_length:
        raise ContractError(
            f"raw_window must hold context_length={config.context_length} represented bars, "
            f"got {represented}"
        )

    _validate_raw_window(values)
    normalized = n0_endpoint_log_ratios(values).values
    if not np.all(np.isfinite(normalized)):
        raise ContractError("normalized N0 output must be finite for a validated raw_window")
    return normalized


__all__ = ["N0_CHANNELS", "normalize_market_window"]
