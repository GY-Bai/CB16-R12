"""R12 VS-A market input: one canonical N0 window normalizer.

The N0 transform itself is the already-reviewed
:func:`cb16_science.normalization.transforms.n0_endpoint_log_ratios`.  This
module only adapts the caller-facing contract and never re-implements,
duplicates or alters the N0 math.

Contract (``docs/tasks/R12_VS_A_PHYSICS_R0.md`` section 2):

* input contains exactly ``L`` represented bars plus the one retained
  predecessor used by the common normalization package;
* output is the represented ``[L, 5]`` N0 tensor;
* the predecessor is not part of the returned tensor;
* no raw OHLCV is returned in parallel;
* normalization errors remain explicit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..normalization.transforms import OHLCV_COLUMNS, n0_endpoint_log_ratios
from .contracts import ContractError, PhysicsConfig

#: N0 output channel order, unchanged from the normalization package.
N0_CHANNELS = ("pO", "pH", "pL", "pC", "v")


def normalize_market_window(
    raw_window: np.ndarray, config: Optional[PhysicsConfig] = None
) -> np.ndarray:
    """Normalize one raw OHLCV window to the represented ``[L, 5]`` N0 tensor.

    ``raw_window`` has shape ``(L + 1, 5)``: one retained predecessor bar
    followed by the ``L`` represented bars, columns ``(open, high, low, close,
    volume)`` in ``float64``.  The returned tensor has shape ``(L, 5)`` and
    drops the predecessor; no raw OHLCV is returned alongside it.

    ``config`` is optional and only pins the expected ``context_length`` when
    supplied (``L == config.context_length``); the N0 formula does not depend
    on it.
    """

    values = np.asarray(raw_window, dtype=np.float64)
    if values.ndim != 2 or values.shape[-1] != OHLCV_COLUMNS:
        raise ContractError("raw_window must have shape (L + 1, 5)")
    represented = values.shape[0] - 1
    if represented < 1:
        raise ContractError("raw_window must retain one predecessor and at least one represented bar")
    if config is not None:
        if not isinstance(config, PhysicsConfig):
            raise ContractError(f"config must be a PhysicsConfig, got {type(config).__name__}")
        if represented != config.context_length:
            raise ContractError(
                f"raw_window must hold context_length={config.context_length} represented bars, "
                f"got {represented}"
            )
    return n0_endpoint_log_ratios(values).values


__all__ = ["N0_CHANNELS", "normalize_market_window"]
