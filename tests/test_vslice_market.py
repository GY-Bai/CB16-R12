"""Known-answer tests for the R12 VS-A market wiring (required tests 29-30).

The VS-A market function must be a thin adapter over the already-reviewed N0
transform: identical output, no duplicated math, no parallel raw OHLCV.

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "science") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "science"))

from cb16_science.normalization.contract import (  # noqa: E402
    PRICE_SCALE_FACTOR,
    SCALE_INVARIANCE_ATOL,
    VOLUME_SCALE_FACTOR,
)
from cb16_science.normalization.transforms import n0_endpoint_log_ratios  # noqa: E402
from cb16_science.vslice.contracts import ContractError, PhysicsConfig  # noqa: E402
from cb16_science.vslice.market import normalize_market_window  # noqa: E402

CONTEXT_LENGTH = 64

#: One retained predecessor bar plus three represented bars.
HAND_RAW = np.array(
    [
        [99.0, 101.0, 98.0, 100.0, 10.0],
        [100.0, 105.0, 95.0, 101.0, 20.0],
        [101.0, 110.0, 100.0, 109.0, 30.0],
        [109.0, 115.0, 108.0, 111.0, 40.0],
    ]
)


def walk_window(represented: int, *, seed: int = 20200101) -> np.ndarray:
    """A deterministic, strictly positive, correctly ordered OHLCV window."""

    generator = np.random.Generator(np.random.PCG64(seed))
    count = represented + 1
    close = 7200.0 * np.exp(np.cumsum(generator.normal(0.0, 0.0009, count)))
    previous = np.concatenate(([7200.0], close[:-1]))
    open_ = previous * np.exp(generator.normal(0.0, 0.0002, count))
    high = np.maximum(open_, close) * np.exp(np.abs(generator.normal(0.0, 0.0004, count)))
    low = np.minimum(open_, close) * np.exp(-np.abs(generator.normal(0.0, 0.0004, count)))
    volume = 5.0 + np.abs(generator.normal(0.0, 3.0, count))
    return np.column_stack([open_, high, low, close, volume])


class VsliceMarketWiringTests(unittest.TestCase):
    def test_29_vslice_market_output_is_identical_to_direct_n0(self):
        raw = walk_window(CONTEXT_LENGTH)
        direct = n0_endpoint_log_ratios(raw)
        through_vslice = normalize_market_window(raw)
        self.assertEqual(through_vslice.shape, (CONTEXT_LENGTH, 5))
        self.assertEqual(through_vslice.dtype, np.float64)
        self.assertTrue(np.array_equal(through_vslice, direct.values))
        self.assertTrue((through_vslice == direct.values).all())
        self.assertEqual(float(np.max(np.abs(through_vslice - direct.values))), 0.0)

        # The same identity holds on the literal hand window and on the
        # extreme-value case where N0 legitimately returns NaN.
        for window in (HAND_RAW, walk_window(4, seed=7)):
            with self.subTest(shape=window.shape):
                self.assertTrue(
                    np.array_equal(
                        normalize_market_window(window), n0_endpoint_log_ratios(window).values
                    )
                )

    def test_29b_config_context_length_is_checked_not_reinterpreted(self):
        raw = walk_window(CONTEXT_LENGTH)
        config = PhysicsConfig()
        self.assertTrue(np.array_equal(normalize_market_window(raw, config), normalize_market_window(raw)))
        with self.assertRaises(ContractError):
            normalize_market_window(raw, PhysicsConfig(context_length=32))
        with self.assertRaises(ContractError):
            normalize_market_window(walk_window(31), config)

    def test_30_price_and_volume_rescaling_leave_the_state_unchanged(self):
        raw = walk_window(CONTEXT_LENGTH)
        base = normalize_market_window(raw)
        for label, scaled in (
            ("price", np.column_stack([raw[:, :4] * PRICE_SCALE_FACTOR, raw[:, 4]])),
            ("volume", np.column_stack([raw[:, :4], raw[:, 4] * VOLUME_SCALE_FACTOR])),
            (
                "both",
                np.column_stack(
                    [raw[:, :4] * PRICE_SCALE_FACTOR, raw[:, 4] * VOLUME_SCALE_FACTOR]
                ),
            ),
        ):
            with self.subTest(rescale=label):
                rescaled = normalize_market_window(scaled)
                difference = float(np.max(np.abs(rescaled - base)))
                self.assertLessEqual(difference, SCALE_INVARIANCE_ATOL)
                self.assertTrue(np.allclose(rescaled, base, rtol=0.0, atol=SCALE_INVARIANCE_ATOL))

    def test_29c_predecessor_is_not_returned_and_no_raw_ohlcv_escapes(self):
        raw = walk_window(CONTEXT_LENGTH)
        output = normalize_market_window(raw)
        self.assertEqual(output.shape[0], raw.shape[0] - 1)
        # The last output row is anchored to the represented endpoint close.
        endpoint_close = raw[-1, 3]
        self.assertTrue(
            np.allclose(np.exp(output[-1, :4]) * endpoint_close, raw[-1, :4], rtol=1e-12, atol=0.0)
        )
        self.assertFalse(np.array_equal(output[:, :4], raw[1:, :4]))

    def test_29d_malformed_windows_raise_explicitly(self):
        with self.assertRaises(ContractError):
            normalize_market_window(np.zeros((CONTEXT_LENGTH, 4)))
        with self.assertRaises(ContractError):
            normalize_market_window(np.zeros((1, 5)))
        with self.assertRaises(ContractError):
            normalize_market_window(np.zeros(CONTEXT_LENGTH))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
