"""Known-answer tests for the R12 VS-A market wiring (required tests 29-30).

The VS-A market function must be a thin adapter over the already-reviewed N0
transform: identical output, no duplicated math, no parallel raw OHLCV.

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import math
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

        # The same identity holds on the literal hand windows; short windows
        # carry an explicit context_length because the canonical default is 64.
        for window in (HAND_RAW, walk_window(4, seed=7)):
            with self.subTest(shape=window.shape):
                config = PhysicsConfig(context_length=window.shape[0] - 1)
                self.assertTrue(
                    np.array_equal(
                        normalize_market_window(window, config),
                        n0_endpoint_log_ratios(window).values,
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


class MarketBoundaryClosureTests(unittest.TestCase):
    """Regression tests for the raw-window and context-length closures.

    The VS-A market function is the canonical raw-market boundary: it must fail
    closed before N0 (validated raw window), after N0 (finite normalized
    output) and on the canonical default ``context_length=64``.
    """

    VALID_PREDECESSOR = [101.0, 106.0, 96.0, 103.0, 11.0]

    def test_default_config_enforces_the_frozen_context_length(self):
        # No config supplied -> the frozen default PhysicsConfig() semantics.
        self.assertEqual(
            normalize_market_window(walk_window(CONTEXT_LENGTH)).shape,
            (CONTEXT_LENGTH, 5),
        )
        for represented in (1, 4, 63, 65):
            with self.subTest(represented=represented):
                with self.assertRaises(ContractError):
                    normalize_market_window(walk_window(represented))
        # An explicit default config is the same contract, not a second rule.
        self.assertEqual(
            normalize_market_window(walk_window(CONTEXT_LENGTH), PhysicsConfig()).shape,
            (CONTEXT_LENGTH, 5),
        )
        with self.assertRaises(ContractError):
            normalize_market_window(walk_window(CONTEXT_LENGTH), object())  # type: ignore[arg-type]

    def test_small_hand_windows_need_an_explicit_context_length(self):
        with self.assertRaises(ContractError):
            normalize_market_window(HAND_RAW)
        explicit = PhysicsConfig(context_length=HAND_RAW.shape[0] - 1)
        self.assertTrue(
            np.array_equal(
                normalize_market_window(HAND_RAW, explicit),
                n0_endpoint_log_ratios(HAND_RAW).values,
            )
        )

    def test_one_represented_bar_plus_predecessor_is_valid(self):
        # One represented bar is semantically valid: the predecessor is an
        # extra input bar, so context_length >= 1.
        config = PhysicsConfig(context_length=1)
        window = walk_window(1, seed=99)
        output = normalize_market_window(window, config)
        self.assertEqual(output.shape, (1, 5))
        self.assertTrue(np.array_equal(output, n0_endpoint_log_ratios(window).values))
        with self.assertRaises(ContractError):
            PhysicsConfig(context_length=0)
        with self.assertRaises(ContractError):
            PhysicsConfig(context_length=-1)

    def test_non_finite_raw_values_are_rejected(self):
        config = PhysicsConfig()
        base = walk_window(CONTEXT_LENGTH, seed=5)
        for bad in (math.nan, math.inf, -math.inf):
            for row, column in ((0, 0), (0, 4), (CONTEXT_LENGTH, 3), (CONTEXT_LENGTH, 4)):
                with self.subTest(value=bad, row=row, column=column):
                    corrupted = base.copy()
                    corrupted[row, column] = bad
                    with self.assertRaises(ContractError):
                        normalize_market_window(corrupted, config)

    def test_non_positive_ohlc_prices_are_rejected(self):
        config = PhysicsConfig()
        base = walk_window(CONTEXT_LENGTH, seed=6)
        for row in (0, 7):  # predecessor row first: N0 never looks at it
            for column in range(4):
                for bad in (0.0, -1.0):
                    with self.subTest(row=row, column=column, value=bad):
                        corrupted = base.copy()
                        corrupted[row, column] = bad
                        with self.assertRaises(ContractError):
                            normalize_market_window(corrupted, config)

    def test_invalid_per_bar_ohlc_ordering_is_rejected(self):
        config = PhysicsConfig(context_length=1)
        for label, represented in (
            # low above min(open, close); the high bound is still satisfied.
            ("low_above_min", [100.0, 105.0, 101.0, 102.0, 10.0]),
            # high below max(open, close); the low bound is still satisfied.
            ("high_below_max", [100.0, 101.0, 95.0, 102.0, 10.0]),
        ):
            with self.subTest(violation=label):
                window = np.array([self.VALID_PREDECESSOR, represented], dtype=np.float64)
                with self.assertRaises(ContractError):
                    normalize_market_window(window, config)

    def test_negative_volume_is_rejected(self):
        window = np.array(
            [self.VALID_PREDECESSOR, [102.0, 107.0, 97.0, 104.0, -1.0]], dtype=np.float64
        )
        with self.assertRaises(ContractError):
            normalize_market_window(window, PhysicsConfig(context_length=1))

    def test_zero_causal_median_volume_is_rejected_without_repair(self):
        # Represented volumes [0, 10, 0] have median 0; the single positive bar
        # is not repaired into a positive statistic.
        window = np.array(
            [
                self.VALID_PREDECESSOR,
                [102.0, 107.0, 97.0, 104.0, 0.0],
                [103.0, 108.0, 98.0, 105.0, 10.0],
                [104.0, 109.0, 99.0, 106.0, 0.0],
            ],
            dtype=np.float64,
        )
        with self.assertRaises(ContractError):
            normalize_market_window(window, PhysicsConfig(context_length=3))

    def test_median_uses_only_the_causal_represented_window(self):
        # A zero predecessor volume is excluded, exactly as the N0 statistic does.
        window = np.array(
            [
                [101.0, 106.0, 96.0, 103.0, 0.0],
                [102.0, 107.0, 97.0, 104.0, 10.0],
                [103.0, 108.0, 98.0, 105.0, 20.0],
            ],
            dtype=np.float64,
        )
        output = normalize_market_window(window, PhysicsConfig(context_length=2))
        self.assertTrue(np.all(np.isfinite(output)))

    def test_non_finite_n0_output_is_rejected_after_normalization(self):
        # A finite, strictly positive, correctly ordered price bar can still
        # overflow the endpoint log-ratio; the normalized tensor must be finite.
        price_overflow = np.array(
            [
                self.VALID_PREDECESSOR,
                [1e308, 1e308, 1e-308, 1e-308, 10.0],
            ],
            dtype=np.float64,
        )
        # A positive but denormal represented median volume still overflows the
        # relative-volume ratio of one large bar.
        volume_overflow = np.array(
            [
                self.VALID_PREDECESSOR,
                [102.0, 107.0, 97.0, 104.0, 1e-308],
                [103.0, 108.0, 98.0, 105.0, 1e-308],
                [104.0, 109.0, 99.0, 106.0, 1e308],
            ],
            dtype=np.float64,
        )
        with np.errstate(over="ignore"):
            for label, window, config in (
                ("price", price_overflow, PhysicsConfig(context_length=1)),
                ("volume", volume_overflow, PhysicsConfig(context_length=3)),
            ):
                with self.subTest(overflow=label):
                    with self.assertRaises(ContractError):
                        normalize_market_window(window, config)

    def test_normalized_output_is_finite_for_valid_windows(self):
        for represented in (1, 3, CONTEXT_LENGTH):
            with self.subTest(represented=represented):
                output = normalize_market_window(
                    walk_window(represented, seed=13),
                    PhysicsConfig(context_length=represented),
                )
                self.assertTrue(np.all(np.isfinite(output)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
