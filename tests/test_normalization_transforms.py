"""Deterministic tests for the pure N0..N4 transforms and the invariant probes.

Expected values are computed independently with ``math``/``statistics`` from
literal hand windows, so a bug in the vectorized transform cannot be hidden by
re-using the same NumPy expression in the test.

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import importlib.util
import math
import statistics
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "science") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "science"))


def _load_support():
    path = Path(__file__).resolve().parent / "cb16_test_support.py"
    spec = importlib.util.spec_from_file_location("cb16_test_support", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


support = _load_support()
fixtures = support.fixtures

from cb16_science.normalization import diagnostics, transforms  # noqa: E402
from cb16_science.normalization.contract import (  # noqa: E402
    EPS_STD,
    EPS_V,
    SCALE_INVARIANCE_ATOL,
)

#: One retained predecessor bar plus three represented bars.
HAND_RAW = np.array(
    [
        [99.0, 101.0, 98.0, 100.0, 10.0],
        [100.0, 105.0, 95.0, 101.0, 20.0],
        [101.0, 110.0, 100.0, 109.0, 30.0],
        [109.0, 115.0, 108.0, 111.0, 40.0],
    ]
)

#: A raw-valid hand window whose per-channel affine maps distort OHLC geometry.
GEOMETRY_RAW = np.array(
    [
        [99.0, 100.0, 97.0, 99.0, 10.0],
        [100.0, 105.0, 95.0, 101.0, 20.0],
        [101.0, 110.0, 100.0, 109.0, 30.0],
        [109.0, 115.0, 108.0, 111.0, 40.0],
    ]
)


def relative_volume(window: np.ndarray) -> list:
    median = statistics.median(window[:, 4].tolist())
    return [math.log(max(float(value) / median, EPS_V)) for value in window[:, 4]]


class TransformShapeTests(unittest.TestCase):
    def test_every_candidate_returns_l_by_5_float64(self):
        for candidate_id, transform in transforms.TRANSFORMS.items():
            with self.subTest(candidate=candidate_id):
                output = transform(HAND_RAW)
                self.assertEqual(output.values.shape, (3, 5))
                self.assertEqual(output.values.dtype, np.float64)
                self.assertEqual(output.invalid_price_sigma.shape, ())
                self.assertFalse(bool(output.invalid_price_sigma))

    def test_batched_call_matches_single_window_call(self):
        batch = np.stack([HAND_RAW, HAND_RAW * 1.5])
        for candidate_id, transform in transforms.TRANSFORMS.items():
            with self.subTest(candidate=candidate_id):
                single = transform(HAND_RAW).values
                batched = transform(batch).values
                self.assertTrue(np.array_equal(single, batched[0]))

    def test_malformed_windows_are_rejected(self):
        with self.assertRaises(ValueError):
            transforms.n0_endpoint_log_ratios(np.zeros((3, 4)))
        with self.assertRaises(ValueError):
            transforms.n0_endpoint_log_ratios(np.zeros((1, 5)))

    def test_unknown_candidate_id_is_rejected(self):
        with self.assertRaises(KeyError):
            transforms.transform_for("N9")


class N0Tests(unittest.TestCase):
    def test_exact_endpoint_anchored_log_ratios(self):
        output = transforms.n0_endpoint_log_ratios(HAND_RAW)
        window = HAND_RAW[1:]
        anchor = 111.0
        expected_price = [
            [math.log(row[0] / anchor), math.log(row[1] / anchor),
             math.log(row[2] / anchor), math.log(row[3] / anchor)]
            for row in window
        ]
        for index, expected in enumerate(expected_price):
            for channel, value in enumerate(expected):
                self.assertAlmostEqual(output.values[index, channel], value, places=12)
        for index, expected in enumerate(relative_volume(window)):
            self.assertAlmostEqual(float(output.values[index, 4]), expected, places=12)

    def test_volume_floor_uses_eps_v(self):
        raw = np.array(
            [
                [100.0, 101.0, 99.0, 100.0, 5.0],
                [100.0, 101.0, 99.0, 100.0, 0.0],
                [100.0, 101.0, 99.0, 100.0, 10.0],
            ]
        )
        output = transforms.n0_endpoint_log_ratios(raw)
        # median(V_window) = 5, so V=0 hits the eps_v floor and V=10 gives log(2).
        self.assertAlmostEqual(float(output.values[0, 4]), math.log(EPS_V), places=12)
        self.assertAlmostEqual(float(output.values[1, 4]), math.log(2.0), places=12)


class N1Tests(unittest.TestCase):
    def test_exact_per_bar_log_returns_include_the_predecessor(self):
        output = transforms.n1_per_bar_log_returns(HAND_RAW)
        closes = [row[3] for row in HAND_RAW]
        expected_price = [
            [math.log(row[0] / closes[index]), math.log(row[1] / closes[index]),
             math.log(row[2] / closes[index]), math.log(row[3] / closes[index])]
            for index, row in enumerate(HAND_RAW[1:])
        ]
        for index, expected in enumerate(expected_price):
            for channel, value in enumerate(expected):
                self.assertAlmostEqual(output.values[index, channel], value, places=12)

    def test_first_represented_bar_uses_the_retained_predecessor_close(self):
        output = transforms.n1_per_bar_log_returns(HAND_RAW)
        self.assertAlmostEqual(float(output.values[0, 0]), math.log(100.0 / 100.0), places=12)
        self.assertAlmostEqual(float(output.values[0, 3]), math.log(101.0 / 100.0), places=12)
        # The endpoint-anchored N0 value for the same bar is different, which
        # proves the predecessor really is used.
        n0 = transforms.n0_endpoint_log_ratios(HAND_RAW)
        self.assertNotAlmostEqual(float(output.values[0, 3]), float(n0.values[0, 3]), places=6)


class N2Tests(unittest.TestCase):
    def test_shared_close_statistics_normalize_all_four_price_channels(self):
        output = transforms.n2_causal_window_zscore(HAND_RAW)
        log_closes = [math.log(row[3]) for row in HAND_RAW[1:]]
        mean = statistics.fmean(log_closes)
        std = statistics.pstdev(log_closes)
        self.assertGreater(std, EPS_STD)
        for index, row in enumerate(HAND_RAW[1:]):
            for channel in range(4):
                expected = (math.log(row[channel]) - mean) / std
                self.assertAlmostEqual(output.values[index, channel], expected, places=12)

    def test_shared_statistics_differ_from_per_channel_normalization(self):
        shared = transforms.n2_causal_window_zscore(HAND_RAW).values
        per_channel = transforms.n3_revin_per_channel(HAND_RAW).values
        std = statistics.pstdev([math.log(row[3]) for row in HAND_RAW[1:]])
        # Under N2 the channel gap is the raw log ratio divided by the SINGLE
        # close standard deviation; under N3 each channel has its own scale.
        for index, row in enumerate(HAND_RAW[1:]):
            n2_gap = shared[index, 1] - shared[index, 3]
            self.assertAlmostEqual(n2_gap, math.log(row[1] / row[3]) / std, places=12)
            self.assertNotAlmostEqual(
                per_channel[index, 1] - per_channel[index, 3], n2_gap, places=6
            )

    def test_zero_close_dispersion_invalidates_price_channels_only(self):
        raw = np.array(
            [
                [100.0, 101.0, 99.0, 100.0, 5.0],
                [100.0, 101.0, 99.0, 100.0, 10.0],
                [100.0, 101.0, 99.0, 100.0, 20.0],
            ]
        )
        output = transforms.n2_causal_window_zscore(raw)
        self.assertTrue(bool(output.invalid_price_sigma))
        self.assertTrue(np.isnan(output.values[:, :4]).all())
        self.assertTrue(np.isfinite(output.values[:, 4]).all())

    def test_volume_is_zscored_on_its_own_log_scale(self):
        output = transforms.n2_causal_window_zscore(HAND_RAW)
        logs = [math.log(row[4]) for row in HAND_RAW[1:]]
        mean = statistics.fmean(logs)
        std = statistics.pstdev(logs)
        for index, value in enumerate(logs):
            self.assertAlmostEqual(output.values[index, 4], (value - mean) / std, places=12)


class N3Tests(unittest.TestCase):
    def test_each_channel_uses_its_own_window_statistics(self):
        output = transforms.n3_revin_per_channel(HAND_RAW)
        log_volume = [math.log(row[4]) for row in HAND_RAW[1:]]
        volume_mean = statistics.fmean(log_volume)
        volume_std = statistics.pstdev(log_volume)
        for channel in range(4):
            logs = [math.log(row[channel]) for row in HAND_RAW[1:]]
            mean = statistics.fmean(logs)
            std = statistics.pstdev(logs)
            for index, value in enumerate(logs):
                self.assertAlmostEqual(output.values[index, channel], (value - mean) / std, places=12)
        for index, value in enumerate(log_volume):
            self.assertAlmostEqual(
                output.values[index, 4], (value - volume_mean) / volume_std, places=12
            )

    def test_per_channel_affine_maps_are_detected_as_geometry_distorting(self):
        output = transforms.n3_revin_per_channel(GEOMETRY_RAW)
        ordering = diagnostics.ohlc_ordering_summary(output.values)
        self.assertLess(ordering["fraction"], 1.0)
        self.assertEqual(ordering["violating_bars"], 3)
        self.assertFalse(ordering["preserves_ohlc_ordering"])

        for candidate_id in ("N0", "N1", "N2"):
            with self.subTest(candidate=candidate_id):
                other = transforms.TRANSFORMS[candidate_id](GEOMETRY_RAW)
                shared = diagnostics.ohlc_ordering_summary(other.values)
                self.assertEqual(shared["fraction"], 1.0)
                self.assertTrue(shared["preserves_ohlc_ordering"])

    def test_eps_std_floor_keeps_a_constant_channel_finite(self):
        raw = np.array(
            [
                [100.0, 101.0, 99.0, 100.0, 5.0],
                [100.0, 100.0, 100.0, 100.0, 5.0],
                [100.0, 100.0, 100.0, 100.0, 5.0],
            ]
        )
        output = transforms.n3_revin_per_channel(raw)
        self.assertTrue(np.isfinite(output.values).all())
        self.assertEqual(float(np.abs(output.values[:, :4]).max()), 0.0)


class N4Tests(unittest.TestCase):
    def test_price_channels_are_n1_returns_divided_by_close_return_sigma(self):
        output = transforms.n4_volatility_normalized_returns(HAND_RAW)
        closes = [row[3] for row in HAND_RAW]
        returns = [math.log(closes[index + 1] / closes[index]) for index in range(3)]
        sigma = statistics.pstdev(returns)
        n1 = transforms.n1_per_bar_log_returns(HAND_RAW)
        self.assertGreater(sigma, EPS_STD)
        for index in range(3):
            for channel in range(4):
                self.assertAlmostEqual(
                    output.values[index, channel], n1.values[index, channel] / sigma, places=12
                )

    def test_zero_sigma_invalidates_price_channels_cleanly(self):
        raw = np.array(
            [
                [100.0, 101.0, 99.0, 100.0, 5.0],
                [100.0, 101.0, 99.0, 100.0, 10.0],
                [100.0, 101.0, 99.0, 100.0, 20.0],
            ]
        )
        output = transforms.n4_volatility_normalized_returns(raw)
        self.assertTrue(bool(output.invalid_price_sigma))
        self.assertTrue(np.isnan(output.values[:, :4]).all())
        self.assertTrue(np.isfinite(output.values[:, 4]).all())
        # The close-return channel is exactly zero, which is what makes sigma
        # collapse; it is rejected rather than divided by the floor.
        n1 = transforms.n1_per_bar_log_returns(raw)
        self.assertTrue(np.all(n1.values[:, 3] == 0.0))


class InvarianceProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        series = fixtures.generate_ohlcv(400, seed=4)
        endpoints = np.arange(64, 400, dtype=np.int64)
        cls.windows = np.stack([series[index - 64 : index + 1] for index in endpoints])

    def test_price_scale_invariance_for_every_candidate(self):
        for candidate_id, transform in transforms.TRANSFORMS.items():
            with self.subTest(candidate=candidate_id):
                probe = diagnostics.price_scale_invariance_probe(transform, self.windows)
                self.assertEqual(probe["mismatch_count"], 0)
                self.assertLessEqual(probe["max_abs_difference"], SCALE_INVARIANCE_ATOL)
                self.assertTrue(probe["within_tolerance"])
                self.assertTrue(probe["price_validity_mask_match"])

    def test_volume_unit_invariance_for_every_candidate(self):
        for candidate_id, transform in transforms.TRANSFORMS.items():
            with self.subTest(candidate=candidate_id):
                probe = diagnostics.volume_unit_invariance_probe(transform, self.windows)
                self.assertEqual(probe["mismatch_count"], 0)
                self.assertLessEqual(probe["max_abs_difference"], SCALE_INVARIANCE_ATOL)
                self.assertTrue(probe["within_tolerance"])

    def test_a_breaking_rescale_is_reported_not_coerced(self):
        # A zero volume multiplier is not a unit change: every candidate's
        # volume behaviour changes, and the probe must report the measured
        # difference instead of declaring invariance.
        for candidate_id, transform in transforms.TRANSFORMS.items():
            with self.subTest(candidate=candidate_id):
                probe = diagnostics.invariance_probe(
                    transform, self.windows, volume_factor=0.0
                )
                self.assertGreater(probe["mismatch_count"], 0)
                self.assertFalse(probe["within_tolerance"])

    def test_no_lookahead_when_future_rows_are_modified(self):
        series = fixtures.generate_ohlcv(300, seed=9)
        endpoints = np.arange(64, 300, dtype=np.int64)
        for candidate_id, transform in transforms.TRANSFORMS.items():
            with self.subTest(candidate=candidate_id):
                probe = diagnostics.causality_probe(transform, series, endpoints, 64)
                self.assertEqual(probe["probe_status"], "MEASURED")
                self.assertEqual(probe["sample_size"], 8)
                self.assertGreater(probe["perturbed_future_bars"], 0)
                self.assertTrue(probe["source_window_identical_after_perturbation"])
                self.assertEqual(probe["mismatch_count"], 0)
                self.assertTrue(probe["no_lookahead"])

    def test_causality_probe_is_not_applicable_without_a_future_bar(self):
        series = fixtures.generate_ohlcv(80, seed=10)
        probe = diagnostics.causality_probe(
            transforms.n0_endpoint_log_ratios, series, np.array([79]), 64
        )
        self.assertEqual(probe["probe_status"], "NOT_APPLICABLE")
        self.assertIsNone(probe["no_lookahead"])

    def test_finite_rate_and_magnitude_summary(self):
        output = transforms.n0_endpoint_log_ratios(self.windows)
        finite = diagnostics.finite_output_rate(output.values)
        self.assertEqual(finite["finite_entry_rate"], 1.0)
        self.assertEqual(finite["fully_finite_window_rate"], 1.0)
        magnitude = diagnostics.magnitude_summary(output.values)
        self.assertEqual(len(magnitude["per_channel_mean"]), 5)
        self.assertEqual(len(magnitude["per_channel_std"]), 5)
        self.assertLessEqual(magnitude["global_p01"], magnitude["global_p50"])
        self.assertLessEqual(magnitude["global_p50"], magnitude["global_p99"])
        self.assertLessEqual(magnitude["global_p99"], magnitude["global_max_abs"])

    def test_candle_range_correlation_is_exact_for_n0(self):
        output = transforms.n0_endpoint_log_ratios(self.windows)
        correlation = diagnostics.candle_range_correlation(output.values, self.windows)
        self.assertAlmostEqual(correlation["pearson_r"], 1.0, places=9)
        self.assertEqual(correlation["paired_bars"], output.values.shape[0] * output.values.shape[1])

    def test_runtime_probe_warms_up_then_repeats_three_times(self):
        probe = diagnostics.runtime_probe(
            transforms.n0_endpoint_log_ratios, self.windows
        )
        self.assertEqual(probe["warmups"], 1)
        self.assertEqual(probe["repeats"], 3)
        self.assertEqual(len(probe["samples_seconds"]), 3)
        self.assertEqual(
            probe["median_seconds"], float(np.median(probe["samples_seconds"]))
        )


class ProjectionTests(unittest.TestCase):
    def test_projection_is_deterministic_and_row_normalized(self):
        first = diagnostics.projection_matrix()
        second = diagnostics.projection_matrix()
        self.assertEqual(first.shape, (32, 320))
        self.assertTrue(np.array_equal(first, second))
        norms = np.linalg.norm(first, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-12))

    def test_projection_checksum_is_deterministic_and_configuration_sensitive(self):
        first = diagnostics.projection_matrix()
        second = diagnostics.projection_matrix(seed=24680)
        self.assertEqual(
            diagnostics.projection_checksum(first), diagnostics.projection_checksum(second)
        )
        other = diagnostics.projection_matrix(seed=24681)
        self.assertNotEqual(
            diagnostics.projection_checksum(first), diagnostics.projection_checksum(other)
        )
        config = diagnostics.projection_config(first)
        self.assertEqual(config["seed"], 24680)
        self.assertEqual(config["shape"], [32, 320])
        self.assertEqual(config["checksum"], diagnostics.projection_checksum(first))

    def test_projection_output_is_deterministic(self):
        matrix = diagnostics.projection_matrix()
        flattened = np.arange(640, dtype=np.float64).reshape(2, 320)
        first = diagnostics.project(matrix, flattened)
        second = diagnostics.project(matrix, flattened)
        self.assertEqual(first.shape, (2, 32))
        self.assertTrue(np.array_equal(first, second))

    def test_projection_summary_reports_stability_fields(self):
        matrix = diagnostics.projection_matrix()
        generator = np.random.Generator(np.random.PCG64(7))
        states = generator.standard_normal((200, 320))
        z = diagnostics.project(matrix, states)
        summary = diagnostics.projection_summary(
            z, price_scale_z=z.copy(), volume_scale_z=z.copy()
        )
        self.assertEqual(summary["finite_z_row_fraction"], 1.0)
        self.assertEqual(summary["windows"], 200)
        self.assertGreaterEqual(summary["covariance_effective_rank"], 1.0)
        self.assertLessEqual(summary["covariance_effective_rank"], 32.0)
        self.assertEqual(summary["price_scale_representation_max_abs_error"], 0.0)
        self.assertLessEqual(
            summary["representation_norm"]["p01"], summary["representation_norm"]["p99"]
        )

    def test_projection_summary_handles_all_nan_states(self):
        matrix = diagnostics.projection_matrix()
        z = np.full((4, 32), np.nan)
        summary = diagnostics.projection_summary(z)
        self.assertEqual(summary["finite_z_row_fraction"], 0.0)
        self.assertIsNone(summary["covariance_effective_rank"])
        self.assertIsNone(summary["representation_norm"]["p50"])


if __name__ == "__main__":
    unittest.main()
