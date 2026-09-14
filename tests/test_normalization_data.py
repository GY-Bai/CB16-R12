"""Deterministic tests for read-only loading, validation and window selection.

Runs with the standard library plus NumPy:

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
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

from cb16_science.normalization import data as market_data  # noqa: E402


class FixtureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="cb16-data-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.symbol = "BTCUSDT"
        self.month = "2020-01"
        self.root = self.tmp / "klines"

    def build_month(self, month: str = None, **kwargs) -> Path:
        return fixtures.write_month_archive(
            self.root, self.symbol, month or self.month, **kwargs
        )


# ---------------------------------------------------------------------------
# Archive reading
# ---------------------------------------------------------------------------


class ArchiveReadingTests(FixtureTestCase):
    def test_headerless_month_archive_is_read_exactly(self):
        path = self.build_month()
        times, values, info = market_data.read_archive(path)
        self.assertEqual(info.basename, "BTCUSDT-1m-2020-01.zip")
        self.assertEqual(info.member_name, "BTCUSDT-1m-2020-01.csv")
        self.assertFalse(info.header_detected)
        self.assertEqual(info.row_count, fixtures.month_minutes(self.month))
        self.assertEqual(times.shape[0], values.shape[0])
        self.assertEqual(values.shape[1], 5)
        self.assertEqual(values.dtype, np.float64)
        self.assertTrue(info.rows_match_calendar_month)
        self.assertEqual(int(times[0]), fixtures.month_start_ms(self.month))
        self.assertTrue(np.all(np.diff(times) == 60_000))

    def test_header_row_is_detected_and_skipped(self):
        path = self.build_month(header=True)
        times, values, info = market_data.read_archive(path)
        self.assertTrue(info.header_detected)
        self.assertEqual(info.row_count, fixtures.month_minutes(self.month))
        self.assertEqual(int(times[0]), fixtures.month_start_ms(self.month))

    def test_archive_without_a_csv_member_fails_closed(self):
        directory = self.root / self.symbol
        directory.mkdir(parents=True)
        path = directory / "BTCUSDT-1m-2020-01.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("notes.txt", "not a kline csv\n")
        with self.assertRaises(market_data.DataUnavailable):
            market_data.read_archive(path)

    def test_missing_month_fails_closed_and_names_the_month(self):
        fixtures.write_month_archive(self.root, self.symbol, "2020-01")
        with self.assertRaises(market_data.DataUnavailable) as context:
            market_data.load_bars(self.root, self.symbol, ("2020-01", "2020-02"))
        self.assertIn("2020-02", str(context.exception))

    def test_cross_archive_gap_is_rejected(self):
        series = fixtures.generate_ohlcv(1440)
        fixtures.write_series_archive(
            self.root,
            self.symbol,
            "2020-01",
            series,
            start_ms=fixtures.month_start_ms("2020-01"),
        )
        # Second archive is shifted by one minute: the concatenated series is
        # no longer a 1m-continuous interval.
        fixtures.write_series_archive(
            self.root,
            self.symbol,
            "2020-02",
            series,
            start_ms=fixtures.month_start_ms("2020-02") + 60_000,
        )
        with self.assertRaises(market_data.DataUnavailable):
            market_data.load_bars(self.root, self.symbol, ("2020-01", "2020-02"))

    def test_load_bars_reports_counts_and_provenance(self):
        fixtures.build_klines_root(self.root, symbol=self.symbol, months=("2020-01",))
        loaded = market_data.load_bars(
            self.root, self.symbol, ("2020-01",), root_source="explicit"
        )
        self.assertEqual(loaded.count, fixtures.month_minutes("2020-01"))
        self.assertEqual(loaded.valid_count, loaded.count)
        self.assertEqual(loaded.invalid_reasons["non_finite_ohlcv"], 0)
        self.assertEqual(loaded.archive_basenames, ["BTCUSDT-1m-2020-01.zip"])
        self.assertTrue(loaded.continuous_60s_spacing)
        self.assertEqual(loaded.root_source, "explicit")


# ---------------------------------------------------------------------------
# Raw validation
# ---------------------------------------------------------------------------


class ValidationTests(unittest.TestCase):
    def test_validation_rejects_each_contract_violation(self):
        values = np.array(
            [
                [100.0, 105.0, 95.0, 101.0, 10.0],  # valid
                [100.0, 105.0, 95.0, np.nan, 10.0],  # non-finite
                [0.0, 105.0, 95.0, 101.0, 10.0],  # non-positive open
                [100.0, 99.0, 95.0, 101.0, 10.0],  # H < max(O, C)
                [100.0, 105.0, 102.0, 101.0, 10.0],  # L > min(O, C)
                [100.0, 105.0, 95.0, 101.0, -3.0],  # negative volume: floor handles it
            ]
        )
        valid, reasons = market_data.validate_ohlcv(values)
        self.assertEqual(valid.tolist(), [True, False, False, False, False, True])
        self.assertEqual(reasons["non_finite_ohlcv"], 1)
        self.assertEqual(reasons["non_positive_ohlc_price"], 1)
        self.assertEqual(reasons["malformed_ohlc_ordering"], 2)

    def test_an_invalid_bar_removes_only_the_windows_that_contain_it(self):
        series = fixtures.generate_ohlcv(40, seed=11)
        series[20, 1] = series[20, 3] - 1.0  # H below C: malformed ordering
        valid, _ = market_data.validate_ohlcv(series)
        self.assertFalse(valid[20])
        endpoints = market_data.valid_endpoints(valid, context_length=4)
        # Source window is [t-4, t]; every endpoint in 20..24 covers bar 20.
        self.assertNotIn(20, endpoints.tolist())
        self.assertNotIn(24, endpoints.tolist())
        self.assertIn(25, endpoints.tolist())
        self.assertIn(19, endpoints.tolist())
        self.assertEqual(endpoints.tolist(), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39])


# ---------------------------------------------------------------------------
# Endpoint selection
# ---------------------------------------------------------------------------


class EndpointSelectionTests(unittest.TestCase):
    def test_endpoint_mask_requires_context_plus_predecessor(self):
        valid = np.ones(10, dtype=bool)
        mask = market_data.endpoint_valid_mask(valid, context_length=3)
        self.assertEqual(mask.tolist(), [False, False, False] + [True] * 7)

    def test_endpoint_mask_is_empty_when_the_series_is_too_short(self):
        mask = market_data.endpoint_valid_mask(np.ones(4, dtype=bool), context_length=4)
        self.assertFalse(mask.any())

    def test_selection_is_deterministic_even_and_inclusive(self):
        endpoints = np.arange(0, 100_000, dtype=np.int64)
        first = market_data.select_evenly_spaced(endpoints, 4096)
        second = market_data.select_evenly_spaced(endpoints, 4096)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape[0], 4096)
        self.assertEqual(int(first[0]), 0)
        self.assertEqual(int(first[-1]), 99_999)
        self.assertTrue(np.all(np.diff(first) > 0))
        gaps = np.diff(first)
        self.assertLessEqual(int(gaps.max()), 2 * int(gaps.min()))

    def test_selection_returns_every_endpoint_under_the_limit(self):
        endpoints = np.arange(100, dtype=np.int64)
        selected = market_data.select_evenly_spaced(endpoints, 4096)
        self.assertTrue(np.array_equal(selected, endpoints))

    def test_selection_supports_a_single_window_budget(self):
        endpoints = np.arange(10, dtype=np.int64)
        selected = market_data.select_evenly_spaced(endpoints, 1)
        self.assertEqual(selected.tolist(), [0])

    def test_endpoint_digest_is_deterministic(self):
        endpoints = np.arange(50, dtype=np.int64)
        self.assertEqual(
            market_data.selected_endpoint_sha256(endpoints, 8),
            market_data.selected_endpoint_sha256(endpoints.copy(), 8),
        )
        self.assertNotEqual(
            market_data.selected_endpoint_sha256(endpoints, 8),
            market_data.selected_endpoint_sha256(endpoints + 1, 8),
        )


class WindowMatrixTests(unittest.TestCase):
    def test_window_matrix_retains_the_predecessor_bar(self):
        series = fixtures.generate_ohlcv(12, seed=5)
        endpoints = np.array([4, 7], dtype=np.int64)
        windows = market_data.window_matrix(series, endpoints, context_length=3)
        self.assertEqual(windows.shape, (2, 4, 5))
        self.assertTrue(np.array_equal(windows[0], series[1:5]))
        self.assertTrue(np.array_equal(windows[1], series[4:8]))


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------


class RootResolutionTests(FixtureTestCase):
    def test_explicit_root_wins(self):
        fixtures.build_klines_root(self.root, symbol=self.symbol, months=("2020-01",))
        resolved, source = market_data.resolve_klines_root(
            self.symbol, explicit=self.root, env={"CB16_DATA_ROOT": "/nope"}
        )
        self.assertEqual(resolved, self.root)
        self.assertEqual(source, "explicit")

    def test_env_root_is_used_only_when_it_contains_the_symbol(self):
        fixtures.build_klines_root(self.root, symbol=self.symbol, months=("2020-01",))
        resolved, source = market_data.resolve_klines_root(
            self.symbol, env={"CB16_DATA_ROOT": str(self.root)}
        )
        self.assertEqual(resolved, self.root)
        self.assertEqual(source, "env:CB16_DATA_ROOT")

        fallback, fallback_source = market_data.resolve_klines_root(
            self.symbol, env={"CB16_DATA_ROOT": str(self.tmp / "absent")}
        )
        self.assertEqual(fallback, Path(market_data.DEFAULT_KLINES_ROOT))
        self.assertEqual(fallback_source, "default")

    def test_describe_root_hides_non_canonical_paths(self):
        self.assertEqual(
            market_data.describe_root(Path("/cb16/raw/klines_1m")),
            {"is_canonical_cb16_path": True, "path": "/cb16/raw/klines_1m"},
        )
        hidden = market_data.describe_root(self.tmp / "private")
        self.assertEqual(hidden, {"is_canonical_cb16_path": False, "path": None})
        self.assertNotIn(str(self.tmp), json.dumps(hidden))


if __name__ == "__main__":
    unittest.main()
