from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from cb16_science.normalization.transforms import n0_endpoint_log_ratios
from cb16_science.vslice import historical_market_canary_r0 as h
from cb16_science.vslice.contracts import AccountTruth, Direction, NominalAction
from cb16_science.vslice.physics import execute_transition


SPEC_SHA256 = "c33b3a8a0c4c111bfef42f86c74f8233fbb5ab6dd100c8e87dfa46cfc9138cbd"


class HistoricalMarketCanaryR0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = h.load_spec()

    def test_spec_hash_authority_and_seeds(self) -> None:
        self.assertEqual(h.spec_sha256(self.spec), SPEC_SHA256)
        h.validate_spec(self.spec)
        self.assertEqual(h.model_seeds(self.spec), tuple(range(3101, 3109)))
        authority = h.parse_authority(self.spec)
        self.assertEqual(authority.context_length, 64)
        self.assertEqual(authority.torch_num_threads, 1)
        self.assertTrue(authority.deterministic_algorithms)

    def test_science_allowlist_entry(self) -> None:
        path = Path(h.REPO_ROOT) / "config" / "cb16_science_allowlist.json"
        allowlist = json.loads(path.read_text())
        entry = allowlist["entrypoints"][h.RESULT_COMMAND]
        self.assertEqual(
            entry["argv"][-2:],
            ["-m", "cb16_science.vslice.historical_market_canary_r0"],
        )
        self.assertEqual(
            entry["produces"],
            ["experiment_spec.json", "RESULT.json", "REPORT.md"],
        )

    def test_utc_hourly_aggregation_is_exact(self) -> None:
        start = h._parse_utc_ms("2020-01-01T00:00:00Z", "start")
        times = start + np.arange(120, dtype=np.int64) * 60_000
        values = np.empty((120, 5), dtype=np.float64)
        values[:, 0] = 100.0 + np.arange(120) / 100.0
        values[:, 1] = values[:, 0] + 2.0
        values[:, 2] = values[:, 0] - 2.0
        values[:, 3] = values[:, 0] + 0.5
        values[:, 4] = 3.0
        hourly_times, hourly = h.aggregate_utc_hourly(times, values)
        self.assertEqual(hourly.shape, (2, 5))
        self.assertEqual(hourly_times.tolist(), [start, start + h.MILLISECONDS_PER_HOUR])
        self.assertEqual(hourly[0, 0], values[0, 0])
        self.assertEqual(hourly[0, 3], values[59, 3])
        self.assertEqual(hourly[0, 4], 180.0)

    def test_dataset_context_is_strictly_before_consequence(self) -> None:
        start = h._parse_utc_ms("2020-01-01T00:00:00Z", "start")
        hourly_times = start + np.arange(2184, dtype=np.int64) * h.MILLISECONDS_PER_HOUR
        index = np.arange(2184, dtype=np.float64)
        close = 100.0 * np.exp(index / 100_000.0)
        hourly = np.column_stack(
            (
                close * 0.999,
                close * 1.002,
                close * 0.998,
                close,
                10.0 + (index % 7),
            )
        )
        dataset = h.build_dataset_from_hourly(self.spec, hourly_times, hourly)
        self.assertEqual(dataset.train_market.shape, (1375, 64, 5))
        self.assertEqual(dataset.validation_market.shape, (744, 64, 5))
        first_consequence = 65
        raw = hourly[first_consequence - 65 : first_consequence]
        expected = n0_endpoint_log_ratios(raw).values
        np.testing.assert_allclose(dataset.train_market[0], expected, rtol=0, atol=0)
        self.assertEqual(dataset.train_open[0], hourly[first_consequence, 0])
        self.assertEqual(dataset.validation_day_index.tolist()[:24], [0] * 24)
        self.assertEqual(dataset.validation_day_index.tolist()[-24:], [30] * 24)

    def test_vectorized_reward_matches_canonical_physics(self) -> None:
        directions = torch.tensor([0, 1, 2], dtype=torch.long)
        risks = torch.tensor([0.4, 0.0, 0.7], dtype=torch.float32)
        opens = np.array([100.0, 100.0, 100.0])
        closes = np.array([90.0, 110.0, 110.0])
        actual = h.historical_reward_batch(directions, risks, opens, closes)
        config = h._physics_config(self.spec)
        truth = AccountTruth(equity=1.0, quantity=0.0, mark_price=100.0)
        semantic = [Direction.SHORT, Direction.FLAT, Direction.LONG]
        expected = []
        for direction, risk, open_next, close_next in zip(semantic, risks, opens, closes):
            action = NominalAction(direction=direction, requested_risk=float(risk))
            step = execute_transition(truth, action, float(open_next), float(close_next), config)
            expected.append(step.reward)
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=0, atol=1e-15)

    def test_bootstrap_is_day_blocked_and_deterministic(self) -> None:
        days = np.repeat(np.arange(31), 24)
        values = np.repeat(np.linspace(0.001, 0.031, 31), 24)
        left = h.bootstrap_lcb(values, days, seed=3101, comparison_id=h.COMPARISON_POSITIVE)
        right = h.bootstrap_lcb(values, days, seed=3101, comparison_id=h.COMPARISON_POSITIVE)
        other = h.bootstrap_lcb(values, days, seed=3102, comparison_id=h.COMPARISON_POSITIVE)
        self.assertEqual(left, right)
        self.assertGreater(left, 0.0)
        self.assertNotEqual(left, other)

    def test_seed_and_aggregate_gates_are_mechanical(self) -> None:
        record = {
            "bootstrap_lcb": {
                "post_positive_mean_one_step_log_growth": 0.01,
                "post_positive_minus_control_mean_one_step_log_growth": 0.02,
                "post_positive_minus_pre_mean_one_step_log_growth": 0.03,
            }
        }
        self.assertTrue(h.seed_gate(self.spec, record)["passed"])
        records = []
        for seed in range(8):
            records.append(
                {
                    "seed_gate": {"passed": seed < 6},
                    "point_estimates": {
                        "post_positive_mean_one_step_log_growth": 0.01,
                        "post_positive_minus_control_mean_one_step_log_growth": 0.02,
                        "post_positive_minus_pre_mean_one_step_log_growth": 0.03,
                    },
                }
            )
        gate = h.aggregate_gate(self.spec, records)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["seed_pass_count"], 6)
        records[5]["seed_gate"]["passed"] = False
        self.assertFalse(h.aggregate_gate(self.spec, records)["passed"])


if __name__ == "__main__":
    unittest.main()
