from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cb16_science.vslice import evaluation_adapter_transfer_r1 as r1
from cb16_science.vslice.policy import ActorOutput


ROOT = Path(__file__).resolve().parents[1]


class _FakeActor:
    def __init__(self, output: ActorOutput) -> None:
        self._output = output

    def forward(self, states: torch.Tensor) -> ActorOutput:
        if states.shape[0] != self._output.direction_logits.shape[0]:
            raise AssertionError("fake actor batch mismatch")
        return self._output


class _FakeLearner:
    def __init__(self, output: ActorOutput) -> None:
        self.actor = _FakeActor(output)

    def _require_generation_snapshot(self) -> None:
        return None

def _uniform_symmetric_output(batch: int) -> ActorOutput:
    logits = torch.zeros(batch, 3, dtype=torch.float32)
    # Positive finite parameters; exact values are not important for symmetry.
    alpha = torch.full((batch,), 2.0, dtype=torch.float32)
    beta = torch.full((batch,), 2.0, dtype=torch.float32)
    return ActorOutput(
        direction_logits=logits,
        short_alpha=alpha,
        short_beta=beta,
        long_alpha=alpha.clone(),
        long_beta=beta.clone(),
    )


class R1SpecAlignmentTests(unittest.TestCase):
    def test_preregistered_spec_and_parent_validate(self) -> None:
        spec = r1.load_spec()
        parent = r1.load_parent_spec()
        r1.validate_r1_against_parent(spec, parent)
        self.assertEqual(spec["claim_scope"]["changed_axis"], "evaluation_adapter_only")
        self.assertEqual(spec["research_series_policy"]["freshness_status"], "NOT_FRESH_CONFIRMATION")

    def test_allowlist_entry_is_exact(self) -> None:
        data = json.loads((ROOT / "config" / "cb16_science_allowlist.json").read_text())
        entry = data["entrypoints"][r1.RESULT_COMMAND]
        self.assertEqual(
            entry["argv"][-2:],
            ["-m", "cb16_science.vslice.evaluation_adapter_transfer_r1"],
        )
        self.assertEqual(
            sorted(entry["produces"]),
            ["REPORT.md", "RESULT.json", "experiment_spec.json"],
        )


class PolicyDistributionEvaluatorTests(unittest.TestCase):
    def test_zero_return_has_exact_zero_expected_reward(self) -> None:
        learner = _FakeLearner(_uniform_symmetric_output(3))
        states = torch.zeros(3, 5)
        result = r1.evaluate_policy_distribution(
            learner,
            states,
            np.array([100.0, 100.0, 100.0]),
            np.array([100.0, 100.0, 100.0]),
        )
        self.assertEqual(result["mean_expected_one_step_log_growth"], 0.0)
        self.assertEqual(result["hourly_expected_log_growth"], [0.0, 0.0, 0.0])

    def test_symmetric_policy_is_invariant_to_return_sign(self) -> None:
        learner = _FakeLearner(_uniform_symmetric_output(2))
        states = torch.zeros(2, 5)
        result = r1.evaluate_policy_distribution(
            learner,
            states,
            np.array([100.0, 100.0]),
            np.array([110.0, 90.0]),
        )
        values = result["hourly_expected_log_growth"]
        self.assertAlmostEqual(values[0], values[1], places=12)
        self.assertLess(values[0], 0.0)

    def test_quadrature_is_deterministic_and_well_normalized(self) -> None:
        learner = _FakeLearner(_uniform_symmetric_output(4))
        states = torch.zeros(4, 5)
        open_ = np.full(4, 100.0)
        close = np.array([101.0, 99.0, 103.0, 97.0])
        a = r1.evaluate_policy_distribution(learner, states, open_, close)
        b = r1.evaluate_policy_distribution(learner, states, open_, close)
        self.assertEqual(a, b)
        diag = a["diagnostics"]
        self.assertLess(diag["max_abs_short_quadrature_mass_error"], 1e-12)
        self.assertLess(diag["max_abs_long_quadrature_mass_error"], 1e-12)
        self.assertAlmostEqual(diag["mean_direction_entropy"], math.log(3.0), places=12)


class AggregateGateTests(unittest.TestCase):
    @staticmethod
    def _record(seed: int, *, adapter: bool, distribution: bool, sign: float) -> dict:
        return {
            "seed": seed,
            "seed_gates": {
                "adapter_transfer": adapter,
                "distribution_level_relation": distribution,
            },
            "point_estimates": {
                "adapter_delta_true_minus_control": sign * 0.01,
                "adapter_delta_true_minus_pre": sign * 0.02,
                "expected_positive_minus_control": sign * 0.03,
                "expected_positive_minus_pre": sign * 0.04,
                "expected_positive_mean_log_growth": sign * 0.05,
            },
        }

    def test_six_of_eight_positive_records_pass_both_aggregate_gates(self) -> None:
        records = [
            self._record(i, adapter=i < 6, distribution=i < 6, sign=1.0 if i < 6 else -1.0)
            for i in range(8)
        ]
        gates = r1.aggregate_gates(records)
        self.assertTrue(gates["adapter"]["passed"])
        self.assertTrue(gates["distribution"]["passed"])

    def test_five_of_eight_do_not_pass_seed_count(self) -> None:
        records = [
            self._record(i, adapter=i < 5, distribution=i < 5, sign=1.0)
            for i in range(8)
        ]
        gates = r1.aggregate_gates(records)
        self.assertFalse(gates["adapter"]["passed"])
        self.assertFalse(gates["distribution"]["passed"])


if __name__ == "__main__":
    unittest.main()
