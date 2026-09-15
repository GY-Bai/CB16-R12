"""Implementation tests for VS-D R2A objective-balance diagnostic."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from cb16_science.vslice import historical_market_canary_r0 as r0
from cb16_science.vslice import objective_balance_entropy_gradient_r2a as r2a
from cb16_science.vslice import qualification as q


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "config" / "experiments" / "r12_vs_d_objective_balance_entropy_gradient_r2a.json"
EXPECTED_SPEC_SHA = "756cb79b8c69cf0618c6134c3e29e949080c06d67bf72e4bab87d02c44279823"


class ObjectiveBalanceR2AImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC_PATH.read_text())
        q.configure_deterministic_runtime()

    def test_preregistered_spec_bytes_are_unchanged(self) -> None:
        self.assertEqual(hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(), EXPECTED_SPEC_SHA)
        r2a.validate_spec(self.spec)

    def _learner(self, seed: int = 3101):
        torch.manual_seed(seed)
        return q.build_learner(
            r0.parse_authority(self.spec),
            r0.parse_optimizer(self.spec),
            direction_entropy_coefficient=0.005,
        )
    def _synthetic_batch_inputs(self, n: int = 24):
        states = torch.linspace(-0.5, 0.5, n * 35, dtype=torch.float32).reshape(n, 35)
        open_next = np.linspace(100.0, 102.0, n, dtype=np.float64)
        signed = np.where(np.arange(n) % 2 == 0, 0.001, -0.0015)
        close_next = open_next * (1.0 + signed)
        return states, open_next, close_next

    def test_measurement_is_finite_and_observational(self) -> None:
        learner = self._learner()
        states, open_next, close_next = self._synthetic_batch_inputs()
        torch.manual_seed(r0._collection_seed(3101, r0.ARM_POSITIVE_ID, 0))
        from cb16_science.vslice import vectorized_tasks as vt
        directions, risks = vt.sample_action_batch(learner, states)
        rewards = r0.historical_reward_batch(directions, risks, open_next, close_next)
        rng_before = torch.random.get_rng_state().clone()
        params_before = [p.detach().clone() for p in learner.actor.parameters()]
        metrics = r2a.measure_objective_balance(learner, states, directions, risks, rewards)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        for before, after in zip(params_before, learner.actor.parameters()):
            self.assertTrue(torch.equal(before, after.detach()))

    def test_diagnostic_training_is_bitwise_parent_r0_equivalent_for_one_generation(self) -> None:
        baseline = self._learner()
        observed = self._learner()
        q.require_identical_parameters(baseline, observed)
        states, open_next, close_next = self._synthetic_batch_inputs()
        r0._train_learner(
            baseline, states, open_next, close_next,
            seed=3101, arm_id=r0.ARM_POSITIVE_ID, generations=1,
            nominal_exposure_budget=1.0,
        )
        from cb16_science.vslice import vectorized_tasks as vt
        from cb16_science.vslice.learner import OnPolicyOneStepBatch
        torch.manual_seed(r0._collection_seed(3101, r0.ARM_POSITIVE_ID, 0))
        directions, risks = vt.sample_action_batch(observed, states)
        rewards = r0.historical_reward_batch(directions, risks, open_next, close_next)
        r2a.measure_objective_balance(observed, states, directions, risks, rewards)
        observed.update_one_step_batch(OnPolicyOneStepBatch(
            states=states, direction_indices=directions, requested_risks=risks,
            rewards=rewards, generation_id=0,
        ))
        for left, right in zip(baseline.actor.parameters(), observed.actor.parameters()):
            self.assertTrue(torch.equal(left.detach(), right.detach()))
        for left, right in zip(baseline.critic.parameters(), observed.critic.parameters()):
            self.assertTrue(torch.equal(left.detach(), right.detach()))

    def test_discrete_q25_is_exact_index_63(self) -> None:
        records = []
        for generation in range(256):
            records.append({
                "log10_entropy_to_direction_pg_grad_ratio": float(generation - 100),
                "mean_direction_entropy": 1.0,
                "std_reward": 0.1,
            })
        summary = r2a._summarize_generations(records)
        self.assertEqual(summary["discrete_q25_log10_ratio_index63"], -37.0)
        self.assertAlmostEqual(summary["fraction_generations_entropy_grad_gt_direction_pg"], 155 / 256)

    def test_allowlist_entry_is_exact_and_not_dry_run(self) -> None:
        allowlist = json.loads((ROOT / "config" / "cb16_science_allowlist.json").read_text())
        entry = allowlist["entrypoints"][r2a.RESULT_COMMAND]
        self.assertEqual(entry["argv"][-1], "cb16_science.vslice.objective_balance_entropy_gradient_r2a")
        self.assertEqual(sorted(entry["produces"]), ["REPORT.md", "RESULT.json", "experiment_spec.json"])
        self.assertFalse(entry.get("dry_run_only", False))


if __name__ == "__main__":
    unittest.main()
