from __future__ import annotations

import copy
import unittest

import torch

from cb16_science.vslice import qualification as q
from cb16_science.vslice import task_a_variance_r1 as r1
from cb16_science.vslice import controlled_tasks as tasks


class TaskAVarianceR1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = r1.load_spec()
        q.configure_deterministic_runtime()

    def test_committed_spec_validates(self) -> None:
        r1.validate_spec(self.spec)
        self.assertEqual(r1.arm_definition(self.spec, r1.BASELINE_ARM).batch_trajectories, 90)
        self.assertEqual(r1.arm_definition(self.spec, r1.HIGH_BATCH_ARM).batch_trajectories, 900)
        self.assertEqual(r1.paired_seeds(self.spec), tuple(range(1201, 1209)))

    def test_arm_schedules_are_exactly_balanced(self) -> None:
        baseline_env = tasks.task_a_environment(r1._arm_spec(self.spec, r1.BASELINE_ARM))
        high_env = tasks.task_a_environment(r1._arm_spec(self.spec, r1.HIGH_BATCH_ARM))
        self.assertEqual(len(baseline_env.positive_schedule), 90)
        self.assertEqual(len(baseline_env.control_schedule), 90)
        self.assertEqual(len(high_env.positive_schedule), 900)
        self.assertEqual(len(high_env.control_schedule), 900)
        self.assertEqual(set(baseline_env.control_schedule), {(a, b) for a in range(3) for b in range(3)})
        self.assertEqual(set(high_env.control_schedule), {(a, b) for a in range(3) for b in range(3)})
        for pair in {(a, b) for a in range(3) for b in range(3)}:
            self.assertEqual(baseline_env.control_schedule.count(pair), 10)
            self.assertEqual(high_env.control_schedule.count(pair), 100)

    def test_b90_and_b900_seeded_initializations_are_bitwise_identical(self) -> None:
        authority = r1.parse_authority(self.spec)
        optimizer = q.parse_optimizer(self.spec)
        torch.manual_seed(1201)
        baseline = q.build_learner(authority, optimizer)
        torch.manual_seed(1201)
        high = q.build_learner(authority, optimizer)
        q.require_identical_parameters(baseline, high)
        self.assertTrue(torch.equal(baseline.sensory.projection, high.sensory.projection))

    def test_attribution_gate_supports_only_material_controlled_improvement(self) -> None:
        aggregates = {
            r1.BASELINE_ARM: {"positive_seed_pass_count": 2, "control_seed_pass_count": 0},
            r1.HIGH_BATCH_ARM: {"positive_seed_pass_count": 6, "control_seed_pass_count": 1},
        }
        gate = r1.attribution_gate(self.spec, aggregates)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["outcome"], "VARIANCE_SUPPORTED")
        self.assertEqual(gate["components"]["high_batch_minus_baseline_positive_pass_count"], 4)

    def test_attribution_gate_reports_not_sufficient_without_rescue(self) -> None:
        aggregates = {
            r1.BASELINE_ARM: {"positive_seed_pass_count": 3, "control_seed_pass_count": 0},
            r1.HIGH_BATCH_ARM: {"positive_seed_pass_count": 5, "control_seed_pass_count": 0},
        }
        gate = r1.attribution_gate(self.spec, aggregates)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["outcome"], "VARIANCE_NOT_SUFFICIENT")

    def test_attribution_gate_rejects_invalid_control(self) -> None:
        aggregates = {
            r1.BASELINE_ARM: {"positive_seed_pass_count": 2, "control_seed_pass_count": 3},
            r1.HIGH_BATCH_ARM: {"positive_seed_pass_count": 7, "control_seed_pass_count": 0},
        }
        gate = r1.attribution_gate(self.spec, aggregates)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["outcome"], "CONTROL_INVALID")

    def test_tiny_execution_uses_same_pre_behavior_and_both_controls(self) -> None:
        diagnostic = copy.deepcopy(self.spec)
        diagnostic["task_a"]["generations"] = 1
        diagnostic["task_a"]["arms"][r1.BASELINE_ARM] = {
            "batch_trajectories": 18,
            "positive_per_actual_account": 6,
            "control_per_actual_observed_pair": 2,
        }
        diagnostic["task_a"]["arms"][r1.HIGH_BATCH_ARM] = {
            "batch_trajectories": 90,
            "positive_per_actual_account": 30,
            "control_per_actual_observed_pair": 10,
        }
        record = r1.run_seed(diagnostic, 1201)
        baseline = record["arms"][r1.BASELINE_ARM]
        high = record["arms"][r1.HIGH_BATCH_ARM]
        self.assertEqual(baseline["positive"]["pre"], high["positive"]["pre"])
        self.assertEqual(baseline["control"]["pre"], high["control"]["pre"])
        self.assertEqual(baseline["positive"]["diagnostics"]["generations"], 1)
        self.assertEqual(high["control"]["diagnostics"]["generations"], 1)

    def test_formal_validation_rejects_entropy_rescue(self) -> None:
        changed = copy.deepcopy(self.spec)
        changed["optimizer"]["entropy_bonus"] = 0.01
        with self.assertRaises(Exception):
            r1.validate_spec(changed)


if __name__ == "__main__":
    unittest.main()
