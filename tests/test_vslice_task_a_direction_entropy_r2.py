from __future__ import annotations
import unittest
import torch
from cb16_science.vslice import controlled_tasks as tasks
from cb16_science.vslice import qualification as q
from cb16_science.vslice import task_a_direction_entropy_r2 as r2

PREREGISTERED_SPEC_SHA256 = "d9e6095896bf6440d92478bb100839e1882c6e3e3925de894fb24268dc1474cd"

class TaskADirectionEntropyR2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = r2.load_spec()
        q.configure_deterministic_runtime()

    def test_spec_hash_and_validation(self):
        r2.validate_spec(self.spec)
        self.assertEqual(r2.spec_sha256(self.spec), PREREGISTERED_SPEC_SHA256)
        self.assertEqual(r2.arm_definition(self.spec, r2.BASELINE_ARM).direction_entropy_coefficient, 0.0)
        self.assertEqual(r2.arm_definition(self.spec, r2.ENTROPY_ARM).direction_entropy_coefficient, 0.005)

    def test_b900_schedule_is_exactly_balanced(self):
        env = tasks.task_a_environment(self.spec)
        self.assertEqual(len(env.positive_schedule), 900)
        self.assertEqual(len(env.control_schedule), 900)
        for i in range(3):
            self.assertEqual(env.positive_schedule.count((i, i)), 300)
        for i in range(3):
            for j in range(3):
                self.assertEqual(env.control_schedule.count((i, j)), 100)

    def test_four_learners_start_bitwise_identical(self):
        learners = r2.build_seed_learners(self.spec, 1201)
        bpos, bctl = learners[r2.BASELINE_ARM]
        epos, ectl = learners[r2.ENTROPY_ARM]
        q.require_identical_parameters(bpos, bctl)
        q.require_identical_parameters(epos, ectl)
        q.require_identical_parameters(bpos, epos)
        q.require_identical_parameters(bctl, ectl)
        self.assertTrue(torch.equal(bpos.sensory.projection, epos.sensory.projection))
        self.assertEqual(bpos.direction_entropy_coefficient, 0.0)
        self.assertEqual(epos.direction_entropy_coefficient, 0.005)

    def test_attribution_truth_table(self):
        good = {
            r2.BASELINE_ARM: {"positive_seed_pass_count": 2, "control_seed_pass_count": 0},
            r2.ENTROPY_ARM: {"positive_seed_pass_count": 6, "control_seed_pass_count": 1},
        }
        gate = r2.attribution_gate(self.spec, good)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["outcome"], "DIRECTION_EXPLORATION_SUPPORTED")
        weak = {
            r2.BASELINE_ARM: {"positive_seed_pass_count": 3, "control_seed_pass_count": 0},
            r2.ENTROPY_ARM: {"positive_seed_pass_count": 5, "control_seed_pass_count": 0},
        }
        self.assertEqual(r2.attribution_gate(self.spec, weak)["outcome"], "DIRECTION_ENTROPY_NOT_SUFFICIENT")
        bad_control = {
            r2.BASELINE_ARM: {"positive_seed_pass_count": 2, "control_seed_pass_count": 3},
            r2.ENTROPY_ARM: {"positive_seed_pass_count": 8, "control_seed_pass_count": 0},
        }
        self.assertEqual(r2.attribution_gate(self.spec, bad_control)["outcome"], "CONTROL_INVALID")

if __name__ == "__main__": unittest.main()
