from __future__ import annotations

import copy
import unittest
from unittest import mock

from cb16_science.vslice import controlled_tasks as tasks
from cb16_science.vslice import joint_confirmation_r3 as r3
from cb16_science.vslice import qualification as q
from cb16_science.vslice import qualification_vectorized as qv
from cb16_science.vslice import task_a_direction_entropy_r2 as r2


SPEC_SHA256 = "ae0f5778a3d509eb9a31d657cfa29665a3eced0b5463b7e53b79aaefe94827c8"


class JointConfirmationR3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = r3.load_spec()

    def test_spec_hash_fresh_seeds_and_validation(self) -> None:
        self.assertEqual(r3.spec_sha256(self.spec), SPEC_SHA256)
        self.assertEqual(r3.paired_seeds(self.spec), tuple(range(2201, 2209)))
        self.assertTrue(set(r3.paired_seeds(self.spec)).isdisjoint(range(1201, 1209)))
        r3.validate_spec(self.spec)
        optimizer = r3.parse_optimizer(self.spec)
        self.assertEqual(optimizer.actor_lr, 0.001)
        self.assertEqual(optimizer.critic_lr, 0.001)

    def test_vectorized_helper_defaults_to_zero_entropy(self) -> None:
        authority = r2.parse_authority(self.spec)
        optimizer = r3.parse_optimizer(self.spec)
        env_a = tasks.task_a_environment(self.spec)
        env_b = tasks.task_b_environment(self.spec)
        seen = []
        original = q.build_learner

        def capture(*args, **kwargs):
            seen.append(kwargs.get("direction_entropy_coefficient", 0.0))
            return original(*args, **kwargs)

        with mock.patch.object(q, "build_learner", side_effect=capture), \
             mock.patch.object(qv, "_train_task_a", return_value={"generations": 0}), \
             mock.patch.object(qv, "_train_task_b", return_value={"generations": 0}):
            qv.run_paired_seed_vectorized(self.spec, 2201, authority, optimizer, env_a, env_b)
        self.assertEqual(seen, [0.0, 0.0, 0.0, 0.0])

    def test_r3_passes_entropy_to_all_four_paired_learners(self) -> None:
        authority = r2.parse_authority(self.spec)
        optimizer = r3.parse_optimizer(self.spec)
        env_a = tasks.task_a_environment(self.spec)
        env_b = tasks.task_b_environment(self.spec)
        seen = []
        original = q.build_learner

        def capture(*args, **kwargs):
            seen.append(kwargs.get("direction_entropy_coefficient", 0.0))
            return original(*args, **kwargs)

        with mock.patch.object(q, "build_learner", side_effect=capture), \
             mock.patch.object(qv, "_train_task_a", return_value={"generations": 0}), \
             mock.patch.object(qv, "_train_task_b", return_value={"generations": 0}):
            qv.run_paired_seed_vectorized(
                self.spec, 2201, authority, optimizer, env_a, env_b,
                direction_entropy_coefficient=r3.DIRECTION_ENTROPY_COEFFICIENT,
            )
        self.assertEqual(seen, [0.005, 0.005, 0.005, 0.005])

    def test_joint_gate_truth_table(self) -> None:
        records = [self._record(True, False, 0.40, True, False, 0.04, 0.04) for _ in range(8)]
        passed = r3.joint_gate(self.spec, records)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["outcome"], "JOINT_LEARNABILITY_CONFIRMED")

        control_bad = copy.deepcopy(records)
        control_bad[0]["task_a"]["control"]["seed_gate"]["passed"] = True
        control_bad[1]["task_a"]["control"]["seed_gate"]["passed"] = True
        failed = r3.joint_gate(self.spec, control_bad)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["outcome"], "CONTROL_INVALID")

        confirmation_bad = copy.deepcopy(records)
        confirmation_bad[0]["task_b"]["positive"]["seed_gate"]["passed"] = False
        confirmation_bad[1]["task_b"]["positive"]["seed_gate"]["passed"] = False
        failed = r3.joint_gate(self.spec, confirmation_bad)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["outcome"], "JOINT_CONFIRMATION_FAILED")

    @staticmethod
    def _record(a_pos, a_ctl, a_improvement, b_pos, b_ctl, b_improvement, b_diff):
        return {
            "task_a": {
                "positive": {"seed_gate": {"passed": a_pos}},
                "control": {"seed_gate": {"passed": a_ctl}},
                "paired": {
                    "positive_target_mae_improvement": a_improvement,
                    "positive_post_target_mae_strictly_better_than_control": True,
                },
            },
            "task_b": {
                "positive": {"seed_gate": {"passed": b_pos}},
                "control": {"seed_gate": {"passed": b_ctl}},
                "paired": {
                    "positive_true_delayed_log_growth_improvement": b_improvement,
                    "positive_minus_control_true_delayed_log_growth": b_diff,
                },
            },
        }


if __name__ == "__main__":
    unittest.main()
