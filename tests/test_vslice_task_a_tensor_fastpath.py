from __future__ import annotations
import copy
import unittest
import torch
from cb16_science.vslice import controlled_tasks as tasks
from cb16_science.vslice import qualification as q
from cb16_science.vslice import task_a_direction_entropy_r2 as r2
from cb16_science.vslice import vectorized_tasks as vt
from cb16_science.vslice.learner import LearnerContractError, OnPolicyOneStepBatch


class TaskATensorFastPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = copy.deepcopy(r2.load_spec())
        cls.env = tasks.task_a_environment(cls.spec)
        cls.authority = r2.parse_authority(cls.spec)
        cls.optimizer = r2.parse_optimizer(cls.spec)
        q.configure_deterministic_runtime()

    def build(self, seed: int, coefficient: float):
        torch.manual_seed(seed)
        return q.build_learner(
            self.authority, self.optimizer, direction_entropy_coefficient=coefficient
        )

    def assert_modules_exact(self, left, right) -> None:
        for a, b in zip(left.parameters(), right.parameters()):
            self.assertTrue(torch.equal(a.detach(), b.detach()))

    def test_reference_and_tensor_fast_path_are_bitwise_exact_for_r2_arms_and_modes(self) -> None:
        seed = 1201
        for arm in r2.ARM_ORDER:
            coefficient = r2.arm_definition(self.spec, arm).direction_entropy_coefficient
            for mode in (tasks.ARM_POSITIVE, tasks.ARM_CONTROL):
                with self.subTest(arm=arm, mode=mode):
                    reference = self.build(seed, coefficient)
                    fast = self.build(seed, coefficient)
                    q.require_identical_parameters(reference, fast)
                    stream = q._collection_seed(seed, "task_a", mode, 0)
                    torch.manual_seed(stream)
                    object_batch = vt.build_task_a_generation(reference, self.env, 0, mode=mode)
                    reference_report = reference.update(object_batch)
                    torch.manual_seed(stream)
                    tensor_batch = vt.build_task_a_one_step_batch(fast, self.env, 0, mode=mode)
                    fast_report = fast.update_one_step_batch(tensor_batch)
                    self.assertEqual(reference_report, fast_report)
                    self.assert_modules_exact(reference.actor, fast.actor)
                    self.assert_modules_exact(reference.critic, fast.critic)
                    self.assertTrue(torch.equal(reference.sensory.projection, fast.sensory.projection))

    def test_tensor_batch_replay_and_stale_generation_are_rejected(self) -> None:
        learner = self.build(1201, 0.005)
        torch.manual_seed(q._collection_seed(1201, "task_a", tasks.ARM_POSITIVE, 0))
        batch = vt.build_task_a_one_step_batch(learner, self.env, 0, mode=tasks.ARM_POSITIVE)
        learner.update_one_step_batch(batch)
        with self.assertRaisesRegex(LearnerContractError, "already consumed"):
            learner.update_one_step_batch(batch)
        torch.manual_seed(q._collection_seed(1201, "task_a", tasks.ARM_POSITIVE, 1))
        future = vt.build_task_a_one_step_batch(learner, self.env, 7, mode=tasks.ARM_POSITIVE)
        with self.assertRaisesRegex(LearnerContractError, "does not match learner generation"):
            learner.update_one_step_batch(future)

    def test_tensor_batch_is_revalidated_after_external_mutation(self) -> None:
        learner = self.build(1201, 0.005)
        torch.manual_seed(q._collection_seed(1201, "task_a", tasks.ARM_POSITIVE, 0))
        batch = vt.build_task_a_one_step_batch(learner, self.env, 0, mode=tasks.ARM_POSITIVE)
        batch.states[0, 0] = float("nan")
        with self.assertRaisesRegex(LearnerContractError, "finite floating"):
            learner.update_one_step_batch(batch)

    def test_tensor_batch_generation_snapshot_guard_is_preserved(self) -> None:
        learner = self.build(1201, 0.005)
        torch.manual_seed(q._collection_seed(1201, "task_a", tasks.ARM_POSITIVE, 0))
        batch = vt.build_task_a_one_step_batch(learner, self.env, 0, mode=tasks.ARM_POSITIVE)
        with torch.no_grad():
            next(learner.actor.parameters()).view(-1)[0].add_(1.0)
        with self.assertRaisesRegex(LearnerContractError, "generation boundary"):
            learner.update_one_step_batch(batch)

    def test_tensor_batch_fails_closed_on_malformed_inputs(self) -> None:
        states = torch.zeros(3, self.authority.sensory_z_dim + 3)
        directions = torch.tensor([0, 1, 2], dtype=torch.long)
        risks = torch.tensor([0.5, 0.0, 0.5])
        rewards = torch.zeros(3, dtype=torch.float64)
        good = dict(
            states=states, direction_indices=directions, requested_risks=risks,
            rewards=rewards, generation_id=0,
        )
        OnPolicyOneStepBatch(**good)
        cases = [
            {**good, "states": states.clone().fill_(float("nan"))},
            {**good, "direction_indices": torch.tensor([0, 3, 2])},
            {**good, "requested_risks": torch.tensor([0.5, 0.1, 0.5])},
            {**good, "rewards": torch.tensor([0.0, float("inf"), 0.0])},
            {**good, "generation_id": -1},
        ]
        for bad in cases:
            with self.subTest(bad=list(bad)):
                with self.assertRaises(LearnerContractError):
                    OnPolicyOneStepBatch(**bad)


if __name__ == "__main__":
    unittest.main()
