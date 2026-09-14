from __future__ import annotations

import unittest

import torch

from cb16_science.vslice import controlled_tasks as tasks
from cb16_science.vslice import qualification as q
from cb16_science.vslice import vectorized_tasks as vt
from cb16_science.vslice.contracts import Direction, NominalAction
from cb16_science.vslice.physics import execute_transition
from cb16_science.vslice.policy import direction_index


class VectorizedTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = q.load_spec()
        cls.authority = q.parse_authority(cls.spec)
        cls.optimizer = q.parse_optimizer(cls.spec)
        cls.env_a = tasks.task_a_environment(cls.spec)
        cls.env_b = tasks.task_b_environment(cls.spec)
        q.configure_deterministic_runtime()

    def _learner(self, seed: int = 1234):
        torch.manual_seed(seed)
        return q.build_learner(self.authority, self.optimizer)

    def test_task_a_vectorized_rewards_match_canonical_physics(self) -> None:
        # Include risk=1.0 in both directions so the R0 kappa=0.1 case
        # actually exercises cost-aware clipping/bisection instead of only
        # comparing interior feasible targets.
        actions = (
            NominalAction(Direction.SHORT, 0.2),
            NominalAction(Direction.SHORT, 0.8),
            NominalAction(Direction.SHORT, 1.0),
            NominalAction(Direction.FLAT, 0.0),
            NominalAction(Direction.LONG, 0.2),
            NominalAction(Direction.LONG, 0.8),
            NominalAction(Direction.LONG, 1.0),
        )
        actual = []
        indices = []
        risks = []
        expected = []
        for case_index, case in enumerate(self.env_a.cases):
            for action in actions:
                actual.append(case_index)
                indices.append(direction_index(action.direction))
                risks.append(action.requested_risk)
                expected.append(
                    execute_transition(
                        case.truth,
                        action,
                        self.env_a.open_next,
                        self.env_a.close_next,
                        self.env_a.config,
                    ).reward
                )
        observed = vt.task_a_reward_batch(
            self.env_a,
            torch.tensor(actual, dtype=torch.long),
            torch.tensor(indices, dtype=torch.long),
            # The fixture actions above are Python float authority values. Keep
            # them in float64 here so the equivalence check measures Physics,
            # not an intentional float32 quantization of 0.2/0.8.
            torch.tensor(risks, dtype=torch.float64),
        )
        self.assertEqual(observed.dtype, torch.float64)
        for got, want in zip(observed.tolist(), expected):
            self.assertAlmostEqual(got, want, places=12)

    def test_task_a_vectorized_generation_is_consumable(self) -> None:
        learner = self._learner()
        torch.manual_seed(99)
        batch = vt.build_task_a_generation(
            learner, self.env_a, 0, mode=tasks.ARM_POSITIVE
        )
        self.assertEqual(len(batch), len(self.env_a.positive_schedule))
        self.assertTrue(all(t.generation_id == 0 for t in batch))
        report = learner.update(batch)
        self.assertEqual(report.generation_id, 0)
        self.assertEqual(report.next_generation_id, 1)

    def test_task_b_vectorized_rewards_match_canonical_physics(self) -> None:
        learner = self._learner()
        torch.manual_seed(101)
        batch = vt.build_task_b_generation(
            learner, self.env_b, 0, mode=tasks.ARM_POSITIVE, seed=1201
        )
        self.assertEqual(len(batch), len(self.env_b.cue_schedule))
        for cue, trajectory in zip(self.env_b.cue_schedule, batch):
            step0 = trajectory.steps[0]
            step1 = trajectory.steps[1]
            canonical0 = execute_transition(
                self.env_b.initial_truth,
                step0.nominal_action,
                self.env_b.step0_open_next,
                self.env_b.step0_close_next,
                self.env_b.config,
            )
            gap = tasks.task_b_gap(self.env_b, cue)
            canonical1 = execute_transition(
                canonical0.truth_next,
                step1.nominal_action,
                gap,
                gap,
                self.env_b.config,
            )
            self.assertEqual(step0.reward, 0.0)
            self.assertAlmostEqual(step1.reward, canonical1.reward, places=12)

    def test_task_b_control_preserves_reward_multiset(self) -> None:
        positive = self._learner(555)
        control = self._learner(555)
        torch.manual_seed(777)
        pos = vt.build_task_b_generation(
            positive, self.env_b, 0, mode=tasks.ARM_POSITIVE, seed=1202
        )
        torch.manual_seed(777)
        ctl = vt.build_task_b_generation(
            control, self.env_b, 0, mode=tasks.ARM_CONTROL, seed=1202
        )
        pos_rewards = sorted(t.steps[1].reward for t in pos)
        ctl_rewards = sorted(t.steps[1].reward for t in ctl)
        self.assertEqual(pos_rewards, ctl_rewards)
        self.assertTrue(any(a.steps[1].reward != b.steps[1].reward for a, b in zip(pos, ctl)))

    def test_sample_batch_semantics(self) -> None:
        learner = self._learner()
        states = torch.stack(
            [
                learner.build_state(
                    torch.from_numpy(self.env_a.market),
                    tasks.task_a_account_state(self.env_a, case),
                ).vector
                for case in self.env_a.cases
            ]
        )
        torch.manual_seed(1)
        directions, risks = vt.sample_action_batch(learner, states)
        self.assertEqual(tuple(directions.shape), (3,))
        self.assertEqual(tuple(risks.shape), (3,))
        flat = directions == 1
        self.assertTrue(bool((risks[flat] == 0.0).all()))
        self.assertTrue(bool(((risks[~flat] > 0.0) & (risks[~flat] < 1.0)).all()))


if __name__ == "__main__":
    unittest.main()
