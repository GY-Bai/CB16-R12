from __future__ import annotations

import math
import unittest

import torch

from cb16_science.vslice.contracts import ContractError, Direction
from cb16_science.vslice.learner import LearnerContractError, OnPolicyLearner
from cb16_science.vslice.policy import Actor, ValueCritic, direction_index
from cb16_science.vslice.sensory import FrozenSensory
from cb16_science.vslice.trajectory import Trajectory, TrajectoryStep


def make_learner(seed: int, coefficient: float) -> OnPolicyLearner:
    torch.manual_seed(seed)
    return OnPolicyLearner(
        FrozenSensory(),
        Actor(),
        ValueCritic(),
        actor_lr=1e-3,
        critic_lr=1e-3,
        direction_entropy_coefficient=coefficient,
    )


def one_step_batch(learner: OnPolicyLearner, reward: float = 0.0) -> tuple[Trajectory, ...]:
    state = torch.linspace(-0.2, 0.2, learner.state_dim, dtype=torch.float32)
    trajectory = Trajectory(
        steps=(
            TrajectoryStep(
                state=state,
                direction=Direction.FLAT,
                requested_risk=0.0,
                reward=reward,
                terminal=False,
                truncated=False,
                generation_id=0,
            ),
        ),
        generation_id=0,
        complete=True,
    )
    return (trajectory,)


class DirectionEntropyLearnerTests(unittest.TestCase):
    def test_zero_coefficient_reproduces_existing_objective_exactly(self) -> None:
        default = make_learner(1201, 0.0)
        explicit = make_learner(1201, 0.0)
        batch = one_step_batch(default, reward=0.125)
        new_loss = default.actor_objective(batch)
        states = torch.stack([step.state for trajectory in batch for step in trajectory.steps])
        indices = torch.tensor(
            [direction_index(step.direction) for trajectory in batch for step in trajectory.steps],
            dtype=torch.long,
        )
        risks = torch.tensor(
            [step.requested_risk for trajectory in batch for step in trajectory.steps],
            dtype=states.dtype,
        )
        returns = torch.cat([trajectory.returns_to_go() for trajectory in batch])
        values = default.critic(states)
        old_formula = -(default.actor.log_prob_batch(states, indices, risks) * (returns - values.detach())).mean()
        self.assertTrue(torch.equal(new_loss, old_formula))
        self.assertTrue(torch.equal(default.actor_objective(batch), explicit.actor_objective(batch)))
        self.assertTrue(torch.equal(default.critic_objective(batch), explicit.critic_objective(batch)))

    def test_positive_coefficient_changes_only_actor_loss_by_direction_entropy(self) -> None:
        baseline = make_learner(1202, 0.0)
        entropy = make_learner(1202, 0.005)
        batch = one_step_batch(baseline, reward=-0.05)
        states = torch.stack([step.state for trajectory in batch for step in trajectory.steps])
        baseline_loss = baseline.actor_objective(batch)
        entropy_loss = entropy.actor_objective(batch)
        direction_entropy = entropy.actor.direction_entropy_batch(states).mean()
        expected = baseline_loss - 0.005 * direction_entropy
        torch.testing.assert_close(entropy_loss, expected, rtol=0.0, atol=1e-8)
        self.assertTrue(torch.equal(baseline.critic_objective(batch), entropy.critic_objective(batch)))

    def test_zero_advantage_entropy_gradient_targets_direction_not_beta_output_rows(self) -> None:
        torch.manual_seed(1203)
        actor = Actor()
        critic = ValueCritic()
        with torch.no_grad():
            for parameter in critic.parameters():
                parameter.zero_()
            final = actor.mlp[-1]
            final.bias[:3].copy_(torch.tensor([1.0, 0.25, -0.5], dtype=final.bias.dtype))
        learner = OnPolicyLearner(
            FrozenSensory(),
            actor,
            critic,
            actor_lr=1e-3,
            critic_lr=1e-3,
            direction_entropy_coefficient=0.005,
        )
        batch = one_step_batch(learner, reward=0.0)
        learner.actor_optimizer.zero_grad(set_to_none=True)
        learner.actor_objective(batch).backward()
        for parameter in learner.critic.parameters():
            self.assertTrue(parameter.grad is None or bool(torch.all(parameter.grad == 0)))
        final = learner.actor.mlp[-1]
        self.assertIsNotNone(final.bias.grad)
        self.assertGreater(float(final.bias.grad[:3].abs().max()), 0.0)
        self.assertEqual(float(final.bias.grad[3:].abs().max()), 0.0)
        self.assertEqual(float(final.weight.grad[3:].abs().max()), 0.0)

    def test_direction_entropy_coefficient_must_be_finite_and_nonnegative(self) -> None:
        for bad in (-1e-9, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ContractError):
                    make_learner(1204, bad)
        self.assertTrue(math.isclose(make_learner(1204, 0.005).direction_entropy_coefficient, 0.005))


if __name__ == "__main__":
    unittest.main()
