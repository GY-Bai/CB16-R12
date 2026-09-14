"""Trajectory / return-to-go / generation-boundary regression tests: 4, 18-31.

Required by ``docs/tasks/R12_VS_B_LEARNER_SPINE_R0.md`` (sections "Trajectory /
returns" and "Generation boundary / update"), plus required test 4 (the frozen
sensory buffer survives a valid update).

    .venv/bin/python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

try:  # CPU PyTorch lives in the task venv; a bare interpreter records a skip.
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without PyTorch
    raise unittest.SkipTest(f"CPU PyTorch is required for the R12 VS-B learner spine: {exc}")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "science") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "science"))

from cb16_science.vslice import learner as learner_module  # noqa: E402
from cb16_science.vslice.contracts import (  # noqa: E402
    AccountState,
    AccountTruth,
    ContractError,
    Direction,
    NominalAction,
    PhysicsConfig,
    account_state_from_truth,
)
from cb16_science.vslice.learner import LearnerContractError, OnPolicyLearner  # noqa: E402
from cb16_science.vslice.physics import execute_transition  # noqa: E402
from cb16_science.vslice.policy import Actor, ValueCritic  # noqa: E402
from cb16_science.vslice.sensory import MARKET_CHANNELS, Z_DIM, FrozenSensory  # noqa: E402
from cb16_science.vslice.trajectory import (  # noqa: E402
    Trajectory,
    TrajectoryStep,
    returns_to_go,
)

CONTEXT_LENGTH = 64

#: A bounded three-step VS-A fixture: one retained price path, three actions.
PHYSICS_ACTIONS = (
    NominalAction(direction=Direction.LONG, requested_risk=0.5),
    NominalAction(direction=Direction.FLAT, requested_risk=0.0),
    NominalAction(direction=Direction.SHORT, requested_risk=0.25),
)
PHYSICS_PRICES = ((101.0, 101.5), (102.0, 101.0), (101.0, 100.4))
PHYSICS_EQUITY = 1000.0


def market_window(seed: int = 12012) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(CONTEXT_LENGTH, MARKET_CHANNELS, generator=generator) * 0.05


def account(signed_exposure: float = 0.25) -> AccountState:
    return AccountState(
        signed_exposure=signed_exposure,
        survival_cushion=1.0,
        new_risk_capacity=1.0 - abs(signed_exposure),
    )


def make_learner(*, seed: int = 0) -> OnPolicyLearner:
    torch.manual_seed(seed)
    return OnPolicyLearner(FrozenSensory(), Actor(), ValueCritic())


def make_trajectory(
    learner: OnPolicyLearner,
    generation_id: int = 0,
    rewards: tuple = (0.5, -0.25, 2.0),
    *,
    complete: bool = True,
    terminal: bool = False,
    truncated: bool = False,
) -> Trajectory:
    """A synthetic complete-or-incomplete trajectory with hand-set rewards."""

    state = learner.build_state(market_window(), account()).vector
    last = len(rewards) - 1
    steps = tuple(
        TrajectoryStep(
            state=state,
            direction=Direction.LONG if position % 2 == 0 else Direction.SHORT,
            requested_risk=0.4 if position % 2 == 0 else 0.6,
            reward=reward,
            terminal=terminal and position == last,
            truncated=truncated and position == last,
            generation_id=generation_id,
        )
        for position, reward in enumerate(rewards)
    )
    return Trajectory(steps=steps, generation_id=generation_id, complete=complete)


def physics_records() -> tuple:
    """Three real VS-A transitions from one fixed price path."""

    truth = AccountTruth(equity=PHYSICS_EQUITY, quantity=0.0, mark_price=100.0)
    records = []
    for (open_next, close_next), action in zip(PHYSICS_PRICES, PHYSICS_ACTIONS):
        record = execute_transition(truth, action, open_next, close_next, PhysicsConfig())
        truth = record.truth_next
        records.append(record)
    return tuple(records), truth


def make_physics_trajectory(
    learner: OnPolicyLearner,
    generation_id: int = 0,
    *,
    complete: bool = True,
    terminal: bool = False,
    truncated: bool = False,
) -> Trajectory:
    records, _ = physics_records()
    state = learner.build_state(market_window(), account()).vector
    last = len(records) - 1
    steps = tuple(
        TrajectoryStep(
            state=state,
            direction=record.nominal_action.direction,
            requested_risk=record.nominal_action.requested_risk,
            reward=record.reward,
            terminal=terminal and position == last,
            truncated=truncated and position == last,
            generation_id=generation_id,
        )
        for position, record in enumerate(records)
    )
    return Trajectory(steps=steps, generation_id=generation_id, complete=complete)


def snapshot(module: torch.nn.Module) -> list:
    return [parameter.detach().clone() for parameter in module.parameters()]


def legacy_aggregate_statistics(module: torch.nn.Module) -> list:
    """The retired ``(numel, sum, abs_sum)`` fingerprint, kept for the adversarial test."""

    statistics = []
    for parameter in module.parameters():
        values = parameter.detach().to(torch.float64)
        statistics.append((int(values.numel()), float(values.sum()), float(values.abs().sum())))
    return statistics


def swap_two_unequal_values_in_place(parameter: torch.Tensor) -> None:
    """Balanced mutation: swap two unequal entries, preserving sum and abs_sum exactly."""

    with torch.no_grad():
        flat = parameter.reshape(-1)
        moved = flat[0].clone()
        partner = int((flat != flat[0]).nonzero(as_tuple=False)[0, 0].item())
        flat[0].copy_(flat[partner])
        flat[partner].copy_(moved)


class ReturnToGoTests(unittest.TestCase):
    def test_18_reverse_cumulative_return_to_go_known_answer(self):
        values = returns_to_go([0.5, -0.25, 2.0])
        self.assertEqual(values.dtype, torch.float64)
        self.assertEqual(values.tolist(), [2.25, 1.75, 2.0])
        self.assertEqual(returns_to_go([0.125]).tolist(), [0.125])

        # Hand-computed running sums for a sequence that is not binary-exact.
        running = returns_to_go([0.1, 0.2, 0.3])
        self.assertAlmostEqual(running[0].item(), 0.6, places=12)
        self.assertAlmostEqual(running[1].item(), 0.5, places=12)
        self.assertAlmostEqual(running[2].item(), 0.3, places=12)

        # The helper is exactly the reverse cumulative sum of the rewards.
        rewards = [0.3, -0.7, 1.1, 0.05]
        expected = [sum(rewards[position:]) for position in range(len(rewards))]
        for value, hand in zip(returns_to_go(rewards).tolist(), expected):
            self.assertAlmostEqual(value, hand, places=12)

        with self.assertRaises(ContractError):
            returns_to_go([])
        with self.assertRaises(ContractError):
            returns_to_go([0.1, float("nan")])
        with self.assertRaises(ContractError):
            returns_to_go([0.1, float("inf")])

    def test_19_vs_a_log_equity_rewards_telescope(self):
        records, final_truth = physics_records()
        rewards = [record.reward for record in records]
        terminal_log_equity = math.log(final_truth.equity / PHYSICS_EQUITY)
        self.assertAlmostEqual(sum(rewards), terminal_log_equity, places=12)

        values = returns_to_go(rewards)
        self.assertAlmostEqual(values[0].item(), terminal_log_equity, places=12)

        equities = [PHYSICS_EQUITY] + [record.truth_next.equity for record in records]
        for position, value in enumerate(values.tolist()):
            self.assertAlmostEqual(
                value, math.log(equities[-1] / equities[position]), places=12
            )


class TrajectoryContractTests(unittest.TestCase):
    def test_20_terminal_and_truncation_stay_distinct(self):
        learner = make_learner()
        state = learner.build_state(market_window(), account()).vector

        # A declared terminal endpoint is a valid complete trajectory.
        report = learner.update([make_trajectory(learner, terminal=True)])
        self.assertEqual(report.generation_id, 0)
        self.assertEqual(report.next_generation_id, 1)

        # Truncation is not a terminal reward event and is rejected.
        truncated = make_trajectory(learner, generation_id=1, complete=False, truncated=True)
        self.assertTrue(truncated.truncated)
        self.assertFalse(truncated.terminal)
        with self.assertRaises(LearnerContractError) as context:
            learner.update([truncated])
        self.assertIn("truncated", str(context.exception))

        # Terminal and truncated cannot both hold on one step.
        with self.assertRaises(ContractError):
            TrajectoryStep(
                state=state,
                direction=Direction.SHORT,
                requested_risk=0.5,
                reward=0.1,
                terminal=True,
                truncated=True,
                generation_id=1,
            )

        # A terminal step may only be the declared endpoint.
        with self.assertRaises(ContractError):
            Trajectory(
                steps=(
                    TrajectoryStep(
                        state=state,
                        direction=Direction.LONG,
                        requested_risk=0.5,
                        reward=0.1,
                        terminal=True,
                        truncated=False,
                        generation_id=1,
                    ),
                    TrajectoryStep(
                        state=state,
                        direction=Direction.FLAT,
                        requested_risk=0.0,
                        reward=0.1,
                        terminal=False,
                        truncated=False,
                        generation_id=1,
                    ),
                ),
                generation_id=1,
                complete=True,
            )

    def test_21_empty_incomplete_truncated_mixed_and_non_finite_batches_are_rejected(self):
        learner = make_learner()

        with self.assertRaises(LearnerContractError) as context:
            learner.update([])
        self.assertIn("empty", str(context.exception))
        with self.assertRaises(LearnerContractError):
            learner.update(())
        # One bare trajectory is not a batch.
        with self.assertRaises(LearnerContractError):
            learner.update(make_trajectory(learner))
        with self.assertRaises(LearnerContractError):
            learner.update([make_trajectory(learner), "not a trajectory"])

        # An empty trajectory cannot even be constructed.
        with self.assertRaises(ContractError):
            Trajectory(steps=(), generation_id=0, complete=True)
        # Incomplete (no declared endpoint) is rejected by the learner.
        with self.assertRaises(LearnerContractError) as context:
            learner.update([make_trajectory(learner, complete=False)])
        self.assertIn("incomplete", str(context.exception))
        # Truncated is rejected even when declared complete.
        with self.assertRaises(LearnerContractError) as context:
            learner.update([make_trajectory(learner, complete=True, truncated=True)])
        self.assertIn("truncated", str(context.exception))

        # Mixed generations cannot share one update.
        with self.assertRaises(LearnerContractError) as context:
            learner.update(
                [make_trajectory(learner, generation_id=0), make_trajectory(learner, generation_id=1)]
            )
        self.assertIn("mixed-generation", str(context.exception))
        # A foreign (stale or future) generation is rejected.
        with self.assertRaises(LearnerContractError) as context:
            learner.update([make_trajectory(learner, generation_id=3)])
        self.assertIn("does not match the current learner generation", str(context.exception))

        # Non-finite rewards, risks and states fail at construction.
        with self.assertRaises(ContractError):
            make_trajectory(learner, rewards=(0.1, float("nan"), 0.2))
        with self.assertRaises(ContractError):
            make_trajectory(learner, rewards=(0.1, float("inf"), 0.2))
        state = learner.build_state(market_window(), account()).vector
        with self.assertRaises(ContractError):
            TrajectoryStep(
                state=torch.full_like(state, float("nan")),
                direction=Direction.LONG,
                requested_risk=0.5,
                reward=0.1,
                terminal=False,
                truncated=False,
                generation_id=0,
            )
        with self.assertRaises(ContractError):
            TrajectoryStep(
                state=state,
                direction=Direction.LONG,
                requested_risk=1.5,
                reward=0.1,
                terminal=False,
                truncated=False,
                generation_id=0,
            )

        # A corrupted record is re-validated and rejected by the learner.
        corrupted = make_trajectory(learner)
        object.__setattr__(corrupted.steps[0], "reward", float("nan"))
        with self.assertRaises(ContractError):
            learner.update([corrupted])

        # A state of the wrong width cannot enter the actor boundary.
        narrow = TrajectoryStep(
            state=torch.zeros(4),
            direction=Direction.LONG,
            requested_risk=0.5,
            reward=0.1,
            terminal=False,
            truncated=False,
            generation_id=0,
        )
        with self.assertRaises(LearnerContractError):
            learner.update([Trajectory(steps=(narrow,), generation_id=0, complete=True)])

        # Nothing above consumed or advanced the learner.
        self.assertEqual(learner.generation_id, 0)


class GenerationBoundaryTests(unittest.TestCase):
    def test_04_sensory_projection_buffer_unchanged_after_valid_update(self):
        learner = make_learner()
        before = learner.sensory.projection.detach().clone()
        before_state_dict = {
            name: tensor.detach().clone() for name, tensor in learner.sensory.state_dict().items()
        }
        report = learner.update([make_trajectory(learner)])
        self.assertEqual(report.next_generation_id, 1)

        self.assertTrue(torch.equal(before, learner.sensory.projection))
        self.assertTrue(torch.equal(before_state_dict["projection"], learner.sensory.projection))
        self.assertFalse(learner.sensory.projection.requires_grad)
        self.assertIsNone(learner.sensory.projection.grad)
        self.assertEqual(sum(parameter.numel() for parameter in learner.sensory.parameters()), 0)

    def test_22_generation_begins_at_zero(self):
        learner = make_learner()
        self.assertEqual(learner.generation_id, 0)
        self.assertIsNot(learner.actor_optimizer, learner.critic_optimizer)
        self.assertEqual(len(learner.actor_optimizer.param_groups), 1)
        self.assertEqual(len(learner.critic_optimizer.param_groups), 1)
        self.assertEqual(learner.actor_optimizer.param_groups[0]["lr"], 3e-4)
        self.assertEqual(learner.critic_optimizer.param_groups[0]["lr"], 1e-3)
        actor_grouped = {
            id(p) for group in learner.actor_optimizer.param_groups for p in group["params"]
        }
        critic_grouped = {
            id(p) for group in learner.critic_optimizer.param_groups for p in group["params"]
        }
        self.assertFalse(actor_grouped & critic_grouped)

    def test_23_collection_takes_no_optimizer_step(self):
        learner = make_learner()
        state = learner.build_state(market_window(), account())
        actor_before = snapshot(learner.actor)
        critic_before = snapshot(learner.critic)

        torch.manual_seed(99)
        samples = [learner.sample_action(state) for _ in range(16)]

        for before, after in zip(actor_before, learner.actor.parameters()):
            self.assertTrue(torch.equal(before, after.detach()))
        for before, after in zip(critic_before, learner.critic.parameters()):
            self.assertTrue(torch.equal(before, after.detach()))
        for parameter in list(learner.actor.parameters()) + list(learner.critic.parameters()):
            self.assertIsNone(parameter.grad)
        self.assertTrue(all(0.0 <= sample.requested_risk <= 1.0 for sample in samples))
        self.assertEqual(learner.generation_id, 0)

    def test_24_one_valid_current_generation_batch_updates_actor_and_critic(self):
        learner = make_learner()
        batch = [make_trajectory(learner), make_physics_trajectory(learner)]
        actor_before = snapshot(learner.actor)
        critic_before = snapshot(learner.critic)

        report = learner.update(batch)

        self.assertEqual(report.generation_id, 0)
        self.assertEqual(report.next_generation_id, 1)
        self.assertEqual(report.trajectory_count, 2)
        self.assertEqual(report.step_count, 6)
        for value in (
            report.actor_loss,
            report.critic_loss,
            report.actor_grad_norm,
            report.critic_grad_norm,
            report.mean_return_to_go,
            report.mean_value,
        ):
            self.assertTrue(math.isfinite(value))
        self.assertGreater(report.actor_grad_norm, 0.0)
        self.assertGreater(report.critic_grad_norm, 0.0)

        self.assertTrue(
            any(
                not torch.equal(before, after.detach())
                for before, after in zip(actor_before, learner.actor.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before, after.detach())
                for before, after in zip(critic_before, learner.critic.parameters())
            )
        )
        for parameter in list(learner.actor.parameters()) + list(learner.critic.parameters()):
            self.assertIsNotNone(parameter.grad)

    def test_25_consumed_batch_cannot_be_updated_twice(self):
        learner = make_learner()
        batch = [make_trajectory(learner)]
        learner.update(batch)

        with self.assertRaises(LearnerContractError) as context:
            learner.update(batch)
        self.assertIn("consumed", str(context.exception))
        self.assertEqual(learner.generation_id, 1)

    def test_26_generation_increments_exactly_once_per_update(self):
        learner = make_learner()
        self.assertEqual(learner.generation_id, 0)

        first = learner.update([make_trajectory(learner, generation_id=0)])
        self.assertEqual(first.generation_id, 0)
        self.assertEqual(first.next_generation_id, 1)
        self.assertEqual(learner.generation_id, 1)

        second = learner.update([make_trajectory(learner, generation_id=1)])
        self.assertEqual(second.generation_id, 1)
        self.assertEqual(second.next_generation_id, 2)
        self.assertEqual(learner.generation_id, 2)

    def test_27_old_generation_trajectories_are_rejected_after_increment(self):
        learner = make_learner()
        learner.update([make_trajectory(learner, generation_id=0)])
        self.assertEqual(learner.generation_id, 1)

        stale = make_trajectory(learner, generation_id=0)
        actor_before = snapshot(learner.actor)
        with self.assertRaises(LearnerContractError) as context:
            learner.update([stale])
        self.assertIn("does not match the current learner generation", str(context.exception))
        self.assertEqual(learner.generation_id, 1)
        for before, after in zip(actor_before, learner.actor.parameters()):
            self.assertTrue(torch.equal(before, after.detach()))

    def test_28_sensory_buffer_unchanged_through_update(self):
        learner = make_learner()
        before = learner.sensory.projection.detach().clone()
        actor_before = snapshot(learner.actor)
        learner.update([make_trajectory(learner)])

        self.assertTrue(torch.equal(before, learner.sensory.projection))
        # The update really moved the actor, so the frozen buffer survived a
        # non-trivial optimizer step.
        self.assertTrue(
            any(
                not torch.equal(before_param, after.detach())
                for before_param, after in zip(actor_before, learner.actor.parameters())
            )
        )
        self.assertEqual(sum(parameter.numel() for parameter in learner.sensory.parameters()), 0)
        self.assertEqual(learner.sensory.projection.dtype, torch.float32)
        self.assertEqual(learner.sensory.projection.device.type, "cpu")

    def test_29_actor_advantage_uses_a_detached_critic_baseline(self):
        learner = make_learner()
        batch = [make_trajectory(learner)]

        learner.actor_optimizer.zero_grad(set_to_none=True)
        learner.critic_optimizer.zero_grad(set_to_none=True)
        learner.actor_objective(batch).backward()

        self.assertTrue(
            any(parameter.grad is not None for parameter in learner.actor.parameters())
        )
        for parameter in learner.critic.parameters():
            self.assertTrue(
                parameter.grad is None or bool(torch.all(parameter.grad == 0)),
                "L_actor populated a critic gradient",
            )

        # The update itself reports the same isolation.
        report = learner.update(batch)
        self.assertEqual(report.critic_grad_from_actor_max_abs, 0.0)
        self.assertGreater(report.critic_grad_norm, 0.0)

    def test_30_critic_target_is_realized_undiscounted_return_to_go(self):
        learner = make_learner()
        batch = [make_trajectory(learner, rewards=(0.5, -0.25, 2.0))]
        expected_returns = torch.tensor([2.25, 1.75, 2.0], dtype=torch.float64)
        self.assertEqual(batch[0].returns_to_go().tolist(), expected_returns.tolist())

        states = torch.stack([step.state for step in batch[0].steps])
        values = learner.critic(states).to(torch.float64)
        expected_loss = ((values - expected_returns) ** 2).mean().detach()
        self.assertAlmostEqual(
            float(learner.critic_objective(batch).detach()), float(expected_loss), places=6
        )

        # After a real update the critic moved against the realized return-to-go,
        # not against a discounted or bootstrapped target.
        values_before = learner.critic(states).detach().clone()
        learner.update(batch)
        values_after = learner.critic(states).detach()
        self.assertFalse(torch.equal(values_before, values_after))
        before_error = float(((values_before.to(torch.float64) - expected_returns) ** 2).mean())
        after_error = float(((values_after.to(torch.float64) - expected_returns) ** 2).mean())
        self.assertLess(after_error, before_error)

    def test_31_no_replay_buffer_or_off_policy_object(self):
        learner = make_learner()
        forbidden = (
            "replay",
            "priority",
            "importance",
            "v_trace",
            "vtrace",
            "gae",
            "clip",
            "target_",
            "q_value",
            "behaviour",
            "behavior",
            "off_policy",
            "offpolicy",
            "scheduler",
        )
        self.assertEqual(learner.direction_entropy_coefficient, 0.0)
        self.assertFalse(hasattr(learner, "beta_entropy_coefficient"))
        self.assertFalse(hasattr(learner, "risk_entropy_coefficient"))
        public_names = [name for name in dir(learner) if not name.startswith("_")]
        self.assertTrue(public_names)
        for name in public_names:
            for token in forbidden:
                self.assertNotIn(token, name.lower(), f"learner exposes {name!r}")
        for name in (
            "replay_buffer",
            "priority_buffer",
            "importance_weights",
            "target_network",
            "buffer",
        ):
            self.assertFalse(hasattr(learner, name))

        module_names = [name for name in vars(learner_module) if not name.startswith("_")]
        for name in module_names:
            for token in forbidden:
                self.assertNotIn(token, name.lower(), f"learner module exposes {name!r}")


class LearnerBoundaryGuardTests(unittest.TestCase):
    def test_construction_requires_a_matching_frozen_spine(self):
        sensory = FrozenSensory()
        with self.assertRaises(LearnerContractError):
            OnPolicyLearner(FrozenSensory(z_dim=8), Actor(), ValueCritic())
        with self.assertRaises(LearnerContractError):
            OnPolicyLearner(sensory, Actor(z_dim=8), ValueCritic())
        with self.assertRaises(LearnerContractError):
            OnPolicyLearner(sensory, Actor(), ValueCritic(z_dim=8))
        with self.assertRaises(LearnerContractError):
            OnPolicyLearner(sensory, Actor(), ValueCritic(), actor_lr=0.0)
        with self.assertRaises(LearnerContractError):
            OnPolicyLearner("sensory", Actor(), ValueCritic())

    def test_actor_and_critic_may_not_share_parameters(self):
        sensory = FrozenSensory()
        actor = Actor()
        with self.assertRaises(LearnerContractError):
            OnPolicyLearner(sensory, actor, actor)

    def test_external_actor_mutation_breaks_the_frozen_generation(self):
        learner = make_learner()
        batch = [make_trajectory(learner)]
        with torch.no_grad():
            next(learner.actor.parameters()).add_(1e-3)
        with self.assertRaises(LearnerContractError) as context:
            learner.update(batch)
        self.assertIn("generation boundary", str(context.exception))
        self.assertEqual(learner.generation_id, 0)

    def test_balanced_actor_mutation_that_preserves_aggregate_statistics_is_rejected(self):
        learner = make_learner()
        batch = [make_trajectory(learner)]
        retired_statistics = legacy_aggregate_statistics(learner.actor)
        actor_before = snapshot(learner.actor)

        swap_two_unequal_values_in_place(next(learner.actor.parameters()))

        # The retired aggregate fingerprint is blind to this mutation: every
        # numel/sum/abs_sum survives even though theta_g changed.
        self.assertEqual(legacy_aggregate_statistics(learner.actor), retired_statistics)
        self.assertTrue(
            any(
                not torch.equal(before, after.detach())
                for before, after in zip(actor_before, learner.actor.parameters())
            )
        )

        mutated = snapshot(learner.actor)
        with self.assertRaises(LearnerContractError) as context:
            learner.update(batch)
        self.assertIn("generation boundary", str(context.exception))

        # The exact snapshot guard fires before any optimizer step: the mutation
        # is the only difference and the generation did not advance.
        self.assertEqual(learner.generation_id, 0)
        for before, after in zip(mutated, learner.actor.parameters()):
            self.assertTrue(torch.equal(before, after.detach()))
        for parameter in learner.actor.parameters():
            self.assertIsNone(parameter.grad)

    def test_non_finite_objectives_are_rejected_before_any_optimizer_step(self):
        learner = make_learner()
        batch = [make_trajectory(learner)]
        actor_before = snapshot(learner.actor)
        with torch.no_grad():
            learner.critic.mlp[-1].bias.fill_(float("nan"))
        with self.assertRaises(ContractError):
            learner.update(batch)
        for before, after in zip(actor_before, learner.actor.parameters()):
            self.assertTrue(torch.equal(before, after.detach()))
        self.assertEqual(learner.generation_id, 0)

    def test_non_finite_returns_are_rejected(self):
        learner = make_learner()
        # Individually finite rewards whose float64 reverse cumulative sum
        # overflows to +inf at the first step.
        batch = [make_trajectory(learner, rewards=(1e308, 1e308))]
        with self.assertRaises(LearnerContractError) as context:
            learner.update(batch)
        self.assertIn("non-finite return-to-go", str(context.exception))
        self.assertEqual(learner.generation_id, 0)

    def test_non_finite_losses_are_rejected(self):
        learner = make_learner()
        # Finite returns whose float64 squares overflow inside the objectives.
        batch = [make_trajectory(learner, rewards=(1e308, 1e308, -1e308))]
        with self.assertRaises(LearnerContractError) as context:
            learner.update(batch)
        self.assertIn("non-finite", str(context.exception))
        self.assertEqual(learner.generation_id, 0)


class EndToEndSpineTests(unittest.TestCase):
    """A tiny fixture that runs the whole VS-B boundary twice on CPU.

    This is not a Task A/B learnability environment: it only proves that the
    frozen sensory -> actor -> Permission/Physics -> trajectory -> return ->
    actor/critic update -> next generation pipeline is exercisable and
    deterministic.
    """

    def _run_two_generations(self) -> tuple:
        learner = make_learner()
        torch.manual_seed(20240712)
        config = PhysicsConfig()
        reports = []
        telescoped = []
        for generation in range(2):
            market = market_window(seed=100 + generation)
            truth = AccountTruth(equity=PHYSICS_EQUITY, quantity=0.0, mark_price=100.0)
            steps = []
            for open_next, close_next in PHYSICS_PRICES:
                state = learner.build_state(market, account_state_from_truth(truth, config))
                sample = learner.sample_action(state)
                record = execute_transition(
                    truth, sample.nominal_action, open_next, close_next, config
                )
                truth = record.truth_next
                steps.append(
                    TrajectoryStep(
                        state=state.vector,
                        direction=sample.direction,
                        requested_risk=sample.requested_risk,
                        reward=record.reward,
                        terminal=False,
                        truncated=False,
                        generation_id=generation,
                    )
                )
            trajectory = Trajectory(
                steps=tuple(steps), generation_id=generation, complete=True
            )
            telescoped.append(
                (trajectory.returns_to_go()[0].item(), math.log(truth.equity / PHYSICS_EQUITY))
            )
            reports.append(learner.update([trajectory]))
        return learner, reports, telescoped

    def test_spine_runs_two_generations_deterministically_on_cpu(self):
        first_learner, first_reports, first_telescoped = self._run_two_generations()
        second_learner, second_reports, second_telescoped = self._run_two_generations()

        self.assertEqual(first_learner.generation_id, 2)
        self.assertEqual(second_learner.generation_id, 2)
        self.assertEqual([report.next_generation_id for report in first_reports], [1, 2])
        self.assertEqual(
            [report.actor_loss for report in first_reports],
            [report.actor_loss for report in second_reports],
        )
        self.assertEqual(
            [report.critic_loss for report in first_reports],
            [report.critic_loss for report in second_reports],
        )
        for value, terminal_log_equity in first_telescoped + second_telescoped:
            self.assertAlmostEqual(value, terminal_log_equity, places=10)
        for parameter in first_learner.actor.parameters():
            self.assertEqual(parameter.device.type, "cpu")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
