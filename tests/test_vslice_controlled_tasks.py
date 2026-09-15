"""R12 VS-C controlled-task regression tests (contract requirements 1-12).

Required by ``docs/experiments/frozen/R12_VS_C_CONTROLLED_LEARNABILITY_R0.md`` section
"Required implementation tests":

1. Task-A flat raw window passes canonical N0 and yields identical normalized
   market for all accounts.
2. Task-A actual account states are exactly ``-0.5/0/+0.5`` exposure and market
   representation is identical.
3. Positive Task-A batch has exact 30/30/30 balance.
4. Task-A control has every actual x observed pair exactly 10 times and
   independent marginals.
5. Task-A reward is produced only through merged VS-A Physics; maintaining the
   target gives zero turnover cost.
6. Task-B UP/DOWN raw windows pass canonical market validation and are distinct
   after N0/FrozenSensory.
7. Task-B ``reward_0 == 0`` for sampled valid actions with kappa zero.
8. Task-B delayed reward changes sign consistently for a hand-set LONG versus
   SHORT action_0 under UP/DOWN gaps.
9. Changing action_1 with fixed action_0 does not change the two-step final
   equity/reward under the frozen Task-B construction.
10. Positive Task-B batch is exactly 64 UP / 64 DOWN.
11. Shuffled control permutation has no fixed points, preserves the reward
    multiset exactly, and is deterministic for seed+generation.
12. Shuffled control does not mutate original trajectory objects.

    .venv/bin/python -m unittest tests.test_vslice_controlled_tasks -v
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

try:  # CPU PyTorch lives in the task venv; a bare interpreter records a skip.
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without PyTorch
    raise unittest.SkipTest(f"CPU PyTorch is required for the R12 VS-C tasks: {exc}")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "science") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "science"))

from cb16_science.vslice import controlled_tasks as controlled  # noqa: E402
from cb16_science.vslice.contracts import (  # noqa: E402
    Direction,
    NominalAction,
    nominal_target_from_action,
)
from cb16_science.vslice.learner import OnPolicyLearner  # noqa: E402
from cb16_science.vslice.market import normalize_market_window  # noqa: E402
from cb16_science.vslice.physics import execute_transition  # noqa: E402
from cb16_science.vslice.policy import (  # noqa: E402
    Actor,
    ValueCritic,
    account_state_vector,
    build_learner_state,
)
from cb16_science.vslice.sensory import FrozenSensory  # noqa: E402

SPEC_PATH = REPO_ROOT / "config" / "experiments" / "r12_vs_c_r0.json"


def committed_spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def make_learner(seed: int = 0) -> OnPolicyLearner:
    torch.manual_seed(seed)
    return OnPolicyLearner(FrozenSensory(), Actor(), ValueCritic())


def two_step(env, action_0, action_1, cue):
    """Hand-run the frozen two-step Task-B economics for one cue."""

    step_0 = execute_transition(
        env.initial_truth, action_0, env.step0_open_next, env.step0_close_next, env.config
    )
    gap = controlled.task_b_gap(env, cue)
    step_1 = execute_transition(step_0.truth_next, action_1, gap, gap, env.config)
    return step_0, step_1


class TaskAMarketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = committed_spec()
        self.env = controlled.task_a_environment(self.spec)

    def test_flat_window_passes_n0_and_is_identical_for_every_account(self):
        """Requirement 1."""

        context_length = self.env.config.context_length
        raw = controlled.flat_raw_window(
            context_length, price=controlled.TASK_A_PRICE, volume=controlled.TASK_A_VOLUME
        )
        self.assertEqual(raw.shape, (context_length + 1, 5))
        self.assertTrue(np.all(raw[:, :4] == controlled.TASK_A_PRICE))
        self.assertTrue(np.all(raw[:, 4] == controlled.TASK_A_VOLUME))

        expected = normalize_market_window(raw, self.env.config)
        self.assertTrue(np.array_equal(self.env.market, expected))
        self.assertEqual(self.env.market.shape, (context_length, 5))
        self.assertTrue(np.all(np.isfinite(self.env.market)))
        self.assertTrue(np.all(self.env.market == 0.0))

        learner = make_learner(11)
        z_market = learner.encode(torch.from_numpy(self.env.market))
        for case in self.env.cases:
            state = build_learner_state(
                z_market, controlled.task_a_account_state(self.env, case)
            )
            self.assertTrue(torch.equal(state.z, z_market))

    def test_actual_account_states_are_exact_half_exposures(self):
        """Requirement 2."""

        self.assertEqual([case.name for case in self.env.cases], ["SHORT_HALF", "FLAT", "LONG_HALF"])
        self.assertEqual(
            [case.truth.exposure for case in self.env.cases], [-0.5, 0.0, 0.5]
        )
        self.assertEqual(
            [case.desired_target_exposure for case in self.env.cases], [-0.5, 0.0, 0.5]
        )
        for case in self.env.cases:
            state = controlled.task_a_account_state(self.env, case)
            self.assertEqual(state.signed_exposure, case.desired_target_exposure)
            self.assertEqual(state.survival_cushion, 1.0)
            self.assertEqual(state.new_risk_capacity, 1.0 - abs(state.signed_exposure))

        learner = make_learner(12)
        z_market = learner.encode(torch.from_numpy(self.env.market))
        states = [
            build_learner_state(z_market, controlled.task_a_account_state(self.env, case))
            for case in self.env.cases
        ]
        for state in states[1:]:
            self.assertTrue(torch.equal(state.z, states[0].z))
        self.assertEqual(
            [state.account.tolist() for state in states],
            [[-0.5, 1.0, 0.5], [0.0, 1.0, 1.0], [0.5, 1.0, 0.5]],
        )

    def test_positive_batch_is_exactly_30_30_30(self):
        """Requirement 3."""

        self.assertEqual(len(self.env.positive_schedule), 90)
        self.assertEqual(
            Counter(actual for actual, _ in self.env.positive_schedule),
            Counter({0: 30, 1: 30, 2: 30}),
        )
        self.assertTrue(
            all(actual == observed for actual, observed in self.env.positive_schedule)
        )

        learner = make_learner(13)
        generation = controlled.build_task_a_generation(
            learner, self.env, 0, mode=controlled.ARM_POSITIVE
        )
        self.assertEqual(len(generation), 90)
        for index, trajectory in enumerate(generation):
            case = self.env.cases[index // 30]
            expected = account_state_vector(controlled.task_a_account_state(self.env, case))
            self.assertEqual(trajectory.length, 1)
            self.assertTrue(trajectory.complete)
            self.assertFalse(trajectory.truncated)
            self.assertTrue(torch.equal(trajectory.steps[0].state[-3:], expected))

    def test_control_has_every_pair_ten_times_and_independent_marginals(self):
        """Requirement 4."""

        schedule = self.env.control_schedule
        self.assertEqual(len(schedule), 90)
        pairs = Counter(schedule)
        for actual in range(3):
            for observed in range(3):
                self.assertEqual(pairs[(actual, observed)], 10)
        self.assertEqual(
            Counter(actual for actual, _ in schedule), Counter({0: 30, 1: 30, 2: 30})
        )
        self.assertEqual(
            Counter(observed for _, observed in schedule), Counter({0: 30, 1: 30, 2: 30})
        )
        # Exact product independence over the balanced generation.
        actual_marginal = Counter(actual for actual, _ in schedule)
        observed_marginal = Counter(observed for _, observed in schedule)
        for actual in range(3):
            for observed in range(3):
                self.assertEqual(
                    pairs[(actual, observed)] * len(schedule),
                    actual_marginal[actual] * observed_marginal[observed],
                )

        learner = make_learner(14)
        captured = []

        def capture(truth, action, open_next, close_next, config):
            record = execute_transition(truth, action, open_next, close_next, config)
            captured.append((truth, action, record))
            return record

        with mock.patch.object(controlled, "execute_transition", side_effect=capture):
            generation = controlled.build_task_a_generation(
                learner, self.env, 0, mode=controlled.ARM_CONTROL
            )

        self.assertEqual(len(captured), 90)
        self.assertEqual(len(generation), 90)
        for index, ((actual, observed), trajectory) in enumerate(
            zip(self.env.control_schedule, generation)
        ):
            self.assertEqual(captured[index][0], self.env.cases[actual].truth)
            expected = account_state_vector(
                controlled.task_a_account_state(self.env, self.env.cases[observed])
            )
            self.assertTrue(torch.equal(trajectory.steps[0].state[-3:], expected))
            self.assertEqual(trajectory.rewards[0], captured[index][2].reward)

    def test_reward_comes_only_from_merged_physics_and_maintenance_is_free(self):
        """Requirement 5."""

        for case in self.env.cases:
            direction = controlled.desired_direction(case.desired_target_exposure)
            action = NominalAction(
                direction=direction, requested_risk=abs(case.desired_target_exposure)
            )
            self.assertEqual(
                nominal_target_from_action(action, self.env.config),
                case.desired_target_exposure,
            )
            step = execute_transition(
                case.truth, action, self.env.open_next, self.env.close_next, self.env.config
            )
            self.assertEqual(step.trade_cost, 0.0)
            self.assertEqual(step.reward, 0.0)

        learner = make_learner(15)
        captured = []

        def capture(truth, action, open_next, close_next, config):
            record = execute_transition(truth, action, open_next, close_next, config)
            captured.append((truth, action, record))
            return record

        with mock.patch.object(controlled, "execute_transition", side_effect=capture):
            generation = controlled.build_task_a_generation(
                learner, self.env, 0, mode=controlled.ARM_POSITIVE
            )

        self.assertEqual(len(captured), 90)
        self.assertEqual(
            [record[0] for record in captured],
            [self.env.cases[actual].truth for actual, _ in self.env.positive_schedule],
        )
        self.assertEqual(
            [trajectory.rewards[0] for trajectory in generation],
            [record[2].reward for record in captured],
        )


class TaskBRawWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = committed_spec()
        self.env = controlled.task_b_environment(self.spec)

    def test_cue_windows_match_the_preregistered_formula(self):
        """Requirement 6 (construction half)."""

        context_length = self.env.config.context_length
        for cue, sign in ((controlled.UP, -1.0), (controlled.DOWN, 1.0)):
            raw = controlled.cue_raw_window(
                cue,
                context_length,
                endpoint_price=100.0,
                ramp_magnitude=0.04,
                volume=10.0,
            )
            self.assertEqual(raw.shape, (context_length + 1, 5))
            self.assertAlmostEqual(raw[0, 3], raw[1, 3], places=12)
            self.assertAlmostEqual(raw[-1, 3], 100.0, places=12)
            self.assertTrue(np.all(raw[:, 4] == 10.0))
            for position in range(1, context_length + 1):
                expected_close = 100.0 * math.exp(
                    sign * 0.04 * (context_length - position) / (context_length - 1)
                )
                self.assertAlmostEqual(raw[position, 3], expected_close, places=12)
                self.assertAlmostEqual(raw[position, 0], raw[position - 1, 3], places=12)
                self.assertAlmostEqual(
                    raw[position, 1], max(raw[position, 0], raw[position, 3]) * 1.001, places=12
                )
                self.assertAlmostEqual(
                    raw[position, 2], min(raw[position, 0], raw[position, 3]) / 1.001, places=12
                )

    def test_cue_windows_normalize_and_stay_distinct_through_frozen_sensory(self):
        """Requirement 6 (validation half)."""

        context_length = self.env.config.context_length
        up = self.env.cue_markets[controlled.UP]
        down = self.env.cue_markets[controlled.DOWN]
        self.assertEqual(up.shape, (context_length, 5))
        self.assertEqual(down.shape, (context_length, 5))
        self.assertTrue(np.all(np.isfinite(up)))
        self.assertTrue(np.all(np.isfinite(down)))
        self.assertFalse(np.allclose(up, down))
        # The UP cue carries a below-anchor open channel and a below-anchor close
        # channel except at the endpoint; the DOWN cue is the mirror image.
        self.assertTrue(np.all(up[:, 0] < 0.0))
        self.assertTrue(np.all(up[:-1, 3] < 0.0))
        self.assertEqual(up[-1, 3], 0.0)
        self.assertTrue(np.all(down[:, 0] > 0.0))
        self.assertTrue(np.all(down[:-1, 3] > 0.0))
        self.assertEqual(down[-1, 3], 0.0)
        self.assertTrue(np.all(up[:, 4] == 0.0))
        self.assertTrue(np.all(down[:, 4] == 0.0))

        for cue in (controlled.UP, controlled.DOWN):
            raw = controlled.cue_raw_window(
                cue,
                context_length,
                endpoint_price=100.0,
                ramp_magnitude=0.04,
                volume=10.0,
            )
            self.assertTrue(
                np.array_equal(
                    normalize_market_window(raw, self.env.config),
                    self.env.cue_markets[cue],
                )
            )

        self.assertTrue(np.all(self.env.neutral_market == 0.0))
        neutral_raw = controlled.flat_raw_window(context_length, price=100.0, volume=10.0)
        self.assertTrue(
            np.array_equal(
                normalize_market_window(neutral_raw, self.env.config), self.env.neutral_market
            )
        )

        sensory = FrozenSensory()
        z_up = sensory(torch.from_numpy(up))
        z_down = sensory(torch.from_numpy(down))
        self.assertEqual(z_up.shape, z_down.shape)
        self.assertTrue(bool(torch.isfinite(z_up).all()))
        self.assertTrue(bool(torch.isfinite(z_down).all()))
        self.assertFalse(bool(torch.allclose(z_up, z_down)))


class TaskBEconomicsTests(unittest.TestCase):
    FLAT = NominalAction(direction=Direction.FLAT, requested_risk=0.0)
    LONG_HALF = NominalAction(direction=Direction.LONG, requested_risk=0.5)
    SHORT_HALF = NominalAction(direction=Direction.SHORT, requested_risk=0.5)

    def setUp(self) -> None:
        self.spec = committed_spec()
        self.env = controlled.task_b_environment(self.spec)

    def test_immediate_reward_is_exactly_zero_for_valid_actions(self):
        """Requirement 7."""

        actions = (
            NominalAction(direction=Direction.LONG, requested_risk=1.0),
            NominalAction(direction=Direction.SHORT, requested_risk=0.5),
            NominalAction(direction=Direction.LONG, requested_risk=0.25),
            self.FLAT,
        )
        for cue in (controlled.UP, controlled.DOWN):
            for action in actions:
                step_0, _ = two_step(self.env, action, self.FLAT, cue)
                self.assertEqual(step_0.reward, 0.0)

        learner = make_learner(21)
        records = controlled.collect_task_b_records(learner, self.env)
        self.assertEqual(len(records), 128)
        self.assertTrue(all(record.immediate_reward == 0.0 for record in records))

    def test_delayed_reward_sign_follows_hand_set_direction(self):
        """Requirement 8."""

        expectations = {
            (controlled.UP, "LONG"): math.log(1.05),
            (controlled.UP, "SHORT"): math.log(0.95),
            (controlled.DOWN, "LONG"): math.log(0.95),
            (controlled.DOWN, "SHORT"): math.log(1.05),
        }
        actions = {"LONG": self.LONG_HALF, "SHORT": self.SHORT_HALF}
        for (cue, name), expected in expectations.items():
            _, delayed = two_step(self.env, actions[name], self.FLAT, cue)
            self.assertAlmostEqual(delayed.reward, expected, places=12)
        # The sign flips with the gap for a fixed early action.
        _, up_long = two_step(self.env, self.LONG_HALF, self.FLAT, controlled.UP)
        _, down_long = two_step(self.env, self.LONG_HALF, self.FLAT, controlled.DOWN)
        self.assertGreater(up_long.reward, 0.0)
        self.assertLess(down_long.reward, 0.0)
        _, up_short = two_step(self.env, self.SHORT_HALF, self.FLAT, controlled.UP)
        _, down_short = two_step(self.env, self.SHORT_HALF, self.FLAT, controlled.DOWN)
        self.assertLess(up_short.reward, 0.0)
        self.assertGreater(down_short.reward, 0.0)

    def test_action_1_does_not_change_the_two_step_consequence(self):
        """Requirement 9."""

        action_0 = NominalAction(direction=Direction.LONG, requested_risk=0.5)
        alternatives = (
            self.FLAT,
            NominalAction(direction=Direction.LONG, requested_risk=1.0),
            NominalAction(direction=Direction.SHORT, requested_risk=0.25),
        )
        for cue in (controlled.UP, controlled.DOWN):
            _, reference = two_step(self.env, action_0, alternatives[0], cue)
            for action_1 in alternatives[1:]:
                _, candidate = two_step(self.env, action_0, action_1, cue)
                self.assertEqual(candidate.reward, reference.reward)
                self.assertEqual(candidate.truth_next.equity, reference.truth_next.equity)
            quantities = {
                two_step(self.env, action_0, action_1, cue)[1].post_cost_quantity
                for action_1 in alternatives
            }
            self.assertEqual(len(quantities), len(alternatives))

    def test_two_step_return_to_go_credits_action_0_with_the_delayed_reward(self):
        """Requirement 8 (credit path): G_0 of the two-step record is the delayed reward."""

        learner = make_learner(24)
        records = controlled.collect_task_b_records(learner, self.env)
        generation = controlled.task_b_trajectories(records, 0)
        self.assertEqual(len(records), len(generation))
        for record, trajectory in zip(records, generation):
            returns = trajectory.returns_to_go()
            self.assertEqual(returns[0].item(), record.immediate_reward + record.delayed_reward)
            self.assertEqual(returns[1].item(), record.delayed_reward)
            cue_sign = 1.0 if record.cue == controlled.UP else -1.0
            target_0 = int(record.direction_0) * record.requested_risk_0
            self.assertAlmostEqual(
                record.delayed_reward, math.log(1.0 + 0.1 * cue_sign * target_0), places=12
            )

    def test_zero_friction_makes_the_bisection_count_immaterial(self):
        """Task B's inherited VS-A bisection count cannot affect any number."""

        from cb16_science.vslice.contracts import PERMISSION_BISECTION_ITERATIONS
        from cb16_science.vslice.physics import feasible_target_interval, mark_to_next_open

        self.assertEqual(self.env.config.permission_bisection_iterations, PERMISSION_BISECTION_ITERATIONS)
        self.assertEqual(self.env.config.proportional_friction_kappa, 0.0)
        alternative = type(self.env.config)(
            context_length=self.env.config.context_length,
            nominal_exposure_budget=self.env.config.nominal_exposure_budget,
            hard_exposure_limit=self.env.config.hard_exposure_limit,
            proportional_friction_kappa=0.0,
            permission_bisection_iterations=1,
        )
        learner = make_learner(23)
        records = controlled.collect_task_b_records(learner, self.env)
        action_0 = NominalAction(direction=Direction.LONG, requested_risk=0.5)
        reference = execute_transition(
            self.env.initial_truth, action_0, self.env.step0_open_next, self.env.step0_close_next, self.env.config
        )
        candidate = execute_transition(
            self.env.initial_truth, action_0, self.env.step0_open_next, self.env.step0_close_next, alternative
        )
        self.assertEqual(reference.truth_next, candidate.truth_next)
        self.assertEqual(reference.reward, candidate.reward)
        pretrade = mark_to_next_open(self.env.initial_truth, self.env.step0_open_next)
        self.assertEqual(feasible_target_interval(pretrade, self.env.config), (-1.0, 1.0))
        self.assertEqual(feasible_target_interval(pretrade, alternative), (-1.0, 1.0))
        self.assertTrue(all(record.immediate_reward == 0.0 for record in records))

    def test_positive_batch_is_exactly_64_up_and_64_down(self):
        """Requirement 10."""

        self.assertEqual(len(self.env.cue_schedule), 128)
        self.assertEqual(
            Counter(self.env.cue_schedule), Counter({controlled.UP: 64, controlled.DOWN: 64})
        )
        learner = make_learner(22)
        records = controlled.collect_task_b_records(learner, self.env)
        self.assertEqual(len(records), 128)
        self.assertEqual(
            Counter(record.cue for record in records),
            Counter({controlled.UP: 64, controlled.DOWN: 64}),
        )
        generation = controlled.task_b_trajectories(records, 0)
        self.assertEqual(len(generation), 128)
        for trajectory in generation:
            self.assertEqual(trajectory.length, 2)
            self.assertTrue(trajectory.complete)
            self.assertFalse(trajectory.truncated)


class ShuffledControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = committed_spec()
        self.env = controlled.task_b_environment(self.spec)

    def test_permutation_is_a_deterministic_derangement(self):
        """Requirement 11 (permutation half)."""

        permutation = controlled.no_fixed_point_permutation(128, seed=1201, generation=0)
        self.assertEqual(len(permutation), 128)
        self.assertEqual(sorted(permutation), list(range(128)))
        self.assertTrue(all(permutation[index] != index for index in range(128)))
        self.assertEqual(
            controlled.no_fixed_point_permutation(128, seed=1201, generation=0), permutation
        )
        self.assertNotEqual(
            controlled.no_fixed_point_permutation(128, seed=1202, generation=0), permutation
        )
        self.assertNotEqual(
            controlled.no_fixed_point_permutation(128, seed=1201, generation=1), permutation
        )
        self.assertNotEqual(
            controlled.no_fixed_point_permutation(128, seed=1201, generation=2), permutation
        )
        self.assertEqual(controlled.control_permutation_seed(1201, 0), controlled.derived_stream_seed(
            controlled.PERMUTATION_DOMAIN, 1201, 0
        ))

    def test_shuffled_control_preserves_the_reward_multiset(self):
        """Requirement 11 (multiset half)."""

        learner = make_learner(31)
        records = controlled.collect_task_b_records(learner, self.env)
        own = [record.delayed_reward for record in records]
        permutation = controlled.no_fixed_point_permutation(
            len(records), seed=1201, generation=0
        )
        assigned = controlled.shuffled_delayed_rewards(records, seed=1201, generation=0)
        self.assertEqual(assigned, tuple(records[source].delayed_reward for source in permutation))
        self.assertEqual(sorted(assigned), sorted(own))

        generation = controlled.task_b_trajectories(
            records, 0, delayed_rewards=assigned
        )
        self.assertEqual([trajectory.rewards[1] for trajectory in generation], list(assigned))
        self.assertTrue(all(trajectory.rewards[0] == 0.0 for trajectory in generation))

    def test_shuffled_control_does_not_mutate_collected_records(self):
        """Requirement 12."""

        learner = make_learner(32)
        records = controlled.collect_task_b_records(learner, self.env)
        own_rewards = tuple(record.delayed_reward for record in records)
        own_states = tuple((record.state_0.clone(), record.state_1.clone()) for record in records)

        positive = controlled.task_b_trajectories(records, 0)
        assigned = controlled.shuffled_delayed_rewards(records, seed=1201, generation=0)
        control = controlled.task_b_trajectories(records, 0, delayed_rewards=assigned)

        # Collected records are untouched.
        self.assertEqual(tuple(record.delayed_reward for record in records), own_rewards)
        for record, (state_0, state_1) in zip(records, own_states):
            self.assertTrue(torch.equal(record.state_0, state_0))
            self.assertTrue(torch.equal(record.state_1, state_1))

        # The positive generation still carries every trajectory's own reward.
        self.assertEqual(
            [trajectory.rewards[1] for trajectory in positive], list(own_rewards)
        )
        # The control generation is built from fresh immutable step records.
        for positive_trajectory, control_trajectory in zip(positive, control):
            self.assertIsNot(positive_trajectory, control_trajectory)
            self.assertIsNot(positive_trajectory.steps[0], control_trajectory.steps[0])
            self.assertIsNot(positive_trajectory.steps[1], control_trajectory.steps[1])
            self.assertTrue(
                torch.equal(positive_trajectory.steps[0].state, control_trajectory.steps[0].state)
            )
            self.assertEqual(
                positive_trajectory.steps[0].reward, control_trajectory.steps[0].reward
            )
            self.assertEqual(
                positive_trajectory.steps[1].direction, control_trajectory.steps[1].direction
            )
            self.assertEqual(
                positive_trajectory.steps[1].requested_risk,
                control_trajectory.steps[1].requested_risk,
            )


if __name__ == "__main__":
    unittest.main()
