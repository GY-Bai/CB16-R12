"""Actor/Critic/state-boundary regression tests: required tests 6-17.

Required by ``docs/experiments/frozen/R12_VS_B_LEARNER_SPINE_R0.md`` (section "Required
tests", "State boundary", "Actor / policy distribution", "Critic").

    .venv/bin/python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
import unittest
from pathlib import Path

try:  # CPU PyTorch lives in the task venv; a bare interpreter records a skip.
    import torch
except ImportError as exc:  # pragma: no cover - exercised only without PyTorch
    raise unittest.SkipTest(f"CPU PyTorch is required for the R12 VS-B learner spine: {exc}")

from torch.distributions import Beta, Categorical

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "science") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "science"))

from cb16_science.vslice import learner as learner_module  # noqa: E402
from cb16_science.vslice import policy as policy_module  # noqa: E402
from cb16_science.vslice import sensory as sensory_module  # noqa: E402
from cb16_science.vslice import trajectory as trajectory_module  # noqa: E402
from cb16_science.vslice.contracts import (  # noqa: E402
    AccountState,
    ContractError,
    Direction,
)
from cb16_science.vslice.policy import (  # noqa: E402
    ACCOUNT_STATE_DIM,
    ACCOUNT_STATE_FIELDS,
    BETA_FLOOR,
    DIRECTION_INDEX_FLAT,
    DIRECTION_INDEX_LONG,
    DIRECTION_INDEX_SHORT,
    DIRECTION_ORDER,
    HIDDEN_LAYERS,
    HIDDEN_WIDTH,
    ActionSample,
    Actor,
    LearnerState,
    ValueCritic,
    account_state_vector,
    build_learner_state,
    direction_from_index,
    direction_index,
)
from cb16_science.vslice.sensory import MARKET_CHANNELS, Z_DIM, FrozenSensory  # noqa: E402

CONTEXT_LENGTH = 64


def account(
    signed_exposure: float = 0.25,
    survival_cushion: float = 1.0,
    new_risk_capacity: float = 0.75,
) -> AccountState:
    return AccountState(
        signed_exposure=signed_exposure,
        survival_cushion=survival_cushion,
        new_risk_capacity=new_risk_capacity,
    )


def state_vector(*, seed: int = 7, signed_exposure: float = 0.25) -> torch.Tensor:
    z = torch.randn(Z_DIM, generator=torch.Generator().manual_seed(seed))
    return build_learner_state(z, account(signed_exposure=signed_exposure)).vector


def state_batch(count: int, *, seed: int = 11) -> torch.Tensor:
    z = torch.randn(count, Z_DIM, generator=torch.Generator().manual_seed(seed))
    return build_learner_state(z, account()).vector


def pin_last_layer(module: torch.nn.Module, bias: list) -> None:
    """Make the actor/critic output independent of the state and fully pinned."""

    with torch.no_grad():
        last = module.mlp[-1]
        last.weight.zero_()
        last.bias.copy_(torch.tensor(bias, dtype=last.bias.dtype))


class StateBoundaryTests(unittest.TestCase):
    def test_06_actor_and_critic_input_dimension_is_z_dim_plus_three(self):
        actor = Actor()
        critic = ValueCritic()
        self.assertEqual(ACCOUNT_STATE_DIM, 3)
        self.assertEqual(actor.input_dim, Z_DIM + ACCOUNT_STATE_DIM)
        self.assertEqual(critic.input_dim, Z_DIM + ACCOUNT_STATE_DIM)
        self.assertEqual(actor.mlp[0].in_features, Z_DIM + ACCOUNT_STATE_DIM)
        self.assertEqual(critic.mlp[0].in_features, Z_DIM + ACCOUNT_STATE_DIM)
        self.assertEqual(actor.z_dim, Z_DIM)

        # Default Central Brain shape is experiment-scoped 64-wide, two hidden layers.
        self.assertEqual(actor.hidden_width, HIDDEN_WIDTH)
        self.assertEqual(actor.hidden_layers, HIDDEN_LAYERS)
        linears = [layer for layer in actor.mlp if isinstance(layer, torch.nn.Linear)]
        self.assertEqual(len(linears), HIDDEN_LAYERS + 1)
        self.assertEqual(linears[-1].out_features, len(DIRECTION_ORDER) + 4)

        # Raw OHLCV can never be fed in parallel: a [L, 5] market tensor is not
        # a valid actor/critic state.
        raw_market = torch.randn(CONTEXT_LENGTH, MARKET_CHANNELS)
        with self.assertRaises(ContractError):
            actor(raw_market)
        with self.assertRaises(ContractError):
            critic(raw_market)

    def test_07_account_state_changes_only_the_final_three_coordinates(self):
        z = torch.randn(Z_DIM, generator=torch.Generator().manual_seed(7))
        first = build_learner_state(z, account(0.2, 1.0, 0.8))
        second = build_learner_state(z, account(-0.4, 0.5, 0.6))

        self.assertEqual(tuple(first.vector.shape), (Z_DIM + ACCOUNT_STATE_DIM,))
        self.assertTrue(torch.equal(first.z, second.z))
        self.assertTrue(torch.equal(first.vector[:Z_DIM], second.vector[:Z_DIM]))
        expected_account = torch.tensor([-0.4, 0.5, 0.6], dtype=torch.float32)
        self.assertTrue(torch.equal(second.vector[Z_DIM:], expected_account))
        self.assertFalse(torch.equal(first.vector[Z_DIM:], second.vector[Z_DIM:]))

        batched = build_learner_state(z.unsqueeze(0).repeat(3, 1), account(0.2, 1.0, 0.8))
        self.assertEqual(tuple(batched.vector.shape), (3, Z_DIM + ACCOUNT_STATE_DIM))

        # The account vector is exactly the canonical three-coordinate order.
        self.assertTrue(
            torch.equal(
                account_state_vector(account(0.2, 1.0, 0.8)),
                torch.tensor([0.2, 1.0, 0.8], dtype=torch.float32),
            )
        )
        with self.assertRaises(ContractError):
            LearnerState(z=z, account=torch.zeros(4))
        with self.assertRaises(ContractError):
            build_learner_state(z, (0.2, 1.0, 0.8))

    def test_08_public_learner_state_api_has_no_raw_market_or_monetary_fields(self):
        self.assertEqual(tuple(f.name for f in dataclasses.fields(LearnerState)), ("z", "account"))
        self.assertEqual(
            ACCOUNT_STATE_FIELDS,
            ("signed_exposure", "survival_cushion", "new_risk_capacity"),
        )
        self.assertEqual(
            tuple(inspect.signature(build_learner_state).parameters), ("z", "account_state")
        )

        forbidden = (
            "ohlc",
            "open",
            "high",
            "low",
            "close",
            "price",
            "equity",
            "quantity",
            "notional",
            "pnl",
            "symbol",
            "asset",
            "entry",
            "holding",
            "drawdown",
            "trade_count",
            "volume",
        )
        inspected = set()
        for module in (sensory_module, policy_module, trajectory_module, learner_module):
            inspected.update(name for name in vars(module) if not name.startswith("_"))
        for record in (LearnerState, ActionSample):
            inspected.update(field.name for field in dataclasses.fields(record))
        inspected.update(
            name for name in dir(LearnerState) if not name.startswith("_")
        )
        for name in inspected:
            lowered = name.lower()
            for token in forbidden:
                self.assertNotIn(token, lowered, f"public learner-state name {name!r}")

    def test_float64_states_are_cast_to_the_module_dtype(self):
        # A canonical NumPy pipeline hands over float64; the policy modules
        # compute in their own dtype instead of failing on a raw dtype mismatch.
        state = state_vector().double()
        actor = Actor()
        with torch.no_grad():
            output = actor(state)
        self.assertEqual(output.direction_logits.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(output.direction_logits).all()))
        self.assertEqual(
            actor.deterministic_action(state).direction,
            direction_from_index(int(torch.argmax(output.direction_logits).item())),
        )
        self.assertTrue(
            torch.isfinite(actor.log_prob(state, Direction.FLAT, 0.0)).all()
        )

        critic = ValueCritic()
        with torch.no_grad():
            value = critic(state)
        self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(tuple(value.shape), ())

        # Integer states are still rejected explicitly.
        with self.assertRaises(ContractError):
            actor(state.to(torch.int64))

        # A finite float64 value beyond float32 range must fail, not saturate.
        overflow = torch.zeros(Z_DIM + ACCOUNT_STATE_DIM, dtype=torch.float64)
        overflow[0] = 1e300
        with self.assertRaises(ContractError):
            actor(overflow)
        with self.assertRaises(ContractError):
            critic(overflow)


class ActorDistributionTests(unittest.TestCase):
    def test_09_direction_semantic_order_is_short_flat_long(self):
        self.assertEqual(DIRECTION_ORDER, (Direction.SHORT, Direction.FLAT, Direction.LONG))
        self.assertEqual(DIRECTION_INDEX_SHORT, 0)
        self.assertEqual(DIRECTION_INDEX_FLAT, 1)
        self.assertEqual(DIRECTION_INDEX_LONG, 2)
        for index, expected in enumerate(DIRECTION_ORDER):
            self.assertEqual(direction_from_index(index), expected)
            self.assertEqual(direction_index(expected), index)

        # Slot 0/1/2 of the real head must mean SHORT/FLAT/LONG.
        state = state_vector()
        for index, expected in enumerate(DIRECTION_ORDER):
            actor = Actor()
            bias = [0.0] * (len(DIRECTION_ORDER) + 4)
            bias[index] = 5.0
            pin_last_layer(actor, bias)
            self.assertEqual(actor.deterministic_action(state).direction, expected)

    def test_10_direction_logits_and_beta_parameters_are_finite(self):
        actor = Actor()
        for label, state in {
            "single": state_vector(),
            "batch": state_batch(4),
            "large positive": torch.full((Z_DIM + ACCOUNT_STATE_DIM,), 1e3),
            "large negative": torch.full((Z_DIM + ACCOUNT_STATE_DIM,), -1e3),
        }.items():
            with self.subTest(state=label):
                output = actor(state)
                for tensor in (
                    output.direction_logits,
                    output.short_alpha,
                    output.short_beta,
                    output.long_alpha,
                    output.long_beta,
                ):
                    self.assertTrue(bool(torch.isfinite(tensor).all()))

    def test_11_beta_parameters_are_positive_and_obey_the_frozen_floor(self):
        actor = Actor()
        output = actor(state_batch(6))
        for tensor in (output.short_alpha, output.short_beta, output.long_alpha, output.long_beta):
            self.assertTrue(bool((tensor > 0.0).all()))
            self.assertTrue(bool((tensor >= BETA_FLOOR).all()))

        # A saturated negative raw head must sit exactly on the frozen floor.
        floor_actor = Actor()
        pin_last_layer(floor_actor, [-1e3] * (len(DIRECTION_ORDER) + 4))
        floor_output = floor_actor(state_vector())
        for tensor in (
            floor_output.short_alpha,
            floor_output.short_beta,
            floor_output.long_alpha,
            floor_output.long_beta,
        ):
            self.assertEqual(float(tensor.detach()), BETA_FLOOR)

    def test_12_flat_sample_has_exactly_zero_risk_and_no_beta_density(self):
        actor = Actor()
        pin_last_layer(actor, [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state = state_vector()

        sample = actor.sample(state)
        self.assertEqual(sample.direction, Direction.FLAT)
        self.assertEqual(sample.requested_risk, 0.0)
        self.assertIsNone(sample.selected_alpha)
        self.assertIsNone(sample.selected_beta)
        self.assertEqual(sample.nominal_action.direction, Direction.FLAT)

        # The FLAT log-probability is the categorical term alone.  The pinned
        # Beta parameters are both > 1, so their density at r = 0 is -inf: an
        # implementation that included the Beta term could not reproduce the
        # finite value asserted below.  The counterfactual is checked explicitly
        # so this test cannot silently stop discriminating.
        with torch.no_grad():
            reference = actor(state)
            counterfactual = Beta(reference.short_alpha, reference.short_beta).log_prob(
                torch.tensor(0.0)
            )
            self.assertFalse(bool(torch.isfinite(counterfactual)))
            expected = float(
                Categorical(logits=reference.direction_logits)
                .log_prob(torch.tensor(DIRECTION_INDEX_FLAT))
                .item()
            )
        self.assertAlmostEqual(sample.log_prob, expected, places=5)
        self.assertAlmostEqual(
            float(actor.log_prob(state, Direction.FLAT, 0.0).item()), expected, places=5
        )
        self.assertTrue(torch.isfinite(actor.log_prob(state, Direction.FLAT, 0.0)))

    def test_13_non_flat_sample_risk_is_bounded_and_log_prob_adds_beta_density(self):
        for index, direction in (
            (DIRECTION_INDEX_SHORT, Direction.SHORT),
            (DIRECTION_INDEX_LONG, Direction.LONG),
        ):
            bias = [0.0] * (len(DIRECTION_ORDER) + 4)
            bias[index] = 5.0
            actor = Actor()
            pin_last_layer(actor, bias)
            state = state_vector()

            torch.manual_seed(20240101 + index)
            sample = actor.sample(state)
            self.assertEqual(sample.direction, direction)
            self.assertGreaterEqual(sample.requested_risk, 0.0)
            self.assertLessEqual(sample.requested_risk, 1.0)

            with torch.no_grad():
                reference = actor(state)
                categorical_log_prob = float(
                    Categorical(logits=reference.direction_logits)
                    .log_prob(torch.tensor(index))
                    .item()
                )
                alpha = (
                    reference.short_alpha
                    if direction is Direction.SHORT
                    else reference.long_alpha
                )
                beta = (
                    reference.short_beta if direction is Direction.SHORT else reference.long_beta
                )
                beta_log_prob = float(
                    Beta(alpha, beta)
                    .log_prob(torch.tensor(sample.requested_risk, dtype=alpha.dtype))
                    .item()
                )
                recomputed = float(
                    actor.log_prob(state, direction, sample.requested_risk).item()
                )
            self.assertAlmostEqual(sample.log_prob, categorical_log_prob + beta_log_prob, places=5)
            self.assertAlmostEqual(sample.selected_alpha, float(alpha), places=6)
            self.assertAlmostEqual(sample.selected_beta, float(beta), places=6)
            self.assertAlmostEqual(recomputed, categorical_log_prob + beta_log_prob, places=5)

    def test_14_deterministic_adapter_uses_argmax_and_beta_mean_exactly(self):
        actor = Actor()
        for signed_exposure in (0.5, -0.5, 0.0):
            state = state_vector(signed_exposure=signed_exposure)
            output = actor(state)
            index = int(torch.argmax(output.direction_logits).item())
            action = actor.deterministic_action(state)
            self.assertEqual(action.direction, direction_from_index(index))
            if action.direction is Direction.FLAT:
                self.assertEqual(action.requested_risk, 0.0)
            else:
                alpha = float(
                    (output.short_alpha if index == DIRECTION_INDEX_SHORT else output.long_alpha).item()
                )
                beta = float(
                    (output.short_beta if index == DIRECTION_INDEX_SHORT else output.long_beta).item()
                )
                self.assertEqual(action.requested_risk, alpha / (alpha + beta))

        # Ties resolve to the ordinary first-index argmax, deterministically.
        tied = Actor()
        pin_last_layer(tied, [1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        state = state_vector()
        first = tied.deterministic_action(state)
        self.assertEqual(first.direction, Direction.SHORT)
        self.assertEqual(first, tied.deterministic_action(state))

    def test_15_invalid_or_non_finite_distribution_outputs_fail_closed(self):
        state = state_vector()
        actor = Actor()

        corrupted_actor = Actor()
        with torch.no_grad():
            corrupted_actor.mlp[-1].weight.fill_(float("nan"))
        with self.assertRaises(ContractError):
            corrupted_actor(state)

        crossed_actor = Actor()
        with torch.no_grad():
            crossed_actor.mlp[-1].bias.fill_(float("inf"))
        with self.assertRaises(ContractError):
            crossed_actor.sample(state)

        # Out-of-domain or semantically impossible risks never produce a value.
        for direction, risk in (
            (Direction.LONG, 1.5),
            (Direction.LONG, -0.25),
            (Direction.LONG, float("nan")),
            (Direction.LONG, 0.0),
            (Direction.FLAT, 0.25),
            (Direction.LONG, 1.0),  # Beta density is zero at the boundary -> -inf
        ):
            with self.subTest(direction=direction, risk=risk):
                with self.assertRaises(ContractError):
                    actor.log_prob(state, direction, risk)

        corrupted_state = state.clone()
        corrupted_state[0] = float("inf")
        with self.assertRaises(ContractError):
            actor(corrupted_state)
        with self.assertRaises(ContractError):
            actor.log_prob(corrupted_state, Direction.FLAT, 0.0)


class CriticTests(unittest.TestCase):
    def test_16_critic_returns_exactly_one_finite_scalar_per_state(self):
        critic = ValueCritic()
        single = critic(state_vector())
        self.assertEqual(tuple(single.shape), ())
        self.assertEqual(single.numel(), 1)
        self.assertTrue(bool(torch.isfinite(single)))

        states = state_batch(5)
        batch = critic(states)
        self.assertEqual(tuple(batch.shape), (5,))
        self.assertEqual(batch.numel(), 5)
        self.assertTrue(bool(torch.isfinite(batch).all()))
        # Batched and single-sample GEMM paths may differ in the last float32 bit.
        self.assertTrue(torch.allclose(batch[0], critic(states[0]), rtol=0.0, atol=1e-6))

        corrupted = ValueCritic()
        with torch.no_grad():
            corrupted.mlp[-1].weight.fill_(float("nan"))
        with self.assertRaises(ContractError):
            corrupted(state_vector())

    def test_17_actor_and_critic_share_no_parameter_objects(self):
        actor = Actor()
        critic = ValueCritic()
        actor_parameters = {id(parameter) for parameter in actor.parameters()}
        critic_parameters = {id(parameter) for parameter in critic.parameters()}
        self.assertTrue(actor_parameters)
        self.assertTrue(critic_parameters)
        self.assertFalse(actor_parameters & critic_parameters)

        # Separate modules, separate optimizers, disjoint parameter groups.
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        self.assertIsNot(actor_optimizer, critic_optimizer)
        actor_grouped = {id(p) for group in actor_optimizer.param_groups for p in group["params"]}
        critic_grouped = {id(p) for group in critic_optimizer.param_groups for p in group["params"]}
        self.assertFalse(actor_grouped & critic_grouped)


class FrozenSensoryBoundaryTests(unittest.TestCase):
    def test_sensory_is_an_explicit_part_of_the_policy_input_boundary(self):
        sensory = FrozenSensory()
        z = sensory(torch.randn(CONTEXT_LENGTH, MARKET_CHANNELS) * 0.05)
        state = build_learner_state(z, account())
        self.assertEqual(tuple(state.vector.shape), (Z_DIM + ACCOUNT_STATE_DIM,))
        self.assertTrue(torch.equal(state.vector[:Z_DIM], z))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
