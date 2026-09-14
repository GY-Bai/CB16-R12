"""Known-answer tests for the R12 VS-A canonical contracts (required tests 1-11).

Every expected value is derived by hand from
``docs/tasks/R12_VS_A_PHYSICS_R0.md`` (sections 1 and 3) with explicit
``math``/literal arithmetic, so the tests cannot be satisfied by re-running the
implementation's own expression.

    python3 -m unittest discover -s tests -t .
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "science") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "science"))

from cb16_science.vslice import contracts  # noqa: E402
from cb16_science.vslice.contracts import (  # noqa: E402
    AccountSolvencyError,
    AccountState,
    AccountTruth,
    ContractError,
    Direction,
    NominalAction,
    PhysicsConfig,
    account_state_from_truth,
    nominal_target_from_action,
)
from cb16_science.vslice.physics import execute_transition  # noqa: E402

FROZEN_DEFAULTS = {
    "context_length": 64,
    "nominal_exposure_budget": 1.0,
    "hard_exposure_limit": 1.0,
    "proportional_friction_kappa": 0.001,
    "permission_bisection_iterations": 80,
    "finite_tolerance": 1e-12,
}


class FrozenConfigTests(unittest.TestCase):
    def test_frozen_controlled_slice_defaults_are_exposed(self):
        config = PhysicsConfig()
        for name, expected in FROZEN_DEFAULTS.items():
            self.assertEqual(getattr(config, name), expected, name)

    def test_config_is_immutable(self):
        config = PhysicsConfig()
        with self.assertRaises(Exception):
            config.hard_exposure_limit = 2.0  # type: ignore[misc]

    def test_config_requirements_are_enforced(self):
        cases = {
            "nominal_exposure_budget": {"nominal_exposure_budget": 0.0},
            "hard_exposure_limit": {"hard_exposure_limit": -1.0},
            "negative_kappa": {"proportional_friction_kappa": -0.001},
            "kappa_limit_product": {
                "proportional_friction_kappa": 1.0,
                "hard_exposure_limit": 1.0,
            },
            "iterations": {"permission_bisection_iterations": 0},
        }
        for name, overrides in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ContractError):
                    PhysicsConfig(**overrides)

    def test_non_finite_configuration_is_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            for name in (
                "nominal_exposure_budget",
                "hard_exposure_limit",
                "proportional_friction_kappa",
                "finite_tolerance",
            ):
                with self.subTest(value=bad, name=name):
                    with self.assertRaises(ContractError):
                        PhysicsConfig(**{name: bad})

    def test_alternate_known_answer_config_is_allowed(self):
        config = PhysicsConfig(
            context_length=2,
            nominal_exposure_budget=2.5,
            hard_exposure_limit=4.0,
            proportional_friction_kappa=0.2,
            permission_bisection_iterations=3,
            finite_tolerance=0.0,
        )
        self.assertEqual(config.nominal_exposure_budget, 2.5)
        self.assertEqual(config.permission_bisection_iterations, 3)

    def test_context_length_counts_represented_bars(self):
        # The retained predecessor is an extra input bar, so one represented
        # bar is semantically valid and zero is not.
        self.assertEqual(PhysicsConfig(context_length=1).context_length, 1)
        for bad in (0, -1):
            with self.subTest(context_length=bad):
                with self.assertRaises(ContractError):
                    PhysicsConfig(context_length=bad)


class DirectionTests(unittest.TestCase):
    def test_direction_values_are_exact(self):
        self.assertEqual(int(Direction.SHORT), -1)
        self.assertEqual(int(Direction.FLAT), 0)
        self.assertEqual(int(Direction.LONG), 1)

    def test_1_long_short_flat_canonical_valid_cases(self):
        self.assertEqual(NominalAction(Direction.LONG, 1.0).requested_risk, 1.0)
        self.assertEqual(NominalAction(Direction.SHORT, 0.25).requested_risk, 0.25)
        self.assertEqual(NominalAction(Direction.FLAT, 0.0).requested_risk, 0.0)

    def test_2_reject_flat_with_non_zero_risk(self):
        for risk in (1e-300, 0.5, 1.0):
            with self.subTest(risk=risk):
                with self.assertRaises(ContractError):
                    NominalAction(Direction.FLAT, risk)

    def test_3_reject_long_short_with_zero_risk(self):
        for direction in (Direction.LONG, Direction.SHORT):
            with self.subTest(direction=direction):
                with self.assertRaises(ContractError):
                    NominalAction(direction, 0.0)

    def test_4a_reject_risk_outside_unit_interval(self):
        for bad in (-1e-9, -1.0, 1.0 + 1e-9, 2.0):
            with self.subTest(risk=bad):
                with self.assertRaises(ContractError):
                    NominalAction(Direction.LONG, bad)

    def test_4b_reject_nan_and_inf_risk(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(risk=bad):
                with self.assertRaises(ContractError):
                    NominalAction(Direction.LONG, bad)

    def test_unknown_direction_is_rejected(self):
        for bad in (2, -2, "LONG", None, True):
            with self.subTest(direction=bad):
                with self.assertRaises(ContractError):
                    NominalAction(bad, 0.5)  # type: ignore[arg-type]

    def test_5_exact_nominal_target_formula(self):
        config = PhysicsConfig()
        self.assertEqual(nominal_target_from_action(NominalAction(Direction.LONG, 0.6), config), 0.6)
        self.assertEqual(nominal_target_from_action(NominalAction(Direction.SHORT, 0.4), config), -0.4)
        self.assertEqual(nominal_target_from_action(NominalAction(Direction.FLAT, 0.0), config), 0.0)
        wide = PhysicsConfig(nominal_exposure_budget=2.5)
        self.assertEqual(nominal_target_from_action(NominalAction(Direction.LONG, 0.4), wide), 0.4 * 2.5)
        self.assertEqual(nominal_target_from_action(NominalAction(Direction.SHORT, 0.4), wide), -0.4 * 2.5)


class AccountTruthTests(unittest.TestCase):
    def test_notional_and_exposure_are_the_marked_quantities(self):
        truth = AccountTruth(equity=2000.0, quantity=-3.0, mark_price=110.0)
        self.assertEqual(truth.notional, -330.0)
        self.assertEqual(truth.exposure, -330.0 / 2000.0)

    def test_non_positive_equity_or_mark_price_is_rejected(self):
        with self.assertRaises(AccountSolvencyError):
            AccountTruth(equity=0.0, quantity=0.0, mark_price=100.0)
        with self.assertRaises(AccountSolvencyError):
            AccountTruth(equity=-1.0, quantity=0.0, mark_price=100.0)
        with self.assertRaises(ContractError):
            AccountTruth(equity=1000.0, quantity=0.0, mark_price=0.0)

    def test_non_finite_truth_is_rejected(self):
        for kwargs in (
            {"equity": math.nan, "quantity": 0.0, "mark_price": 100.0},
            {"equity": math.inf, "quantity": 0.0, "mark_price": 100.0},
            {"equity": 1000.0, "quantity": math.nan, "mark_price": 100.0},
            {"equity": 1000.0, "quantity": math.inf, "mark_price": 100.0},
            {"equity": 1000.0, "quantity": 0.0, "mark_price": math.nan},
            {"equity": 1000.0, "quantity": 0.0, "mark_price": math.inf},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ContractError):
                    AccountTruth(**kwargs)

    def test_quantity_may_be_negative_zero_or_positive(self):
        for quantity in (-5.0, 0.0, 5.0):
            with self.subTest(quantity=quantity):
                self.assertEqual(AccountTruth(1000.0, quantity, 100.0).quantity, quantity)


class AccountStateTests(unittest.TestCase):
    def setUp(self):
        self.config = PhysicsConfig()

    def test_6_flat_account_state(self):
        state = account_state_from_truth(AccountTruth(1000.0, 0.0, 100.0), self.config)
        self.assertEqual(state, AccountState(signed_exposure=0.0, survival_cushion=1.0, new_risk_capacity=1.0))

    def test_7_signed_half_exposures(self):
        long_state = account_state_from_truth(AccountTruth(1000.0, 5.0, 100.0), self.config)
        short_state = account_state_from_truth(AccountTruth(1000.0, -5.0, 100.0), self.config)
        for state, expected_exposure in ((long_state, 0.5), (short_state, -0.5)):
            self.assertEqual(state.signed_exposure, expected_exposure)
            self.assertEqual(state.survival_cushion, 1.0)
            self.assertEqual(state.new_risk_capacity, 0.5)

    def test_8_beyond_envelope_is_not_clipped(self):
        state = account_state_from_truth(AccountTruth(1000.0, 25.0, 100.0), self.config)
        self.assertEqual(state.signed_exposure, 2.5)
        self.assertEqual(state.new_risk_capacity, 0.0)
        self.assertEqual(state.survival_cushion, 1.0)

    def test_9_account_state_scale_invariance(self):
        base = AccountTruth(equity=1000.0, quantity=6.0, mark_price=100.0)
        base_state = account_state_from_truth(base, self.config)
        for factor in (1e-6, 0.5, 3.0, 1e6):
            with self.subTest(factor=factor):
                scaled = AccountTruth(
                    equity=base.equity * factor,
                    quantity=base.quantity * factor,
                    mark_price=base.mark_price,
                )
                self.assertAlmostEqual(scaled.notional, base.notional * factor, places=12)
                self.assertAlmostEqual(scaled.exposure, base.exposure, places=12)
                scaled_state = account_state_from_truth(scaled, self.config)
                self.assertAlmostEqual(scaled_state.signed_exposure, base_state.signed_exposure, places=12)
                self.assertAlmostEqual(scaled_state.survival_cushion, base_state.survival_cushion, places=12)
                self.assertAlmostEqual(scaled_state.new_risk_capacity, base_state.new_risk_capacity, places=12)

    def test_state_fields_are_exactly_the_v1_triple(self):
        state = account_state_from_truth(AccountTruth(1000.0, 5.0, 100.0), self.config)
        self.assertEqual(
            tuple(field for field in state.__dataclass_fields__),
            ("signed_exposure", "survival_cushion", "new_risk_capacity"),
        )


class TimingAndContinuityTests(unittest.TestCase):
    def test_10_old_quantity_earns_the_gap(self):
        config = PhysicsConfig(proportional_friction_kappa=0.0)
        truth = AccountTruth(equity=1000.0, quantity=10.0, mark_price=100.0)
        step = execute_transition(
            truth, NominalAction(Direction.FLAT, 0.0), open_next=104.0, close_next=104.0, config=config
        )
        expected_equity_minus = 1000.0 + 10.0 * (104.0 - 100.0)
        self.assertEqual(step.pretrade_equity, expected_equity_minus)
        self.assertEqual(step.pretrade_notional, 10.0 * 104.0)
        self.assertEqual(step.pretrade_exposure, step.pretrade_notional / step.pretrade_equity)
        # The gap is banked before execution, so it survives flattening.
        self.assertEqual(step.post_cost_equity, expected_equity_minus)
        self.assertEqual(step.truth_next.equity, expected_equity_minus)
        self.assertEqual(step.truth_next.quantity, 0.0)
        self.assertEqual(step.reward, math.log(expected_equity_minus / 1000.0))

    def test_11_consecutive_transitions_continue_the_account(self):
        config = PhysicsConfig(proportional_friction_kappa=0.001)
        truth_0 = AccountTruth(equity=1000.0, quantity=0.0, mark_price=100.0)
        first = execute_transition(
            truth_0, NominalAction(Direction.LONG, 0.6), open_next=101.0, close_next=105.0, config=config
        )
        # Hand arithmetic: 1000 -> LONG 0.6 at open 101 (cost 0.606) -> 6 units -> close 105.
        equity_minus_1 = 1000.0
        expected_cost_1 = 0.001 * abs(0.6 * equity_minus_1 - 0.0)
        equity_plus_1 = equity_minus_1 - expected_cost_1
        quantity_1 = (0.6 * equity_minus_1) / 101.0
        equity_1 = equity_plus_1 + quantity_1 * (105.0 - 101.0)
        self.assertEqual(first.trade_cost, expected_cost_1)
        self.assertEqual(first.post_cost_quantity, quantity_1)
        self.assertEqual(first.truth_next, AccountTruth(equity_1, quantity_1, 105.0))

        # The second call starts from the first call's next truth; nothing is reset.
        second = execute_transition(
            first.truth_next, NominalAction(Direction.FLAT, 0.0), open_next=105.5, close_next=106.0, config=config
        )
        self.assertEqual(second.truth_t, first.truth_next)
        equity_minus_2 = equity_1 + quantity_1 * (105.5 - 105.0)
        self.assertEqual(second.pretrade_equity, equity_minus_2)
        self.assertEqual(second.nominal_target, 0.0)
        expected_cost_2 = 0.001 * abs(0.0 - quantity_1 * 105.5)
        equity_2 = equity_minus_2 - expected_cost_2
        self.assertEqual(second.trade_cost, expected_cost_2)
        self.assertEqual(second.post_cost_quantity, 0.0)
        self.assertEqual(second.truth_next, AccountTruth(equity_2, 0.0, 106.0))
        self.assertEqual(second.reward, math.log(equity_2 / equity_1))

        # A third call re-opens from the second call's truth: capital continues to compound.
        third = execute_transition(
            second.truth_next, NominalAction(Direction.LONG, 0.5), open_next=106.0, close_next=107.5, config=config
        )
        expected_quantity_3 = (0.5 * equity_2) / 106.0
        self.assertEqual(third.pretrade_notional, 0.0)
        self.assertEqual(third.trade_cost, 0.001 * abs(0.5 * equity_2))
        self.assertEqual(third.post_cost_quantity, expected_quantity_3)
        self.assertEqual(third.truth_next.quantity, expected_quantity_3)
        self.assertEqual(third.truth_next.equity, third.post_cost_equity + expected_quantity_3 * 1.5)

    def test_account_truth_objects_are_never_mutated_by_a_transition(self):
        config = PhysicsConfig()
        truth = AccountTruth(equity=1000.0, quantity=4.0, mark_price=100.0)
        snapshot = (truth.equity, truth.quantity, truth.mark_price)
        execute_transition(
            truth, NominalAction(Direction.SHORT, 0.5), open_next=99.0, close_next=98.0, config=config
        )
        self.assertEqual((truth.equity, truth.quantity, truth.mark_price), snapshot)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
