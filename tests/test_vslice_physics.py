"""Known-answer tests for R12 VS-A Permission, Execution, and account physics.

Required tests 12-28 of ``docs/tasks/R12_VS_A_PHYSICS_R0.md`` section 9.
Expected values are hand-computed from the contract formulas with literal
arithmetic; the predicate is never re-derived from a second risk rule.

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

from cb16_science.vslice.contracts import (  # noqa: E402
    AccountSolvencyError,
    AccountTruth,
    ContractError,
    Direction,
    NominalAction,
    PhysicsConfig,
)
from cb16_science.vslice.physics import (  # noqa: E402
    PreTradeTruth,
    candidate_target_feasible,
    execute_transition,
    feasible_target_interval,
    mark_to_next_open,
    permit_target,
)

DEFAULT_CONFIG = PhysicsConfig()
#: The exact closed form of the flat pre-trade bisection bound 1 / (1 + kappa).
ANALYTIC_FLAT_BOUND = 1.0 / (1.0 + 1e-3)


def flat_truth(equity: float = 1000.0) -> AccountTruth:
    return AccountTruth(equity=equity, quantity=0.0, mark_price=100.0)


class PermissionIntervalTests(unittest.TestCase):
    def test_12_zero_friction_interval_is_exactly_the_hard_limit(self):
        config = PhysicsConfig(proportional_friction_kappa=0.0)
        pretrade = mark_to_next_open(flat_truth(), 101.0)
        self.assertEqual(feasible_target_interval(pretrade, config), (-1.0, 1.0))
        # Also exactly the hard limit for a non-flat book: no cost means no adjustment.
        held = mark_to_next_open(AccountTruth(1000.0, 2.0, 100.0), 100.0)
        self.assertEqual(feasible_target_interval(held, config), (-1.0, 1.0))

    def test_13_positive_friction_pulls_the_boundary_target_slightly_inward(self):
        pretrade = mark_to_next_open(flat_truth(), 101.0)
        self.assertFalse(candidate_target_feasible(pretrade, 1.0, DEFAULT_CONFIG))
        result = permit_target(pretrade, 1.0, DEFAULT_CONFIG)
        self.assertTrue(result.was_clipped)
        self.assertLess(result.feasible_max, 1.0)
        self.assertAlmostEqual(result.feasible_max, ANALYTIC_FLAT_BOUND, delta=1e-9)
        self.assertEqual(result.permitted_target, result.feasible_max)

    def test_14_every_returned_endpoint_is_feasible_within_tolerance(self):
        for quantity in (-3.0, -0.5, 0.0, 0.5, 3.0):
            with self.subTest(quantity=quantity):
                pretrade = mark_to_next_open(AccountTruth(1000.0, quantity, 100.0), 100.0)
                e_min, e_max = feasible_target_interval(pretrade, DEFAULT_CONFIG)
                self.assertTrue(candidate_target_feasible(pretrade, e_min, DEFAULT_CONFIG))
                self.assertTrue(candidate_target_feasible(pretrade, e_max, DEFAULT_CONFIG))
                self.assertLessEqual(e_min, 0.0)
                self.assertGreaterEqual(e_max, 0.0)

    def test_15_a_point_infinitesimally_beyond_the_bisection_endpoint_is_infeasible(self):
        pretrade = mark_to_next_open(flat_truth(), 101.0)
        e_min, e_max = feasible_target_interval(pretrade, DEFAULT_CONFIG)
        self.assertTrue(candidate_target_feasible(pretrade, e_max, DEFAULT_CONFIG))
        beyond = e_max * (1.0 + 1e-6)
        self.assertGreater(beyond, e_max + 1e-9)
        self.assertFalse(candidate_target_feasible(pretrade, beyond, DEFAULT_CONFIG))
        self.assertFalse(candidate_target_feasible(pretrade, e_min * (1.0 + 1e-6), DEFAULT_CONFIG))
        # The bisection result is also consistent with the closed-form boundary.
        self.assertLess(abs(e_max - ANALYTIC_FLAT_BOUND), 1e-9)
        # An infeasible boundary triggers the full configured bisection count.
        exact = PhysicsConfig(permission_bisection_iterations=1)
        looped = feasible_target_interval(pretrade, exact)
        self.assertEqual(looped, (-0.5, 0.5))

    def test_16_flat_target_remains_exactly_zero(self):
        for quantity in (-5.0, 0.0, 5.0):
            with self.subTest(quantity=quantity):
                pretrade = mark_to_next_open(AccountTruth(1000.0, quantity, 100.0), 100.0)
                action = NominalAction(Direction.FLAT, 0.0)
                step = execute_transition(
                    AccountTruth(1000.0, quantity, 100.0),
                    action,
                    open_next=100.0,
                    close_next=100.0,
                    config=DEFAULT_CONFIG,
                )
                result = permit_target(pretrade, step.nominal_target, DEFAULT_CONFIG)
                self.assertEqual(step.nominal_target, 0.0)
                self.assertEqual(result.permitted_target, 0.0)
                self.assertFalse(result.was_clipped)
                self.assertEqual(step.post_cost_quantity, 0.0)
                self.assertEqual(step.truth_next.quantity, 0.0)

    def test_17_out_of_envelope_target_clips_without_reversing_sign(self):
        config = PhysicsConfig(hard_exposure_limit=0.05, proportional_friction_kappa=0.001)
        truth = AccountTruth(equity=1000.0, quantity=0.0, mark_price=100.0)
        long_step = execute_transition(
            truth, NominalAction(Direction.LONG, 1.0), open_next=100.0, close_next=100.0, config=config
        )
        self.assertEqual(long_step.nominal_target, 1.0)
        self.assertGreater(long_step.permission.feasible_max, 0.0)
        self.assertLess(long_step.permission.feasible_max, long_step.nominal_target)
        self.assertTrue(long_step.permission.was_clipped)
        self.assertEqual(long_step.permission.permitted_target, long_step.permission.feasible_max)
        self.assertGreater(long_step.permission.permitted_target, 0.0)
        self.assertGreater(long_step.post_cost_quantity, 0.0)
        self.assertGreater(long_step.state_next.signed_exposure, 0.0)

        short_step = execute_transition(
            truth, NominalAction(Direction.SHORT, 0.9), open_next=100.0, close_next=100.0, config=config
        )
        self.assertEqual(short_step.nominal_target, -0.9)
        self.assertLess(short_step.post_cost_quantity, 0.0)
        self.assertLess(short_step.state_next.signed_exposure, 0.0)
        self.assertGreaterEqual(
            short_step.permission.permitted_target, short_step.permission.feasible_min
        )
        self.assertLessEqual(
            short_step.permission.permitted_target, short_step.permission.feasible_max
        )

    def test_18_exposure_above_the_hard_limit_after_a_gap_can_still_flatten(self):
        config = PhysicsConfig()
        truth = AccountTruth(equity=1000.0, quantity=20.0, mark_price=100.0)
        self.assertEqual(truth.exposure, 2.0)
        self.assertGreater(abs(truth.exposure), config.hard_exposure_limit)
        state = execute_transition(
            truth, NominalAction(Direction.FLAT, 0.0), open_next=100.0, close_next=100.0, config=config
        ).state_t
        self.assertEqual(state.signed_exposure, 2.0)
        self.assertEqual(state.new_risk_capacity, 0.0)

        # The gap pushes equity down and exposure further out; zero stays feasible.
        step = execute_transition(
            truth, NominalAction(Direction.FLAT, 0.0), open_next=95.0, close_next=95.0, config=config
        )
        self.assertEqual(step.pretrade_equity, 900.0)
        self.assertAlmostEqual(step.pretrade_exposure, 1900.0 / 900.0, places=12)
        self.assertEqual(step.permission.permitted_target, 0.0)
        self.assertEqual(step.truth_next.quantity, 0.0)
        self.assertAlmostEqual(step.truth_next.equity, 900.0 - 0.001 * 1900.0, places=12)

        # Moving only part of the way toward zero is also allowed under exhausted capacity.
        partial = execute_transition(
            truth, NominalAction(Direction.SHORT, 0.5), open_next=100.0, close_next=100.0, config=config
        )
        self.assertLess(partial.nominal_target, 0.0)
        self.assertLess(partial.post_cost_quantity, 20.0)
        self.assertLess(abs(partial.state_next.signed_exposure), 2.0)


class ExecutionCostTests(unittest.TestCase):
    def test_19_repeating_the_pretrade_target_has_zero_turnover_cost(self):
        config = PhysicsConfig(proportional_friction_kappa=0.001)
        truth = AccountTruth(equity=1000.0, quantity=0.2, mark_price=100.0)
        step = execute_transition(
            truth, NominalAction(Direction.LONG, 0.02), open_next=100.0, close_next=100.0, config=config
        )
        self.assertEqual(step.nominal_target, 0.02)
        self.assertEqual(step.permission.permitted_target, 0.02)
        self.assertEqual(step.executed_delta_notional, 0.0)
        self.assertEqual(step.trade_cost, 0.0)
        self.assertEqual(step.post_cost_equity, 1000.0)
        self.assertEqual(step.post_cost_quantity, 0.2)
        self.assertEqual(step.reward, 0.0)

    def test_20_full_reversal_charges_the_full_signed_notional_change(self):
        config = PhysicsConfig(proportional_friction_kappa=0.001)
        truth = AccountTruth(equity=1000.0, quantity=5.0, mark_price=100.0)
        step = execute_transition(
            truth, NominalAction(Direction.SHORT, 0.5), open_next=100.0, close_next=100.0, config=config
        )
        self.assertEqual(step.nominal_target, -0.5)
        self.assertEqual(step.pretrade_notional, 500.0)
        self.assertEqual(step.permission.permitted_target, -0.5)
        # Full reversal: |Delta N| = |-500 - 500| = 1000, i.e. one equity unit of turnover.
        self.assertEqual(step.executed_delta_notional, -1000.0)
        self.assertEqual(step.trade_cost, 1.0)
        self.assertEqual(step.post_cost_equity, 999.0)
        self.assertEqual(step.post_cost_quantity, -5.0)
        self.assertEqual(step.truth_next.quantity, -5.0)

    def test_20b_expensive_reversal_walks_inward_by_predicate(self):
        # 1000 -> 500 is either a 500-unit de-risk or a 1500-unit reversal; with kappa=0.2
        # the reversal costs 300, which lifts post-cost exposure to 1.42 > 1, so the
        # -1.0 boundary is infeasible and Permission must walk the short side inward.
        config = PhysicsConfig(proportional_friction_kappa=0.2)
        truth = AccountTruth(equity=1000.0, quantity=5.0, mark_price=100.0)
        pretrade = mark_to_next_open(truth, 100.0)
        self.assertFalse(candidate_target_feasible(pretrade, -1.0, config))
        self.assertFalse(candidate_target_feasible(pretrade, 1.0, config))
        # The short side is squeezed much harder: reversing pays both legs of turnover.
        # Closed forms follow from |x| = 1 - kappa|x - 0.5|: x<0 gives |x| = 3/4,
        # 0 <= x < 0.5 gives 3/4 too, and x >= 0.5 gives 11/12.
        e_min, e_max = feasible_target_interval(pretrade, config)
        self.assertAlmostEqual(e_min, -3.0 / 4.0, delta=1e-6)
        self.assertAlmostEqual(e_max, 11.0 / 12.0, delta=1e-6)
        self.assertGreater(abs(e_min), 0.0)
        self.assertLess(abs(e_min), e_max)
        step = execute_transition(
            truth, NominalAction(Direction.SHORT, 1.0), open_next=100.0, close_next=100.0, config=config
        )
        self.assertEqual(step.nominal_target, -1.0)
        self.assertTrue(step.permission.was_clipped)
        self.assertEqual(step.permission.permitted_target, step.permission.feasible_min)
        self.assertGreater(step.permission.permitted_target, -1.0)
        self.assertTrue(candidate_target_feasible(pretrade, step.permission.permitted_target, config))

    def test_21_greater_absolute_turnover_never_costs_less(self):
        config = PhysicsConfig(proportional_friction_kappa=0.001)
        truth = AccountTruth(equity=1000.0, quantity=0.0, mark_price=100.0)
        measured = []
        for requested_risk in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            direction = Direction.LONG if requested_risk else Direction.FLAT
            step = execute_transition(
                truth,
                NominalAction(direction, requested_risk),
                open_next=100.0,
                close_next=100.0,
                config=config,
            )
            measured.append((abs(step.executed_delta_notional), step.trade_cost))
        for (turnover_a, cost_a), (turnover_b, cost_b) in zip(measured, measured[1:]):
            with self.subTest(turnover_a=turnover_a, turnover_b=turnover_b):
                self.assertLess(turnover_a, turnover_b)
                self.assertLess(cost_a, cost_b)

        pretrade = mark_to_next_open(truth, 100.0)
        for target in (0.1, 0.25, 0.5, 0.75):
            with self.subTest(target=target):
                self.assertTrue(candidate_target_feasible(pretrade, target, config))
                notional_star = target * pretrade.equity
                expected = 0.001 * abs(notional_star - pretrade.notional)
                self.assertAlmostEqual(expected, target * 1000.0 * 0.001, places=12)
                step = execute_transition(
                    truth,
                    NominalAction(Direction.LONG, target),
                    open_next=100.0,
                    close_next=100.0,
                    config=config,
                )
                self.assertEqual(step.trade_cost, expected)
        # The boundary target 1.0 is *not* feasible here because its own cost would
        # push post-cost exposure above the hard limit; the executed ladder therefore
        # ends one rung lower and still costs strictly more than the rung below it.
        self.assertFalse(candidate_target_feasible(pretrade, 1.0, config))

    def test_22_actual_post_cost_exposure_is_recomputed_from_truth(self):
        config = PhysicsConfig(proportional_friction_kappa=0.001)
        step = execute_transition(
            flat_truth(),
            NominalAction(Direction.LONG, 0.5),
            open_next=100.0,
            close_next=100.0,
            config=config,
        )
        permitted = step.permission.permitted_target
        self.assertEqual(permitted, 0.5)
        self.assertEqual(step.trade_cost, 0.5)
        self.assertEqual(step.post_cost_equity, 999.5)
        self.assertEqual(step.post_cost_quantity, 5.0)
        self.assertAlmostEqual(step.post_cost_exposure, 500.0 / 999.5, places=12)
        self.assertNotAlmostEqual(step.post_cost_exposure, permitted, places=12)
        # The recomputed exposure is what the next decision state reports.
        self.assertAlmostEqual(
            step.state_next.signed_exposure, step.post_cost_exposure, places=12
        )

    def test_23_monetary_scaling_scales_dollars_and_leaves_normalized_values(self):
        config = PhysicsConfig(proportional_friction_kappa=0.001)
        action = NominalAction(Direction.SHORT, 0.4)
        base_truth = AccountTruth(equity=1000.0, quantity=0.0, mark_price=100.0)
        base = execute_transition(base_truth, action, open_next=98.0, close_next=101.0, config=config)
        for factor in (0.25, 3.0, 1e6):
            with self.subTest(factor=factor):
                # Scaling every monetary quantity (equity, prices) by one positive
                # constant is a pure change of money unit.
                scaled_truth = AccountTruth(
                    equity=base_truth.equity * factor,
                    quantity=base_truth.quantity,
                    mark_price=base_truth.mark_price * factor,
                )
                scaled = execute_transition(
                    scaled_truth,
                    action,
                    open_next=98.0 * factor,
                    close_next=101.0 * factor,
                    config=config,
                )
                # Dollar magnitudes scale; normalized exposures and reward do not.
                self.assertAlmostEqual(
                    scaled.trade_cost / base.trade_cost, factor, delta=1e-9 * factor
                )
                self.assertAlmostEqual(
                    scaled.post_cost_equity / base.post_cost_equity, factor, delta=1e-9 * factor
                )
                self.assertAlmostEqual(
                    scaled.truth_next.equity / base.truth_next.equity, factor, delta=1e-9 * factor
                )
                self.assertAlmostEqual(
                    scaled.post_cost_quantity, base.post_cost_quantity, places=12
                )
                self.assertAlmostEqual(
                    scaled.post_cost_exposure, base.post_cost_exposure, places=12
                )
                self.assertAlmostEqual(
                    scaled.state_next.signed_exposure, base.state_next.signed_exposure, places=12
                )
                self.assertAlmostEqual(
                    scaled.state_next.new_risk_capacity,
                    base.state_next.new_risk_capacity,
                    places=12,
                )
                self.assertAlmostEqual(scaled.reward, base.reward, delta=1e-12)


class RewardAndFailureTests(unittest.TestCase):
    def test_24_flat_market_with_no_turnover_yields_exactly_zero_reward(self):
        config = PhysicsConfig(proportional_friction_kappa=0.0)
        truth = AccountTruth(equity=1000.0, quantity=0.0, mark_price=100.0)
        step = execute_transition(
            truth, NominalAction(Direction.FLAT, 0.0), open_next=100.0, close_next=100.0, config=config
        )
        self.assertEqual(step.reward, 0.0)
        self.assertEqual(step.truth_next.equity, 1000.0)
        # A held flat book in a flat market also produces exactly zero reward.
        held = execute_transition(
            AccountTruth(1000.0, 2.0, 100.0),
            NominalAction(Direction.LONG, 0.2),
            open_next=100.0,
            close_next=100.0,
            config=config,
        )
        self.assertEqual(held.reward, 0.0)

    def test_25_reward_equals_log_equity_ratio_on_a_hand_computed_transition(self):
        config = PhysicsConfig(proportional_friction_kappa=0.001)
        truth = AccountTruth(equity=1000.0, quantity=10.0, mark_price=100.0)
        step = execute_transition(
            truth, NominalAction(Direction.FLAT, 0.0), open_next=102.0, close_next=105.0, config=config
        )
        # Hand arithmetic: gap 10 * 2 = 20 -> W- 1020; flatten 10 units at 102 -> cost 1.02.
        self.assertEqual(step.pretrade_equity, 1020.0)
        self.assertEqual(step.trade_cost, 1.02)
        self.assertEqual(step.post_cost_equity, 1018.98)
        self.assertEqual(step.truth_next.equity, 1018.98)
        self.assertEqual(step.reward, math.log(1018.98 / 1000.0))
        self.assertAlmostEqual(step.reward, math.log(1018.98) - math.log(1000.0), delta=1e-12)

        # A profitable long marked at the next close, with no rebalancing cost.
        held = execute_transition(
            AccountTruth(1000.0, 5.0, 100.0),
            NominalAction(Direction.LONG, 0.5),
            open_next=100.0,
            close_next=106.0,
            config=config,
        )
        self.assertEqual(held.trade_cost, 0.0)
        self.assertEqual(held.truth_next.equity, 1030.0)
        self.assertEqual(held.reward, math.log(1030.0 / 1000.0))

    def test_26_invalid_prices_equity_and_non_finite_values_fail_explicitly(self):
        for bad_price in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(open_next=bad_price):
                with self.assertRaises(ContractError):
                    execute_transition(
                        flat_truth(), NominalAction(Direction.FLAT, 0.0), bad_price, 100.0, DEFAULT_CONFIG
                    )
            with self.subTest(close_next=bad_price):
                with self.assertRaises(ContractError):
                    execute_transition(
                        flat_truth(), NominalAction(Direction.FLAT, 0.0), 100.0, bad_price, DEFAULT_CONFIG
                    )
            with self.subTest(mark_to_next_open=bad_price):
                with self.assertRaises(ContractError):
                    mark_to_next_open(flat_truth(), bad_price)
        with self.assertRaises(ContractError):
            AccountTruth(equity=math.nan, quantity=0.0, mark_price=100.0)
        with self.assertRaises(AccountSolvencyError):
            AccountTruth(equity=0.0, quantity=0.0, mark_price=100.0)

    def test_27_no_free_principal_reset_on_any_failure_path(self):
        config = PhysicsConfig(proportional_friction_kappa=0.001)
        losing_truth = AccountTruth(equity=1000.0, quantity=-10.0, mark_price=100.0)
        snapshot = (losing_truth.equity, losing_truth.quantity, losing_truth.mark_price)
        failures = (
            (
                ContractError,
                lambda: execute_transition(
                    losing_truth, NominalAction(Direction.FLAT, 0.0), math.nan, 103.0, config
                ),
            ),
            (
                ContractError,
                lambda: execute_transition(
                    losing_truth, NominalAction(Direction.FLAT, 0.0), 103.0, 0.0, config
                ),
            ),
            (
                AccountSolvencyError,
                lambda: AccountTruth(equity=-5.0, quantity=0.0, mark_price=100.0),
            ),
            (
                AccountSolvencyError,
                # An adverse gap makes the gap-adjusted equity non-positive before execution.
                lambda: execute_transition(
                    losing_truth, NominalAction(Direction.FLAT, 0.0), 320.0, 320.0, config
                ),
            ),
        )
        for index, (expected_error, failing) in enumerate(failures):
            with self.subTest(failure=index):
                with self.assertRaises(expected_error):
                    failing()
                self.assertEqual(
                    (losing_truth.equity, losing_truth.quantity, losing_truth.mark_price), snapshot
                )
        # A legal continuation after those failures marks the loss honestly and resets nothing.
        step = execute_transition(
            losing_truth, NominalAction(Direction.FLAT, 0.0), open_next=103.0, close_next=103.0, config=config
        )
        expected_equity_minus = 1000.0 + (-10.0) * (103.0 - 100.0)
        expected_cost = 0.001 * abs(-10.0 * 103.0)
        self.assertEqual(step.pretrade_equity, expected_equity_minus)
        self.assertEqual(step.trade_cost, expected_cost)
        self.assertEqual(step.truth_next.equity, expected_equity_minus - expected_cost)
        self.assertEqual(step.reward, math.log((expected_equity_minus - expected_cost) / 1000.0))
        self.assertLess(step.reward, 0.0)

    def test_28_terminal_and_truncated_remain_false_for_ordinary_transitions(self):
        config = PhysicsConfig()
        transitions = (
            (flat_truth(), NominalAction(Direction.LONG, 0.5), 101.0, 104.0),
            (flat_truth(), NominalAction(Direction.FLAT, 0.0), 101.0, 99.0),
            (AccountTruth(1000.0, -4.0, 100.0), NominalAction(Direction.SHORT, 1.0), 102.0, 101.0),
            (AccountTruth(1000.0, 4.0, 100.0), NominalAction(Direction.LONG, 1.0), 98.0, 99.0),
        )
        for truth, action, open_next, close_next in transitions:
            with self.subTest(action=action, open_next=open_next):
                step = execute_transition(truth, action, open_next, close_next, config)
                self.assertFalse(step.terminal)
                self.assertFalse(step.truncated)
                self.assertFalse(step.forced_risk)
                self.assertFalse(step.permission.forced_risk)


class PreTradeTruthBoundaryTests(unittest.TestCase):
    """Regression tests for the fail-closed pre-trade truth closure.

    ``PreTradeTruth`` stores only primitives and derives ``notional`` and
    ``exposure``; a malformed record must be rejected explicitly instead of
    being reinterpreted as ordinary infeasibility.
    """

    def test_construction_derives_notional_and_exposure(self):
        pretrade = PreTradeTruth(equity=1000.0, quantity=3.0, open_next=105.0)
        self.assertEqual(pretrade.notional, 315.0)
        self.assertEqual(pretrade.exposure, 0.315)
        # Only primitives are stored; the derived fields cannot contradict them.
        self.assertEqual(
            tuple(pretrade.__dataclass_fields__), ("equity", "quantity", "open_next")
        )

    def test_malformed_construction_fails_closed(self):
        cases = (
            (ContractError, {"equity": math.nan, "quantity": 0.0, "open_next": 100.0}),
            (ContractError, {"equity": math.inf, "quantity": 0.0, "open_next": 100.0}),
            (ContractError, {"equity": -math.inf, "quantity": 0.0, "open_next": 100.0}),
            (AccountSolvencyError, {"equity": 0.0, "quantity": 0.0, "open_next": 100.0}),
            (AccountSolvencyError, {"equity": -1.0, "quantity": 4.0, "open_next": 100.0}),
            (ContractError, {"equity": 1000.0, "quantity": math.nan, "open_next": 100.0}),
            (ContractError, {"equity": 1000.0, "quantity": math.inf, "open_next": 100.0}),
            (ContractError, {"equity": 1000.0, "quantity": 0.0, "open_next": 0.0}),
            (ContractError, {"equity": 1000.0, "quantity": 0.0, "open_next": -1.0}),
            (ContractError, {"equity": 1000.0, "quantity": 0.0, "open_next": math.nan}),
            (ContractError, {"equity": 1000.0, "quantity": 0.0, "open_next": math.inf}),
            # Derived quantities must stay finite as well.
            (ContractError, {"equity": 1.0, "quantity": 1e308, "open_next": 1e308}),
            (ContractError, {"equity": 1e-300, "quantity": 1.0, "open_next": 1e300}),
        )
        for expected_error, fields in cases:
            with self.subTest(**fields):
                with self.assertRaises(expected_error):
                    PreTradeTruth(**fields)

    def test_public_physics_functions_reject_forged_malformed_pretrade(self):
        config = PhysicsConfig()
        for fields in (
            {"equity": -1000.0, "quantity": 0.0, "open_next": 100.0},
            {"equity": math.nan, "quantity": 0.0, "open_next": 100.0},
            {"equity": 1000.0, "quantity": math.inf, "open_next": 100.0},
            {"equity": 1000.0, "quantity": 0.0, "open_next": 0.0},
            {"equity": 1.0, "quantity": 1e308, "open_next": 1e308},
        ):
            # Bypass construction to prove the public functions re-validate.
            forged = object.__new__(PreTradeTruth)
            for name, value in fields.items():
                object.__setattr__(forged, name, value)
            with self.subTest(**fields):
                with self.assertRaises(ContractError):
                    candidate_target_feasible(forged, 0.0, config)
                with self.assertRaises(ContractError):
                    feasible_target_interval(forged, config)
                with self.assertRaises(ContractError):
                    permit_target(forged, 0.5, config)

    def test_non_pretrade_values_are_rejected(self):
        config = PhysicsConfig()
        for bad in (None, 0.0, {"equity": 1000.0, "quantity": 0.0, "open_next": 100.0}):
            with self.subTest(bad=bad):
                with self.assertRaises(ContractError):
                    candidate_target_feasible(bad, 0.0, config)  # type: ignore[arg-type]

    def test_mark_to_next_open_reports_insolvency_without_building_a_record(self):
        truth = AccountTruth(equity=1000.0, quantity=-10.0, mark_price=100.0)
        with self.assertRaises(AccountSolvencyError):
            mark_to_next_open(truth, 320.0)
        valid = mark_to_next_open(truth, 103.0)
        self.assertEqual(valid.notional, -10.0 * 103.0)
        self.assertEqual(valid.exposure, valid.notional / valid.equity)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
