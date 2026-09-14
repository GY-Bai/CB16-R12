"""Deterministic R12 VS-A account physics: mark, permission, execution, reward.

Formulas are copied literally from ``docs/tasks/R12_VS_A_PHYSICS_R0.md``
sections 3-7 and from ``docs/R12_ACTION_PERMISSION_ACCOUNT_PHYSICS.md``.  The
module is pure: every function is deterministic, stateless and locally
testable.  There is no hidden mutable account object, no venue identity, no
market forecast, no strategy logic, no forced liquidation and no principal
reset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .contracts import (
    AccountSolvencyError,
    AccountState,
    AccountTruth,
    ContractError,
    NominalAction,
    PhysicsConfig,
    _require_finite_float,
    _require_positive_float,
    account_state_from_truth,
    nominal_target_from_action,
)

#: Reserved for a later forced-risk rule.  VS-A never liquidates, so every
#: ordinary controlled-slice transition reports the neutral value.
FORCED_RISK_DEFAULT = False

#: VS-A ordinary transitions are neither economically terminal nor
#: computationally truncated.
TERMINAL_DEFAULT = False
TRUNCATED_DEFAULT = False


def _validate_pretrade(pretrade: "PreTradeTruth") -> None:
    """Fail closed on a malformed pre-trade record.

    ``notional`` and ``exposure`` are derived from the primitive fields, so the
    only possible contradictions are non-finite or non-positive primitives and
    non-finite derived quantities.  Every rule is checked here; callers must
    never reinterpret a malformed record as ordinary infeasibility.
    """

    equity = _require_finite_float(pretrade.equity, "pre-trade equity")
    if not equity > 0.0:
        raise AccountSolvencyError(
            f"pre-trade equity must be strictly positive, got {pretrade.equity!r}"
        )
    quantity = _require_finite_float(pretrade.quantity, "pre-trade quantity")
    price = _require_positive_float(pretrade.open_next, "open_next")
    notional = quantity * price
    _require_finite_float(notional, "pre-trade notional")
    _require_finite_float(notional / equity, "pre-trade exposure")


@dataclass(frozen=True)
class PreTradeTruth:
    """Execution-time truth at ``open_{t+1}`` for the position inherited from ``t``.

    Only the primitive fields ``equity`` (``W_minus``), ``quantity`` (``q_t``)
    and ``open_next`` are stored; ``notional`` (``N_minus``) and ``exposure``
    (``a_minus``) are derived from them, so one record can never hold
    contradictory redundant values.  Construction fails closed on any
    non-finite value, non-positive equity or non-positive execution price.
    """

    equity: float
    quantity: float
    open_next: float

    def __post_init__(self) -> None:
        _validate_pretrade(self)
        object.__setattr__(self, "equity", float(self.equity))
        object.__setattr__(self, "quantity", float(self.quantity))
        object.__setattr__(self, "open_next", float(self.open_next))

    @property
    def notional(self) -> float:
        """``N_minus = q_t * open_next``, derived and never independently stored."""

        return self.quantity * self.open_next

    @property
    def exposure(self) -> float:
        """``a_minus = N_minus / W_minus``, derived and never independently stored."""

        return self.notional / self.equity


@dataclass(frozen=True)
class PermissionResult:
    """Nominal intent, feasible envelope and permitted target, kept separate."""

    nominal_target: float
    feasible_min: float
    feasible_max: float
    permitted_target: float
    was_clipped: bool
    forced_risk: bool = FORCED_RISK_DEFAULT


@dataclass(frozen=True)
class PhysicsStep:
    """One complete VS-A transition record (section 7, required return record)."""

    truth_t: AccountTruth
    state_t: AccountState
    nominal_action: NominalAction
    nominal_target: float
    open_next: float
    pretrade_equity: float
    pretrade_notional: float
    pretrade_exposure: float
    permission: PermissionResult
    executed_delta_notional: float
    trade_cost: float
    post_cost_equity: float
    post_cost_quantity: float
    post_cost_exposure: float
    close_next: float
    truth_next: AccountTruth
    state_next: AccountState
    reward: float
    forced_risk: bool = FORCED_RISK_DEFAULT
    terminal: bool = TERMINAL_DEFAULT
    truncated: bool = TRUNCATED_DEFAULT


# ---------------------------------------------------------------------------
# 3. Next-open pre-trade mark
# ---------------------------------------------------------------------------


def mark_to_next_open(truth: AccountTruth, open_next: float) -> PreTradeTruth:
    """Mark the inherited position at ``open_{t+1}`` before any execution.

    ``W_minus = W_t + q_t * (open_next - mark_price_t)``
    ``N_minus = q_t * open_next``
    """

    if not isinstance(truth, AccountTruth):
        raise ContractError(f"truth must be an AccountTruth, got {type(truth).__name__}")
    price = _require_positive_float(open_next, "open_next")
    equity = truth.equity + truth.quantity * (price - truth.mark_price)
    if not equity > 0.0:
        raise AccountSolvencyError(
            f"pre-trade equity must be strictly positive, got {equity!r}"
        )
    return PreTradeTruth(equity=equity, quantity=truth.quantity, open_next=price)


# ---------------------------------------------------------------------------
# 4. Cost-aware target feasibility
# ---------------------------------------------------------------------------


def _require_pretrade(pretrade: Any) -> PreTradeTruth:
    if not isinstance(pretrade, PreTradeTruth):
        raise ContractError(f"pretrade must be a PreTradeTruth, got {type(pretrade).__name__}")
    _validate_pretrade(pretrade)
    return pretrade


def _require_finite_target(target: float) -> float:
    return _require_finite_float(target, "target")


def _notional_plan(
    pretrade: PreTradeTruth, target: float
) -> tuple[float, float, float]:
    """``(N_star, delta_notional, |delta_notional|)`` for one candidate target.

    ``N_star = target * W_minus`` and ``delta_notional = N_star - N_minus`` are
    the contract formulas.  The signed change is always formed from the signed
    notionals, so an exact reversal pays the full turnover of both legs
    (``+0.5 -> -0.5`` charges ``|-500 - 500| = 1000``) instead of cancelling.
    """

    notional_star = target * pretrade.equity
    delta_notional = notional_star - pretrade.notional
    return notional_star, delta_notional, abs(delta_notional)


def candidate_target_feasible(
    pretrade: PreTradeTruth, target: float, config: PhysicsConfig
) -> bool:
    """Exact cost-aware predicate for a candidate target exposure ``x``.

    ``N_star = x * W_minus``; ``delta_notional = N_star - N_minus``;
    ``trade_cost = kappa * abs(delta_notional)``; ``W_plus = W_minus - trade_cost``.

    Feasible iff ``W_plus > 0`` and
    ``abs(N_star / W_plus) <= hard_exposure_limit + finite_tolerance``.
    This checks post-cost actual exposure, not merely the nominal target.
    """

    pretrade = _require_pretrade(pretrade)
    if not isinstance(config, PhysicsConfig):
        raise ContractError(f"config must be a PhysicsConfig, got {type(config).__name__}")
    candidate = _require_finite_target(target)
    notional_star, _, delta_magnitude = _notional_plan(pretrade, candidate)
    trade_cost = config.proportional_friction_kappa * delta_magnitude
    equity_plus = pretrade.equity - trade_cost
    if not equity_plus > 0.0:
        return False
    return abs(notional_star / equity_plus) <= (
        config.hard_exposure_limit + config.finite_tolerance
    )


def _boundary_magnitude(pretrade: PreTradeTruth, sign: float, config: PhysicsConfig) -> float:
    """Largest feasible magnitude on one side, by the contract's exact predicate.

    ``0`` is the known feasible lower magnitude.  If the boundary
    ``+/-hard_exposure_limit`` is feasible it is used directly; otherwise
    exactly ``permission_bisection_iterations`` bisection steps walk inward
    from it.  The value returned is always a magnitude the exact predicate has
    accepted, and it is the largest such magnitude the frozen iteration count
    resolves.
    """

    limit = config.hard_exposure_limit
    if candidate_target_feasible(pretrade, sign * limit, config):
        return limit
    low = 0.0
    if not candidate_target_feasible(pretrade, 0.0, config):
        raise AccountSolvencyError(
            "zero target exposure is not feasible at open_next; "
            "the account is insolvent before execution"
        )
    high = limit
    for _ in range(config.permission_bisection_iterations):
        mid = 0.5 * (low + high)
        # ``sign * mid`` is a signed candidate; the predicate itself is the
        # only feasibility rule used on either side.
        if candidate_target_feasible(pretrade, sign * mid, config):
            low = mid
        else:
            high = mid
    return low


def feasible_target_interval(
    pretrade: PreTradeTruth, config: PhysicsConfig
) -> tuple[float, float]:
    """Signed feasible interval ``[e_min, e_max]`` from the exact predicate.

    Each direction is bisected independently, because friction makes the
    envelope asymmetric: reversing an existing position pays the turnover of
    both legs, while de-risking toward zero pays less.
    """

    pretrade = _require_pretrade(pretrade)
    if not isinstance(config, PhysicsConfig):
        raise ContractError(f"config must be a PhysicsConfig, got {type(config).__name__}")
    if not candidate_target_feasible(pretrade, 0.0, config):
        raise AccountSolvencyError(
            "zero target exposure is not feasible at open_next; "
            "the account is insolvent before execution"
        )
    feasible_min = -_boundary_magnitude(pretrade, -1.0, config)
    feasible_max = _boundary_magnitude(pretrade, 1.0, config)
    return (feasible_min, feasible_max)


def permit_target(
    pretrade: PreTradeTruth, nominal_target: float, config: PhysicsConfig
) -> PermissionResult:
    """Project the nominal target onto the feasible interval (clipping).

    Deterministic feasibility control only: it cannot create non-zero risk from
    ``FLAT``, cannot reverse the requested direction as a strategy choice, and
    cannot block a feasible movement toward zero.
    """

    pretrade = _require_pretrade(pretrade)
    if not isinstance(config, PhysicsConfig):
        raise ContractError(f"config must be a PhysicsConfig, got {type(config).__name__}")
    nominal = _require_finite_target(nominal_target)
    feasible_min, feasible_max = feasible_target_interval(pretrade, config)
    if nominal < feasible_min:
        permitted = feasible_min
    elif nominal > feasible_max:
        permitted = feasible_max
    else:
        permitted = nominal
    return PermissionResult(
        nominal_target=nominal,
        feasible_min=feasible_min,
        feasible_max=feasible_max,
        permitted_target=permitted,
        was_clipped=permitted != nominal,
        forced_risk=FORCED_RISK_DEFAULT,
    )


# ---------------------------------------------------------------------------
# 5-6. Execution, next-close mark and log-equity reward
# ---------------------------------------------------------------------------


def execute_transition(
    truth_t: AccountTruth,
    action: NominalAction,
    open_next: float,
    close_next: float,
    config: PhysicsConfig,
) -> PhysicsStep:
    """Execute one deterministic VS-A transition and return its full record.

    observe through ``close_t`` -> decide -> execute at ``open_{t+1}`` ->
    mark at ``close_{t+1}``.  The policy never observes ``open_next`` or
    ``close_next`` when it chooses the nominal action.
    """

    if not isinstance(truth_t, AccountTruth):
        raise ContractError(f"truth_t must be an AccountTruth, got {type(truth_t).__name__}")
    if not isinstance(action, NominalAction):
        raise ContractError(f"action must be a NominalAction, got {type(action).__name__}")
    if not isinstance(config, PhysicsConfig):
        raise ContractError(f"config must be a PhysicsConfig, got {type(config).__name__}")
    execution_price = _require_positive_float(open_next, "open_next")
    marking_price = _require_positive_float(close_next, "close_next")

    state_t = account_state_from_truth(truth_t, config)
    nominal_target = nominal_target_from_action(action, config)
    pretrade = mark_to_next_open(truth_t, execution_price)
    permission = permit_target(pretrade, nominal_target, config)
    permitted = permission.permitted_target
    if not candidate_target_feasible(pretrade, permitted, config):
        raise AccountSolvencyError(
            f"permitted target {permitted!r} is not feasible after its trade cost"
        )

    notional_star, delta_notional, delta_magnitude = _notional_plan(pretrade, permitted)
    trade_cost = config.proportional_friction_kappa * delta_magnitude
    equity_plus = pretrade.equity - trade_cost
    if not equity_plus > 0.0:
        raise AccountSolvencyError(
            f"post-cost equity must be strictly positive, got {equity_plus!r}"
        )
    quantity_plus = notional_star / execution_price
    post_cost_exposure = (quantity_plus * execution_price) / equity_plus

    equity_next = equity_plus + quantity_plus * (marking_price - execution_price)
    if not equity_next > 0.0:
        raise AccountSolvencyError(
            f"next equity must be strictly positive, got {equity_next!r}"
        )
    truth_next = AccountTruth(
        equity=equity_next,
        quantity=quantity_plus,
        mark_price=marking_price,
    )
    state_next = account_state_from_truth(truth_next, config)
    reward = math.log(equity_next / truth_t.equity)

    return PhysicsStep(
        truth_t=truth_t,
        state_t=state_t,
        nominal_action=action,
        nominal_target=nominal_target,
        open_next=execution_price,
        pretrade_equity=pretrade.equity,
        pretrade_notional=pretrade.notional,
        pretrade_exposure=pretrade.exposure,
        permission=permission,
        executed_delta_notional=delta_notional,
        trade_cost=trade_cost,
        post_cost_equity=equity_plus,
        post_cost_quantity=quantity_plus,
        post_cost_exposure=post_cost_exposure,
        close_next=marking_price,
        truth_next=truth_next,
        state_next=state_next,
        reward=reward,
        forced_risk=FORCED_RISK_DEFAULT,
        terminal=TERMINAL_DEFAULT,
        truncated=TRUNCATED_DEFAULT,
    )


__all__ = [
    "FORCED_RISK_DEFAULT",
    "TERMINAL_DEFAULT",
    "TRUNCATED_DEFAULT",
    "PermissionResult",
    "PhysicsStep",
    "PreTradeTruth",
    "candidate_target_feasible",
    "execute_transition",
    "feasible_target_interval",
    "mark_to_next_open",
    "permit_target",
]
