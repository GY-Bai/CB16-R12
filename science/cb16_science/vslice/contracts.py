"""Canonical R12 VS-A contracts: direction, action, account truth/state, config.

Every semantic rule in this module is copied literally from
``docs/experiments/frozen/R12_VS_A_PHYSICS_R0.md`` (sections 0, 1 and 3).  The module is
pure: no market data, no dispatcher/GitHub/OCI code, no Central Brain, no
mutable account object, no learning.

The frozen controlled-slice defaults are experiment-scoped values for the first
vertical slice.  They are not production authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

# ---------------------------------------------------------------------------
# Frozen controlled-slice defaults (experiment-scoped, not production authority)
# ---------------------------------------------------------------------------

CONTEXT_LENGTH = 64
NOMINAL_EXPOSURE_BUDGET = 1.0
HARD_EXPOSURE_LIMIT = 1.0
PROPORTIONAL_FRICTION_KAPPA = 0.001
PERMISSION_BISECTION_ITERATIONS = 80
FINITE_TOLERANCE = 1e-12

#: The sandbox has no maintenance-liquidation rule, so the decision state uses
#: the neutral safe survival value.
NEUTRAL_SURVIVAL_CUSHION = 1.0


class ContractError(ValueError):
    """A contract-invalid value (shape, domain, finiteness) was supplied."""


class AccountSolvencyError(ContractError):
    """An account/solvency condition failed explicitly; capital is never repaired."""


class Direction(IntEnum):
    """Canonical signed direction with the exact contract values."""

    SHORT = -1
    FLAT = 0
    LONG = 1


def _coerce_direction(value: Any) -> Direction:
    if isinstance(value, Direction):
        return value
    if isinstance(value, bool):
        raise ContractError(f"direction must be one of -1/0/+1, got {value!r}")
    try:
        return Direction(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"direction must be one of -1/0/+1, got {value!r}") from exc


def _require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a real number, got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ContractError(f"{name} must be finite, got {value!r}")
    return number


def _require_positive_float(value: Any, name: str) -> float:
    number = _require_finite_float(value, name)
    if not number > 0.0:
        raise ContractError(f"{name} must be strictly positive, got {value!r}")
    return number


# ---------------------------------------------------------------------------
# Nominal action
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NominalAction:
    """The Central Brain intent ``(direction, requested_risk)``.

    ``requested_risk`` is not confidence and is not permission: it is the
    requested fraction of the frozen nominal exposure budget.
    """

    direction: Direction
    requested_risk: float

    def __post_init__(self) -> None:
        direction = _coerce_direction(self.direction)
        risk = _require_finite_float(self.requested_risk, "requested_risk")
        if not 0.0 <= risk <= 1.0:
            raise ContractError(f"requested_risk must lie in [0, 1], got {self.requested_risk!r}")
        is_flat = direction is Direction.FLAT
        if is_flat != (risk == 0.0):
            raise ContractError(
                "FLAT holds iff requested_risk == 0; got "
                f"direction={direction.name} requested_risk={risk!r}"
            )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "requested_risk", risk)


def nominal_target_from_action(action: NominalAction, config: "PhysicsConfig") -> float:
    """``nominal_target = int(direction) * requested_risk * nominal_exposure_budget``."""

    if not isinstance(action, NominalAction):
        raise ContractError(f"action must be a NominalAction, got {type(action).__name__}")
    config = _require_config(config)
    return float(int(action.direction)) * action.requested_risk * config.nominal_exposure_budget


# ---------------------------------------------------------------------------
# Account truth and decision state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountTruth:
    """Minimal marked account truth: equity, signed quantity, mark price."""

    equity: float
    quantity: float
    mark_price: float

    def __post_init__(self) -> None:
        equity = _require_finite_float(self.equity, "equity")
        quantity = _require_finite_float(self.quantity, "quantity")
        mark_price = _require_positive_float(self.mark_price, "mark_price")
        if not equity > 0.0:
            raise AccountSolvencyError(f"equity must be strictly positive, got {self.equity!r}")
        object.__setattr__(self, "equity", equity)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "mark_price", mark_price)

    @property
    def notional(self) -> float:
        """Current signed marked notional ``quantity * mark_price``."""

        return self.quantity * self.mark_price

    @property
    def exposure(self) -> float:
        """Current exposure ``notional / equity`` (never clipped)."""

        return self.notional / self.equity


@dataclass(frozen=True)
class AccountState:
    """Exact v1 decision state derived from marked truth."""

    signed_exposure: float
    survival_cushion: float
    new_risk_capacity: float


def account_state_from_truth(truth: AccountTruth, config: "PhysicsConfig") -> AccountState:
    """Derive the v1 ``AccountState`` from marked truth.

    ``signed_exposure`` is never clipped: market movement may temporarily push
    it beyond the hard new-risk envelope.
    """

    if not isinstance(truth, AccountTruth):
        raise ContractError(f"truth must be an AccountTruth, got {type(truth).__name__}")
    config = _require_config(config)
    signed_exposure = truth.quantity * truth.mark_price / truth.equity
    new_risk_capacity = max(
        0.0, 1.0 - abs(signed_exposure) / config.hard_exposure_limit
    )
    return AccountState(
        signed_exposure=signed_exposure,
        survival_cushion=NEUTRAL_SURVIVAL_CUSHION,
        new_risk_capacity=new_risk_capacity,
    )


# ---------------------------------------------------------------------------
# Frozen configuration
# ---------------------------------------------------------------------------


def validate_physics_config(
    *,
    context_length: int,
    nominal_exposure_budget: float,
    hard_exposure_limit: float,
    proportional_friction_kappa: float,
    permission_bisection_iterations: int,
    finite_tolerance: float,
) -> None:
    """Require the frozen controlled-slice configuration contract."""

    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise ContractError(f"context_length must be an int, got {context_length!r}")
    if context_length < 1:
        raise ContractError(
            "context_length counts represented bars and must be >= 1 "
            f"(the retained predecessor is an extra input bar), got {context_length!r}"
        )
    budget = _require_finite_float(nominal_exposure_budget, "nominal_exposure_budget")
    if not budget > 0.0:
        raise ContractError(f"nominal_exposure_budget must be > 0, got {nominal_exposure_budget!r}")
    limit = _require_finite_float(hard_exposure_limit, "hard_exposure_limit")
    if not limit > 0.0:
        raise ContractError(f"hard_exposure_limit must be > 0, got {hard_exposure_limit!r}")
    kappa = _require_finite_float(proportional_friction_kappa, "proportional_friction_kappa")
    if kappa < 0.0:
        raise ContractError(f"proportional_friction_kappa must be >= 0, got {proportional_friction_kappa!r}")
    if not kappa * limit < 1.0:
        raise ContractError(
            "proportional_friction_kappa * hard_exposure_limit must be < 1, got "
            f"{kappa * limit!r}"
        )
    if isinstance(permission_bisection_iterations, bool) or not isinstance(
        permission_bisection_iterations, int
    ):
        raise ContractError(
            "permission_bisection_iterations must be an int, got "
            f"{permission_bisection_iterations!r}"
        )
    if permission_bisection_iterations < 1:
        raise ContractError(
            "permission_bisection_iterations must be >= 1, got "
            f"{permission_bisection_iterations!r}"
        )
    tolerance = _require_finite_float(finite_tolerance, "finite_tolerance")
    if tolerance < 0.0:
        raise ContractError(f"finite_tolerance must be >= 0, got {finite_tolerance!r}")


@dataclass(frozen=True)
class PhysicsConfig:
    """Immutable frozen configuration for one controlled slice."""

    context_length: int = CONTEXT_LENGTH
    nominal_exposure_budget: float = NOMINAL_EXPOSURE_BUDGET
    hard_exposure_limit: float = HARD_EXPOSURE_LIMIT
    proportional_friction_kappa: float = PROPORTIONAL_FRICTION_KAPPA
    permission_bisection_iterations: int = PERMISSION_BISECTION_ITERATIONS
    finite_tolerance: float = FINITE_TOLERANCE

    def __post_init__(self) -> None:
        validate_physics_config(
            context_length=self.context_length,
            nominal_exposure_budget=self.nominal_exposure_budget,
            hard_exposure_limit=self.hard_exposure_limit,
            proportional_friction_kappa=self.proportional_friction_kappa,
            permission_bisection_iterations=self.permission_bisection_iterations,
            finite_tolerance=self.finite_tolerance,
        )


def _require_config(config: Any) -> PhysicsConfig:
    if not isinstance(config, PhysicsConfig):
        raise ContractError(f"config must be a PhysicsConfig, got {type(config).__name__}")
    return config


__all__ = [
    "CONTEXT_LENGTH",
    "FINITE_TOLERANCE",
    "HARD_EXPOSURE_LIMIT",
    "NEUTRAL_SURVIVAL_CUSHION",
    "NOMINAL_EXPOSURE_BUDGET",
    "PERMISSION_BISECTION_ITERATIONS",
    "PROPORTIONAL_FRICTION_KAPPA",
    "AccountSolvencyError",
    "AccountState",
    "AccountTruth",
    "ContractError",
    "Direction",
    "NominalAction",
    "PhysicsConfig",
    "account_state_from_truth",
    "nominal_target_from_action",
    "validate_physics_config",
]
