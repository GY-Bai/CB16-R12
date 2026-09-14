"""R12 VS-A: canonical contracts, N0 market input and account physics.

First non-learning segment of the R12 vertical slice:

```text
raw/synthetic OHLCV window
  -> canonical N0 normalization          (market.normalize_market_window)
  -> AccountTruth -> AccountState        (contracts.account_state_from_truth)
  -> NominalAction -> nominal target     (contracts.nominal_target_from_action)
  -> next-open pre-trade mark            (physics.mark_to_next_open)
  -> cost-aware Permission               (physics.permit_target)
  -> deterministic execution             (physics.execute_transition)
  -> next-close mark + log-equity reward (physics.execute_transition)
```

The package is deliberately narrow.  It uses only the Python standard library
plus the existing NumPy, imports no PyTorch, and implements no Central Brain,
actor/critic, learner, replay, teacher, checkpoint, database or throughput
machinery.  Every physics function is pure and deterministic; there is no
hidden mutable account object.
"""

from __future__ import annotations

from .contracts import (
    CONTEXT_LENGTH,
    FINITE_TOLERANCE,
    HARD_EXPOSURE_LIMIT,
    NEUTRAL_SURVIVAL_CUSHION,
    NOMINAL_EXPOSURE_BUDGET,
    PERMISSION_BISECTION_ITERATIONS,
    PROPORTIONAL_FRICTION_KAPPA,
    AccountSolvencyError,
    AccountState,
    AccountTruth,
    ContractError,
    Direction,
    NominalAction,
    PhysicsConfig,
    account_state_from_truth,
    nominal_target_from_action,
    validate_physics_config,
)
from .market import N0_CHANNELS, normalize_market_window
from .physics import (
    FORCED_RISK_DEFAULT,
    TERMINAL_DEFAULT,
    TRUNCATED_DEFAULT,
    PermissionResult,
    PhysicsStep,
    PreTradeTruth,
    candidate_target_feasible,
    execute_transition,
    feasible_target_interval,
    mark_to_next_open,
    permit_target,
)

__all__ = [
    "CONTEXT_LENGTH",
    "FINITE_TOLERANCE",
    "FORCED_RISK_DEFAULT",
    "HARD_EXPOSURE_LIMIT",
    "N0_CHANNELS",
    "NEUTRAL_SURVIVAL_CUSHION",
    "NOMINAL_EXPOSURE_BUDGET",
    "PERMISSION_BISECTION_ITERATIONS",
    "PROPORTIONAL_FRICTION_KAPPA",
    "TERMINAL_DEFAULT",
    "TRUNCATED_DEFAULT",
    "AccountSolvencyError",
    "AccountState",
    "AccountTruth",
    "ContractError",
    "Direction",
    "NominalAction",
    "PermissionResult",
    "PhysicsConfig",
    "PhysicsStep",
    "PreTradeTruth",
    "account_state_from_truth",
    "candidate_target_feasible",
    "execute_transition",
    "feasible_target_interval",
    "mark_to_next_open",
    "nominal_target_from_action",
    "normalize_market_window",
    "permit_target",
    "validate_physics_config",
]
