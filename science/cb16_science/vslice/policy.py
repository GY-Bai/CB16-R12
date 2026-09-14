"""R12 VS-B policy boundary: runtime state, Central Brain actor, value critic.

The runtime state for the controlled slice is

```text
s_t = (Z_t, A_t)
```

with ``Z_t`` the frozen sensory representation and ``A_t`` exactly the
canonical three-coordinate account vector

```text
[signed_exposure, survival_cushion, new_risk_capacity].
```

``X_t`` is omitted because execution economics are fixed by ``PhysicsConfig``.
No raw OHLCV, raw price, equity, position quantity, symbol identity, entry
price, PnL history, holding time, previous action, drawdown or trade count is
part of this boundary.

The actor is a small CPU MLP that emits a three-way categorical direction head
in the fixed semantic order ``SHORT, FLAT, LONG`` plus direction-conditioned
``Beta`` parameters for the two non-FLAT directions.  The critic is a separate
scalar MLP with no shared parameters.  Both consume exactly ``Z_t + A_t``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
from torch.distributions import Beta, Categorical

from .contracts import (
    AccountState,
    ContractError,
    Direction,
    NominalAction,
    _coerce_direction,
    _require_finite_float,
)
from .sensory import Z_DIM

#: Experiment-scoped Central Brain MLP shape for VS-B.
HIDDEN_WIDTH = 64
HIDDEN_LAYERS = 2
#: Frozen positive floor for every direction-conditioned Beta parameter.
BETA_FLOOR = 1.0
#: Exact size of the account part of the runtime state.
ACCOUNT_STATE_DIM = 3
#: Exact canonical semantic order of the categorical direction head.
DIRECTION_ORDER: Tuple[Direction, ...] = (Direction.SHORT, Direction.FLAT, Direction.LONG)
#: Exact field order of the account part of the runtime state.
ACCOUNT_STATE_FIELDS: Tuple[str, ...] = (
    "signed_exposure",
    "survival_cushion",
    "new_risk_capacity",
)

DIRECTION_INDEX_SHORT = 0
DIRECTION_INDEX_FLAT = 1
DIRECTION_INDEX_LONG = 2
#: Raw Beta scalars emitted per state: short alpha/beta, long alpha/beta.
BETA_RAW_WIDTH = 4


def _require_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ContractError(f"{name} must be >= {minimum}, got {value!r}")
    return value


def _require_floating_dtype(dtype: Any) -> torch.dtype:
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ContractError(f"dtype must be a floating torch dtype, got {dtype!r}")
    return dtype


def direction_index(direction: Union[Direction, int]) -> int:
    """Canonical head index for a semantic direction: ``SHORT, FLAT, LONG``."""

    return DIRECTION_ORDER.index(_coerce_direction(direction))


def direction_from_index(index: int) -> Direction:
    """Semantic direction for a canonical head index in ``0, 1, 2``."""

    if isinstance(index, bool) or not isinstance(index, int):
        raise ContractError(f"direction index must be an int, got {index!r}")
    if not 0 <= index < len(DIRECTION_ORDER):
        raise ContractError(
            f"direction index must lie in 0..{len(DIRECTION_ORDER) - 1}, got {index!r}"
        )
    return DIRECTION_ORDER[index]


def account_state_vector(account_state: AccountState) -> torch.Tensor:
    """The exact three-coordinate account vector, in canonical field order."""

    if not isinstance(account_state, AccountState):
        raise ContractError(
            f"account_state must be an AccountState, got {type(account_state).__name__}"
        )
    values = tuple(
        _require_finite_float(getattr(account_state, name), name) for name in ACCOUNT_STATE_FIELDS
    )
    return torch.tensor(values, dtype=torch.float32)


@dataclass(frozen=True)
class LearnerState:
    """Runtime state ``s_t = (Z_t, A_t)`` for the controlled VS-B slice.

    The only two components are the frozen sensory representation ``z`` and the
    canonical ``account`` vector.  Constructing this record requires nothing
    else, so no raw market or monetary field can enter the learner state.
    """

    z: torch.Tensor
    account: torch.Tensor

    def __post_init__(self) -> None:
        for name, tensor in (("z", self.z), ("account", self.account)):
            if not isinstance(tensor, torch.Tensor):
                raise ContractError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
            if not tensor.dtype.is_floating_point:
                raise ContractError(f"{name} must be a floating tensor, got dtype {tensor.dtype}")
        if self.z.ndim < 1:
            raise ContractError("z must carry at least one coordinate dimension")
        if self.account.shape[-1] != ACCOUNT_STATE_DIM:
            raise ContractError(
                f"account must end in exactly {ACCOUNT_STATE_DIM} coordinates, got shape "
                f"{tuple(self.account.shape)}"
            )
        if self.account.ndim != self.z.ndim or self.account.shape[:-1] != self.z.shape[:-1]:
            raise ContractError(
                "z and account must share leading dimensions, got "
                f"{tuple(self.z.shape)} and {tuple(self.account.shape)}"
            )
        for name, tensor in (("z", self.z), ("account", self.account)):
            if not bool(torch.isfinite(tensor).all()):
                raise ContractError(f"{name} must be finite")
        object.__setattr__(self, "z", self.z.detach().clone())
        object.__setattr__(self, "account", self.account.detach().clone())

    @property
    def vector(self) -> torch.Tensor:
        """Concatenated actor/critic input ``[Z_t, A_t]``."""

        return torch.cat((self.z, self.account), dim=-1)


def build_learner_state(z: torch.Tensor, account_state: AccountState) -> LearnerState:
    """Assemble ``(Z_t, A_t)`` from one sensory representation and one account state."""

    if not isinstance(z, torch.Tensor):
        raise ContractError(f"z must be a torch.Tensor, got {type(z).__name__}")
    account = account_state_vector(account_state)
    if z.ndim == 1:
        return LearnerState(z=z, account=account)
    return LearnerState(z=z, account=account.expand(*z.shape[:-1], ACCOUNT_STATE_DIM))


def _require_state_vector(state: Any, input_dim: int, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(state, LearnerState):
        state = state.vector
    if not isinstance(state, torch.Tensor):
        raise ContractError(
            f"state must be a torch.Tensor or LearnerState, got {type(state).__name__}"
        )
    if state.ndim < 1 or state.shape[-1] != input_dim:
        raise ContractError(
            f"state vector must end in exactly {input_dim} coordinates, got shape "
            f"{tuple(state.shape)}"
        )
    if not state.dtype.is_floating_point:
        raise ContractError(f"state vector must be a floating tensor, got dtype {state.dtype}")
    if not bool(torch.isfinite(state).all()):
        raise ContractError("state vector must be finite")
    # A state may arrive as float64 from a canonical NumPy pipeline; the policy
    # modules compute in their own dtype.  Fail if a finite float64 value beyond
    # the module dtype's range would cast to infinity.
    if state.dtype != dtype:
        state = state.to(dtype)
        if not bool(torch.isfinite(state).all()):
            raise ContractError(f"state vector must stay finite in the module dtype {dtype}")
    return state


def _direction_index_value(value: Any) -> int:
    if isinstance(value, Direction):
        return direction_index(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(
            f"direction must be a Direction or a canonical index 0..2, got {value!r}"
        )
    direction_from_index(value)
    return value


def _require_direction_indices(directions: Any, count: int) -> torch.Tensor:
    if isinstance(directions, torch.Tensor):
        if directions.ndim != 1 or directions.numel() != count:
            raise ContractError(
                f"direction indices must be a [{count}] tensor, got shape "
                f"{tuple(directions.shape)}"
            )
        if directions.dtype not in (torch.int64, torch.int32):
            raise ContractError(
                f"direction indices must be an integer tensor, got {directions.dtype}"
            )
        indices = directions.to(torch.long)
        if bool(((indices < 0) | (indices >= len(DIRECTION_ORDER))).any()):
            raise ContractError(
                f"direction indices must lie in 0..{len(DIRECTION_ORDER) - 1}"
            )
        return indices
    if isinstance(directions, (str, bytes)) or not isinstance(directions, Sequence):
        raise ContractError(
            f"direction indices must be a sequence of {count} entries, got "
            f"{type(directions).__name__}"
        )
    values = [_direction_index_value(item) for item in directions]
    if len(values) != count:
        raise ContractError(
            f"direction indices must supply exactly {count} entries, got {len(values)}"
        )
    return torch.tensor(values, dtype=torch.long)


def _require_risk_values(requested_risks: Any, count: int, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(requested_risks, torch.Tensor):
        risks = requested_risks
        if risks.ndim != 1 or risks.numel() != count:
            raise ContractError(
                f"requested risk must be a [{count}] tensor, got shape {tuple(risks.shape)}"
            )
        if not risks.dtype.is_floating_point:
            raise ContractError(f"requested risk must be a floating tensor, got {risks.dtype}")
        risks = risks.to(dtype)
    else:
        if isinstance(requested_risks, (str, bytes)) or not isinstance(requested_risks, Sequence):
            raise ContractError(
                f"requested risk must be a sequence of {count} entries, got "
                f"{type(requested_risks).__name__}"
            )
        values = [_require_finite_float(item, "requested_risk") for item in requested_risks]
        if len(values) != count:
            raise ContractError(
                f"requested risk must supply exactly {count} entries, got {len(values)}"
            )
        risks = torch.tensor(values, dtype=dtype)
    if not bool(torch.isfinite(risks).all()):
        raise ContractError("requested risk must be finite")
    if bool(((risks < 0.0) | (risks > 1.0)).any()):
        raise ContractError("requested risk must lie in [0, 1]")
    return risks


def _require_scalar_risk(requested_risk: Any, dtype: torch.dtype) -> torch.Tensor:
    value = _require_finite_float(requested_risk, "requested_risk")
    if not 0.0 <= value <= 1.0:
        raise ContractError(f"requested_risk must lie in [0, 1], got {requested_risk!r}")
    return torch.tensor(value, dtype=dtype)


def _require_semantic_consistency(index: torch.Tensor, risks: torch.Tensor) -> None:
    """Reject direction/risk combinations the semantic action cannot represent."""

    flat = index == DIRECTION_INDEX_FLAT
    if bool((flat & (risks != 0.0)).any()):
        raise ContractError("FLAT actions must request exactly zero risk")
    if bool(((~flat) & (risks == 0.0)).any()):
        raise ContractError("SHORT/LONG actions must request strictly positive risk")


def _require_actor_output(output: "ActorOutput") -> None:
    tensors = (
        ("direction_logits", output.direction_logits),
        ("short_alpha", output.short_alpha),
        ("short_beta", output.short_beta),
        ("long_alpha", output.long_alpha),
        ("long_beta", output.long_beta),
    )
    for name, tensor in tensors:
        if not bool(torch.isfinite(tensor).all()):
            raise ContractError(f"actor output {name} must be finite")
    for name, tensor in tensors[1:]:
        if not bool((tensor > 0.0).all()):
            raise ContractError(f"actor output {name} must be strictly positive")


def _selected_beta(output: "ActorOutput", index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    short = index == DIRECTION_INDEX_SHORT
    alpha = torch.where(short, output.short_alpha, output.long_alpha)
    beta = torch.where(short, output.short_beta, output.long_beta)
    return alpha, beta


@dataclass(frozen=True)
class ActionSample:
    """One typed stochastic action sample from the training behavior policy.

    ``requested_risk`` is exactly zero for ``FLAT`` and the sampled Beta value
    otherwise.  ``selected_alpha``/``selected_beta`` are diagnostics for the
    direction-conditioned Beta that produced the risk and are absent for FLAT.
    """

    direction: Direction
    requested_risk: float
    log_prob: float
    selected_alpha: Optional[float] = None
    selected_beta: Optional[float] = None

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
        log_prob = _require_finite_float(self.log_prob, "log_prob")
        if is_flat:
            if self.selected_alpha is not None or self.selected_beta is not None:
                raise ContractError("FLAT samples carry no Beta parameters")
        else:
            alpha = _require_finite_float(self.selected_alpha, "selected_alpha")
            beta = _require_finite_float(self.selected_beta, "selected_beta")
            if not alpha > 0.0 or not beta > 0.0:
                raise ContractError("selected Beta parameters must be strictly positive")
            object.__setattr__(self, "selected_alpha", alpha)
            object.__setattr__(self, "selected_beta", beta)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "requested_risk", risk)
        object.__setattr__(self, "log_prob", log_prob)

    @property
    def nominal_action(self) -> NominalAction:
        """The semantic ``NominalAction`` carried by this sample."""

        return NominalAction(direction=self.direction, requested_risk=self.requested_risk)


@dataclass(frozen=True)
class ActorOutput:
    """Raw actor head outputs for one state batch."""

    direction_logits: torch.Tensor
    short_alpha: torch.Tensor
    short_beta: torch.Tensor
    long_alpha: torch.Tensor
    long_beta: torch.Tensor


class Actor(nn.Module):
    """Central Brain actor: categorical direction head + conditional Beta head.

    Input is exactly ``Z_t + A_t`` (``z_dim + 3``).  Output is three direction
    logits in the fixed ``SHORT, FLAT, LONG`` order plus strictly positive
    ``Beta`` parameters for ``SHORT`` and ``LONG``:

    ```text
    alpha = softplus(raw) + beta_floor
    beta  = softplus(raw) + beta_floor
    ```
    """

    def __init__(
        self,
        *,
        z_dim: int = Z_DIM,
        hidden_width: int = HIDDEN_WIDTH,
        hidden_layers: int = HIDDEN_LAYERS,
        beta_floor: float = BETA_FLOOR,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        z_dim = _require_int(z_dim, "z_dim")
        hidden_width = _require_int(hidden_width, "hidden_width")
        hidden_layers = _require_int(hidden_layers, "hidden_layers")
        floor = _require_finite_float(beta_floor, "beta_floor")
        if not floor > 0.0:
            raise ContractError(f"beta_floor must be strictly positive, got {beta_floor!r}")
        dtype = _require_floating_dtype(dtype)

        self.z_dim = z_dim
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.beta_floor = floor
        self.dtype = dtype
        self.input_dim = z_dim + ACCOUNT_STATE_DIM

        layers = []
        width = self.input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(width, hidden_width))
            layers.append(nn.Tanh())
            width = hidden_width
        layers.append(nn.Linear(width, len(DIRECTION_ORDER) + BETA_RAW_WIDTH))
        self.mlp = nn.Sequential(*layers).to(dtype)

    def forward(self, state: Any) -> ActorOutput:
        vector = _require_state_vector(state, self.input_dim, self.dtype)
        raw = self.mlp(vector)
        logits = raw[..., : len(DIRECTION_ORDER)]
        beta_raw = raw[..., len(DIRECTION_ORDER) :]
        output = ActorOutput(
            direction_logits=logits,
            short_alpha=torch.nn.functional.softplus(beta_raw[..., 0]) + self.beta_floor,
            short_beta=torch.nn.functional.softplus(beta_raw[..., 1]) + self.beta_floor,
            long_alpha=torch.nn.functional.softplus(beta_raw[..., 2]) + self.beta_floor,
            long_beta=torch.nn.functional.softplus(beta_raw[..., 3]) + self.beta_floor,
        )
        _require_actor_output(output)
        return output

    def _log_prob_impl(
        self, states: torch.Tensor, index: torch.Tensor, risks: torch.Tensor
    ) -> torch.Tensor:
        output = self.forward(states)
        flat = index == DIRECTION_INDEX_FLAT
        selected_alpha, selected_beta = _selected_beta(output, index)
        # FLAT contributes no Beta density.  The discarded branch is evaluated
        # at 0.5, strictly inside the Beta support, so its forward value stays
        # finite and cannot leak through ``torch.where`` (whose unselected
        # branch receives exactly zero gradient).
        safe_risk = torch.where(flat, torch.full_like(risks, 0.5), risks)
        categorical_log_prob = Categorical(logits=output.direction_logits).log_prob(index)
        beta_log_prob = Beta(selected_alpha, selected_beta).log_prob(safe_risk)
        log_prob = torch.where(flat, categorical_log_prob, categorical_log_prob + beta_log_prob)
        if not bool(torch.isfinite(log_prob).all()):
            raise ContractError("action log-probability must be finite")
        return log_prob

    def log_prob(self, state: Any, direction: Any, requested_risk: Any) -> torch.Tensor:
        """Log-probability of one semantic action under this policy."""

        vector = _require_state_vector(state, self.input_dim, self.dtype)
        if vector.ndim != 1:
            raise ContractError("log_prob expects one state vector; use log_prob_batch for a batch")
        index = torch.tensor(_direction_index_value(direction), dtype=torch.long)
        risk = _require_scalar_risk(requested_risk, self.dtype)
        _require_semantic_consistency(index, risk)
        return self._log_prob_impl(vector, index, risk)

    def log_prob_batch(
        self, states: Any, directions: Any, requested_risks: Any
    ) -> torch.Tensor:
        """Log-probabilities ``[N]`` for ``N`` stored semantic actions."""

        vector = _require_state_vector(states, self.input_dim, self.dtype)
        if vector.ndim != 2:
            raise ContractError("log_prob_batch expects states of shape [N, input_dim]")
        index = _require_direction_indices(directions, vector.shape[0])
        risks = _require_risk_values(requested_risks, vector.shape[0], self.dtype)
        _require_semantic_consistency(index, risks)
        return self._log_prob_impl(vector, index, risks)

    def sample(self, state: Any) -> ActionSample:
        """Sample one semantic action from the stochastic training policy.

        Sampling draws on the ambient torch RNG; callers that need reproducibility
        seed it (``torch.manual_seed``) before collection.
        """

        vector = _require_state_vector(state, self.input_dim, self.dtype)
        if vector.ndim != 1:
            raise ContractError("sample expects one state vector")
        with torch.no_grad():
            output = self.forward(vector)
            index = int(Categorical(logits=output.direction_logits).sample().item())
            index_tensor = torch.tensor(index, dtype=torch.long)
            direction = direction_from_index(index)
            if direction is Direction.FLAT:
                requested_risk = 0.0
                selected_alpha = None
                selected_beta = None
            else:
                alpha, beta = _selected_beta(output, index_tensor)
                sampled = float(Beta(alpha, beta).sample())
                if not math.isfinite(sampled) or not 0.0 <= sampled <= 1.0:
                    raise ContractError(
                        "sampled requested_risk must be finite and inside [0, 1], got "
                        f"{sampled!r}"
                    )
                if not 0.0 < sampled < 1.0:
                    raise ContractError(
                        "sampled requested_risk must lie strictly inside (0, 1) for a non-FLAT "
                        f"direction; the tensor runtime returned {sampled!r}"
                    )
                requested_risk = sampled
                selected_alpha = float(alpha)
                selected_beta = float(beta)
            risk_tensor = torch.tensor(requested_risk, dtype=self.dtype)
            log_prob = float(self._log_prob_impl(vector, index_tensor, risk_tensor).item())
        return ActionSample(
            direction=direction,
            requested_risk=requested_risk,
            log_prob=log_prob,
            selected_alpha=selected_alpha,
            selected_beta=selected_beta,
        )

    def deterministic_action(self, state: Any) -> NominalAction:
        """Preregistered deterministic evaluation adapter (evaluation only).

        ``direction = argmax(direction_logits)`` with the tensor runtime's
        ordinary first-index tie rule; ``FLAT`` implies zero requested risk and
        a non-FLAT direction uses the Beta mean ``alpha / (alpha + beta)``.
        The stochastic training policy never uses this adapter.
        """

        vector = _require_state_vector(state, self.input_dim, self.dtype)
        if vector.ndim != 1:
            raise ContractError("deterministic_action expects one state vector")
        with torch.no_grad():
            output = self.forward(vector)
            index = int(torch.argmax(output.direction_logits).item())
            direction = direction_from_index(index)
            if direction is Direction.FLAT:
                return NominalAction(direction=direction, requested_risk=0.0)
            alpha, beta = _selected_beta(output, torch.tensor(index, dtype=torch.long))
            mean = (alpha.double() / (alpha.double() + beta.double())).item()
        return NominalAction(direction=direction, requested_risk=float(mean))


class ValueCritic(nn.Module):
    """Scalar state-value baseline ``V_phi(Z_t + A_t)``.

    Separate parameters from the actor; the critic is a variance-reduction
    baseline only and never economic truth or promotion authority.
    """

    def __init__(
        self,
        *,
        z_dim: int = Z_DIM,
        hidden_width: int = HIDDEN_WIDTH,
        hidden_layers: int = HIDDEN_LAYERS,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        z_dim = _require_int(z_dim, "z_dim")
        hidden_width = _require_int(hidden_width, "hidden_width")
        hidden_layers = _require_int(hidden_layers, "hidden_layers")
        dtype = _require_floating_dtype(dtype)

        self.z_dim = z_dim
        self.hidden_width = hidden_width
        self.hidden_layers = hidden_layers
        self.dtype = dtype
        self.input_dim = z_dim + ACCOUNT_STATE_DIM

        layers = []
        width = self.input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(width, hidden_width))
            layers.append(nn.Tanh())
            width = hidden_width
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers).to(dtype)

    def forward(self, state: Any) -> torch.Tensor:
        vector = _require_state_vector(state, self.input_dim, self.dtype)
        value = self.mlp(vector).squeeze(-1)
        if not bool(torch.isfinite(value).all()):
            raise ContractError("critic value must be finite")
        return value


__all__ = [
    "ACCOUNT_STATE_DIM",
    "ACCOUNT_STATE_FIELDS",
    "BETA_FLOOR",
    "BETA_RAW_WIDTH",
    "DIRECTION_INDEX_FLAT",
    "DIRECTION_INDEX_LONG",
    "DIRECTION_INDEX_SHORT",
    "DIRECTION_ORDER",
    "HIDDEN_LAYERS",
    "HIDDEN_WIDTH",
    "ActionSample",
    "Actor",
    "ActorOutput",
    "LearnerState",
    "ValueCritic",
    "account_state_vector",
    "build_learner_state",
    "direction_from_index",
    "direction_index",
]
