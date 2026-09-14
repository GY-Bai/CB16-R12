"""R12 VS-B trajectory records and undiscounted Monte-Carlo return-to-go.

A trajectory is the unit the on-policy learner consumes.  Every step retains
the frozen state vector, the semantic action, the reward and the
terminal/truncation flags, so the actor log-probability can be recomputed under
the same frozen generation before the update.  A crash, timeout, logging cutoff
or arbitrary chunk end is *not* a terminal event: it is represented as an
incomplete trajectory (``complete=False``) and is rejected by the learner.

Return-to-go is undiscounted:

```text
G_t = sum_{k=t}^{T-1} reward_k
```

Because the VS-A reward is ``log(W_{t+1} / W_t)``, the cumulative return
telescopes to ``log(W_T / W_0)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Tuple, Union

import torch

from .contracts import (
    ContractError,
    Direction,
    NominalAction,
    _coerce_direction,
    _require_finite_float,
)


def _require_generation_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"generation_id must be an int, got {value!r}")
    if value < 0:
        raise ContractError(f"generation_id must be >= 0, got {value!r}")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a bool, got {value!r}")
    return value


def returns_to_go(rewards: Union[Sequence[float], torch.Tensor]) -> torch.Tensor:
    """Undiscounted return-to-go ``G_t`` for one complete reward sequence.

    Computed with a reverse cumulative sum and returned in ``float64``;
    rewards must be finite and the sequence must be non-empty.
    """

    if isinstance(rewards, torch.Tensor):
        if rewards.ndim != 1:
            raise ContractError(
                f"rewards must be one-dimensional, got shape {tuple(rewards.shape)}"
            )
        values = rewards.to(torch.float64)
    else:
        if isinstance(rewards, (str, bytes)) or not isinstance(rewards, Sequence):
            raise ContractError(
                f"rewards must be a sequence of real numbers, got {type(rewards).__name__}"
            )
        try:
            values = torch.tensor(
                [_require_finite_float(item, "reward") for item in rewards], dtype=torch.float64
            )
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ContractError("rewards must be a sequence of real numbers") from exc
    if values.numel() == 0:
        raise ContractError("return-to-go requires at least one reward")
    if not bool(torch.isfinite(values).all()):
        raise ContractError("rewards must be finite")
    reversed_cumulative = torch.cumsum(torch.flip(values, dims=[0]), dim=0)
    return torch.flip(reversed_cumulative, dims=[0])


def _as_state_tensor(state: Any) -> torch.Tensor:
    if not isinstance(state, torch.Tensor):
        raise ContractError(f"state must be a torch.Tensor, got {type(state).__name__}")
    if state.ndim != 1:
        raise ContractError(f"state must be one vector, got shape {tuple(state.shape)}")
    if state.numel() < 1:
        raise ContractError("state must carry at least one coordinate")
    if not state.dtype.is_floating_point:
        raise ContractError(f"state must be a floating tensor, got dtype {state.dtype}")
    if not bool(torch.isfinite(state).all()):
        raise ContractError("state must be finite")
    return state.detach().clone()


@dataclass(frozen=True, eq=False)
class TrajectoryStep:
    """One retained ``(state, action, reward, flags)`` transition record.

    Instances compare and hash by identity, which lets the learner keep a
    consumption ledger of *used* batches without ever storing transition data
    for replay.
    """

    state: torch.Tensor
    direction: Direction
    requested_risk: float
    reward: float
    terminal: bool
    truncated: bool
    generation_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _as_state_tensor(self.state))
        object.__setattr__(self, "direction", _coerce_direction(self.direction))
        object.__setattr__(
            self, "requested_risk", _require_finite_float(self.requested_risk, "requested_risk")
        )
        object.__setattr__(self, "reward", _require_finite_float(self.reward, "reward"))
        validate_step(self)

    @property
    def nominal_action(self) -> NominalAction:
        """The semantic action carried by this step."""

        return NominalAction(direction=self.direction, requested_risk=self.requested_risk)


def validate_step(step: Any) -> None:
    """Fail closed on a malformed or corrupted step record."""

    if not isinstance(step, TrajectoryStep):
        raise ContractError(f"step must be a TrajectoryStep, got {type(step).__name__}")
    state = step.state
    if not isinstance(state, torch.Tensor) or state.ndim != 1 or state.numel() < 1:
        raise ContractError("trajectory state must be a non-empty vector")
    if not state.dtype.is_floating_point:
        raise ContractError(f"trajectory state must be a floating tensor, got dtype {state.dtype}")
    if not bool(torch.isfinite(state).all()):
        raise ContractError("trajectory state must be finite")
    direction = _coerce_direction(step.direction)
    risk = _require_finite_float(step.requested_risk, "requested_risk")
    if not 0.0 <= risk <= 1.0:
        raise ContractError(f"requested_risk must lie in [0, 1], got {step.requested_risk!r}")
    is_flat = direction is Direction.FLAT
    if is_flat != (risk == 0.0):
        raise ContractError(
            "FLAT holds iff requested_risk == 0; got "
            f"direction={direction.name} requested_risk={risk!r}"
        )
    _require_finite_float(step.reward, "reward")
    _require_bool(step.terminal, "terminal")
    _require_bool(step.truncated, "truncated")
    if step.terminal and step.truncated:
        raise ContractError(
            "a step cannot be terminal and truncated at once; truncation is not a terminal "
            "reward event"
        )
    _require_generation_id(step.generation_id)


@dataclass(frozen=True, eq=False)
class Trajectory:
    """A complete-or-incomplete sequence of steps from exactly one generation.

    ``complete`` states that the declared scientific endpoint (or a true
    terminal event) was reached.  An incomplete record is representable so that
    a timeout or chunk boundary stays visible, but the learner rejects it.
    """

    steps: Tuple[TrajectoryStep, ...]
    generation_id: int
    complete: bool

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "generation_id", _require_generation_id(self.generation_id))
        object.__setattr__(self, "complete", _require_bool(self.complete, "complete"))
        validate_trajectory(self)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def rewards(self) -> Tuple[float, ...]:
        return tuple(step.reward for step in self.steps)

    @property
    def truncated(self) -> bool:
        return any(step.truncated for step in self.steps)

    @property
    def terminal(self) -> bool:
        return bool(self.steps) and self.steps[-1].terminal

    def returns_to_go(self) -> torch.Tensor:
        """Undiscounted ``G_t`` for every step of this trajectory."""

        return returns_to_go(self.rewards)


def validate_trajectory(trajectory: Any) -> None:
    """Fail closed on a malformed, empty, mixed-generation or mis-flagged record."""

    if not isinstance(trajectory, Trajectory):
        raise ContractError(f"trajectory must be a Trajectory, got {type(trajectory).__name__}")
    steps = trajectory.steps
    if not isinstance(steps, tuple) or len(steps) == 0:
        raise ContractError("trajectory must hold at least one step")
    _require_generation_id(trajectory.generation_id)
    _require_bool(trajectory.complete, "complete")
    last = len(steps) - 1
    for position, step in enumerate(steps):
        validate_step(step)
        if step.generation_id != trajectory.generation_id:
            raise ContractError(
                f"step {position} carries generation_id {step.generation_id}, but the "
                f"trajectory declares {trajectory.generation_id}"
            )
        if (step.terminal or step.truncated) and position != last:
            raise ContractError(
                f"step {position} is terminal or truncated before the final step; only the "
                "declared endpoint may end the trajectory"
            )


__all__ = ["Trajectory", "TrajectoryStep", "returns_to_go", "validate_step", "validate_trajectory"]
