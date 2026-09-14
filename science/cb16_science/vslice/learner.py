"""R12 VS-B strict on-policy generation boundary and Monte-Carlo update.

One learner owns one actor, one critic and one frozen sensory module.  The
required lifecycle for generation ``g`` is:

```text
freeze actor theta_g
collect complete trajectories carrying generation_id = g
finish collection
recompute log_prob under unchanged theta_g
compute undiscounted return-to-go G_t
compute detached advantage A_hat_t = G_t - stopgrad(V_phi(s_t))
actor optimizer step
critic optimizer step
mark the consumed batch used
increment generation_id to g + 1
```

There is no replay: a consumed batch can never be updated twice, stale
generations and mixed-generation batches are rejected, and a truncated or
incomplete trajectory is never converted into a terminal sample.

Generation ``g`` is guarded by an *exact* snapshot of the actor parameters:
detached clones compared tensor-by-tensor with ``torch.equal`` (never with a
reduced aggregate statistic).  Any mutation between collection and update --
including a balanced in-place edit that preserves every tensor sum and absolute
sum -- is rejected before an optimizer step, and the snapshot is refreshed only
by a successful learner-owned update.

Losses are the preregistered on-policy Monte-Carlo objectives:

```text
L_actor = -mean(log_pi(a_t | s_t) * A_hat_t)
L_value =  mean((V_phi(s_t) - G_t)^2)
```

with ``A_hat_t = G_t - stopgrad(V_phi(s_t))``, no discount, no clipping, no
bootstrapping, no target network and no behaviour-policy correction.  The
default direction-entropy coefficient is exactly zero; later controlled
experiments may opt into a categorical-direction-only entropy term without
adding any Beta-risk entropy.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Tuple

import torch
from torch import nn

from .contracts import AccountState, ContractError, _require_finite_float
from .policy import (
    ACCOUNT_STATE_DIM,
    ActionSample,
    Actor,
    LearnerState,
    ValueCritic,
    build_learner_state,
    direction_index,
    DIRECTION_INDEX_FLAT,
)
from .sensory import FrozenSensory
from .trajectory import Trajectory, validate_trajectory

#: Experiment-scoped optimizer defaults (not Task A/B qualification hyperparameters).
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3


class LearnerContractError(ContractError):
    """An on-policy generation-boundary rule was violated."""


@dataclass(frozen=True)
class UpdateReport:
    """Diagnostics of exactly one consumed generation update."""

    generation_id: int
    next_generation_id: int
    trajectory_count: int
    step_count: int
    actor_loss: float
    critic_loss: float
    actor_grad_norm: float
    critic_grad_norm: float
    critic_grad_from_actor_max_abs: float
    mean_return_to_go: float
    mean_value: float


def _validate_one_step_tensor_contract(
    states: Any, directions: Any, risks: Any, rewards: Any, generation_id: Any
) -> torch.Tensor:
    if not isinstance(states, torch.Tensor) or states.ndim != 2 or states.shape[0] < 1:
        raise LearnerContractError("one-step states must be a non-empty [N, state_dim] tensor")
    n = states.shape[0]
    if not states.dtype.is_floating_point or not bool(torch.isfinite(states).all()):
        raise LearnerContractError("one-step states must be finite floating values")
    if not isinstance(directions, torch.Tensor) or directions.ndim != 1 or directions.numel() != n:
        raise LearnerContractError("one-step direction_indices must be a [N] tensor")
    if directions.dtype not in (torch.int32, torch.int64):
        raise LearnerContractError("one-step direction_indices must be integer typed")
    directions = directions.to(torch.long)
    if bool(((directions < 0) | (directions > 2)).any()):
        raise LearnerContractError("one-step direction indices must lie in 0..2")
    if not isinstance(risks, torch.Tensor) or risks.ndim != 1 or risks.numel() != n:
        raise LearnerContractError("one-step requested_risks must be a [N] tensor")
    if not risks.dtype.is_floating_point or not bool(torch.isfinite(risks).all()):
        raise LearnerContractError("one-step requested_risks must be finite floating values")
    if bool(((risks < 0.0) | (risks > 1.0)).any()):
        raise LearnerContractError("one-step requested_risks must lie in [0, 1]")
    flat = directions == DIRECTION_INDEX_FLAT
    if bool((flat & (risks != 0.0)).any() | ((~flat) & (risks == 0.0)).any()):
        raise LearnerContractError("one-step FLAT iff requested_risk == 0 semantic contract failed")
    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 1 or rewards.numel() != n:
        raise LearnerContractError("one-step rewards must be a [N] tensor")
    if not rewards.dtype.is_floating_point or not bool(torch.isfinite(rewards).all()):
        raise LearnerContractError("one-step rewards must be finite floating values")
    if isinstance(generation_id, bool) or not isinstance(generation_id, int) or generation_id < 0:
        raise LearnerContractError("one-step generation_id must be a non-negative int")
    return directions


@dataclass(frozen=True, eq=False)
class OnPolicyOneStepBatch:
    """Tensor-native batch of complete one-step trajectories."""

    states: torch.Tensor
    direction_indices: torch.Tensor
    requested_risks: torch.Tensor
    rewards: torch.Tensor
    generation_id: int

    def __post_init__(self) -> None:
        directions = _validate_one_step_tensor_contract(
            self.states, self.direction_indices, self.requested_risks, self.rewards, self.generation_id
        )
        object.__setattr__(self, "states", self.states.detach().clone())
        object.__setattr__(self, "direction_indices", directions.detach().clone())
        object.__setattr__(self, "requested_risks", self.requested_risks.detach().clone())
        object.__setattr__(self, "rewards", self.rewards.detach().clone())

    @property
    def trajectory_count(self) -> int:
        return int(self.states.shape[0])


@dataclass(frozen=True)
class _Objectives:
    actor_loss: torch.Tensor
    critic_loss: torch.Tensor
    log_prob: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    direction_entropy: torch.Tensor


def _grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().to(torch.float64)
        total += float(gradient.pow(2).sum())
    return math.sqrt(total)


def _max_abs_grad(module: nn.Module) -> float:
    maximum = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        maximum = max(maximum, float(parameter.grad.detach().abs().max()))
    return maximum


class OnPolicyLearner:
    """Strict on-policy learner over the frozen sensory / actor / critic spine."""

    def __init__(
        self,
        sensory: FrozenSensory,
        actor: Actor,
        critic: ValueCritic,
        *,
        actor_lr: float = ACTOR_LR,
        critic_lr: float = CRITIC_LR,
        direction_entropy_coefficient: float = 0.0,
    ) -> None:
        if not isinstance(sensory, FrozenSensory):
            raise LearnerContractError(
                f"sensory must be a FrozenSensory, got {type(sensory).__name__}"
            )
        if not isinstance(actor, Actor):
            raise LearnerContractError(f"actor must be an Actor, got {type(actor).__name__}")
        if not isinstance(critic, ValueCritic):
            raise LearnerContractError(f"critic must be a ValueCritic, got {type(critic).__name__}")
        if list(sensory.parameters()):
            raise LearnerContractError("frozen sensory must expose zero trainable parameters")
        if actor.input_dim != sensory.z_dim + ACCOUNT_STATE_DIM:
            raise LearnerContractError(
                f"actor input_dim {actor.input_dim} must equal sensory z_dim "
                f"{sensory.z_dim} + {ACCOUNT_STATE_DIM}"
            )
        if critic.input_dim != actor.input_dim:
            raise LearnerContractError(
                f"critic input_dim {critic.input_dim} must equal actor input_dim {actor.input_dim}"
            )
        shared = set(map(id, actor.parameters())) & set(map(id, critic.parameters()))
        if shared:
            raise LearnerContractError("actor and critic must not share parameter objects")

        self._actor_lr = _require_positive_lr(actor_lr, "actor_lr")
        self._critic_lr = _require_positive_lr(critic_lr, "critic_lr")
        self._direction_entropy_coefficient = _require_nonnegative_coefficient(
            direction_entropy_coefficient, "direction_entropy_coefficient"
        )
        self._sensory = sensory
        self._actor = actor
        self._critic = critic
        # Separate optimizer instances: no weighted actor/critic scalarization.
        self._actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self._actor_lr)
        self._critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self._critic_lr)
        self._generation_id = 0
        # A consumption ledger keyed by trajectory identity.  Weak references
        # keep no transition data alive, so this is not a replay buffer.
        self._consumed: "weakref.WeakSet[Any]" = weakref.WeakSet()
        # Exact frozen generation snapshot theta_g: detached clones of every
        # actor parameter tensor, refreshed only by a learner-owned update.
        self._actor_snapshot = self._snapshot_actor_parameters()

    # -- read-only surface -------------------------------------------------

    @property
    def generation_id(self) -> int:
        return self._generation_id

    @property
    def sensory(self) -> FrozenSensory:
        return self._sensory

    @property
    def actor(self) -> Actor:
        return self._actor

    @property
    def critic(self) -> ValueCritic:
        return self._critic

    @property
    def actor_optimizer(self) -> torch.optim.Optimizer:
        return self._actor_optimizer

    @property
    def critic_optimizer(self) -> torch.optim.Optimizer:
        return self._critic_optimizer

    @property
    def direction_entropy_coefficient(self) -> float:
        return self._direction_entropy_coefficient

    @property
    def state_dim(self) -> int:
        return self._actor.input_dim

    # -- collection --------------------------------------------------------

    def encode(self, market: torch.Tensor) -> torch.Tensor:
        """Frozen sensory representation ``Z_t`` of one normalized market tensor."""

        return self._sensory(market)

    def build_state(self, market: torch.Tensor, account_state: AccountState) -> LearnerState:
        """Runtime state ``(Z_t, A_t)`` from normalized market plus account state."""

        return build_learner_state(self.encode(market), account_state)

    def sample_action(self, state: Any) -> ActionSample:
        """Sample one behavior action for the current generation.

        Collection performs no optimizer step, so actor parameters are unchanged
        by any number of calls.
        """

        self._require_generation_snapshot()
        return self._actor.sample(state)

    # -- objectives --------------------------------------------------------

    def actor_objective(self, trajectories: Any) -> torch.Tensor:
        """``L_actor`` for a valid current-generation batch (grad-enabled).

        The caller owns gradient buffers; :meth:`update` performs its own
        zero/backward/step sequence.
        """

        return self._objectives(self._validated_batch(trajectories)).actor_loss

    def critic_objective(self, trajectories: Any) -> torch.Tensor:
        """``L_value`` against the realized undiscounted return-to-go."""

        return self._objectives(self._validated_batch(trajectories)).critic_loss

    # -- update ------------------------------------------------------------

    def update(self, trajectories: Any) -> UpdateReport:
        """Consume one current-generation trajectory batch and commit ``g + 1``."""

        batch = self._validated_batch(trajectories)
        objectives = self._objectives(batch)
        return self._commit_update(objectives, trajectory_count=len(batch), consumed=batch)

    def update_one_step_batch(self, batch: OnPolicyOneStepBatch) -> UpdateReport:
        """Consume a tensor-native batch of complete one-step trajectories.

        This is an execution fast path only.  The actor/critic objectives,
        generation snapshot, no-replay rule and optimizer steps are shared with
        :meth:`update`.
        """

        batch = self._validated_one_step_batch(batch)
        objectives = self._objectives_from_tensors(
            batch.states,
            batch.direction_indices,
            batch.requested_risks.to(batch.states.dtype),
            batch.rewards.to(torch.float64),
        )
        return self._commit_update(objectives, trajectory_count=batch.trajectory_count, consumed=(batch,))

    def _commit_update(
        self, objectives: _Objectives, *, trajectory_count: int, consumed: Sequence[Any]
    ) -> UpdateReport:
        self._actor_optimizer.zero_grad(set_to_none=True)
        self._critic_optimizer.zero_grad(set_to_none=True)
        objectives.actor_loss.backward()
        critic_grad_from_actor = _max_abs_grad(self._critic)
        objectives.critic_loss.backward()
        actor_grad_norm = _grad_norm(self._actor)
        critic_grad_norm = _grad_norm(self._critic)
        if not math.isfinite(actor_grad_norm) or not math.isfinite(critic_grad_norm):
            raise LearnerContractError(
                "actor and critic gradients must be finite; no optimizer step was taken"
            )
        self._actor_optimizer.step()
        self._critic_optimizer.step()
        for item in consumed:
            self._consumed.add(item)
        updated_generation = self._generation_id
        self._generation_id += 1
        self._actor_snapshot = self._snapshot_actor_parameters()
        return UpdateReport(
            generation_id=updated_generation,
            next_generation_id=self._generation_id,
            trajectory_count=trajectory_count,
            step_count=int(objectives.returns.numel()),
            actor_loss=float(objectives.actor_loss.detach()),
            critic_loss=float(objectives.critic_loss.detach()),
            actor_grad_norm=actor_grad_norm,
            critic_grad_norm=critic_grad_norm,
            critic_grad_from_actor_max_abs=critic_grad_from_actor,
            mean_return_to_go=float(objectives.returns.detach().mean()),
            mean_value=float(objectives.values.detach().mean()),
        )

    # -- internals ---------------------------------------------------------

    def _validated_batch(self, trajectories: Any) -> Tuple[Trajectory, ...]:
        if isinstance(trajectories, Trajectory):
            raise LearnerContractError(
                "update and objective calls require a batch (sequence) of trajectories, got one "
                "Trajectory"
            )
        if isinstance(trajectories, (str, bytes)) or not isinstance(trajectories, Sequence):
            raise LearnerContractError(
                f"trajectories must be a sequence of Trajectory records, got "
                f"{type(trajectories).__name__}"
            )
        batch = tuple(trajectories)
        if not batch:
            raise LearnerContractError("empty update batch")
        for trajectory in batch:
            if not isinstance(trajectory, Trajectory):
                raise LearnerContractError(
                    f"every batch entry must be a Trajectory, got {type(trajectory).__name__}"
                )
        if any(trajectory in self._consumed for trajectory in batch):
            raise LearnerContractError(
                "batch contains trajectories already consumed by an earlier update; there is no "
                "replay"
            )
        generations = {trajectory.generation_id for trajectory in batch}
        if len(generations) > 1:
            raise LearnerContractError(
                f"mixed-generation batch {sorted(generations)} cannot be used in one update"
            )
        batch_generation = batch[0].generation_id
        if batch_generation != self._generation_id:
            raise LearnerContractError(
                f"trajectory generation_id {batch_generation} does not match the current learner "
                f"generation {self._generation_id}"
            )
        self._require_generation_snapshot()
        for trajectory in batch:
            validate_trajectory(trajectory)
            if trajectory.truncated:
                raise LearnerContractError(
                    "truncated trajectories cannot be used for an on-policy Monte-Carlo update"
                )
            if not trajectory.complete:
                raise LearnerContractError(
                    "incomplete trajectory cannot be used for an on-policy Monte-Carlo update; "
                    "declare the scientific endpoint instead of treating a cutoff as terminal"
                )
            for step in trajectory.steps:
                if step.state.shape[-1] != self.state_dim:
                    raise LearnerContractError(
                        f"trajectory state width {step.state.shape[-1]} does not match learner "
                        f"state width {self.state_dim}"
                    )
        return batch

    def _validated_one_step_batch(self, batch: Any) -> OnPolicyOneStepBatch:
        if not isinstance(batch, OnPolicyOneStepBatch):
            raise LearnerContractError(
                f"one-step fast path requires OnPolicyOneStepBatch, got {type(batch).__name__}"
            )
        _validate_one_step_tensor_contract(
            batch.states, batch.direction_indices, batch.requested_risks, batch.rewards, batch.generation_id
        )
        if batch in self._consumed:
            raise LearnerContractError("one-step batch was already consumed; there is no replay")
        if batch.generation_id != self._generation_id:
            raise LearnerContractError(
                f"one-step generation_id {batch.generation_id} does not match learner generation {self._generation_id}"
            )
        if batch.states.shape[1] != self.state_dim:
            raise LearnerContractError(
                f"one-step state width {batch.states.shape[1]} does not match learner state width {self.state_dim}"
            )
        self._require_generation_snapshot()
        return batch

    def _objectives(self, batch: Tuple[Trajectory, ...]) -> _Objectives:
        states = torch.stack([step.state for trajectory in batch for step in trajectory.steps])
        index = torch.tensor(
            [
                direction_index(step.direction)
                for trajectory in batch
                for step in trajectory.steps
            ],
            dtype=torch.long,
        )
        risks = torch.tensor(
            [step.requested_risk for trajectory in batch for step in trajectory.steps],
            dtype=states.dtype,
        )
        returns = torch.cat([trajectory.returns_to_go() for trajectory in batch])
        return self._objectives_from_tensors(states, index, risks, returns)

    def _objectives_from_tensors(
        self, states: torch.Tensor, index: torch.Tensor, risks: torch.Tensor, returns: torch.Tensor
    ) -> _Objectives:
        log_prob = self._actor.log_prob_batch(states, index, risks)
        values = self._critic(states)
        advantages = returns - values.detach()
        direction_entropy = self._actor.direction_entropy_batch(states)
        actor_loss = -(log_prob * advantages).mean() - (
            self._direction_entropy_coefficient * direction_entropy.mean()
        )
        critic_loss = ((values - returns) ** 2).mean()
        objectives = _Objectives(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            log_prob=log_prob,
            values=values,
            returns=returns,
            advantages=advantages,
            direction_entropy=direction_entropy,
        )
        self._require_finite_objectives(objectives)
        return objectives

    @staticmethod
    def _require_finite_objectives(objectives: _Objectives) -> None:
        tensors = (
            ("log-probability", objectives.log_prob),
            ("return-to-go", objectives.returns),
            ("critic value", objectives.values),
            ("advantage", objectives.advantages),
            ("direction entropy", objectives.direction_entropy),
            ("actor loss", objectives.actor_loss),
            ("critic loss", objectives.critic_loss),
        )
        for name, tensor in tensors:
            if not bool(torch.isfinite(tensor).all()):
                raise LearnerContractError(f"non-finite {name}; refusing to update")

    def _snapshot_actor_parameters(self) -> Tuple[torch.Tensor, ...]:
        """Exact detached clone of every actor parameter tensor (theta_g).

        The snapshot is compared with :func:`torch.equal` element by element, so
        no reduced statistic (sum, absolute sum, norm, hash of a few moments)
        can hide a permutation or any other balanced parameter mutation.
        """

        with torch.no_grad():
            return tuple(parameter.detach().clone() for parameter in self._actor.parameters())

    def _actor_snapshot_intact(self) -> bool:
        parameters = tuple(self._actor.parameters())
        if len(parameters) != len(self._actor_snapshot):
            return False
        with torch.no_grad():
            for expected, parameter in zip(self._actor_snapshot, parameters):
                if expected.shape != parameter.shape:
                    return False
                if expected.dtype != parameter.dtype or expected.device != parameter.device:
                    return False
                if not torch.equal(expected, parameter.detach()):
                    return False
        return True

    def _require_generation_snapshot(self) -> None:
        if not self._actor_snapshot_intact():
            raise LearnerContractError(
                "actor parameters changed outside the learner's generation boundary; the exact "
                f"snapshot of generation theta_{self._generation_id} is no longer intact"
            )


def _require_nonnegative_coefficient(value: Any, name: str) -> float:
    coefficient = _require_finite_float(value, name)
    if coefficient < 0.0:
        raise LearnerContractError(f"{name} must be non-negative, got {value!r}")
    return coefficient


def _require_positive_lr(value: Any, name: str) -> float:
    learning_rate = _require_finite_float(value, name)
    if not learning_rate > 0.0:
        raise LearnerContractError(f"{name} must be strictly positive, got {value!r}")
    return learning_rate


__all__ = [
    "ACTOR_LR",
    "CRITIC_LR",
    "LearnerContractError",
    "OnPolicyLearner",
    "OnPolicyOneStepBatch",
    "UpdateReport",
]
