"""Vectorized controlled-task collection for R12 VS-C+.

This module is an execution optimization only. It does not change the action
space, account semantics, Physics equations, learner objective, or scientific
gates. It batches actor inference and controlled-task Physics across a whole
on-policy generation so the CPU runs matrix/tensor operations instead of one
small Python/MLP call per trajectory.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.distributions import Beta, Categorical

from . import controlled_tasks as tasks
from .contracts import (
    ContractError,
    Direction,
    PhysicsConfig,
    account_state_from_truth,
)
from .learner import OnPolicyLearner, OnPolicyOneStepBatch
from .policy import (
    DIRECTION_INDEX_FLAT,
    DIRECTION_INDEX_SHORT,
    DIRECTION_ORDER,
    account_state_vector,
)
from .trajectory import Trajectory, TrajectoryStep


def _direction_values(indices: torch.Tensor) -> Tuple[Direction, ...]:
    return tuple(DIRECTION_ORDER[int(i)] for i in indices.tolist())


def _nominal_targets(indices: torch.Tensor, risks: torch.Tensor, config: PhysicsConfig) -> torch.Tensor:
    signed = indices.to(torch.float64) - 1.0
    return signed * risks.to(torch.float64) * float(config.nominal_exposure_budget)


def sample_action_batch(learner: OnPolicyLearner, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one generation's semantic actions in one actor forward pass."""
    if not isinstance(states, torch.Tensor) or states.ndim != 2:
        raise ContractError("states must be a [N, state_dim] tensor")
    if states.shape[0] < 1:
        raise ContractError("states batch must be non-empty")
    if states.shape[1] != learner.state_dim:
        raise ContractError(
            f"state width {states.shape[1]} does not match learner state width {learner.state_dim}"
        )

    learner._require_generation_snapshot()
    with torch.no_grad():
        output = learner.actor.forward(states)
        indices = Categorical(logits=output.direction_logits).sample().to(torch.long)
        flat = indices == DIRECTION_INDEX_FLAT
        short = indices == DIRECTION_INDEX_SHORT
        alpha = torch.where(short, output.short_alpha, output.long_alpha)
        beta = torch.where(short, output.short_beta, output.long_beta)
        risks = torch.zeros(indices.shape[0], dtype=learner.actor.dtype)
        nonflat = ~flat
        if bool(nonflat.any()):
            sampled = Beta(alpha[nonflat], beta[nonflat]).sample()
            if not bool(torch.isfinite(sampled).all()):
                raise ContractError("sampled requested_risk must be finite")
            if bool(((sampled <= 0.0) | (sampled >= 1.0)).any()):
                raise ContractError(
                    "sampled requested_risk must lie strictly inside (0, 1) for non-FLAT actions"
                )
            risks[nonflat] = sampled
    return indices, risks


def _candidate_feasible_batch(
    pretrade_equity: torch.Tensor,
    pretrade_notional: torch.Tensor,
    target: torch.Tensor,
    config: PhysicsConfig,
) -> torch.Tensor:
    n_star = target * pretrade_equity
    cost = float(config.proportional_friction_kappa) * torch.abs(n_star - pretrade_notional)
    equity_plus = pretrade_equity - cost
    safe_equity = torch.where(equity_plus > 0.0, equity_plus, torch.ones_like(equity_plus))
    exposure = torch.abs(n_star / safe_equity)
    return (equity_plus > 0.0) & (
        exposure <= float(config.hard_exposure_limit + config.finite_tolerance)
    )


def _boundary_batch(
    pretrade_equity: torch.Tensor,
    pretrade_notional: torch.Tensor,
    sign: float,
    config: PhysicsConfig,
) -> torch.Tensor:
    """Vectorized form of the frozen per-account bisection contract."""
    limit = float(config.hard_exposure_limit)
    boundary = torch.full_like(pretrade_equity, sign * limit)
    boundary_ok = _candidate_feasible_batch(pretrade_equity, pretrade_notional, boundary, config)
    zero = torch.zeros_like(pretrade_equity)
    if not bool(_candidate_feasible_batch(pretrade_equity, pretrade_notional, zero, config).all()):
        raise ContractError("zero target exposure is not feasible for at least one batch entry")
    low = torch.zeros_like(pretrade_equity)
    high = torch.full_like(pretrade_equity, limit)
    for _ in range(config.permission_bisection_iterations):
        mid = 0.5 * (low + high)
        ok = _candidate_feasible_batch(pretrade_equity, pretrade_notional, mid * sign, config)
        low = torch.where(ok, mid, low)
        high = torch.where(ok, high, mid)
    magnitude = torch.where(boundary_ok, torch.full_like(low, limit), low)
    return magnitude


def task_a_reward_batch(
    env: tasks.TaskAEnvironment,
    actual_indices: torch.Tensor,
    direction_indices: torch.Tensor,
    requested_risks: torch.Tensor,
) -> torch.Tensor:
    """Canonical Task-A one-step log-equity rewards, tensorized across the batch."""
    if actual_indices.ndim != 1 or direction_indices.ndim != 1 or requested_risks.ndim != 1:
        raise ContractError("Task-A batch inputs must be one-dimensional")
    n = actual_indices.numel()
    if direction_indices.numel() != n or requested_risks.numel() != n:
        raise ContractError("Task-A batch inputs must have equal length")

    equity_t = torch.tensor(
        [env.cases[int(i)].truth.equity for i in actual_indices.tolist()], dtype=torch.float64
    )
    quantity_t = torch.tensor(
        [env.cases[int(i)].truth.quantity for i in actual_indices.tolist()], dtype=torch.float64
    )
    mark_t = torch.tensor(
        [env.cases[int(i)].truth.mark_price for i in actual_indices.tolist()], dtype=torch.float64
    )
    open_next = float(env.open_next)
    close_next = float(env.close_next)
    pretrade_equity = equity_t + quantity_t * (open_next - mark_t)
    if not bool((pretrade_equity > 0.0).all()):
        raise ContractError("Task-A vectorized pre-trade equity must stay strictly positive")
    pretrade_notional = quantity_t * open_next
    nominal = _nominal_targets(direction_indices, requested_risks, env.config)

    feasible_min = -_boundary_batch(pretrade_equity, pretrade_notional, -1.0, env.config)
    feasible_max = _boundary_batch(pretrade_equity, pretrade_notional, +1.0, env.config)
    permitted = torch.minimum(torch.maximum(nominal, feasible_min), feasible_max)

    n_star = permitted * pretrade_equity
    delta = n_star - pretrade_notional
    cost = float(env.config.proportional_friction_kappa) * torch.abs(delta)
    equity_plus = pretrade_equity - cost
    if not bool((equity_plus > 0.0).all()):
        raise ContractError("Task-A vectorized post-cost equity must stay strictly positive")
    quantity_plus = n_star / open_next
    equity_next = equity_plus + quantity_plus * (close_next - open_next)
    if not bool((equity_next > 0.0).all()):
        raise ContractError("Task-A vectorized next equity must stay strictly positive")
    rewards = torch.log(equity_next / equity_t)
    if not bool(torch.isfinite(rewards).all()):
        raise ContractError("Task-A vectorized rewards must be finite")
    return rewards


def _task_a_generation_tensors(
    learner: OnPolicyLearner,
    env: tasks.TaskAEnvironment,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == tasks.ARM_POSITIVE:
        schedule = env.positive_schedule
    elif mode == tasks.ARM_CONTROL:
        schedule = env.control_schedule
    else:
        raise ContractError(f"unknown Task-A arm {mode!r}")
    actual_indices = torch.tensor([a for a, _ in schedule], dtype=torch.long)
    observed_indices = torch.tensor([o for _, o in schedule], dtype=torch.long)
    z = learner.encode(torch.from_numpy(env.market)).to(torch.float32)
    account_rows = torch.stack(
        [account_state_vector(account_state_from_truth(case.truth, env.config)) for case in env.cases]
    )
    states = torch.cat(
        (z.expand(len(schedule), z.shape[-1]), account_rows.index_select(0, observed_indices)), dim=-1
    )
    direction_indices, risks = sample_action_batch(learner, states)
    rewards = task_a_reward_batch(env, actual_indices, direction_indices, risks)
    return states, direction_indices, risks, rewards


def build_task_a_one_step_batch(
    learner: OnPolicyLearner, env: tasks.TaskAEnvironment, generation_id: int, *, mode: str
) -> OnPolicyOneStepBatch:
    """Tensor-native complete one-step Task-A batch for the learner fast path."""
    states, direction_indices, risks, rewards = _task_a_generation_tensors(learner, env, mode=mode)
    return OnPolicyOneStepBatch(
        states=states,
        direction_indices=direction_indices,
        requested_risks=risks,
        rewards=rewards,
        generation_id=generation_id,
    )


def build_task_a_generation(
    learner: OnPolicyLearner,
    env: tasks.TaskAEnvironment,
    generation_id: int,
    *,
    mode: str,
) -> Tuple[Trajectory, ...]:
    """Reference object path for Task A; retained as semantic authority/equivalence oracle."""
    states, direction_indices, risks, rewards = _task_a_generation_tensors(learner, env, mode=mode)
    directions = _direction_values(direction_indices)
    return tuple(
        Trajectory(
            steps=(TrajectoryStep(
                state=states[i], direction=directions[i], requested_risk=float(risks[i]),
                reward=float(rewards[i]), terminal=False, truncated=False, generation_id=generation_id,
            ),),
            generation_id=generation_id, complete=True,
        )
        for i in range(states.shape[0])
    )


def _task_b_step0_states(learner: OnPolicyLearner, env: tasks.TaskBEnvironment) -> torch.Tensor:
    z_up = learner.encode(torch.from_numpy(env.cue_markets[tasks.UP])).to(torch.float32)
    z_down = learner.encode(torch.from_numpy(env.cue_markets[tasks.DOWN])).to(torch.float32)
    z_rows = torch.stack([z_up if cue == tasks.UP else z_down for cue in env.cue_schedule])
    account = account_state_vector(account_state_from_truth(env.initial_truth, env.config))
    return torch.cat((z_rows, account.expand(len(env.cue_schedule), account.shape[-1])), dim=-1)


def _task_b_delayed_reward(
    env: tasks.TaskBEnvironment,
    direction_indices: torch.Tensor,
    requested_risks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = _nominal_targets(direction_indices, requested_risks, env.config)
    gaps = torch.tensor(
        [env.up_gap if cue == tasks.UP else env.down_gap for cue in env.cue_schedule],
        dtype=torch.float64,
    )
    base = float(env.step0_open_next)
    growth_factor = 1.0 + target * (gaps / base - 1.0)
    if not bool((growth_factor > 0.0).all()):
        raise ContractError("Task-B vectorized delayed account value must stay positive")
    reward = torch.log(growth_factor)
    if not bool(torch.isfinite(reward).all()):
        raise ContractError("Task-B vectorized delayed rewards must be finite")
    return target, reward


def build_task_b_generation(
    learner: OnPolicyLearner,
    env: tasks.TaskBEnvironment,
    generation_id: int,
    *,
    mode: str,
    seed: int,
) -> Tuple[Trajectory, ...]:
    """Task B with two batched actor forwards and tensorized delayed Physics."""
    states0 = _task_b_step0_states(learner, env)
    dir0, risk0 = sample_action_batch(learner, states0)
    target0, delayed = _task_b_delayed_reward(env, dir0, risk0)

    z_neutral = learner.encode(torch.from_numpy(env.neutral_market)).to(torch.float32)
    exposure = target0.to(torch.float32)
    capacity = torch.clamp(
        1.0 - torch.abs(exposure) / float(env.config.hard_exposure_limit), min=0.0
    )
    account1 = torch.stack((exposure, torch.ones_like(exposure), capacity), dim=-1)
    states1 = torch.cat(
        (z_neutral.expand(states0.shape[0], z_neutral.shape[-1]), account1), dim=-1
    )
    dir1, risk1 = sample_action_batch(learner, states1)

    if mode == tasks.ARM_POSITIVE:
        assigned = delayed
    elif mode == tasks.ARM_CONTROL:
        order = tasks.no_fixed_point_permutation(
            delayed.numel(), seed=seed, generation=generation_id
        )
        assigned = delayed.index_select(0, torch.tensor(order, dtype=torch.long))
    else:
        raise ContractError(f"unknown Task-B arm {mode!r}")

    directions0 = _direction_values(dir0)
    directions1 = _direction_values(dir1)
    return tuple(
        Trajectory(
            steps=(
                TrajectoryStep(
                    state=states0[i],
                    direction=directions0[i],
                    requested_risk=float(risk0[i]),
                    reward=0.0,
                    terminal=False,
                    truncated=False,
                    generation_id=generation_id,
                ),
                TrajectoryStep(
                    state=states1[i],
                    direction=directions1[i],
                    requested_risk=float(risk1[i]),
                    reward=float(assigned[i]),
                    terminal=False,
                    truncated=False,
                    generation_id=generation_id,
                ),
            ),
            generation_id=generation_id,
            complete=True,
        )
        for i in range(states0.shape[0])
    )


__all__ = [
    "build_task_a_generation",
    "build_task_b_generation",
    "sample_action_batch",
    "task_a_reward_batch",
]
