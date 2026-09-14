"""R12 VS-C controlled synthetic tasks: Task A, Task B and their controls.

Authority (used literally):

* ``config/experiments/r12_vs_c_r0.json`` -- the preregistered machine-readable
  authority.  Every budget, formula, seed and control construction below is
  read from, or pinned to, that file; nothing here invents a scientific
  parameter.
* ``docs/tasks/R12_VS_C_CONTROLLED_LEARNABILITY_R0.md`` -- the task contract.

The module builds two known-answer synthetic environments on top of the merged
VS-A Physics and VS-B learner spine and never modifies either:

Task A -- account-conditioned maintenance
    One identical flat market (one predecessor plus ``context_length``
    represented bars, every OHLC price ``100``, every volume ``10``) is passed
    through the canonical ``normalize_market_window`` N0 boundary and then
    through the frozen sensory projection.  Three account truths
    (``SHORT_HALF``, ``FLAT``, ``LONG_HALF``) share that identical market
    representation and differ only in the policy-observed ``AccountState``.
    There is no price edge: the only economic reward is the VS-A
    transaction-cost consequence, so maintaining the existing target exposure
    costs zero and unnecessary movement costs capital.

    The matched negative control keeps both account marginals balanced but
    makes the actual account and the policy-observed account exactly
    independent (every ``3 x 3`` pair exactly ``batch_trajectories / 9`` times);
    reward/Physics always use the actual ``AccountTruth``.

Task B -- delayed consequence credit
    Two exponentially-ramped cue windows (``UP``/``DOWN``) and one neutral
    flat window are normalized through the same N0 boundary.  A two-step
    trajectory starts from a flat account, the step-0 market carries the cue
    but has ``open_1 == close_1 == 100`` with ``kappa == 0`` so the immediate
    reward is exactly zero, and the step-1 market is neutral while the gap
    (``+10`` for ``UP``, ``-10`` for ``DOWN``) reprices the position created by
    action 0.  With ``kappa == 0`` and ``close_2 == open_2``, action 1 has no
    contemporaneous PnL effect, so the delayed reward is the economic
    consequence that must credit action 0 through undiscounted return-to-go.

    The matched negative control preserves every collected state, action and
    the complete multiset of delayed rewards, but reassigns the delayed rewards
    with a deterministic seed-and-generation-specific permutation that has no
    fixed points.

Both environments are constructed from the parsed preregistered JSON, so the
runner that consumes them offers no independent scientific parameter path.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from .contracts import (
    PERMISSION_BISECTION_ITERATIONS,
    AccountState,
    AccountTruth,
    ContractError,
    Direction,
    PhysicsConfig,
    account_state_from_truth,
    nominal_target_from_action,
)
from .learner import OnPolicyLearner
from .market import normalize_market_window
from .physics import execute_transition
from .policy import DIRECTION_ORDER, build_learner_state, direction_index
from .trajectory import Trajectory, TrajectoryStep

# ---------------------------------------------------------------------------
# Frozen controlled-slice constants (pinned to the preregistered construction)
# ---------------------------------------------------------------------------

#: Cue classification labels.  ``UP`` means "the up cue", not an economic
#: prediction; the environment, never the learner, owns this label.
UP = "UP"
DOWN = "DOWN"

#: Arm labels for the positive task and its matched negative control.
ARM_POSITIVE = "positive"
ARM_CONTROL = "control"

#: Task identifiers used when deriving deterministic RNG streams.
TASK_A_ID = 0
TASK_B_ID = 1

#: Task A market constants (the preregistered flat window).
TASK_A_PRICE = 100.0
TASK_A_VOLUME = 10.0

#: Cue candle wick factors, exactly as written in the preregistered JSON.
CUE_HIGH_FACTOR = 1.001
CUE_LOW_FACTOR = 1.001

#: Task B step-1 gap magnitudes are read from the JSON; the cue construction
#: uses the JSON's endpoint price and volume.

#: Stream domains keep independent purposes on disjoint RNG streams.
PERMUTATION_DOMAIN = 0x5045524D  # "PERM"
COLLECTION_DOMAIN = 0x434F4C4C  # "COLL"

_MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# Preregistered-JSON access helpers (fail closed on anything unexpected)
# ---------------------------------------------------------------------------


def spec_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Require one JSON object."""

    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a JSON object, got {type(value).__name__}")
    return value


def spec_sequence(value: Any, name: str) -> Sequence[Any]:
    """Require one JSON array."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError(f"{name} must be a JSON array, got {type(value).__name__}")
    return value


def spec_int(value: Any, name: str, *, minimum: int = 0) -> int:
    """Require one JSON integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ContractError(f"{name} must be >= {minimum}, got {value!r}")
    return value


def spec_float(value: Any, name: str) -> float:
    """Require one finite JSON number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{name} must be finite, got {value!r}")
    return number


def spec_str(value: Any, name: str) -> str:
    """Require one non-empty JSON string."""

    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string, got {value!r}")
    return value


def spec_bool(value: Any, name: str) -> bool:
    """Require one JSON boolean."""

    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a boolean, got {value!r}")
    return value


def _require_formula(text: str, needle: str, name: str) -> None:
    """Fail closed when the preregistered formula text is not the implemented one."""

    if needle not in text:
        raise ContractError(
            f"preregistered {name} formula {text!r} is not the implemented construction "
            f"(expected it to contain {needle!r})"
        )


# ---------------------------------------------------------------------------
# Deterministic local streams (no wall-clock entropy)
# ---------------------------------------------------------------------------


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def derived_stream_seed(*parts: int) -> int:
    """Deterministic 64-bit stream seed from integer parts.

    Only preregistered integers (seeds, generation indices, domain tags) may be
    supplied.  No wall-clock or process entropy is read anywhere in this task.
    """

    state = 0xCB16000000000001
    for part in parts:
        if isinstance(part, bool) or not isinstance(part, int):
            raise ContractError(f"stream seed parts must be integers, got {part!r}")
        state = _splitmix64(state ^ ((part & _MASK64) + 0x9E3779B97F4A7C15))
    return state


def control_permutation_seed(seed: int, generation: int) -> int:
    """Permutation stream derived only from the preregistered seed and generation."""

    return derived_stream_seed(PERMUTATION_DOMAIN, seed, generation)


def no_fixed_point_permutation(count: int, *, seed: int, generation: int) -> Tuple[int, ...]:
    """Deterministic derangement of ``range(count)`` for one seed and generation.

    ``result[i]`` is the source index whose delayed reward is assigned to
    trajectory ``i``.  Every call with the same ``(count, seed, generation)``
    returns exactly the same permutation, every index is used exactly once and
    no trajectory keeps its own delayed reward (``result[i] != i``).
    """

    if isinstance(count, bool) or not isinstance(count, int):
        raise ContractError(f"permutation count must be an integer, got {count!r}")
    if count < 2:
        raise ContractError(f"a fixed-point-free permutation requires count >= 2, got {count!r}")
    stream = random.Random(control_permutation_seed(seed, generation))
    order = list(range(count))
    # A uniformly shuffled permutation is a derangement with probability
    # ~1/e, so rejection terminates almost immediately.  The accepted draw is a
    # deterministic function of the stream seed.
    for _ in range(10_000):
        stream.shuffle(order)
        if all(order[index] != index for index in range(count)):
            return tuple(order)
    raise ContractError("failed to draw a fixed-point-free permutation")  # pragma: no cover


# ---------------------------------------------------------------------------
# Raw synthetic windows
# ---------------------------------------------------------------------------


def flat_raw_window(context_length: int, *, price: float, volume: float) -> np.ndarray:
    """One predecessor plus ``context_length`` represented flat OHLCV bars."""

    _require_context_length(context_length)
    if not price > 0.0 or not volume > 0.0:
        raise ContractError("flat window price and volume must be strictly positive")
    window = np.empty((context_length + 1, 5), dtype=np.float64)
    window[:, :4] = float(price)
    window[:, 4] = float(volume)
    return window


def cue_raw_window(
    cue: str,
    context_length: int,
    *,
    endpoint_price: float,
    ramp_magnitude: float,
    volume: float,
) -> np.ndarray:
    """One predecessor plus ``context_length`` represented exponentially-ramped bars.

    For represented index ``j = 0..L-1``::

        UP   close_j = endpoint_price * exp(-ramp * (L - 1 - j) / (L - 1))
        DOWN close_j = endpoint_price * exp(+ramp * (L - 1 - j) / (L - 1))

    ``open_j`` is the previous represented close (the predecessor close for
    ``j = 0``), ``high = max(open, close) * 1.001``,
    ``low = min(open, close) / 1.001`` and every volume is the same constant.
    """

    if cue not in (UP, DOWN):
        raise ContractError(f"cue must be {UP!r} or {DOWN!r}, got {cue!r}")
    _require_context_length(context_length)
    if not endpoint_price > 0.0:
        raise ContractError("cue endpoint_price must be strictly positive")
    if not ramp_magnitude > 0.0:
        raise ContractError("cue ramp_magnitude must be strictly positive")
    if not volume > 0.0:
        raise ContractError("cue volume must be strictly positive")

    sign = -1.0 if cue == UP else 1.0
    index = np.arange(context_length, dtype=np.float64)
    closes = endpoint_price * np.exp(sign * ramp_magnitude * (context_length - 1 - index) / (context_length - 1))

    window = np.empty((context_length + 1, 5), dtype=np.float64)
    predecessor_close = float(closes[0])
    window[0, :4] = predecessor_close
    window[0, 4] = float(volume)
    previous = predecessor_close
    for position, close in enumerate(closes):
        close_value = float(close)
        window[position + 1, 0] = previous
        window[position + 1, 1] = max(previous, close_value) * CUE_HIGH_FACTOR
        window[position + 1, 2] = min(previous, close_value) / CUE_LOW_FACTOR
        window[position + 1, 3] = close_value
        window[position + 1, 4] = float(volume)
        previous = close_value
    return window


def _require_context_length(context_length: Any) -> None:
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise ContractError(f"context_length must be an integer, got {context_length!r}")
    if context_length < 2:
        raise ContractError(f"context_length must be >= 2, got {context_length!r}")


# ---------------------------------------------------------------------------
# Task A
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountCase:
    """One preregistered Task-A account truth plus its known-answer target."""

    name: str
    truth: AccountTruth
    desired_target_exposure: float


@dataclass(frozen=True)
class TaskAEnvironment:
    """Preregistered Task-A environment (flat market, three account truths)."""

    config: PhysicsConfig
    cases: Tuple[AccountCase, ...]
    market: np.ndarray
    open_next: float
    close_next: float
    positive_schedule: Tuple[Tuple[int, int], ...]
    control_schedule: Tuple[Tuple[int, int], ...]


def desired_direction(desired_target_exposure: float) -> Direction:
    """Known-answer direction for a desired target exposure."""

    if desired_target_exposure > 0.0:
        return Direction.LONG
    if desired_target_exposure < 0.0:
        return Direction.SHORT
    return Direction.FLAT


def _authority_values(spec: Mapping[str, Any]) -> Tuple[int, float, float]:
    authority = spec_mapping(spec["authority"], "authority")
    context_length = spec_int(authority["context_length"], "authority.context_length", minimum=2)
    budget = spec_float(authority["nominal_exposure_budget"], "authority.nominal_exposure_budget")
    limit = spec_float(authority["hard_exposure_limit"], "authority.hard_exposure_limit")
    return context_length, budget, limit


def _task_a_case(entry: Any, position: int, config: PhysicsConfig) -> AccountCase:
    mapping = spec_mapping(entry, f"task_a.initial_truths[{position}]")
    name = spec_str(mapping["name"], f"task_a.initial_truths[{position}].name")
    truth = AccountTruth(
        equity=spec_float(mapping["equity"], f"{name}.equity"),
        quantity=spec_float(mapping["quantity"], f"{name}.quantity"),
        mark_price=spec_float(mapping["mark_price"], f"{name}.mark_price"),
    )
    desired = spec_float(
        mapping["desired_target_exposure"], f"{name}.desired_target_exposure"
    )
    exposure = account_state_from_truth(truth, config).signed_exposure
    if exposure != desired:
        raise ContractError(
            f"task_a.initial_truths[{position}] ({name}) declares desired target {desired!r}, "
            f"but the canonical AccountState exposure is {exposure!r}"
        )
    return AccountCase(name=name, truth=truth, desired_target_exposure=desired)


def task_a_environment(spec: Mapping[str, Any]) -> TaskAEnvironment:
    """Build the preregistered Task-A environment from the committed JSON."""

    task = spec_mapping(spec["task_a"], "task_a")
    context_length, budget, limit = _authority_values(spec)
    physics = spec_mapping(task["physics"], "task_a.physics")
    open_next = spec_float(physics["open_next"], "task_a.physics.open_next")
    close_next = spec_float(physics["close_next"], "task_a.physics.close_next")
    if open_next != TASK_A_PRICE or close_next != TASK_A_PRICE:
        raise ContractError(
            "task_a.physics open_next/close_next must equal the flat market price "
            f"{TASK_A_PRICE!r}, got {open_next!r}/{close_next!r}"
        )
    config = PhysicsConfig(
        context_length=context_length,
        nominal_exposure_budget=budget,
        hard_exposure_limit=limit,
        proportional_friction_kappa=spec_float(
            physics["proportional_friction_kappa"], "task_a.physics.proportional_friction_kappa"
        ),
        permission_bisection_iterations=spec_int(
            physics["permission_bisection_iterations"],
            "task_a.physics.permission_bisection_iterations",
            minimum=1,
        ),
    )

    entries = spec_sequence(task["initial_truths"], "task_a.initial_truths")
    cases = tuple(_task_a_case(entry, position, config) for position, entry in enumerate(entries))
    if len(cases) != 3:
        raise ContractError(
            f"Task A is the preregistered three-account task and requires exactly 3 "
            f"initial_truths, got {len(cases)}"
        )

    if spec_int(task["trajectory_length"], "task_a.trajectory_length") != 1:
        raise ContractError("Task A trajectories are exactly one decision/transition long")
    batch = spec_int(task["batch_trajectories"], "task_a.batch_trajectories", minimum=1)
    if batch % len(cases):
        raise ContractError(
            f"task_a.batch_trajectories {batch} is not divisible across {len(cases)} account cases"
        )
    per_case = batch // len(cases)
    positive_schedule = tuple(
        (case_index, case_index) for case_index in range(len(cases)) for _ in range(per_case)
    )
    pairs = len(cases) * len(cases)
    if batch % pairs:
        raise ContractError(
            f"task_a.batch_trajectories {batch} is not divisible across {pairs} account pairs"
        )
    per_pair = batch // pairs
    control_schedule = tuple(
        (actual, observed)
        for actual in range(len(cases))
        for observed in range(len(cases))
        for _ in range(per_pair)
    )

    market = normalize_market_window(
        flat_raw_window(context_length, price=TASK_A_PRICE, volume=TASK_A_VOLUME), config
    )
    return TaskAEnvironment(
        config=config,
        cases=cases,
        market=market,
        open_next=open_next,
        close_next=close_next,
        positive_schedule=positive_schedule,
        control_schedule=control_schedule,
    )


def task_a_account_state(env: TaskAEnvironment, case: AccountCase) -> AccountState:
    """Canonical decision state for one Task-A account truth."""

    return account_state_from_truth(case.truth, env.config)


def build_task_a_generation(
    learner: OnPolicyLearner,
    env: TaskAEnvironment,
    generation_id: int,
    *,
    mode: str,
) -> Tuple[Trajectory, ...]:
    """Collect one complete Task-A generation of one-decision trajectories.

    ``mode=ARM_POSITIVE`` observes the account state derived from each actual
    truth; ``mode=ARM_CONTROL`` observes the paired independent account state.
    Reward always comes from the actual ``AccountTruth`` through the merged
    VS-A ``execute_transition``.
    """

    if mode == ARM_POSITIVE:
        schedule = env.positive_schedule
    elif mode == ARM_CONTROL:
        schedule = env.control_schedule
    else:
        raise ContractError(f"mode must be {ARM_POSITIVE!r} or {ARM_CONTROL!r}, got {mode!r}")

    z_market = learner.encode(torch.from_numpy(env.market))
    trajectories = []
    for actual_index, observed_index in schedule:
        actual = env.cases[actual_index]
        observed = env.cases[observed_index]
        state = build_learner_state(z_market, task_a_account_state(env, observed))
        sample = learner.sample_action(state)
        step = execute_transition(
            actual.truth, sample.nominal_action, env.open_next, env.close_next, env.config
        )
        trajectories.append(
            Trajectory(
                steps=(
                    TrajectoryStep(
                        state=state.vector,
                        direction=sample.direction,
                        requested_risk=sample.requested_risk,
                        reward=step.reward,
                        terminal=False,
                        truncated=False,
                        generation_id=generation_id,
                    ),
                ),
                generation_id=generation_id,
                complete=True,
            )
        )
    return tuple(trajectories)


def evaluate_task_a(learner: OnPolicyLearner, env: TaskAEnvironment) -> Dict[str, Any]:
    """Deterministic-adapter evaluation on the true Task-A account states."""

    z_market = learner.encode(torch.from_numpy(env.market))
    per_case = []
    for case in env.cases:
        state = build_learner_state(z_market, task_a_account_state(env, case))
        action = learner.actor.deterministic_action(state)
        target = nominal_target_from_action(action, env.config)
        step = execute_transition(
            case.truth, action, env.open_next, env.close_next, env.config
        )
        per_case.append(
            {
                "case": case.name,
                "direction": action.direction.name,
                "requested_risk": action.requested_risk,
                "target_exposure": target,
                "desired_target_exposure": case.desired_target_exposure,
                "target_absolute_error": abs(target - case.desired_target_exposure),
                "one_step_log_growth": step.reward,
                "direction_correct": action.direction is desired_direction(
                    case.desired_target_exposure
                ),
            }
        )
    return {
        "cases": per_case,
        "direction_correct_count": sum(1 for case in per_case if case["direction_correct"]),
        "target_exposure_mae": float(
            np.mean([case["target_absolute_error"] for case in per_case])
        ),
        "mean_one_step_log_growth": float(
            np.mean([case["one_step_log_growth"] for case in per_case])
        ),
    }


# ---------------------------------------------------------------------------
# Task B
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskBEnvironment:
    """Preregistered Task-B environment (cue windows plus a delayed gap)."""

    config: PhysicsConfig
    initial_truth: AccountTruth
    step0_open_next: float
    step0_close_next: float
    up_gap: float
    down_gap: float
    cue_markets: Mapping[str, np.ndarray]
    neutral_market: np.ndarray
    cue_schedule: Tuple[str, ...]


@dataclass(frozen=True)
class TaskBRecord:
    """One collected two-step Task-B trajectory before reward assignment.

    The record owns the states, actions and the two Physics rewards of one
    complete two-step trajectory.  Trajectory records are built separately, so
    the shuffled control never mutates a collected record.
    """

    cue: str
    state_0: torch.Tensor
    direction_0: Direction
    requested_risk_0: float
    immediate_reward: float
    state_1: torch.Tensor
    direction_1: Direction
    requested_risk_1: float
    delayed_reward: float


def task_b_environment(spec: Mapping[str, Any]) -> TaskBEnvironment:
    """Build the preregistered Task-B environment from the committed JSON."""

    task = spec_mapping(spec["task_b"], "task_b")
    context_length, budget, limit = _authority_values(spec)

    cue = spec_mapping(task["cue_market"], "task_b.cue_market")
    endpoint_price = spec_float(cue["endpoint_price"], "task_b.cue_market.endpoint_price")
    ramp = spec_float(
        cue["represented_log_ramp_magnitude"], "task_b.cue_market.represented_log_ramp_magnitude"
    )
    volume = spec_float(cue["volume"], "task_b.cue_market.volume")
    _require_formula(spec_str(cue["up"], "task_b.cue_market.up"), "exp(-0.04", "cue up")
    _require_formula(spec_str(cue["down"], "task_b.cue_market.down"), "exp(+0.04", "cue down")
    _require_formula(spec_str(cue["open"], "task_b.cue_market.open"), "previous represented close", "cue open")
    _require_formula(spec_str(cue["high"], "task_b.cue_market.high"), "1.001", "cue high")
    _require_formula(spec_str(cue["low"], "task_b.cue_market.low"), "1.001", "cue low")

    step_0 = spec_mapping(task["step_0"], "task_b.step_0")
    initial = spec_mapping(step_0["initial_truth"], "task_b.step_0.initial_truth")
    initial_truth = AccountTruth(
        equity=spec_float(initial["equity"], "task_b.step_0.initial_truth.equity"),
        quantity=spec_float(initial["quantity"], "task_b.step_0.initial_truth.quantity"),
        mark_price=spec_float(initial["mark_price"], "task_b.step_0.initial_truth.mark_price"),
    )
    step0_open_next = spec_float(step_0["open_next"], "task_b.step_0.open_next")
    step0_close_next = spec_float(step_0["close_next"], "task_b.step_0.close_next")
    kappa = spec_float(
        step_0["proportional_friction_kappa"], "task_b.step_0.proportional_friction_kappa"
    )
    if kappa != 0.0:
        raise ContractError(
            "Task B is preregistered with zero transaction friction; the delayed-credit "
            f"construction does not hold for kappa={kappa!r}"
        )
    required_immediate = spec_float(
        step_0["required_immediate_reward"], "task_b.step_0.required_immediate_reward"
    )
    if required_immediate != 0.0:
        raise ContractError(
            f"Task B requires an exactly zero immediate reward, got {required_immediate!r}"
        )

    step_1 = spec_mapping(task["step_1"], "task_b.step_1")
    up_gap = spec_float(step_1["up_gap_open_and_close"], "task_b.step_1.up_gap_open_and_close")
    down_gap = spec_float(
        step_1["down_gap_open_and_close"], "task_b.step_1.down_gap_open_and_close"
    )

    config = PhysicsConfig(
        context_length=context_length,
        nominal_exposure_budget=budget,
        hard_exposure_limit=limit,
        proportional_friction_kappa=kappa,
        permission_bisection_iterations=_task_b_bisection_iterations(),
    )

    up_raw = cue_raw_window(
        UP,
        context_length,
        endpoint_price=endpoint_price,
        ramp_magnitude=ramp,
        volume=volume,
    )
    down_raw = cue_raw_window(
        DOWN,
        context_length,
        endpoint_price=endpoint_price,
        ramp_magnitude=ramp,
        volume=volume,
    )
    neutral_raw = flat_raw_window(context_length, price=endpoint_price, volume=volume)
    cue_markets = {
        UP: normalize_market_window(up_raw, config),
        DOWN: normalize_market_window(down_raw, config),
    }
    neutral_market = normalize_market_window(neutral_raw, config)

    if spec_int(task["trajectory_length"], "task_b.trajectory_length") != 2:
        raise ContractError("Task B trajectories are exactly two decisions long")
    batch = spec_int(task["batch_trajectories"], "task_b.batch_trajectories", minimum=2)
    if batch % 2:
        raise ContractError(f"task_b.batch_trajectories {batch} is not divisible across two cues")
    per_cue = batch // 2
    cue_schedule = (UP,) * per_cue + (DOWN,) * per_cue

    return TaskBEnvironment(
        config=config,
        initial_truth=initial_truth,
        step0_open_next=step0_open_next,
        step0_close_next=step0_close_next,
        up_gap=up_gap,
        down_gap=down_gap,
        cue_markets=cue_markets,
        neutral_market=neutral_market,
        cue_schedule=cue_schedule,
    )


def _task_b_bisection_iterations() -> int:
    """Task-B permission bisection count.

    The preregistered Task-B physics fixes ``kappa == 0``, so the boundary
    ``+/-hard_exposure_limit`` is always feasible and the bisection loop is
    never entered; the frozen VS-A default is therefore used and cannot affect
    any Task-B number.
    """

    return PERMISSION_BISECTION_ITERATIONS


def task_b_gap(env: TaskBEnvironment, cue: str) -> float:
    """Step-1 execution/marking price for one cue."""

    if cue == UP:
        return env.up_gap
    if cue == DOWN:
        return env.down_gap
    raise ContractError(f"cue must be {UP!r} or {DOWN!r}, got {cue!r}")


def collect_task_b_records(
    learner: OnPolicyLearner, env: TaskBEnvironment
) -> Tuple[TaskBRecord, ...]:
    """Collect one complete Task-B generation of two-step trajectories.

    Every trajectory's own Physics rewards are retained, including the exactly
    zero immediate reward, which is asserted here so a broken construction can
    never be trained on silently.
    """

    z_neutral = learner.encode(torch.from_numpy(env.neutral_market))
    z_cue = {
        cue: learner.encode(torch.from_numpy(env.cue_markets[cue])) for cue in (UP, DOWN)
    }
    initial_state = account_state_from_truth(env.initial_truth, env.config)

    records = []
    for cue in env.cue_schedule:
        state_0 = build_learner_state(z_cue[cue], initial_state)
        sample_0 = learner.sample_action(state_0)
        step_0 = execute_transition(
            env.initial_truth,
            sample_0.nominal_action,
            env.step0_open_next,
            env.step0_close_next,
            env.config,
        )
        if step_0.reward != 0.0:
            raise ContractError(
                "Task B immediate reward must be exactly zero under the frozen Physics, got "
                f"{step_0.reward!r}"
            )
        state_1 = build_learner_state(z_neutral, step_0.state_next)
        sample_1 = learner.sample_action(state_1)
        gap = task_b_gap(env, cue)
        step_1 = execute_transition(
            step_0.truth_next, sample_1.nominal_action, gap, gap, env.config
        )
        records.append(
            TaskBRecord(
                cue=cue,
                state_0=state_0.vector,
                direction_0=sample_0.direction,
                requested_risk_0=sample_0.requested_risk,
                immediate_reward=step_0.reward,
                state_1=state_1.vector,
                direction_1=sample_1.direction,
                requested_risk_1=sample_1.requested_risk,
                delayed_reward=step_1.reward,
            )
        )
    return tuple(records)


def shuffled_delayed_rewards(
    records: Sequence[TaskBRecord], *, seed: int, generation: int
) -> Tuple[float, ...]:
    """The collected delayed-reward multiset reassigned by a fixed-point-free permutation."""

    order = no_fixed_point_permutation(len(records), seed=seed, generation=generation)
    return tuple(records[source].delayed_reward for source in order)


def task_b_trajectories(
    records: Sequence[TaskBRecord],
    generation_id: int,
    *,
    delayed_rewards: Sequence[float] | None = None,
) -> Tuple[Trajectory, ...]:
    """Build fresh immutable trajectory records for one Task-B generation.

    ``delayed_rewards`` is the reward assigned to each trajectory in order; when
    omitted every trajectory keeps its own collected delayed reward.  New step
    records are always constructed, so a control assignment can never mutate a
    collected record.
    """

    if delayed_rewards is None:
        assigned = tuple(record.delayed_reward for record in records)
    else:
        assigned = tuple(float(value) for value in delayed_rewards)
    if len(assigned) != len(records):
        raise ContractError(
            f"expected {len(records)} assigned delayed rewards, got {len(assigned)}"
        )
    trajectories = []
    for record, delayed in zip(records, assigned):
        trajectories.append(
            Trajectory(
                steps=(
                    TrajectoryStep(
                        state=record.state_0,
                        direction=record.direction_0,
                        requested_risk=record.requested_risk_0,
                        reward=record.immediate_reward,
                        terminal=False,
                        truncated=False,
                        generation_id=generation_id,
                    ),
                    TrajectoryStep(
                        state=record.state_1,
                        direction=record.direction_1,
                        requested_risk=record.requested_risk_1,
                        reward=delayed,
                        terminal=False,
                        truncated=False,
                        generation_id=generation_id,
                    ),
                ),
                generation_id=generation_id,
                complete=True,
            )
        )
    return tuple(trajectories)


def build_task_b_generation(
    learner: OnPolicyLearner,
    env: TaskBEnvironment,
    generation_id: int,
    *,
    mode: str,
    seed: int,
) -> Tuple[Trajectory, ...]:
    """Collect one complete Task-B generation for the positive task or its control."""

    records = collect_task_b_records(learner, env)
    if mode == ARM_POSITIVE:
        return task_b_trajectories(records, generation_id)
    if mode == ARM_CONTROL:
        return task_b_trajectories(
            records,
            generation_id,
            delayed_rewards=shuffled_delayed_rewards(records, seed=seed, generation=generation_id),
        )
    raise ContractError(f"mode must be {ARM_POSITIVE!r} or {ARM_CONTROL!r}, got {mode!r}")


# ---------------------------------------------------------------------------
# Task B evaluation
# ---------------------------------------------------------------------------


def direction_probabilities(actor: Any, state: Any) -> Dict[str, float]:
    """Categorical direction probabilities for one state, in canonical order."""

    with torch.no_grad():
        output = actor.forward(state)
        probabilities = torch.softmax(output.direction_logits, dim=-1)
    return {
        direction.name: float(probabilities[direction_index(direction)])
        for direction in DIRECTION_ORDER
    }


def evaluate_task_b(learner: OnPolicyLearner, env: TaskBEnvironment) -> Dict[str, Any]:
    """Deterministic-adapter evaluation on the true, unshuffled Task-B environment."""

    z_neutral = learner.encode(torch.from_numpy(env.neutral_market))
    per_cue: Dict[str, Dict[str, Any]] = {}
    for cue in (UP, DOWN):
        z_cue = learner.encode(torch.from_numpy(env.cue_markets[cue]))
        state_0 = build_learner_state(
            z_cue, account_state_from_truth(env.initial_truth, env.config)
        )
        probabilities = direction_probabilities(learner.actor, state_0)
        action_0 = learner.actor.deterministic_action(state_0)
        step_0 = execute_transition(
            env.initial_truth,
            action_0,
            env.step0_open_next,
            env.step0_close_next,
            env.config,
        )
        state_1 = build_learner_state(z_neutral, step_0.state_next)
        action_1 = learner.actor.deterministic_action(state_1)
        gap = task_b_gap(env, cue)
        step_1 = execute_transition(step_0.truth_next, action_1, gap, gap, env.config)
        per_cue[cue] = {
            "direction": action_0.direction.name,
            "requested_risk": action_0.requested_risk,
            "target_exposure": nominal_target_from_action(action_0, env.config),
            "immediate_reward": step_0.reward,
            "delayed_log_growth": step_1.reward,
            "categorical_probabilities": probabilities,
        }

    up = per_cue[UP]
    down = per_cue[DOWN]
    margin = 0.5 * (
        (up["categorical_probabilities"]["LONG"] - up["categorical_probabilities"]["SHORT"])
        + (down["categorical_probabilities"]["SHORT"] - down["categorical_probabilities"]["LONG"])
    )
    return {
        "up": up,
        "down": down,
        "mean_true_delayed_log_growth": float(
            np.mean([up["delayed_log_growth"], down["delayed_log_growth"]])
        ),
        "direction_probability_margin": float(margin),
    }


__all__ = [
    "ARM_CONTROL",
    "ARM_POSITIVE",
    "COLLECTION_DOMAIN",
    "CUE_HIGH_FACTOR",
    "CUE_LOW_FACTOR",
    "DOWN",
    "PERMUTATION_DOMAIN",
    "TASK_A_ID",
    "TASK_A_PRICE",
    "TASK_A_VOLUME",
    "TASK_B_ID",
    "UP",
    "AccountCase",
    "TaskAEnvironment",
    "TaskBEnvironment",
    "TaskBRecord",
    "build_task_a_generation",
    "build_task_b_generation",
    "collect_task_b_records",
    "control_permutation_seed",
    "cue_raw_window",
    "derived_stream_seed",
    "desired_direction",
    "direction_probabilities",
    "evaluate_task_a",
    "evaluate_task_b",
    "flat_raw_window",
    "no_fixed_point_permutation",
    "shuffled_delayed_rewards",
    "spec_bool",
    "spec_float",
    "spec_int",
    "spec_mapping",
    "spec_sequence",
    "spec_str",
    "task_a_account_state",
    "task_a_environment",
    "task_b_environment",
    "task_b_gap",
    "task_b_trajectories",
]
