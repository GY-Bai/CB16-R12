# R12 VS-A — Canonical Contracts, N0 Market Input, AccountState, Permission, and Physics

Status: **READY FOR BOUNDED IMPLEMENTATION**

## Objective

Implement the first non-learning segment of the R12 Vertical Slice:

```text
raw/synthetic OHLCV window
  -> canonical N0 normalization
  -> AccountTruth -> AccountState
  -> NominalAction -> nominal target exposure
  -> next-open pre-trade mark
  -> cost-aware Permission
  -> deterministic execution
  -> next-close account mark
  -> next AccountTruth + AccountState + log-equity reward
```

This task proves deterministic wiring and economic/account semantics only. It must **not** implement Central Brain, Actor/Critic, REINFORCE, replay, PPO, Teacher, historical profitability, pretrained sensory models, checkpoint systems, databases, distributed execution, or throughput optimization.

## Existing authority

Read and follow:

- `docs/R12_CORE_OBJECTIVE_AND_QUALIFICATION_DRAFT.md`
- `docs/R12_ACTION_PERMISSION_ACCOUNT_PHYSICS.md`
- `docs/R12_FIRST_VERTICAL_SLICE_TASK.md`
- `docs/tasks/R12_NORMALIZATION_ABLATION_R0.md`

Normalization R0 has selected N0 endpoint-anchored log-ratio + relative-window volume as the **first controlled Vertical Slice baseline**. Reuse that implementation; do not duplicate or alter its formula.

## Allowed implementation area

Create a narrow package:

```text
science/cb16_science/vslice/
  __init__.py
  contracts.py
  market.py
  physics.py
```

and focused tests, preferably:

```text
tests/test_vslice_contracts.py
tests/test_vslice_market.py
tests/test_vslice_physics.py
```

Do not modify dispatcher/workflows/OCI configuration/Science sandbox. Do not alter the normalization formulas. No new Science command is required in VS-A.

## Dependency rule

VS-A uses only Python standard library + existing NumPy. PyTorch is not required.

## Frozen controlled-slice defaults

These values are experiment-scoped defaults for the first controlled slice, not production authority:

```text
context_length = 64
nominal_exposure_budget = 1.0
hard_exposure_limit = 1.0
proportional_friction_kappa = 0.001
permission_bisection_iterations = 80
finite_tolerance = 1e-12
```

The implementation should expose these through a small immutable `PhysicsConfig`; tests may instantiate alternate values for known-answer cases.

Require:

```text
nominal_exposure_budget > 0
hard_exposure_limit > 0
0 <= kappa
kappa * hard_exposure_limit < 1
permission_bisection_iterations >= 1
```

Reject non-finite configuration values.

## 1. Canonical contracts

### Direction

Use one enum with exact values:

```text
SHORT = -1
FLAT  =  0
LONG  = +1
```

### NominalAction

Fields:

```text
direction
requested_risk
```

Validation:

- `requested_risk` must be finite and in `[0,1]`;
- `FLAT` iff `requested_risk == 0`;
- LONG/SHORT require `requested_risk > 0`;
- no silent clipping or canonicalization of an invalid action.

Nominal target exposure:

```text
nominal_target = int(direction) * requested_risk * nominal_exposure_budget
```

`requested_risk` is not confidence and is not permission.

### AccountTruth

Minimal marked account truth:

```text
equity
quantity
mark_price
```

All fields must be finite. `equity > 0`, `mark_price > 0`. Quantity may be negative, zero, or positive.

Current signed marked notional:

```text
notional = quantity * mark_price
```

Current exposure:

```text
exposure = notional / equity
```

Do not add entry price, holding duration, realized PnL, cumulative fees, symbol, venue, trade count, drawdown, or previous nominal action.

### AccountState

Exact v1 fields:

```text
signed_exposure
survival_cushion
new_risk_capacity
```

For this controlled bounded sandbox:

```text
signed_exposure = quantity * mark_price / equity
survival_cushion = 1.0
new_risk_capacity = max(0.0, 1.0 - abs(signed_exposure) / hard_exposure_limit)
```

Interpretation: the sandbox has no maintenance-liquidation rule, so survival cushion is the neutral safe value. It uses a symmetric initial-margin envelope with `initial_margin_rate = 1 / hard_exposure_limit`; therefore `new_risk_capacity` is the remaining collateral fraction under that bounded envelope. This is a controlled-slice simplification, not a production margin model.

Do not clip `signed_exposure`. Market movement may temporarily make it exceed the hard new-risk envelope; Permission must still allow de-risking toward zero when feasible.

## 2. Market input

`market.py` must expose one small canonical function for the first slice, conceptually:

```python
normalize_market_window(raw_window) -> np.ndarray
```

It must call/reuse the already-reviewed N0 transform from `cb16_science.normalization.transforms`; do not copy the N0 math into `vslice`.

Contract:

- input contains exactly `L` represented bars plus the one retained predecessor used by the common normalization package;
- output is the represented `[L,5]` N0 tensor;
- the predecessor is not part of the returned tensor;
- no raw OHLCV is returned in parallel;
- normalization errors remain explicit.

Tests must prove bit/float equality with direct `n0_endpoint_log_ratios()` on the same window and preserve the existing price-scale / volume-unit invariants.

## 3. Next-open pre-trade mark

Decision is made after `close_t`. The previous position remains active until `open_{t+1}`.

Given close-time `AccountTruth T_t` and `open_next`:

```text
W_minus = W_t + q_t * (open_next - mark_price_t)
N_minus = q_t * open_next
```

Require finite `open_next > 0` and finite `W_minus > 0` for an ordinary transition. Otherwise fail explicitly; do not repair/reset capital.

Pre-trade exposure:

```text
a_minus = N_minus / W_minus
```

The policy did not observe `open_next` when it chose the nominal action.

## 4. Cost-aware target feasibility

For a candidate target exposure `x`, use the authority formula:

```text
N_star(x) = x * W_minus
delta_notional(x) = N_star(x) - N_minus
trade_cost(x) = kappa * abs(delta_notional(x))
W_plus(x) = W_minus - trade_cost(x)
```

The candidate is feasible iff all are true:

```text
W_plus(x) > 0
abs(N_star(x) / W_plus(x)) <= hard_exposure_limit + finite_tolerance
```

This deliberately checks **post-cost actual exposure**, not merely the nominal target.

Zero must be tested first. If zero is not feasible, fail explicitly as an account/solvency condition; do not invent non-zero risk.

### Feasible interval

For the controlled symmetric sandbox, compute a one-dimensional feasible interval:

```text
[e_min, e_max]
```

Use the exact predicate above. Do not derive a second approximate risk rule.

For each direction independently:

1. `0` is the known feasible lower magnitude.
2. Test the boundary `+hard_exposure_limit` or `-hard_exposure_limit`.
3. If that boundary is feasible, use it.
4. Otherwise run exactly `permission_bisection_iterations` iterations over magnitude `[0, hard_exposure_limit]` to find the largest feasible magnitude on that side.
5. Return the negative and positive signed endpoints.

The configuration requirement `kappa * hard_exposure_limit < 1` makes the ray feasibility monotone for this linear sandbox. Do not replace this with an optimizer or grid search.

### Permission

```text
permitted_target = clip(nominal_target, e_min, e_max)
```

Required semantics:

- FLAT target `0` remains exactly `0`;
- Permission cannot reverse target direction as a strategy choice;
- exhausted capacity cannot block a feasible movement toward zero;
- Permission is deterministic and has no market forecast/strategy logic;
- nominal target, permitted target, and executed truth remain separate fields.

## 5. Execution

At `open_next`:

```text
N_star = permitted_target * W_minus
delta_notional = N_star - N_minus
trade_cost = kappa * abs(delta_notional)
W_plus = W_minus - trade_cost
q_plus = N_star / open_next
```

Require finite `W_plus > 0`.

Immediate post-cost exposure is recomputed from truth:

```text
post_cost_exposure = (q_plus * open_next) / W_plus
```

Do not copy `permitted_target` into account state.

## 6. Next-close mark and reward

Given finite `close_next > 0`:

```text
W_next = W_plus + q_plus * (close_next - open_next)
```

For VS-A ordinary transitions require finite `W_next > 0`. A non-positive result must fail explicitly; do not reset principal. Full production ruin handling remains deferred.

Next truth:

```text
AccountTruth(
    equity=W_next,
    quantity=q_plus,
    mark_price=close_next,
)
```

Reward:

```text
reward = log(W_next / W_t)
```

No inactivity penalty, bonus, benchmark subtraction, reward clipping, or hindsight label.

## 7. Required return records

Use small immutable dataclasses. Names may differ slightly, but the information must remain separately inspectable.

### PermissionResult

At minimum:

```text
nominal_target
feasible_min
feasible_max
permitted_target
was_clipped
forced_risk = false
```

VS-A does not implement forced liquidation. The field is reserved and remains false for ordinary controlled-slice transitions.

### PhysicsStep

At minimum:

```text
truth_t
state_t
nominal_action
nominal_target
open_next
pretrade_equity
pretrade_notional
pretrade_exposure
permission
executed_delta_notional
trade_cost
post_cost_equity
post_cost_quantity
post_cost_exposure
close_next
truth_next
state_next
reward
forced_risk
terminal = false
truncated = false
```

No policy generation is needed in VS-A; rollout owns it later.

## 8. Public pure functions

Prefer a small surface equivalent to:

```python
account_state_from_truth(truth, config) -> AccountState
nominal_target_from_action(action, config) -> float
mark_to_next_open(truth, open_next) -> PreTradeTruth
candidate_target_feasible(pretrade, target, config) -> bool
feasible_target_interval(pretrade, config) -> tuple[float, float]
permit_target(pretrade, nominal_target, config) -> PermissionResult
execute_transition(truth_t, action, open_next, close_next, config) -> PhysicsStep
```

Do not make classes with hidden mutable account state. The physics functions should be deterministic and locally testable.

## 9. Required known-answer tests

Add independent tests for all of the following.

### Action semantics

1. LONG/SHORT/FLAT canonical valid cases.
2. Reject FLAT with non-zero risk.
3. Reject LONG/SHORT with zero risk.
4. Reject risk outside `[0,1]` and NaN/Inf.
5. Exact nominal target formula.

### AccountState

6. Flat account gives `e=0`, `s=1`, `c=1` when `hard_exposure_limit=1`.
7. Exposure `+0.5` and `-0.5` both give `c=0.5` under limit 1.
8. Exposure beyond the hard envelope is not clipped and gives `c=0`.
9. Scaling equity and quantity by the same positive factor leaves AccountState unchanged.

### Timing and continuity

10. Old quantity earns/loses the `close_t -> open_{t+1}` gap before execution.
11. Two consecutive `execute_transition` calls use prior `truth_next`; no capital/position reset occurs between calls.

### Permission

12. With `kappa=0`, feasible interval is exactly `[-hard_limit,+hard_limit]`.
13. With positive friction, a boundary target that would violate post-cost exposure is reduced slightly inward.
14. Every returned interval endpoint is feasible within tolerance.
15. A point infinitesimally beyond a bisection-constrained endpoint is infeasible where numerically testable.
16. FLAT remains exactly zero.
17. An out-of-envelope nominal target clips toward the feasible interval without reversing sign.
18. A pre-trade exposure above the hard limit after a gap can still move toward zero when flattening remains feasible.

### Cost/execution

19. Repeating the exact pre-trade target has zero turnover cost when no rebalance is needed.
20. Full reversal uses full signed notional change; e.g. approximately `+0.5 -> -0.5` charges approximately one equity unit of turnover before multiplying by `kappa`.
21. Greater absolute turnover cannot have lower cost at fixed `kappa`.
22. Actual post-cost exposure is recomputed and differs from the permitted target when cost is positive and target is non-zero.
23. Scaling all monetary quantities/notional by a positive constant scales dollar cost/equity but leaves normalized exposures and reward unchanged.

### Reward / failure

24. Flat market + no turnover yields exactly zero log-equity reward.
25. Reward equals `log(W_next/W_t)` on a hand-computable positive transition.
26. Invalid/non-positive open/close/equity and non-finite values fail explicitly.
27. No free principal reset occurs after any failure path.
28. `terminal` and `truncated` remain false for every ordinary VS-A transition.

### Market wiring

29. VS-A market function is identical to direct N0 output on the same source window.
30. Price x100 and volume x1000 leave the VS-A normalized state unchanged within the already-frozen N0 tolerance.

Run the full repository test suite after targeted tests.

## 10. No-change boundaries

Do not:

- alter N0/N1/N2/N3/N4 formulas;
- introduce raw price or symbol into policy-facing state;
- add position return, entry price, holding duration, drawdown, trade count, realized PnL, or cumulative fees to AccountState;
- add a hand-coded regime/cycle/history memory;
- add a market-impact/order-book model;
- add a complex liquidation engine;
- add PyTorch;
- implement the learner;
- add a database/cache/queue/runtime service;
- optimize throughput;
- add venue-specific semantics;
- touch final holdout data.

## 11. Done when

- the narrow VS-A package exists with the pure deterministic surface above;
- all 30 required known-answer/invariant tests are represented and green;
- full repository suite is green;
- no unrelated files or semantics changed;
- Builder emits a compact BUILD_REPORT listing formulas implemented, test commands/results, and any mismatch with this contract;
- one coherent PR is ready for Chat-SOL review.
