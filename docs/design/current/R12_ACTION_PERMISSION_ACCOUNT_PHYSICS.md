# R12 Action, Permission, Execution, and Account Physics

Status: **MASTER-ALIGNED V1 CANDIDATE — CONSTANTS REMAIN EXPERIMENT-SCOPED**

Authority class: `CURRENT_DESIGN_CANDIDATE`; subordinate to `docs/authority/`.

Purpose: define the smallest deterministic contract that converts Central Brain intent into account consequences without venue identity, hindsight, or strategy logic in Physics.

## 1. Runtime and timing

```text
completed bar t + marked account truth
  -> Z_t + A_t
  -> nominal action a_t
  -> open_{t+1}: mark old position
  -> Permission
  -> Execution + cost
  -> close_{t+1}: mark account
  -> A_{t+1} + reward
```

Default v1 timing is:

```text
observe through close_t
-> decide
-> execute at open_{t+1}
-> next decision mark at close_{t+1}
```

The policy cannot use any value from bar `t+1` when choosing `a_t`. Physics may use `open_{t+1}` when execution actually occurs.

## 2. Nominal action

\[
a_t=(d_t,r_t)
\]

where:

\[
d_t\in\{-1,0,+1\}=\{SHORT,FLAT,LONG\},
\qquad r_t\in[0,1].
\]

`requested_risk` is not confidence, probability, or permission. In v1 it is the requested fraction of a frozen nominal exposure budget \(\bar e>0\):

\[
\hat e_t=d_t r_t\bar e.
\]

Canonical action representation is unique:

- `FLAT` iff `requested_risk = 0`;
- non-finite action values are invalid;
- \(\bar e\) is an experiment constant, not policy input.

## 3. Target-exposure semantics

Exposure is dimensionless:

\[
e=\frac{\text{signed marked-to-market risky notional}}{\text{economic equity}}.
\]

The CB emits a **target exposure**, not incremental `BUY/SELL/CLOSE/PYRAMID` commands.

For \(\bar e=1\):

```text
LONG 0.60  -> +0.60 target exposure
FLAT       ->  0.00 target exposure
SHORT 0.40 -> -0.40 target exposure
```

A reversal is one target transition. Full notional turnover is charged. Repeating the same target means maintaining that exposure; if exposure drifts after market movement, rebalancing may occur and incur cost.

## 4. Execution-time truth

The previous position remains economically active until execution.

At `open_{t+1}` Physics first marks the old position and forms pre-trade truth \(T^-_{t+1}\), including actual pre-trade equity \(W^-_{t+1}\), quantity/notional, and any cash/margin/collateral state required by the environment.

The policy did not observe \(T^-_{t+1}\) when it selected `a_t`.

## 5. Permission

Permission is deterministic feasibility control. It does not optimize return and cannot invent a strategy.

Given \(T^-_{t+1}\), the frozen execution-cost convention, and a candidate target exposure \(e\), Physics must test the **projected post-trade state**, including the cost required to reach that target.

Define the feasible target set:

\[
\mathcal E^-_{t+1}
=
\{e:\text{projected execution of }e\text{ remains legal/solvent after deterministic trade cost}\}.
\]

For the initial single-asset system this should be a one-dimensional interval:

\[
\mathcal E^-_{t+1}=[e^{min}_{t+1},e^{max}_{t+1}].
\]

When the account is maintenance-safe:

- zero must be feasible;
- maintaining current exposure remains feasible unless a hard rule requires de-risking;
- movement toward zero is not blocked merely because new-risk capacity is exhausted;
- risk-increasing targets obey cash/initial-margin/new-risk rules;
- deterministic fees/friction are included in the feasibility calculation;
- direction asymmetry may appear through unequal lower/upper bounds.

The permitted target is:

\[
e^{perm}_t=\operatorname{Proj}_{\mathcal E^-_{t+1}}(\hat e_t).
\]

For an interval this is clipping.

Because ordinary feasible sets contain zero, Permission may reduce an infeasible intent toward zero but must not create non-zero risk from FLAT or reverse the requested direction as a strategy choice.

If the account has already crossed a maintenance/solvency boundary, ordinary policy intent no longer owns the transition. Physics invokes the frozen forced-risk rule and records it explicitly.

## 6. Execution v1

Use continuous notional and price-taker execution unless a later experiment requires quantization.

At:

\[
P^{exec}_{t+1}=O_{t+1},
\]

let current signed notional be \(N^-_{t+1}\). For permitted target exposure \(e^{perm}_t\):

\[
N^*_{t+1}=e^{perm}_t W^-_{t+1},
\]

\[
\Delta N_{t+1}=N^*_{t+1}-N^-_{t+1}.
\]

With frozen proportional friction \(\kappa\):

\[
C^{trade}_{t+1}=\kappa|\Delta N_{t+1}|.
\]

Post-cost equity is:

\[
W^+_{t+1}=W^-_{t+1}-C^{trade}_{t+1}.
\]

Continuous post-trade quantity is:

\[
q^+_{t+1}=\frac{N^*_{t+1}}{P^{exec}_{t+1}}.
\]

Because cost changes equity, actual post-trade exposure can differ slightly from `e_perm`. Truth is recomputed from account quantities; requested/permitted exposure is never copied into account truth.

The first slice does not require partial fills, order books, market impact, lot size, venue minimums, or exchange-specific order types.

## 7. Account transition

For the minimal linear single-asset price-taker account, without variable carry:

\[
W^-_{t+1}=W_t+q_t(O_{t+1}-C_t),
\]

\[
W^+_{t+1}=W^-_{t+1}-C^{trade}_{t+1},
\]

\[
W_{t+1}=W^+_{t+1}+q^+_{t+1}(C_{t+1}-O_{t+1}).
\]

If funding/borrow/carry is later modeled, it is an explicit causal account term under a frozen convention.

For positive next equity:

\[
e_{t+1}=\frac{q^+_{t+1}C_{t+1}}{W_{t+1}}.
\]

`AccountState` is derived from this marked truth. It is never advanced by copying the prior action.

## 8. Reward

\[
r_t=\log\frac{W_{t+1}}{W_t}
\]

for valid positive-equity transitions.

The interval reward includes old-position gap PnL before execution, transaction cost, and new-position PnL after execution under the frozen timing convention.

No separate inactivity penalty is added. Unnecessary turnover is already economically costly.

## 9. Solvency and ruin

`survival_cushion` and `new_risk_capacity` are decision-state summaries; raw margin/solvency truth remains in Physics.

Maintenance breach and final ruin are not synonyms. A frozen environment may force reduction/flattening and continue if legitimate positive capital remains.

True ruin is an account-ending state where legitimate trading cannot continue. The production ruin boundary remains open.

The first controlled learnability slice does not require a full exchange liquidation engine. A bounded-exposure sandbox may use neutral survival semantics while preserving the interface for later margin qualification.

No restart, truncation, loss, or forced reduction grants free principal reset.

## 10. Required transition record

```text
state/time
policy generation
nominal direction + requested_risk
nominal target exposure
pre-trade equity/exposure
permitted target exposure
executed notional change / quantity
trade cost
next equity/exposure
reward
forced-risk flag
terminal/truncation flags
```

Nominal intent, permitted target, and actual executed truth remain separate.

## 11. Required invariants

1. **No lookahead** — future bar values cannot change `a_t`.
2. **FLAT uniqueness** — zero requested risk has one canonical FLAT form.
3. **Capital-scale invariance** — scaling all monetary quantities/notional by one positive constant leaves normalized semantics unchanged.
4. **Permission non-invention** — FLAT cannot become non-zero risk through Permission.
5. **Risk-reduction availability** — exhausted new-risk capacity cannot block movement toward zero while still tradable.
6. **Cost-aware feasibility** — every permitted target remains feasible after its deterministic transaction cost is deducted.
7. **Full reversal turnover** — reversal cost uses the full signed notional change.
8. **Cost monotonicity** — fixed \(\kappa\) implies greater absolute turnover cannot cost less.
9. **Truth after cost** — actual exposure is recomputed after cost/marking.
10. **Account continuity** — process/chunk/checkpoint boundaries do not reset capital or position.
11. **Terminal separation** — computational truncation is not economic ruin.
12. **Finite math** — non-finite prices/equity/actions or non-positive execution price fail explicitly.
13. **Inspectability** — nominal, permitted, and executed values are individually recoverable.

## 12. First-slice defaults

Prefer:

- one asset;
- continuous fractional notional;
- bounded nominal exposure budget;
- next-open execution after completed-bar decision;
- price taker;
- fixed proportional friction;
- no variable carry;
- no market impact;
- no lot/min-order rules;
- no complex liquidation engine;
- local deterministic Physics.

These are controlled-learnability simplifications, not production realism claims.

## 13. Still open

- exact nominal exposure budget \(\bar e\);
- exact proportional friction \(\kappa\);
- detailed margin/maintenance/liquidation model after the first slice;
- exact production ruin boundary;
- variable funding/borrow/carry;
- venue quantization/slippage/market impact if later required;
- learning algorithm.
