# R12 Action, Permission, Execution, and Account Physics

Status: **MASTER-ALIGNED V1 CANDIDATE — EXACT ENVIRONMENT CONSTANTS STILL EXPERIMENT-SCOPED**

Purpose: define the smallest deterministic contract that turns Central Brain intent into account consequences without embedding venue identity, hindsight, or strategy logic in Physics.

## 1. Runtime sequence

R12 uses a target-exposure control loop.

```text
completed bar t + current account truth
        -> normalized market representation Z_t
        -> AccountState A_t
        -> Central Brain nominal intent a_t
        -> wait for execution point
        -> mark existing account to execution price
        -> Permission projects intent into feasible target exposure
        -> Execution changes exposure and charges frozen costs
        -> market moves
        -> mark account to next decision point
        -> next AccountState + log-equity reward
```

The Central Brain chooses desired economic exposure. It does not emit exchange operations such as `BUY`, `SELL`, `PYRAMID`, `MARTINGALE`, or `CLOSE` as separate strategy primitives.

## 2. Decision timing

At decision time `t`, bar `t` is complete and all market information used by the policy is observable.

The policy does not execute using the same bar close that completed its input. The default v1 timing is:

```text
observe through close_t
-> produce a_t
-> execute at open_{t+1}
-> mark next decision state at close_{t+1}
```

Therefore the policy cannot use `open_{t+1}`, `high_{t+1}`, `low_{t+1}`, or `close_{t+1}` to choose `a_t`.

Physics may use `open_{t+1}` when the execution event actually occurs. This is execution truth, not policy information.

## 3. Nominal action schema

The canonical nominal action is:

\[
a_t=(d_t,r_t)
\]

with:

\[
d_t\in\{-1,0,+1\}
\]

representing `SHORT`, `FLAT`, `LONG`, and:

\[
r_t\in[0,1]
\]

representing `requested_risk`.

`requested_risk` is not confidence, probability, expected return, or permission. In v1 it is a dimensionless request for a fraction of the experiment's frozen nominal exposure budget.

Canonical representation is unique:

- `FLAT` requires `requested_risk = 0`;
- `requested_risk = 0` canonicalizes to `FLAT`;
- non-finite values are invalid, not clipped silently.

Let \(\bar e>0\) be the frozen nominal exposure budget for the experiment. The nominal target signed exposure is:

\[
\hat e_t=d_t\,r_t\,\bar e.
\]

`requested_risk = 1` therefore requests the full nominal exposure budget; it does not guarantee that Permission will allow it.

The exact value of \(\bar e\) is an experiment parameter, not a learned input and not an asset identity cue.

## 4. Target exposure semantics

Exposure is dimensionless:

\[
e=\frac{\text{signed marked-to-market risky notional}}{\text{economic equity}}.
\]

The action specifies a **target exposure**, not an incremental trade command.

Examples when \(\bar e=1\):

```text
LONG 0.60  -> target exposure +0.60
FLAT       -> target exposure  0.00
SHORT 0.40 -> target exposure -0.40
```

A reversal is one target transition. Moving from `+0.60` to `-0.40` requires the full economic turnover implied by that change; no special reversal primitive is needed.

Repeating the same target means maintaining that target exposure. If market movement causes actual exposure to drift, execution may rebalance it back toward the requested target and incur the corresponding cost.

## 5. Execution-time pre-trade truth

The existing position remains economically active between `close_t` and `open_{t+1}`.

At `open_{t+1}`, Physics first marks the existing account to the execution price and obtains pre-trade truth:

\[
T^-_{t+1}.
\]

This truth includes the actual marked account quantities required by the environment, including pre-trade equity, current signed position/notional, and any margin/collateral values used by Permission.

The policy does not observe `T^-_{t+1}` before choosing `a_t`.

## 6. Permission

Permission is a deterministic safety/feasibility transform. It does not optimize return and does not invent strategy.

Given pre-trade truth \(T^-_{t+1}\), Physics derives the feasible target-exposure set:

\[
\mathcal E^-_{t+1}.
\]

For the initial single-asset system this should be representable as a one-dimensional interval:

\[
\mathcal E^-_{t+1}=[e^{min}_{t+1},e^{max}_{t+1}].
\]

When the account is maintenance-safe:

- zero exposure must be feasible;
- maintaining the current exposure must remain feasible unless a hard rule requires de-risking;
- risk-reducing movement toward zero must not be blocked merely because risk-increasing capacity is exhausted;
- initial-margin/cash/new-risk rules constrain risk-increasing targets;
- direction-specific asymmetry may appear through unequal lower/upper bounds.

The permitted target is the projection of nominal intent into the feasible set:

\[
e^{perm}_t=\operatorname{Proj}_{\mathcal E^-_{t+1}}(\hat e_t).
\]

For an interval this is ordinary clipping.

Because zero belongs to the ordinary feasible interval, Permission may reduce magnitude or reduce an infeasible intent to FLAT, but it must not create a non-zero position from FLAT or reverse the requested direction as a strategy choice.

If the account has already crossed a maintenance/solvency boundary, ordinary policy intent no longer owns the transition. Physics invokes the experiment's frozen forced-risk handling rule and records that event explicitly.

## 7. Execution v1

R12 v1 uses a continuous-notional, price-taker execution model unless a later experiment requires exchange quantization.

At execution price \(P^{exec}_{t+1}=O_{t+1}\), let pre-trade equity be \(W^-_{t+1}>0\). The desired signed notional is:

\[
N^*_{t+1}=e^{perm}_t W^-_{t+1}.
\]

Let current signed marked notional immediately before trading be \(N^-_{t+1}\). Required turnover is:

\[
\Delta N_{t+1}=N^*_{t+1}-N^-_{t+1}.
\]

With frozen proportional friction \(\kappa\):

\[
C^{trade}_{t+1}=\kappa |\Delta N_{t+1}|.
\]

Post-trade equity before subsequent market movement is:

\[
W^+_{t+1}=W^-_{t+1}-C^{trade}_{t+1}.
\]

The v1 quantity after execution is the continuous quantity that corresponds to the permitted target notional at the execution price:

\[
q^+_{t+1}=\frac{N^*_{t+1}}{P^{exec}_{t+1}}.
\]

Because trading cost reduces equity after target notional is computed, actual post-trade exposure may differ slightly from `e_perm`. Actual exposure is always recomputed from account truth; the simulator must not overwrite truth with the requested value.

No partial-fill, order-book, market-impact, lot-size, or venue-specific minimum-order model is required in the initial scientific slice.

## 8. Account transition

For the simple linear single-asset price-taker account, existing quantity earns/losses PnL until execution and the post-trade quantity earns/losses PnL after execution.

Ignoring variable carry for the minimal v1 contract:

\[
W^-_{t+1}=W_t+q_t(O_{t+1}-C_t),
\]

then after execution cost:

\[
W^+_{t+1}=W^-_{t+1}-C^{trade}_{t+1},
\]

and at the next decision mark:

\[
W_{t+1}=W^+_{t+1}+q^+_{t+1}(C_{t+1}-O_{t+1}).
\]

If the experiment models carry/funding/borrow cost, it is deducted as an explicit additional account term under a frozen causal convention.

The next actual signed exposure is recomputed from marked truth:

\[
e_{t+1}=\frac{q^+_{t+1}C_{t+1}}{W_{t+1}}
\]

when \(W_{t+1}>0\).

AccountState for the next policy decision is derived from this truth; it is never advanced by directly copying the previous requested action.

## 9. Reward

The provisional dense economic reward remains:

\[
r_t=\log\frac{W_{t+1}}{W_t}
\]

for valid positive equity states.

This transition includes all economically real consequences assigned to the interval under the frozen timing convention: carry-over PnL from the old position before execution, transaction cost, and PnL from the new position after execution.

Do not subtract a separate activity penalty. Trading cost already makes unnecessary turnover economically costly.

## 10. Solvency and forced-risk events

`survival_cushion` and `new_risk_capacity` remain derived decision-state summaries; raw solvency/margin truth remains in Physics.

A maintenance breach is not automatically identical to final economic ruin. A frozen environment may force reduction or flattening and then continue if legitimate positive account equity remains.

True ruin is an account-ending state in which legitimate trading cannot continue. The exact R12 production ruin boundary remains a separate scientific/physics decision.

For the first controlled learnability slice, complex exchange liquidation mechanics are not required. A bounded-exposure sandbox may use a neutral `survival_cushion` and reserve detailed margin liquidation for a later physics qualification task.

No loss, forced reduction, process restart, or training-chunk boundary grants free principal reset.

## 11. Required records

Each transition must preserve enough factual information to reconstruct semantics without storing a large provenance graph:

```text
state_id / time
policy generation
nominal action: direction + requested_risk
nominal target exposure
pre-trade actual exposure/equity
permitted target exposure
executed quantity/notional change
trade cost
next marked equity/exposure
reward
forced-risk / terminal flags if applicable
```

Nominal intent, permitted target, and actual executed result are distinct fields.

## 12. Required invariants and tests

Implementation must test at least:

1. **No-lookahead timing** — changing any value after the action decision boundary cannot change `a_t`.
2. **FLAT uniqueness** — zero requested risk has one canonical FLAT representation.
3. **Scale invariance** — multiplying all account monetary quantities and risky notional by the same positive factor leaves normalized action/account semantics unchanged.
4. **Permission non-invention** — FLAT cannot become a non-zero risk position through Permission.
5. **Risk-reduction availability** — exhaustion of new-risk capacity does not block movement toward zero while the account remains tradable.
6. **Turnover accounting** — reversal pays cost on the full notional change, not merely the final absolute exposure.
7. **Cost monotonicity** — under fixed \(\kappa\), larger absolute turnover cannot have smaller transaction cost.
8. **Truth after cost** — actual post-trade/next-state exposure is recomputed after costs and marking; it is not copied from the requested target.
9. **Account continuity** — chunk/checkpoint/process boundaries do not reset equity or position.
10. **Terminal separation** — computational truncation cannot masquerade as economic ruin.
11. **Finite math** — non-finite prices/equity/actions or non-positive execution price fail explicitly.
12. **Intent/execution separation** — stored nominal, permitted, and executed values can differ and remain individually inspectable.

## 13. Minimal first-slice defaults

Unless a scientific task explicitly changes them, the first controlled vertical slice should prefer:

- continuous fractional notional;
- one asset;
- price taker;
- next-open execution after completed-bar decision;
- fixed proportional transaction friction;
- no variable carry;
- no market impact;
- no exchange lot/min-order rules;
- no complex liquidation engine;
- bounded nominal exposure budget;
- local deterministic account Physics.

These are simplifications for controlled learnability, not claims about production execution realism.

## 14. Still open

The following are not silently fixed by this contract:

- exact nominal exposure budget \(\bar e\);
- exact proportional friction \(\kappa\);
- detailed margin/maintenance/liquidation model after the first slice;
- exact production ruin boundary;
- whether variable funding/borrow/carry enters later environments;
- whether venue quantization/slippage/market impact is later required;
- learning algorithm used to optimize the policy.
