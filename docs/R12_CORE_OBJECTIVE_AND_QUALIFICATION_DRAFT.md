# CB16-R12 Core Objective and Qualification — Initial Draft

Status: **MASTER-ALIGNED DRAFT — NOT YET FINAL AUTHORITY**

Purpose: define the R12 Trader's product objective, runtime information boundary, learning objective, and candidate qualification logic. Open scientific choices remain explicit. Do not invent placeholder thresholds.

---

## 1. Product objective

CB16 exists to build an **autonomous Trader**.

The Trader is a stateful sequential decision-maker. Its job is to trade the selected asset/account autonomously after the outer experiment/deployment scope has been chosen.

The human remains outside the runtime policy. The human may choose the asset, capital allocation, research direction, deployment permission, and experiment scope. The runtime Trader does **not** receive a human macro-direction signal, discretionary long/short view, or manual regime label.

---

## 2. Decision state

R12 v1 uses:

\[
a_t = \pi(Z_t,A_t)
\]

where:

- \(Z_t\): market representation derived causally from normalized K-line context;
- \(A_t\): current account state;
- \(a_t\): nominal Trader action.

Account state evolves through previous actions, market movement, execution, and costs:

\[
A_{t+1}=F(A_t,a_t,M_t,M_{t+1},\text{execution},\text{costs})
\]

Therefore the same market representation can legitimately require different actions for different account states.

### Reserved learned memory interface

The architecture must leave a clean path toward:

\[
a_t=\pi(Z_t,A_t,S_t)
\]

where \(S_t\) is optional learned strategy memory / recurrent latent state.

R12 v1 does not require explicit StrategyMemory. The reserved interface must not become a handcrafted cycle engine, rule-activation table, chronological-expiry system, or manual regime state machine.

---

## 3. Asset-agnostic market semantics

The Trader should react to **relative market structure**, not asset identity.

Runtime policy input must not include:

- symbol/name tokens such as BTC, ETH, BNB;
- raw absolute price as an identity cue;
- human macro labels or discretionary directional hints.

Price-bearing K-line inputs must be transformed causally into normalized / relative form before they become policy information. The normalization must preserve the geometry and temporal relationships needed to represent K-line context while removing trivial dependence on the asset's nominal price scale.

The exact normalization transform is a scientific/implementation choice to be frozen per experiment. It must not use future information.

Other scale-sensitive inputs should likewise avoid unnecessary asset or capital identity leakage when a dimensionless economic representation is available.

The design target is not that two assets become statistically indistinguishable. It is that the Trader is not handed an explicit or trivial identity shortcut and must respond to the structure present in the normalized context.

---

## 4. Frozen market representation, trainable Central Brain

Market representation modules act as frozen sensory organs during Central Brain learning.

Their role is to transform normalized K-line context into stable representations of relative market structure. Their role is **not** to choose the final trade.

Conceptually:

```text
normalized K-line context
        |
        v
frozen market representation organs
        |
        v
       Z_t  +  AccountState A_t
                  |
                  v
          trainable Central Brain
                  |
                  v
        direction + requested_risk
```

The Central Brain integrates market representation and account condition and owns the economic decision.

Specific sensory-model identities, dimensions, taps, and parameter counts are replaceable components. If a sensory component changes, its information value and interface must be qualified again; the project does not inherit a particular legacy organ by default.

---

## 5. Prediction is perception, not action authority

Predictive or representation models may estimate direction, return distributions, volatility, uncertainty, or latent market structure.

They are inputs to decision-making, not final decision authority.

A correct market forecast does not imply that the same trade is correct for every account state. The final action is account-conditioned.

---

## 6. Initial market-impact assumption

R12 initially treats the Trader as a price taker:

\[
P(M_{t+1}\mid M_t,a_t)\approx P(M_{t+1}\mid M_t)
\]

while account transition remains action-dependent:

\[
P(A_{t+1}\mid A_t,M_t,M_{t+1},a_t)
\]

Endogenous market impact is outside the initial system.

---

## 7. Economic objective

The Trader should remain economically viable while compounding capital over a long horizon.

The highest-level problem is:

\[
\max_\pi J_{growth}(\pi)
\quad\text{subject to acceptable survival/path risk.}
\]

Interpretation:

- survival alone is insufficient because permanent inactivity can survive trivially;
- growth alone is insufficient because unconstrained growth seeking can accept destructive path risk;
- growth and survival/path risk remain distinct concepts.

This defines the problem, not a specific RL algorithm.

---

## 8. Ruin and economic continuity

A genuine account-ending state is a physical/economic failure state, not an ordinary bad timestep.

\[
\tau_{ruin}=\inf\{t:A_t\in\mathcal F\}
\]

where \(\mathcal F\) contains states in which legitimate trading can no longer continue.

Computational segmentation must not silently erase economic continuity. A training chunk boundary is not automatically an account reset boundary. Previous decisions continue to affect future account state unless an experiment explicitly starts a new independent trajectory.

There is no free principal reset after loss or ruin.

---

## 9. Inactivity is not inherently wrong

R12 must not apply an arbitrary penalty merely because the Trader is FLAT.

If no usable opportunity exists, not taking risk may be the correct action.

> **No opportunity -> not taking risk can be correct.**

The system must not reward trading activity for its own sake.

---

## 10. Primary benchmark: Buy & Hold

For a single asset, the primary economic benchmark is that asset's **Buy & Hold** path over the same evaluation interval.

Let Trader equity be \(W_t\), and benchmark wealth be:

\[
B_t=W_0\frac{P_t}{P_0}
\]

under a frozen benchmark execution convention.

Absolute log growth:

\[
G_{abs}=\log\frac{W_T}{W_0}
\]

Benchmark-relative log growth:

\[
G_{rel}
=\log\frac{W_T}{W_0}-\log\frac{B_T}{B_0}
=\log\frac{W_T}{B_T}
\]

Buy & Hold is primarily a qualification anchor: did active decision-making create economic value relative to simply holding the same asset?

A bearish market can make FLAT economically superior to Buy & Hold. The benchmark must not force benchmark-like exposure.

---

## 11. Provisional v1 learning signal

The initial economic learning signal is:

\[
r_t^{growth}=\log\frac{W_{t+1}}{W_t}
\]

with \(W_t\) equal to true net account equity under the experiment's execution and cost model.

Then:

\[
\sum_{t=0}^{T-1}r_t^{growth}=\log\frac{W_T}{W_0}
\]

This provides dense feedback while remaining exactly consistent with compounded terminal wealth.

**v1 rule:** do not add survival/path-risk penalties to this scalar reward unless experiments show that qualification-only risk control is insufficient.

---

## 12. Qualification is multi-objective

Do not reduce qualification to a universal scalar score.

Begin with two conceptual result dimensions:

1. **Economic Value** — long-run economic value, including performance relative to Buy & Hold.
2. **Survival / Path Risk** — how dangerous the account path was while generating that value.

The exact authority metric for the second axis is intentionally **OPEN**.

MDD, CDaR, CVaR, drawdown duration, ruin probability, Sharpe, Sortino, Calmar, turnover, and similar statistics may be computed as diagnostics without automatically becoming qualification authority.

---

## 13. Evidence is not a tradeable objective

Evidence quality answers:

> **How confident are we that the measured Economic Value and Survival / Path Risk are real?**

It is not a third objective that can be sacrificed for more return.

Relevant evidence may include paired evaluation on identical frozen market windows, seed robustness, dependence-aware uncertainty estimation, negative/random/shuffle controls where appropriate, multiple-testing correction when many candidates are tried, out-of-sample consistency, and uncertainty intervals on core objective estimates.

---

## 14. Machine-executable Pareto comparison

Routine candidate comparison must be mechanical.

After orienting all authority objectives so that higher is better, candidate \(A\) Pareto-dominates candidate \(B\) when:

\[
A_i\ge B_i\quad\forall i
\]

and:

\[
A_j>B_j\quad\text{for at least one }j.
\]

If one policy has greater economic value but worse survival/path quality, the policies are non-dominated.

Tiny point-estimate differences are insufficient for promotion. The comparator must support confidence-aware or probability-of-dominance logic. A conservative form is:

\[
LCB_i(A)\ge UCB_i(B)\quad\forall i
\]

with meaningful strict improvement in at least one authority dimension.

Machine comparison returns only:

```text
DOMINATES
DOMINATED
UNRESOLVED / NON_DOMINATED
```

When a promising comparison is unresolved, the evaluator may spend a predeclared bounded evidence budget. If the relation remains unresolved when the budget is exhausted, both candidates remain eligible.

---

## 15. Pareto Archive and Active Champion

The **Pareto Archive** is the machine-maintained set of currently qualified, statistically non-dominated Trader policies.

The **Active Champion** is the single policy chosen for a concrete operational purpose such as continued experimentation, bounded deployment, or parent initialization.

A policy is not deleted merely because it is not the Active Champion.

Human/SOL judgment is required only when a concrete decision demands one policy and surviving candidates express a genuine preference trade-off not already frozen in authority.

---

## 16. Required semantic separations

The system must preserve these distinctions:

- **Truth != Belief != Decision != Permission**.
- Nominal Trader action != permitted/executed action.
- `requested_risk` != epistemic confidence.
- Market representation != account state.
- Prediction quality != economic decision quality.
- Training reward != qualification metrics != diagnostics.
- Computational episode boundary != economic account reset.
- Evidence strength != economic objective value.
- Human outer-loop authority != runtime policy input.
- Asset identity != relative market structure.

---

## 17. Historical knowledge principle

New observations may recalibrate current behavior without requiring learned historical information to be manually erased.

Historical recurrence should primarily be represented through learned model parameters and responses to current normalized inputs.

Do not introduce handcrafted chronological expiry, cycle activation, regime-switch tables, or equivalent manual recurrence machinery unless future evidence establishes a concrete need.

---

## 18. Scientific validity

Claims must remain separated by strength:

```text
wiring works
!=
controlled learning works
!=
historical improvement exists
!=
economic superiority exists
!=
production profitability is established
```

Final holdout data must remain protected until explicitly authorized for the appropriate scientific claim.

---

## 19. Open decisions

Do not silently decide the following in implementation:

1. Exact causal normalization transform for K-line price channels and other scale-sensitive market inputs.
2. Exact minimal account-state schema and normalization.
3. Exact definition of the Survival / Path-Risk authority axis.
4. Which path statistics are authority versus diagnostics.
5. Whether and when survival/path risk must enter the training objective.
6. Economic materiality / epsilon thresholds.
7. Exact confidence and resampling protocol for dominance claims.
8. Exact ruin boundary beyond obvious account-ending/legal failure states.
9. Rule for selecting one Active Champion from multiple non-dominated candidates when a single operational choice is required.
10. Whether evidence eventually justifies explicit learned StrategyMemory.
11. Whether future scale requires market-impact modeling.

These are explicit scientific choices, not implementation gaps.

---

## 20. Minimal implementation shape

```text
market/
  causal_normalization
  frozen_representation

state/
  account_state
  optional_memory_interface

env/
  account_physics
  execution
  ruin_terminal_semantics

policy/
  central_brain
  action_intent

qualification/
  economic_value
  path_risk
  uncertainty
  dominance
  archive
```

No larger infrastructure is justified by this document alone.
