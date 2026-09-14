# CB16-R12 Core Objective and Qualification — Initial Draft

Status: **MASTER-ALIGNED DRAFT — NOT YET FINAL AUTHORITY**

Purpose: define what the R12 Trader is optimizing, what information it acts on, how economic success is evaluated, and how candidate policies are compared. Open decisions remain explicit; no placeholder thresholds are invented.

---

## 1. Product objective

CB16 exists to build an **autonomous Trader**.

The Trader is a stateful sequential decision-maker. It must optimize behavior conditioned on both market state and account state, rather than act as an independent per-bar market-direction predictor.

---

## 2. Decision state

R12 v1 uses:

\[
a_t = \pi(M_t,A_t)
\]

where:

- \(M_t\): causal market information available at decision time;
- \(A_t\): current account state;
- \(a_t\): nominal Trader action.

Account state evolves through previous actions, market movement, execution, and costs:

\[
A_{t+1}=F(A_t,a_t,M_t,M_{t+1},\text{execution},\text{costs})
\]

Therefore the same market state can legitimately require different actions for different account states.

### Reserved learned memory interface

The architecture must leave a clean path toward:

\[
a_t=\pi(M_t,A_t,S_t)
\]

where \(S_t\) is optional learned strategy memory / recurrent latent state.

R12 v1 does not require an explicit StrategyMemory implementation. The reserved interface must not become a handcrafted cycle engine, rule-activation table, or manually maintained regime state machine.

---

## 3. Prediction is perception, not action authority

Predictive or representation models may estimate direction, return distributions, volatility, uncertainty, or latent market structure.

They are inputs to decision-making, not final decision authority.

```text
Market
  -> predictive / representation organs
  -> representation Z_t
                       + AccountState
                             -> Trader
                             -> Action
```

The final action is account-conditioned. A correct market forecast does not imply that the same trade is correct for every account state.

---

## 4. Initial market-impact assumption

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

## 5. Economic objective

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

## 6. Ruin and continuity

A genuine account-ending state is a physical/economic failure state, not an ordinary bad timestep.

\[
\tau_{ruin}=\inf\{t:A_t\in\mathcal F\}
\]

where \(\mathcal F\) contains states in which legitimate trading can no longer continue.

### No free principal reset

Computational segmentation must not silently erase economic continuity.

A training chunk boundary is not automatically an account reset boundary. Previous decisions must continue to affect future account state unless an experiment explicitly defines a new independent trajectory.

---

## 7. Inactivity is not inherently wrong

R12 should not apply an arbitrary penalty merely because the Trader is FLAT.

If no usable opportunity exists, not taking risk may be the correct action.

> **No opportunity -> not taking risk can be correct.**

The system must not reward trading activity for its own sake.

---

## 8. Primary benchmark: Buy & Hold

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
=
\log\frac{W_T}{W_0}
-
\log\frac{B_T}{B_0}
=
\log\frac{W_T}{B_T}
\]

Buy & Hold is primarily a qualification anchor:

> Did active decision-making create economic value relative to simply holding the same asset?

A bearish market can make FLAT economically superior to Buy & Hold. The benchmark must not force benchmark-like exposure.

---

## 9. Provisional v1 learning signal

The initial economic learning signal is:

\[
r_t^{growth}=\log\frac{W_{t+1}}{W_t}
\]

with \(W_t\) equal to true net account equity under the experiment's execution and cost model.

Then:

\[
\sum_{t=0}^{T-1}r_t^{growth}
=
\log\frac{W_T}{W_0}
\]

This provides dense feedback while remaining exactly consistent with compounded terminal wealth.

**v1 rule:** do not add survival/path-risk penalties to this scalar reward unless experiments show that qualification-only risk control is insufficient.

---

## 10. Qualification is multi-objective

Do not reduce qualification to a universal score such as:

\[
Score=w_1Return+w_2Sharpe-w_3MDD-w_4CVaR+\cdots
\]

Instead, begin with two conceptual result dimensions:

1. **Economic Value** — long-run economic value, including performance relative to Buy & Hold.
2. **Survival / Path Risk** — how dangerous the account path was while generating that value.

The exact authority metric for the second axis is intentionally **OPEN**.

MDD, CDaR, CVaR, drawdown duration, ruin probability, Sharpe, Sortino, Calmar, turnover, and similar statistics may be computed as diagnostics without automatically becoming qualification authority.

---

## 11. Evidence is not a tradeable objective

Evidence quality answers:

> **How confident are we that the measured Economic Value and Survival / Path Risk are real?**

It is not a third objective that can be sacrificed for more return.

Relevant evidence may include:

- paired evaluation on identical frozen market windows;
- seed robustness;
- dependence-aware uncertainty estimation;
- negative/random/shuffle controls where appropriate;
- multiple-testing or strategy-selection correction when many candidates are tried;
- out-of-sample consistency;
- uncertainty intervals on core objective estimates.

---

## 12. Machine-executable Pareto comparison

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

### Uncertainty-aware dominance

Tiny point-estimate differences are insufficient for promotion.

The comparator must support confidence-aware or probability-of-dominance logic. A conservative form is:

\[
LCB_i(A)\ge UCB_i(B)\quad\forall i
\]

with a meaningful strict improvement in at least one authority dimension.

Machine comparison returns only:

```text
DOMINATES
DOMINATED
UNRESOLVED / NON_DOMINATED
```

It does not fabricate a total ranking.

### Bounded evidence escalation

When a promising comparison remains unresolved, the evaluator may spend a predeclared bounded budget on additional seeds, resampling, or evaluation.

If the relation remains unresolved when that budget is exhausted, both candidates remain eligible.

---

## 13. Pareto Archive and Active Champion

### Pareto Archive

A machine-maintained set of currently qualified, statistically non-dominated Trader policies.

### Active Champion

The single policy currently chosen for a concrete operational purpose such as continued experimentation, bounded deployment, or parent initialization.

A policy is not deleted merely because it is not the Active Champion.

Human/SOL judgment is required only when a concrete decision demands one policy and the surviving candidates express a genuine preference trade-off that has not already been frozen in authority.

---

## 14. Materiality tolerance

The comparison interface should reserve per-objective tolerances:

\[
\epsilon_i
\]

so that numerically meaningless differences do not fill the Pareto Archive with effectively equivalent candidates.

Do not invent economic thresholds in advance.

Initial implementation may use zero or purely numerical/statistical tolerance. Economic materiality thresholds require explicit justification.

---

## 15. Qualification pipeline

```text
Candidate policy
    |
    v
Frozen evaluation protocol
    |
    +--> hard validity / legality / leakage gates
    |
    +--> Economic Value estimate
    |
    +--> Survival / Path-Risk estimate
    |
    +--> uncertainty / robustness evidence
    v
Confidence-aware Pareto comparator
    |
    +--> DOMINATES
    +--> DOMINATED
    +--> UNRESOLVED / NON_DOMINATED
    v
Pareto Archive update
    v
Active Champion selection only when operationally required
```

The first implementation should remain a small local module. It does not justify an LLM judge, custom service, custom database, or distributed orchestration layer.

---

## 16. Required semantic separations

The system must preserve these distinctions:

- **Truth != Belief != Decision != Permission**.
- Nominal Trader action != permitted/executed action.
- `requested_risk` != epistemic confidence.
- Market information != account information.
- Prediction quality != economic decision quality.
- Training reward != qualification metrics != diagnostics.
- Computational episode boundary != economic account reset.
- Evidence strength != economic objective value.

---

## 17. Historical knowledge principle

New observations may recalibrate current behavior without requiring learned historical information to be manually erased.

Historical recurrence should primarily be represented through learned model parameters and responses to current inputs.

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

1. Exact definition of the Survival / Path-Risk authority axis.
2. Which path statistics are authority versus diagnostics.
3. Whether and when survival/path risk must enter the training objective.
4. Economic materiality / epsilon thresholds.
5. Exact confidence and resampling protocol for dominance claims.
6. Exact ruin boundary beyond obvious account-ending/legal failure states.
7. Rule for selecting one Active Champion from multiple non-dominated candidates when a single operational choice is required.
8. Whether evidence eventually justifies explicit learned StrategyMemory.
9. Whether future scale requires market-impact modeling.

These are explicit scientific choices, not implementation gaps.

---

## 20. Minimal implementation shape

This draft implies only a small initial structure:

```text
state/
  market_state
  account_state
  optional_memory_interface

env/
  account_physics
  execution
  ruin_terminal_semantics

policy/
  trader
  action_intent

qualification/
  economic_value.py
  path_risk.py
  uncertainty.py
  dominance.py
  archive.py
```

No larger infrastructure is justified by this document alone.

---

## 21. Research basis

The design is consistent with established work on:

- long-run log-wealth growth and drawdown-constrained Kelly optimization;
- multi-objective reinforcement learning and Pareto policy sets;
- Pareto comparison under uncertain/noisy objectives;
- drawdown-aware risk measures such as CDaR;
- reward misspecification / Goodhart effects;
- dependence-aware and selection-bias-aware evaluation of trading strategies.

References should support design choices without becoming implementation authority.
