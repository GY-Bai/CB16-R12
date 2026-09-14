# CB16-R12 Core Objective and Qualification — Initial Draft

Status: **MASTER-ALIGNED DRAFT — NOT YET FINAL AUTHORITY**

## 1. Product objective

CB16 builds an autonomous, continuing Trader. Human authority stays outside the runtime policy. Runtime input must not include human macro direction, asset-name tokens, or other identity shortcuts.

## 2. Decision interface

R12 v1 uses:

\[
a_t=\pi(Z_t,A_t,X_t)
\]

- \(Z_t\): frozen representation of causally normalized K-line context;
- \(A_t\): minimal dimensionless account state;
- \(X_t\): minimal normalized execution/constraint context;
- \(a_t\): direction + `requested_risk`.

A future learned-memory interface \(S_t\) is reserved, but R12 v1 does not require it.

## 3. Market state

The Trader reacts to relative K-line structure, not BTC/ETH/BNB identity or nominal price scale. Price-bearing inputs are causally normalized before frozen sensory models produce \(Z_t\). Exact normalization is frozen per experiment.

## 4. AccountState v1

Raw account truth stays in Physics/Execution. The Central Brain receives only dimensionless economic state needed for future decisions.

\[
A_t^{v1}=[e_t,s_t,c_t]
\]

### `signed_exposure` \(e_t\)

\[
e_t=\frac{\text{signed marked-to-market risky exposure}}{\text{economic equity}}
\]

It represents current long/flat/short exposure without raw quantity or nominal price.

### `survival_cushion` \(s_t\)

Let \(E_t^{risk}\) be the risk engine's eligible account equity and \(MM_t\) the current maintenance requirement:

\[
s_t=\frac{E_t^{risk}-MM_t}{E_t^{risk}}
\]

for positive \(E_t^{risk}\). Zero means the maintenance boundary; a breach is terminal/forced-risk handling according to Physics. For an unlevered environment with no maintenance-margin liquidation rule, the environment defines the neutral safe value.

### `new_risk_capacity` \(c_t\)

Let \(F_t^{new}\) be capital/collateral currently available for increasing risk after current positions, reservations, and initial-margin/cash constraints:

\[
c_t=\frac{F_t^{new}}{E_t^{risk}}
\]

`survival_cushion` and `new_risk_capacity` are different semantics:

- maintenance logic: can current risk survive?
- initial-margin / available-funds logic: can new risk be added?

Do not collapse them into one headroom value.

### Excluded by default

Do not feed raw equity, raw quantity, raw entry price, symbol/venue identity, cumulative PnL, cumulative fees, position return, holding duration, previous nominal action, drawdown, running peak, or win/loss streaks merely because they are available.

A history-derived feature enters policy state only if removing it breaks transition sufficiency, reward sufficiency, constraint sufficiency, or a preregistered ablation shows repeatable decision value.

## 5. Execution / permission context

\(X_t\) contains only external friction or feasibility that materially changes action value, in normalized form. A v1 candidate is normalized expected trading friction. If long/short feasibility is materially asymmetric, Permission may expose a normalized direction-specific feasible-action envelope rather than venue-specific rules.

Nominal intent != permitted action != executed action.

## 6. Economic objective

The highest-level objective is long-run account growth subject to acceptable survival/path risk. A provisional dense learning signal is:

\[
r_t=\log\frac{W_{t+1}}{W_t}
\]

with true net account equity \(W_t\). FLAT is not penalized merely for inactivity. Computational chunk boundaries do not reset economic capital.

## 7. Benchmark and qualification

The primary single-asset economic benchmark is the same asset's Buy & Hold path over the same interval.

Qualification is not a weighted grand score. Begin with two conceptual outcome dimensions:

1. Economic Value;
2. Survival / Path Risk.

Evidence strength is not a third tradeable objective. Routine candidate comparison is machine-executable and returns only `DOMINATES`, `DOMINATED`, or `UNRESOLVED / NON_DOMINATED`, using confidence-aware Pareto logic rather than point estimates alone.

The Pareto Archive stores qualified non-dominated policies. Active Champion is the policy selected for a concrete operational purpose.

## 8. Required separations

- Truth != Belief != Decision != Permission
- nominal action != executed action
- `requested_risk` != confidence
- market state != account state != execution context
- prediction quality != economic decision quality
- training reward != qualification != diagnostics
- evidence strength != economic value
- survival cushion != new-risk capacity

## 9. Scientific validity

`wiring works != controlled learning works != historical improvement exists != economic superiority exists != production profitability is established`.

Formal experiments freeze question, data, transforms, metrics, and gates before execution. No silent rescue. Final holdout remains protected.

## 10. Open decisions

- exact K-line normalization;
- exact numerical scaling/clipping of \(e_t,s_t,c_t\);
- exact v1 \(X_t\) friction schema;
- whether direction-specific feasible-action capacity is needed;
- Survival / Path-Risk authority metric;
- whether risk must later enter the training objective;
- Pareto materiality thresholds and confidence protocol;
- exact ruin boundary;
- Active Champion selection among non-dominated candidates;
- whether evidence later justifies learned StrategyMemory or endogenous market impact.
