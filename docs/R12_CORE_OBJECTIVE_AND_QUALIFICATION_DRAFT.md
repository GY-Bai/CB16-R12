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
- \(X_t\): optional minimal execution-economics context, present only when hidden execution conditions would otherwise change action value;
- \(a_t\): direction + `requested_risk`.

If execution economics are fixed and identical across the relevant environment, \(X_t\) may be empty. A future learned-memory interface \(S_t\) is reserved, but R12 v1 does not require it.

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

## 5. Execution economics and permission

Execution economics remain part of Physics. The Central Brain should not receive venue-specific rules or duplicate the execution engine.

Let the target exposure implied by the nominal action be \(e_t^{target}\), and define:

\[
\Delta e_t=e_t^{target}-e_t.
\]

Trading cost is action-dependent:

\[
\frac{C_t^{trade}}{W_t}=g(\Delta e_t;X_t).
\]

For the initial price-taker system, a simple proportional model is sufficient unless evidence requires more:

\[
\frac{C_t^{trade}}{W_t}=\kappa_t|\Delta e_t|.
\]

Here \(\kappa_t\) is a normalized proportional friction coefficient under the frozen execution convention.

### When `X_t` is unnecessary

If \(\kappa\) and all other execution economics are constant across the relevant training/evaluation universe, they remain environment parameters and are not repeated as policy input:

\[
X_t=\varnothing.
\]

The policy can learn their consequence through true account transitions and net-equity reward.

### When `X_t` is required

If two otherwise identical decision states can have different future reward or transition because execution economics vary, the minimal parameters causing that difference must be observable. Examples include:

- time/asset-varying proportional trading friction;
- materially asymmetric long/short execution cost;
- variable carry, funding, or borrow cost if modeled by the environment.

Expose the parameters needed to evaluate future marginal economics, not a venue name and not a post-hoc realized cost.

Do not provide a scalar "cost of the chosen action" as an input before the action is chosen. The Central Brain chooses the target exposure; Physics evaluates that target using the observable cost parameters.

### Carry cost

If holding cost is modeled and varies, it is distinct from transaction cost. A normalized form may be represented conceptually as:

\[
\frac{C_t^{carry}}{W_t}
=
\rho_t^{long}\max(e_t,0)
+
\rho_t^{short}\max(-e_t,0).
\]

Only include \(\rho_t\) in \(X_t\) if the modeled carry economics actually vary and affect decisions. Fixed carry remains an environment parameter.

### Market impact

Nonlinear endogenous market impact is deferred under the initial price-taker assumption. Do not add order-book, liquidity, or impact features to the Central Brain merely to make the execution model look realistic.

### Permission

Hard feasibility and legality remain Permission/Physics semantics, not reward shaping. `new_risk_capacity` provides the account-level ability to increase risk. If materially different long/short feasible capacity exists and cannot be inferred from the minimal state, expose a normalized direction-specific feasible-action envelope rather than venue-specific rules.

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
- market state != account state != execution economics
- prediction quality != economic decision quality
- training reward != qualification != diagnostics
- evidence strength != economic value
- survival cushion != new-risk capacity
- transaction cost != carry cost
- environment parameter != policy state

## 9. Scientific validity

`wiring works != controlled learning works != historical improvement exists != economic superiority exists != production profitability is established`.

Formal experiments freeze question, data, transforms, metrics, and gates before execution. No silent rescue. Final holdout remains protected.

## 10. Open decisions

- exact K-line normalization;
- exact numerical scaling/clipping of \(e_t,s_t,c_t\);
- exact frozen v1 proportional execution convention and whether \(X_t\) can remain empty;
- whether variable carry/funding/borrow economics are modeled in v1;
- whether direction-specific feasible-action capacity is needed;
- Survival / Path-Risk authority metric;
- whether risk must later enter the training objective;
- Pareto materiality thresholds and confidence protocol;
- exact ruin boundary;
- Active Champion selection among non-dominated candidates;
- whether evidence later justifies learned StrategyMemory or endogenous market impact.
