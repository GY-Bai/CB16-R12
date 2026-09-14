# CB16-R12 Core Objective and Qualification — Initial Draft

Status: **MASTER-ALIGNED DRAFT — NOT YET FINAL AUTHORITY**

Purpose: capture the highest-level product/scientific decisions agreed during the R12 reduction review, while explicitly separating durable CB16 principles from R11 implementation choices. Open items remain open; this document must not invent thresholds merely to make the specification look complete.

---

## 1. Ultimate purpose

CB16 exists to build an **autonomous Trader**, not a research platform for its own sake.

Scientific rigor, evaluation, infrastructure, and auxiliary models exist to make the Trader learnable, falsifiable, and trustworthy. They are means, not the product.

The Trader is a stateful sequential decision-maker. It must not collapse into a per-bar market-direction predictor.

---

## 2. Core state model

### 2.1 R12 v1 state

The first implementation should use:

\[
a_t = \pi(M_t, A_t)
\]

where:

- \(M_t\) = current causal market information;
- \(A_t\) = current account state;
- \(a_t\) = nominal Trader action.

Account state evolves through the Trader's previous choices and subsequent market movement:

\[
A_{t+1}=F(A_t,a_t,M_t,M_{t+1},\text{costs},\text{execution})
\]

Account state is therefore a compressed current economic state produced by the previous decision path. It is not merely an extra market feature.

### 2.2 Future learned memory must remain possible

R12 v1 should **not pay the complexity cost of explicit StrategyMemory yet**, but interfaces must not make it impossible to evolve toward:

\[
a_t = \pi(M_t,A_t,S_t)
\]

where \(S_t\) is learned internal strategy memory / latent recurrent state.

The reserved memory path must not become a handcrafted cycle engine, rule-activation table, or manually maintained market-regime state machine.

**Decision:** preserve the interface, defer the mechanism.

---

## 3. Market prediction is perception, not decision authority

Predictive models may act as the Trader's **eyes**. They may estimate direction, return distributions, volatility, uncertainty, latent market representations, or other useful quantities.

They do not replace the Trader's brain.

Conceptually:

```text
Market
  -> predictive / representation organs
  -> learned representation Z_t
                        + AccountState
                              -> Trader
                              -> Action
```

A market prediction can be correct while the corresponding trade is still wrong for the current account. The final action authority therefore belongs to the account-conditioned sequential policy, not to a forecasting model.

---

## 4. Initial market-impact assumption

R12 initially assumes the Trader is a small price taker:

\[
P(M_{t+1}\mid M_t,a_t) \approx P(M_{t+1}\mid M_t)
\]

while account transition depends strongly on the action:

\[
P(A_{t+1}\mid A_t,M_t,M_{t+1},a_t)
\]

This assumption deliberately excludes endogenous market impact from the first R12 system. Market-impact modeling is a later-stage capability, not a prerequisite for learning the core closed loop.

---

## 5. Highest-level economic objective

The Trader should behave like an agent in a continuing single-player economic game:

- it must preserve enough capital and legal trading capacity to remain in the game;
- it must learn from the consequences of previous decisions;
- it should improve future decisions rather than repeatedly resetting away prior mistakes;
- survival alone is not sufficient, because a zero-risk permanently-flat policy can survive trivially;
- growth alone is not sufficient, because unconstrained growth maximization can accept destructive path risk.

The target problem is therefore best expressed as:

> **survival-aware long-run account growth**, with growth and survival/path risk kept conceptually distinct.

A generic mathematical form is:

\[
\max_\pi J_{growth}(\pi)
\quad\text{subject to acceptable path/survival risk.}
\]

This is a problem definition, not yet a commitment to a specific constrained-RL algorithm.

---

## 6. Ruin and continuity

### 6.1 Ruin is not an ordinary negative reward

A genuine account-ending state should be treated as a physical/economic failure state, conceptually absorbing:

\[
\tau_{ruin}=\inf\{t:A_t\in\mathcal F\}
\]

where \(\mathcal F\) contains states in which the account can no longer continue legitimate trading.

R12 should not teach the model that catastrophic failure is merely another bad timestep that is followed by a free restoration of principal.

### 6.2 No free principal reset

Training may be segmented for computational reasons, but segmentation must not silently erase economic continuity.

A computational episode boundary is not automatically an economic reset boundary.

---

## 7. Do not punish inactivity directly

R12 should **not** add an arbitrary per-step penalty merely because the Trader is FLAT.

If the market contains no usable edge, FLAT may be the correct action. A forced-activity penalty would reward trading for its own sake and create an avoidable reward-hacking channel.

Therefore:

> **No opportunity -> not taking risk can be correct.**

The anti-degeneracy mechanism should come from economic comparison and long-run qualification, not from a synthetic "must trade" tax.

---

## 8. Buy-and-hold benchmark

For a single current asset, R12 should use that asset's **Buy & Hold** path as the primary economic benchmark.

Let Trader equity be \(W_t\). Let benchmark wealth be:

\[
B_t = W_0\frac{P_t}{P_0}
\]

under the same evaluation interval and a clearly specified benchmark execution convention.

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

### Important interpretation

Buy & Hold is primarily an **economic anchor / qualification benchmark**:

> Did active decision-making create value relative to simply holding the same asset?

Because the benchmark path is exogenous under the price-taker assumption, subtracting its return from the training objective does not by itself solve the inactivity problem; for a fixed market path it is a policy-independent term. This is acceptable. Its main value is to expose whether the Trader created economically meaningful relative value.

A bearish market is an important example: remaining FLAT while Buy & Hold loses heavily can be a legitimate success. R12 must not force trading merely to imitate benchmark exposure.

---

## 9. Provisional v1 training signal

The cleanest initial economic learning signal is:

\[
r_t^{growth}=\log\frac{W_{t+1}}{W_t}
\]

with \(W_t\) representing true net account equity after the frozen account/execution economics relevant to the experiment.

Then:

\[
\sum_{t=0}^{T-1}r_t^{growth}
=
\log\frac{W_T}{W_0}
\]

This gives dense local feedback while remaining exactly consistent with compounded terminal wealth.

**Status:** recommended R12 v1 default, not a permanent algorithm lock. Survival/path-risk terms should not be added to this scalar reward without evidence that doing so is necessary.

---

## 10. Multi-objective qualification: no universal score

R12 should not create a universal scalar such as:

\[
Score=w_1 Return+w_2 Sharpe-w_3 MDD-w_4 CVaR+\cdots
\]

Such a score hides value judgments in arbitrary weights, creates scale problems, and increases Goodhart/reward-hacking risk.

Instead, qualification should use a small number of distinct semantic dimensions and Pareto dominance.

### Initial objective space

Start with only two conceptual result dimensions:

1. **Economic Value** — did the Trader create meaningful long-run economic value, especially relative to Buy & Hold?
2. **Survival / Path Risk** — how dangerous was the account path while generating that value?

The exact statistic used for the second axis is intentionally **OPEN**. MDD, CDaR, CVaR, drawdown duration, ruin probability, and other diagnostics must not automatically become authority merely because they are familiar metrics.

---

## 11. Evidence is not a third tradeable objective

Evidence quality / robustness is not another objective that can be traded for more return.

It answers a different question:

> **How confident are we that the Economic Value and Survival/Path Risk estimates are real?**

Examples include:

- paired evaluation on the same frozen market windows;
- seed robustness;
- block/stationary bootstrap or another dependence-aware uncertainty method;
- shuffle/random/negative controls where appropriate;
- multiple-testing / strategy-selection correction when many candidates have been tried;
- out-of-sample consistency;
- uncertainty intervals on the core objective estimates.

This layer governs the strength of a promotion claim. It is not itself something the Trader is allowed to sacrifice for more growth.

---

## 12. Machine-executable Pareto qualification

Routine candidate comparison must not require Chat-SOL or Master judgment.

### 12.1 Deterministic Pareto relation

After converting all objective axes to a common direction (higher = better), candidate \(A\) dominates candidate \(B\) if:

\[
A_i\ge B_i\quad\forall i
\]

and:

\[
A_j>B_j\quad\text{for at least one }j.
\]

If one policy has higher growth but worse path risk, the two are non-dominated and both may survive.

### 12.2 Uncertainty-aware dominance

Financial evaluations are noisy. R12 must not declare superiority from tiny point-estimate differences.

The comparison layer should support confidence-aware or probability-of-dominance logic. A deliberately conservative form is:

\[
LCB_i(A)\ge UCB_i(B)\quad\forall i
\]

with a meaningful strict improvement in at least one dimension before declaring confident domination.

If uncertainty intervals overlap such that reliable domination cannot be established, the result is:

```text
UNRESOLVED / NON_DOMINATED
```

not a fabricated first-place ranking.

### 12.3 Automatic evidence escalation

When a promising comparison is unresolved, the evaluator may automatically spend a predeclared bounded evidence budget on additional seeds/resampling/evaluation before giving up.

If uncertainty remains after the budget is exhausted, both candidates remain eligible in the Pareto archive.

---

## 13. Pareto Archive and Active Champion

R12 should distinguish two concepts.

### Pareto Archive

A small machine-maintained set of all currently qualified, statistically non-dominated Trader policies.

Example:

```text
Pareto Archive
  P17  higher growth / higher path risk
  P23  medium growth / lower path risk
  P31  lower growth / strongest survival properties
```

### Active Champion

The single policy currently selected for the next concrete purpose: continued experimentation, bounded deployment, or parent initialization.

A policy does not have to be erased merely because it is not the current Active Champion.

Routine archive maintenance is mechanical. Master/SOL intervention is required only when a real-world choice demands one policy and the surviving Pareto candidates encode a genuine preference trade-off not already frozen in authority.

---

## 14. Epsilon / materiality tolerance

The comparison API should reserve per-objective materiality tolerances:

\[
\epsilon_i
\]

so that numerically trivial differences do not fill the Pareto archive with effectively identical policies.

However, R12 must **not invent economic thresholds yet**.

Initial implementation may use zero or purely numerical/statistical tolerance. Economic materiality thresholds should be introduced only after evidence shows that they are needed and their meaning can be justified.

---

## 15. Qualification pipeline

The intended machine flow is:

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
Active Champion selection only when needed
```

The first implementation can remain a small local module. It does not justify a service, agent judge, custom database, or distributed orchestration layer.

---

## 16. R11 reduction implications

R11 remains valuable archaeology, but its frozen mechanisms are not automatically R12 authority.

| R11 concept | R12 treatment | Reason |
|---|---|---|
| Market + Account as decision input | **CORE** | Defines CB16 as an account-conditioned sequential Trader rather than a market-only predictor. |
| Truth != Belief != Decision != Permission | **CORE** | Preserves semantic/authority separation. |
| Nominal action != executed action | **CORE** | Execution/permission/account physics remain distinct from the brain's proposal. |
| Historical market can be replayed; experience depends on policy/account trajectory | **CORE** | Required for closed-loop learning. |
| Same-account temporal continuity | **CORE** | Previous actions must change future decision state. |
| Historical information should not be manually erased by recency | **CORE PRINCIPLE** | Preserve learned historical knowledge without hand-coded expiry. |
| Hand-engineered cycle/regime activation engine | **DROP / FORBID BY DEFAULT** | Recurrence should primarily live in learned representation/weights. |
| Final-holdout protection | **CORE** | Scientific validity requirement. |
| Fixed `AccountState6` dimensionality | **SIMPLIFY / UNFREEZE** | Account information is core; the old six-dimensional encoding is an implementation choice. |
| Fixed Operator48 / Medium48 sensory stack | **DEFER / REQUALIFY** | Predictive organs are allowed but specific R11 organs are not project essence. |
| Fixed R11 Central Brain layer sizes / 189052 parameters | **DROP AS AUTHORITY** | Specific architecture is not a highest-level invariant. |
| Fixed H72 utility | **DROP AS CORE; MAY RESEARCH LATER** | Long-run account objective is core; 72h is one historical horizon choice. |
| Probabilistic Teacher `P(U | I_t,a)` implementation | **DEFER / REQUALIFY** | A possible learning mechanism, not the definition of the Trader. |
| Teacher target loss as Champion promotion authority | **REPLACE** | R12 qualification should be based on economic/path outcomes plus evidence, not an auxiliary Teacher metric alone. |
| requested_risk != confidence | **CORE SEMANTIC GUARD** | Risk request remains an action-control quantity, not epistemic confidence. |
| Multi-layer authority/receipt chains, custom gate compiler, routine SHA ceremony | **DROP / SIMPLIFY** | Git + tests + bounded artifacts should carry ordinary development authority. |
| Generalized async/distributed orchestration before observed need | **DEFER** | Infra grows only from demonstrated bottlenecks. |
| Explicit StrategyMemory implementation | **DEFER, INTERFACE RESERVED** | Preserve future capability without paying complexity now. |
| Market impact / endogenous market response | **DEFER** | Initial price-taker assumption is sufficient for R12 v1. |
| CVaR critic, Lagrangian risk optimizer, distributional risk head, complex MORL training | **DEFER** | Do not implement before simple log-growth learning exposes a concrete need. |

---

## 17. What remains intentionally open

The following should **not** be silently decided by an implementation agent:

1. Exact definition of the Survival / Path-Risk axis.
2. Which path statistics are authority vs diagnostics.
3. Whether and when survival/risk must enter the training objective rather than qualification only.
4. Economic materiality / epsilon thresholds.
5. Exact statistical confidence and resampling protocol for policy domination.
6. Exact ruin boundary beyond obvious legal/account-ending states.
7. Policy for selecting one Active Champion from multiple non-dominated candidates when a single choice is operationally required.
8. Whether future evidence requires explicit learned StrategyMemory.
9. Whether future scale requires market-impact modeling.

These are future scientific choices, not holes to be filled opportunistically.

---

## 18. Minimal R12 implementation direction implied by this draft

This document does **not** authorize full training implementation yet. It suggests only a minimal future shape:

```text
state/
  market_state
  account_state
  optional_memory_interface

env/
  account_physics
  execution
  ruin/terminal semantics

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

No multi-agent judge, custom Pareto service, complex MORL framework, or distributed qualification system is justified at this stage.

---

## 19. Research basis

These references support the design direction but do not become CB16 authority by themselves.

- Busseti, Ryu, Boyd (2016), **Risk-Constrained Kelly Gambling** — long-run log-growth with explicit drawdown-risk constraints rather than collapsing risk and growth into one ad hoc score.  
  https://stanford.edu/~boyd/papers/pdf/kelly.pdf

- Van Moffaert & Nowé (2014), **Multi-Objective Reinforcement Learning using Sets of Pareto Dominating Policies** — maintenance of Pareto-dominating policy sets instead of assuming a total ordering.  
  https://jmlr.org/papers/v15/vanmoffaert14a.html

- Abdolmaleki et al. (ICML 2020), **A Distributional View on Multi-Objective Policy Optimization** — highlights scale/conflict problems in naive scalarization and develops scale-invariant multi-objective policy optimization.  
  https://proceedings.mlr.press/v119/abdolmaleki20a.html

- Mlakar, Tušar, Filipič (2014), **Comparing Solutions under Uncertainty in Multiobjective Optimization** — extends Pareto comparison when objectives are uncertain and represented by confidence intervals.  
  https://doi.org/10.1155/2014/817964

- Budszuhn, Krallmann, Horn (2025), **Adaptive Resampling with Bootstrap for Noisy Multi-Objective Optimization Problems** — uses bootstrap-estimated probability of dominance to allocate extra evaluation only where uncertainty matters.  
  https://arxiv.org/abs/2503.21495

- Chekhlov, Uryasev, Zabarankin (2003), **Portfolio Optimization with Drawdown Constraints** — motivates path-dependent drawdown risk and CDaR as an alternative to relying on one worst MDD observation.  
  https://www.cis.upenn.edu/~mkearns/finread/drawdown.pdf

- Karwowski et al. (ICLR 2024), **Goodhart's Law in Reinforcement Learning** — supports treating optimized metrics as imperfect proxies and avoiding unnecessary pressure on composite proxy scores.  
  https://proceedings.iclr.cc/paper_files/paper/2024/file/6ad68a54eaa8f9bf6ac698b02ec05048-Paper-Conference.pdf

- Bailey & López de Prado (2014), **The Deflated Sharpe Ratio** — documents selection bias / backtest overfitting when many strategies are tried and motivates uncertainty/multiple-testing awareness in strategy qualification.  
  https://davidhbailey.com/dhbpapers/deflated-sharpe.pdf

---

## 20. Draft summary

The current R12 philosophy can be compressed to:

> **Build an account-aware autonomous Trader that learns to compound capital over a continuing trajectory without treating survival as a trivial inactivity game. Prediction models are eyes, not the brain. The first system remains a price taker and uses Market + Account, while reserving an interface for future learned memory. Train initially on clean net log-equity growth, evaluate economic value against same-asset Buy & Hold, keep survival/path risk distinct, and qualify policies through uncertainty-aware Pareto dominance rather than a universal weighted score. Preserve non-dominated policies in a small archive; automate routine comparison; involve Master/SOL only when a genuine preference trade-off remains.**
