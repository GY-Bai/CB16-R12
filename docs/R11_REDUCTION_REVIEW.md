# R11 -> R12 Reduction Review

Status: **FIRST-PASS CLASSIFICATION COMPLETE — MASTER QUESTION OPEN**

This file is archaeology/reduction material. It is not normal R12 implementation context and must not be imported into canonical R12 documents. Its purpose is only to decide what survives into clean R12 requirements.

Latest R12 decisions override conflicting legacy requirements.

## Classes

- **CORE** — project/product/scientific invariant; removing it changes what CB16 is.
- **SIMPLIFY** — preserve the invariant with a smaller mechanism.
- **DEFER** — potentially useful, but unnecessary for the first working R12 system.
- **DROP** — do not carry forward as an R12 requirement.
- **UNKNOWN** — requires evidence or a Master decision.

---

## 1. CORE

| Requirement | R12 rule | Why | Minimal guard |
|---|---|---|---|
| Autonomous Trader | The product is an autonomous stateful Trader, not a forecasting benchmark. | Defines the product. | End-to-end policy must issue economic actions. |
| Market + Account decision state | Policy decisions depend on causal market information and current account state. | Prior actions must change future decisions through the account. | Known-answer task where identical market state + different account state requires different action. |
| Price-taker initial assumption | Trader actions affect its own account but not the market path in R12 v1. | Keeps the first environment tractable without changing the core closed loop. | Environment test: market transition is action-independent; account transition is action-dependent. |
| Same-account temporal continuity | Account consequences persist across future decisions. | Prevents per-bar independent prediction disguised as trading. | No reset across chunk/checkpoint boundaries unless a new independent trajectory is explicitly declared. |
| No free principal reset | Computational segmentation must not erase economic losses or terminal state. | Otherwise catastrophic behavior is artificially forgiven. | Continuation/restart test preserves equity, position, liabilities and terminal state. |
| Truth != Belief != Decision != Permission | Factual state, learned belief, policy intent and execution permission remain distinct. | Prevents semantic leakage between observation, policy and controls. | Typed interfaces/tests prevent one layer from silently substituting for another. |
| Nominal action != executed action | The Trader proposes an action; permission/execution may modify or reject it. | Required for real account physics and later deployment. | Record both intent and execution outcome. |
| requested_risk != confidence | Risk exposure request is an action-control quantity, not epistemic confidence. | Prevents a dangerous semantic collapse. | Schema/test forbids confidence semantics from driving requested_risk implicitly. |
| Direction + risk action semantics | Initial Trader action remains directional intent plus bounded risk/exposure request, including FLAT. | Matches the autonomous-account control problem. | Action schema and account-physics tests. |
| Market history is reusable environment material | Historical market data may be replayed; experience changes with policy/account trajectory. | Separates fixed exogenous history from endogenous account experience. | Same market segment under different policies produces different account trajectories. |
| Experience preserves real consequences | Successful, ordinary, failed and terminal trajectories remain factual experience. | Failure must not disappear through survivor filtering. | Replay completeness tests include terminal/failure samples. |
| Learning must change future behavior | A learning loop is not qualified merely because gradients ran. | Core learnability requirement. | Controlled known-answer task must show frozen post-training policy behavior changes correctly. |
| Prediction is perception, not final authority | Forecasting/representation models may be inputs; account-conditioned Trader owns the action. | A correct forecast can still imply a wrong trade for a particular account. | Ablation/task shows account state can change action under same prediction. |
| Survival-aware long-run capital growth | Economic goal is long-run account growth subject to acceptable path/survival risk. | Latest R12 product objective. | Qualification separates Economic Value and Survival/Path Risk. |
| Buy & Hold economic benchmark | Current asset Buy & Hold is the primary economic benchmark for qualification. | Tests whether active decisions justify use of capital without forcing activity. | Same-window, same-capital benchmark replay. |
| No forced activity | FLAT is allowed when no usable opportunity exists. | Prevents reward hacking through pointless turnover. | No synthetic per-step penalty for inactivity. |
| Multi-objective qualification | Economic Value and Survival/Path Risk are not collapsed into one arbitrary weighted score. | Preserves real trade-offs and avoids hidden value judgments. | Pareto comparator; no universal scalar rank. |
| Evidence != objective | Statistical evidence/robustness controls confidence in claims; it is not tradeable for return. | Separates epistemic confidence from economic preference. | Qualification report keeps objective estimates and uncertainty separate. |
| Pareto archive | Statistically non-dominated qualified policies may coexist. | Avoids fake total ordering when trade-offs are genuine. | Machine-maintained dominance archive. |
| Scientific claim ladder | Wiring, controlled learning, historical improvement, economic superiority and production profitability are distinct claims. | Prevents evidence inflation. | Stage-specific tests and labels. |
| Final-holdout protection | Final holdout stays unopened until explicitly authorized for the appropriate claim. | Required for scientific validity. | Dataset access test/manifest. |
| Historical knowledge is not manually erased by recency | New observations may recalibrate behavior without mandatory deletion of learned historical information. | Historical patterns may recur. | No default chronological expiry/reset mechanism. |
| No handcrafted recurrence engine | Do not build manual cycle/regime activation tables as the default recurrence mechanism. | Recurrence should primarily be represented by learned parameters and current inputs. | Architecture review/test forbids hidden rule-switch state unless explicitly authorized. |

---

## 2. SIMPLIFY

| Requirement | R12 rule | Smaller mechanism | Minimal guard |
|---|---|---|---|
| Account state schema | Preserve economically sufficient account information; do not freeze a legacy dimension count. | Small typed `AccountState` with only fields required by physics and decision-making. | Reachable-state and transition tests. |
| Account/execution physics | Preserve fees, position, equity, leverage/liability, legality and terminal behavior needed by the experiment. | One local deterministic environment module first. | Accounting invariants and known-answer trajectories. |
| Durable experience | Experience required for learning/restart must be reconstructable and inspectable. | Plain files/compact local store first; no database requirement. | Restart/replay equivalence test. |
| Policy identity during collection | A collected trajectory must identify the behavior policy that generated it. | One checkpoint/generation ID in each trajectory; freeze policy within the declared collection unit. | Reject mixed unidentified policy data. |
| Checkpoint generation continuity | Learned updates become active only through an explicit committed checkpoint boundary. | Atomic local checkpoint + generation counter. | Crash/restart and generation-switch tests. |
| Experiment preregistration | Formal scientific runs freeze data scope, costs, seeds/budgets, objectives and gates before results are inspected. | One compact machine-readable experiment spec. | Validator checks required fields; no receipt chain. |
| Controlled learnability before historical claims | Prove the closed learning loop can solve bounded known-answer tasks before interpreting historical failures. | A very small synthetic test suite aligned with R12 semantics. | Must include account-conditioned action and delayed-consequence learning. |
| Recovery | Restart must preserve required policy/account/experience continuity. | Local checkpoint + state file; no distributed recovery framework initially. | Kill/restart equivalence test. |
| Permission boundary | Invalid or impossible execution must be prevented independently of the Trader's intent. | Simple in-process permission/physics checks first. | Hostile action unit tests. |
| Provenance | Formal experiments/results must be reproducible enough to identify code/config/data used. | Git commit + compact run manifest + hashes only for external artifacts that need identity. | Re-run identity test. |
| CPU-first development | Start with CPU execution to reduce implementation complexity. | Single-host CPU profile and bounded multiprocessing only if measured useful. | Correctness first; performance work only after measured bottleneck. |

---

## 3. DEFER

| Capability | R12 position | Trigger to revisit |
|---|---|---|
| Explicit StrategyMemory | Interface reserved; no implementation in v1. | Market+Account policy shows a concrete memory limitation. |
| Specific pretrained market organs | Predictive/representation organs are allowed, but no legacy organ is automatically authoritative. | R12 first proves minimal learning loop, then qualifies candidate organs on cost/information value. |
| Teacher/distributional target system | Not required for the first R12 learning loop. | Direct policy/critic learning exposes a concrete credit or sample-efficiency blocker. |
| V-trace or any specific off-policy correction | Algorithm choice, not project essence. | Selected learner/replay design actually requires it. |
| Complex replay routing / elite experience taxonomies | Do not prebuild. | Simple replay demonstrably fails on a diagnosed learning problem. |
| CVaR critic / Lagrangian risk optimizer / distributional risk head | Qualification risk stays outside the v1 scalar reward initially. | Evidence shows log-growth training repeatedly produces unacceptable path-risk candidates. |
| Full multi-objective RL training | Pareto is initially a qualification mechanism, not necessarily a multi-objective training algorithm. | Single-objective learning + qualification is insufficient. |
| Market-impact model | Excluded under v1 price-taker assumption. | Capital/market scale makes own-impact non-negligible. |
| GPU dependency | Not a v1 prerequisite. | CPU profiling shows a concrete scientific throughput blocker. |
| Distributed/asynchronous training orchestration | Not a default dependency. | One-host execution becomes a measured bottleneck. |
| Dedicated database/queue/cache stack | Not required initially. | File/local store cannot meet measured correctness or throughput needs. |
| Production-grade leases/fencing/singleton services | Not needed for first scientific R12. | Multiple concurrent writers or live capital create a real concurrency hazard. |
| Hostile cutover infrastructure | Use focused tests instead of a generalized cutover program. | A concrete runtime replacement creates a cutover risk that unit/integration tests cannot cover. |
| Live/simulated real-market deployment | Scientific sandbox comes first. | Separate explicit deployment authorization and evidence. |
| Multi-asset portfolio coordination | R12 core starts with a single-asset Trader/account. | Single-asset policy is qualified and portfolio coordination becomes the next product question. |

---

## 4. DROP / SUPERSEDED

| Legacy requirement | R12 decision | Replacement |
|---|---|---|
| Arithmetic expected return as the sole master objective | **DROP / SUPERSEDED** | Long-run log-equity growth with a distinct Survival/Path-Risk qualification axis. |
| "High ruin probability may win if arithmetic expectation is higher" | **DROP / SUPERSEDED** | High growth does not erase path/survival risk; use Pareto qualification. |
| B&H and FLAT as co-equal master benchmark diagnostics | **DROP / SUPERSEDED** | Current asset Buy & Hold is the primary economic benchmark; FLAT may remain a diagnostic or policy behavior, not a co-equal authority. |
| One scalar promotion score combining many familiar metrics | **DROP** | Confidence-aware Pareto dominance. |
| Fixed H72 utility horizon as project semantics | **DROP** | Long-run account objective; experiment horizons are local scientific choices. |
| Teacher target loss as Champion promotion authority | **DROP** | Economic Value + Survival/Path Risk + evidence govern qualification. |
| Fixed `AccountState6` | **DROP AS AUTHORITY** | Minimal semantic AccountState chosen by R12 requirements. |
| Fixed sensory dimensions/model identities | **DROP AS AUTHORITY** | Predictive organs are replaceable auxiliary components. |
| Fixed Central Brain layer widths/parameter count | **DROP AS AUTHORITY** | Architecture is selected by evidence and resource constraints. |
| Frozen legacy optimizer/network/training hyperparameters as project identity | **DROP AS AUTHORITY** | Freeze parameters per experiment, not forever. |
| Multi-layer authority/receipt hierarchy | **DROP** | Git + concise task/experiment specs + tests/results. |
| Routine SHA ceremony for ordinary development | **DROP** | Git identity normally; hashes only where external scientific artifacts require them. |
| Receipt/gate compiler as a general architecture layer | **DROP** | Small direct validators for concrete experiments. |
| Branch-per-microstep / commit-per-edit protocols | **DROP** | One coherent task PR; normal local edit/test loop. |
| Duplicate current-state/authority documentation | **DROP** | Small canonical forward-looking docs only. |
| Custom ownership/lease system for ordinary development | **DROP** | One active writer per PR/worktree. |
| General async orchestrator before a measured need | **DROP AS DEFAULT** | Thin Actions/OCI/Builder path; add concurrency only when justified. |
| Legacy implementation compatibility as a product requirement | **DROP** | Preserve semantics/tests that matter; do not preserve obsolete implementation shape. |
| Handcrafted cycle/regime activation or chronological belief expiry | **DROP / FORBID BY DEFAULT** | Learned representation/weights + current inputs. |
| Treating realized winning action as a supervised correct-action label | **DROP / FORBID** | Outcomes are consequences/evidence, not hindsight labels. |

---

## 5. OPEN R12 SCIENTIFIC DECISIONS

These are intentionally not filled by an implementation agent:

1. Exact authority definition of the Survival / Path-Risk axis.
2. Exact ruin boundary beyond obvious account-ending or legally impossible states.
3. Economic materiality / epsilon thresholds for Pareto comparison.
4. Exact uncertainty/confidence protocol used to establish domination.
5. Rule for selecting one Active Champion when several non-dominated policies remain and an operational choice is mandatory.
6. Whether/when survival risk must enter training rather than qualification only.
7. Whether later evidence justifies explicit learned StrategyMemory.
8. Whether later scale requires market-impact modeling.

---

## 6. MASTER QUESTION

One legacy product statement is still ambiguous enough to affect the R12 state/interface design:

> Should the Master's **macro direction / macro interpretation** ever enter the Trader at runtime as an explicit observation or constraint, or is the Master only responsible for selecting the asset/capital/research direction while the Trader itself remains autonomous inside the chosen single-asset sandbox?

Until answered, R12 should **not** add a runtime `HumanMacroSignal` input.

---

## Reduction rule

When a requirement is transferred out of this worksheet into a canonical R12 document, rewrite it purely as a forward-looking R12 requirement. Do not carry this archaeology or legacy comparison with it.
