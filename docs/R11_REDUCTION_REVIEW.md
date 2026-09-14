# R11 -> R12 Reduction Review

Status: **SECOND-PASS CLASSIFICATION COMPLETE — NO MASTER QUESTION OPEN**

This file is archaeology/reduction material. It is not normal R12 implementation context. Its only purpose is to decide what survives into clean R12 requirements.

Latest R12 decisions override conflicting legacy requirements.

## Classes

- **CORE** — removing it changes what CB16 is.
- **SIMPLIFY** — preserve the invariant with a smaller mechanism.
- **DEFER** — potentially useful, but unnecessary for the first working R12 system.
- **DROP** — do not carry forward as an R12 requirement.
- **UNKNOWN** — requires evidence or a Master decision.

---

## 1. CORE

| Requirement | R12 rule | Minimal guard |
|---|---|---|
| Autonomous Trader | Runtime Trader acts autonomously inside the selected asset/account scope. Human authority stays outside the runtime policy. | No `HumanMacroSignal`, discretionary long/short view, or manual regime label in policy input. |
| Asset-agnostic policy | Trader is not told whether it is trading BTC, ETH, BNB, or another selected asset. | No symbol/name token or equivalent identity field. |
| Relative K-line semantics | Price-bearing K-line context is causally normalized so policy reacts to relative structure rather than nominal price scale. | No future leakage; no raw absolute-price identity shortcut. |
| Frozen market representation boundary | Market sensory/representation weights stay frozen while the Central Brain learns. | Gradient test shows no sensory-weight updates. |
| Trainable Central Brain | CB combines market representation with account state and owns the final economic action. | Same market representation + different reachable account state can require different action. |
| Market + Account decision state | Policy decisions depend on causal market representation and current account state. | Account-conditioned known-answer task. |
| Price-taker initial assumption | Trader actions affect its account, not the market path, in v1. | Market transition action-independent; account transition action-dependent. |
| Same-account temporal continuity | Account consequences persist across future decisions. | No reset across chunk/checkpoint boundaries unless a new independent trajectory is declared. |
| No free principal reset | Computational segmentation cannot erase losses or terminal state. | Restart preserves economic state. |
| Truth != Belief != Decision != Permission | Factual state, learned representation, policy intent and execution permission remain distinct. | Typed/bounded interfaces. |
| Nominal action != executed action | Trader proposes; permission/execution may modify or reject. | Record intent and execution outcome separately. |
| requested_risk != confidence | Requested exposure is action control, not epistemic confidence. | Semantic/schema test. |
| Direction + bounded risk action | Initial action contains directional intent plus bounded requested exposure, including FLAT. | Action-schema and physics tests. |
| Historical market is reusable environment material | Exogenous market history may be replayed; policy/account experience changes with actions. | Same market segment under different policies yields different account paths. |
| Experience preserves consequences | Success, ordinary outcomes, failure and terminal trajectories remain factual experience. | No survivor-only filtering. |
| Learning must change future behavior | Gradients running is not learnability. | Frozen post-training policy must change correctly on controlled tasks. |
| Prediction is perception | Predictive/representation organs do not own final action. | Account ablation/decision test. |
| Survival-aware long-run growth | Economic goal is long-run account growth subject to acceptable path/survival risk. | Separate Economic Value and Survival/Path Risk. |
| Buy & Hold benchmark | Current asset Buy & Hold is the primary economic benchmark. | Same-window, same-capital benchmark replay. |
| No forced activity | FLAT may be correct when there is no usable opportunity. | No synthetic inactivity penalty. |
| Multi-objective qualification | Economic Value and Survival/Path Risk are not collapsed into one arbitrary weighted score. | Pareto comparator. |
| Evidence != objective | Evidence controls confidence in claims; it is not tradeable for return. | Objective estimates and uncertainty reported separately. |
| Pareto archive | Statistically non-dominated qualified policies may coexist. | Machine-maintained archive. |
| Scientific claim ladder | Wiring, controlled learning, historical improvement, economic superiority and production profitability are distinct claims. | Stage-specific tests. |
| Final-holdout protection | Final holdout remains unopened until explicitly authorized. | Dataset-access guard. |
| Historical knowledge not manually erased by recency | New evidence may recalibrate behavior without mandatory deletion of historical information. | No default chronological expiry/reset. |
| No handcrafted recurrence engine | Recurrence should primarily live in learned parameters and current normalized inputs. | No default manual cycle/regime switch tables. |
| Economic terminal != computational truncation | Account death/ruin, task horizon, replay chunk, BPTT/training truncation and process timeout are semantically distinct. | No single generic `done` flag may silently define all semantics. |
| Behavioral evidence > training diagnostics | Loss, gradients, parameter movement and checkpoint creation are not scientific success by themselves. | Qualification evaluates frozen-policy behavior/economics. |
| No-rescue formal qualification | Question, data scope, objective, controls, budget and gate freeze before qualification results are seen. | Changed method/question gets a new run identity. |
| Negative controls when informative | Leakage/credit claims require matched negative/random/shuffle controls when they materially distinguish explanations. | Positive task must beat the relevant control under frozen criteria. |

---

## 2. SIMPLIFY

| Requirement | R12 rule | Smaller mechanism |
|---|---|---|
| Market normalization | Preserve relative K-line geometry and causality; exact transform is not project identity. | One small causal-normalization module frozen per experiment. |
| Account state schema | Preserve economically sufficient state; do not freeze a legacy dimension count. | Small typed, preferably dimensionless/scale-aware `AccountState`. |
| Account/execution physics | Preserve fees, position, equity, leverage/liability, legality and terminal behavior required by the experiment. | One local deterministic environment module first. |
| Reconstructable experience | Reusable learning experience must be inspectable/reconstructable. | Compact append-only trajectory files or a tiny local store. |
| Observation integrity | Learner input must correspond to the causal observation used for the decision. | Stable trajectory schema + sequence/account IDs; content-addressed storage is optional. |
| Policy identity during collection | Each trajectory identifies the behavior checkpoint/generation that produced it. | One checkpoint/generation ID in each trajectory. |
| Behavior likelihood when required | If the chosen algorithm needs behavior probabilities/ratios, record them at decision time. | Persist the required `log_prob`/distribution metadata; never reconstruct it from execution outcomes. |
| Checkpoint continuity | Learned updates become active through an explicit committed generation boundary. | Atomic local checkpoint + generation counter. |
| Update recovery | Re-running after a crash must not silently corrupt policy/account state. | Atomic checkpoint/run marker; no distributed transaction system by default. |
| Scientific preregistration | Formal runs freeze data scope, costs, seeds/budgets, objectives and gates before results. | One compact machine-readable experiment spec. |
| Controlled learnability before historical claims | Prove the closed loop can learn bounded known-answer tasks first. | Small synthetic suite. |
| Recovery | Restart preserves required policy/account/experience continuity. | Local checkpoint + state file. |
| Permission boundary | Impossible/illegal execution is prevented independently of Trader intent. | Simple in-process checks. |
| Provenance | Formal results identify code/config/data strongly enough to reproduce the claim. | Git commit + compact run manifest + external-artifact hashes only where needed. |
| CPU-first development | Start with CPU to minimize system complexity. | Single host; bounded multiprocessing only when measured useful. |
| Failure taxonomy | Distinguish scientific failure from execution/hardware/evidence problems. | `SCIENTIFIC_FAIL`, `EXECUTION_BLOCKED`, `HARDWARE_LIMIT`, `EVIDENCE_INSUFFICIENT`, `CONTRACT_MISMATCH`. |

---

## 3. DEFER

| Capability | R12 position | Trigger to revisit |
|---|---|---|
| Explicit StrategyMemory | Interface reserved; no v1 implementation. | Market+Account policy shows a concrete memory limitation. |
| Specific pretrained organ identity | Frozen sensory boundary is core; particular model/tap/dimension is replaceable. | Qualify candidate organ for information value and cost. |
| Separate Critic as permanent architecture | Useful candidate, not project essence. | Selected learning algorithm benefits from value estimation. |
| Teacher/distributional target system | Not required initially. | Direct learning exposes a concrete blocker. |
| V-trace / specific off-policy correction | Algorithm choice, not project semantics. | Replay becomes materially off-policy and chosen learner needs correction. |
| Mandatory behavior `log_mu` everywhere | Only needed by algorithms that require behavior likelihood. | Chosen learner uses importance ratios/PPO-style old-policy likelihood. |
| Complex replay routing / elite taxonomies | Do not prebuild. | Simple replay demonstrably fails on a diagnosed problem. |
| Long-term A-B-A retention qualification | Important later, not required for first vertical slice. | Minimal learnability loop works and retention becomes the next hypothesis. |
| CVaR critic / Lagrangian risk optimizer / distributional risk head | Risk remains qualification-only initially. | Log-growth learning repeatedly yields unacceptable path risk. |
| Full multi-objective RL training | Pareto begins as qualification, not necessarily training. | Single-objective learning + qualification is insufficient. |
| Market-impact model | Excluded under price-taker v1. | Scale makes own impact material. |
| GPU dependency | Not a v1 prerequisite. | CPU becomes a measured scientific bottleneck. |
| Distributed/asynchronous training | Not a default dependency. | One-host execution becomes a measured bottleneck. |
| Dedicated database/queue/cache stack | Not required initially. | Local storage cannot meet measured correctness/throughput needs. |
| Production leases/fencing/singletons | Not needed for initial science. | Concurrent writers/live capital create a real hazard. |
| Live/simulated real-market deployment | Sandbox first. | Separate deployment authorization and evidence. |
| Multi-asset portfolio coordination | Single-asset Trader first. | Single-asset policy is qualified. |

---

## 4. DROP / SUPERSEDED

| Legacy requirement | R12 replacement |
|---|---|
| Runtime human macro-direction input/manual regime guidance | Human remains outer-loop authority only; Trader runtime is autonomous. |
| Asset identity as model input | Asset-agnostic normalized market context. |
| Raw absolute price as policy cue | Causal relative/normalized price representation. |
| Arithmetic expected return as sole master objective | Long-run log-equity growth + separate Survival/Path Risk qualification. |
| High ruin probability may win solely because arithmetic expectation is higher | High growth cannot erase path/survival risk; Pareto qualification. |
| B&H and FLAT as co-equal master benchmark authorities | Buy & Hold is primary benchmark; FLAT may remain diagnostic/behavior. |
| Universal weighted promotion score | Confidence-aware Pareto dominance. |
| Fixed H72 utility as project semantics | Long-run account objective; local horizons belong to experiments. |
| Teacher target loss as Champion authority | Economic Value + Survival/Path Risk + evidence. |
| Fixed `AccountState6` | Minimal semantic AccountState selected by R12 needs. |
| Fixed sensory model/tap/dimension as permanent authority | Frozen sensory role remains; concrete organ is replaceable and requalified. |
| Fixed CB layer widths/parameter count | Architecture selected by evidence/resources. |
| Frozen legacy optimizer/hyperparameters as project identity | Freeze parameters per experiment only. |
| Content-addressed observation store as mandatory architecture | Small deterministic trajectory persistence unless stronger identity is actually needed. |
| Durable-replay-only learner truth as universal rule | Reconstructability is required; exact on-policy/off-policy data path depends on the chosen learner. |
| V-trace as canonical project algorithm | Use only if selected off-policy design requires it. |
| Exactly-once distributed learner transaction matrix as a v1 prerequisite | Atomic local update/checkpoint semantics first. |
| Full action/update provenance chain for every ordinary dev run | Compact experiment provenance sufficient for the claim. |
| General gate compiler/receipt architecture | Small direct validators for concrete experiments. |
| Multi-layer authority/receipt hierarchy | Git + concise specs + tests/results. |
| Routine SHA ceremony for ordinary development | Git normally; hashes only where scientific external artifacts require them. |
| Branch-per-microstep / commit-per-edit | One coherent task PR. |
| Duplicate current-state/authority docs | Small canonical forward-looking docs. |
| Custom ownership/lease for ordinary development | One active writer per PR/worktree. |
| General async orchestrator before measured need | Thin GitHub/OCI/Builder path. |
| Legacy implementation compatibility as product requirement | Preserve only useful semantics/tests. |
| Handcrafted cycle/regime activation or chronological belief expiry | Learned representation/weights + current inputs. |
| Realized winning action as supervised correct-action label | Outcomes are consequences/evidence, not hindsight labels. |
| `HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION` as a required known-answer task | Superseded by the R12 survival-aware growth objective. |

---

## 5. FIRST VERTICAL-SLICE SCIENCE

The first R12 learnability slice should stay smaller than the old S1 program.

Required positive capabilities:

1. **Account-conditioned action** — same normalized market structure, different reachable account states, different appropriate actions.
2. **Delayed consequence credit** — an early action must be learned from later account consequence without hindsight labels.

Required controls:

- no-signal / zero-relation control;
- shuffled action-to-consequence control for delayed credit where applicable.

The learner algorithm itself remains open. The slice does **not** require V-trace, off-policy replay, a Teacher, a distributional target system, a database, or a full recovery transaction framework.

Success means frozen-policy behavior changes correctly under the preregistered task criteria. Loss reduction alone is not success.

---

## 6. OPEN R12 SCIENTIFIC DECISIONS

1. Exact causal normalization transform for K-line price channels and other scale-sensitive market inputs.
2. Exact minimal account-state schema and scale normalization.
3. Exact authority definition of the Survival / Path-Risk axis.
4. Exact ruin boundary beyond obvious account-ending/legal failure states.
5. Economic materiality / epsilon thresholds for Pareto comparison.
6. Exact uncertainty/confidence protocol used to establish domination.
7. Rule for choosing one Active Champion when multiple non-dominated policies remain and one operational choice is mandatory.
8. Whether/when survival risk must enter training rather than qualification only.
9. Which minimal learning algorithm should power the first vertical slice.
10. Whether later evidence justifies explicit learned StrategyMemory.
11. Whether later scale requires market-impact modeling.

---

## Closed Master decision: runtime human role

**Decision: human outside the runtime policy.**

The human selects asset/capital/research/deployment scope. Inside the selected single-asset sandbox, the Trader receives no human macro-direction signal and no asset identity token. Its market perception is derived from normalized relative K-line structure; frozen market representations feed the trainable Central Brain, which combines them with account state and trades autonomously.

---

## Reduction rule

When a requirement leaves this worksheet for a canonical R12 document, rewrite it purely as a forward-looking R12 requirement. Do not carry archaeology or legacy comparison into canonical R12 context.
