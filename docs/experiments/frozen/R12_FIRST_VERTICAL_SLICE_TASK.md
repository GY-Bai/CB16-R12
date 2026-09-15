# R12 First Vertical Slice — Implementation Task Contract

Status: **READY TO SCHEDULE — DO NOT EXPAND SCOPE**

## Objective

Implement the smallest CPU-only R12 closed loop that can prove controlled account-conditioned sequential learning before any historical economic claim.

The task is complete only when the system can:

```text
normalized synthetic K-line context
-> frozen sensory representation
-> Central Brain + AccountState
-> direction + requested_risk
-> Permission / Execution / Account Physics
-> next AccountState + log-equity reward
-> complete on-policy trajectory
-> minimal actor/critic update
-> frozen-policy behavioral evaluation
```

and pass the preregistered positive/control tests below.

## Existing authority

Implement the current forward-looking contracts literally:

- `docs/R12_CORE_OBJECTIVE_AND_QUALIFICATION_DRAFT.md`
- `docs/R12_ACTION_PERMISSION_ACCOUNT_PHYSICS.md`
- `docs/R12_MINIMAL_LEARNER_CONTRACT.md`
- `docs/TASK_REVIEW_PROTOCOL.md`
- `AGENTS.md`

If these documents conflict, stop and report the contradiction instead of inventing a replacement.

## Required implementation scope

### 1. Market-state adapter

Implement the current v1 deterministic causal normalization interface required by the synthetic tasks.

The slice must support the same normalized market representation being presented with different AccountStates.

Do not run the deferred normalization challenger comparison in this task.

### 2. Frozen sensory boundary

Provide one small frozen sensory module/interface that maps normalized market input to `Z_t`.

For the first controlled tests, this may be deliberately small and deterministic/preinitialized. The scientific requirement is the frozen boundary and data flow, not proving pretrained-organ superiority.

Required test: optimizer/gradient steps for the Central Brain must not change sensory weights.

### 3. AccountState

Expose the v1 decision state:

```text
signed_exposure
survival_cushion
new_risk_capacity
```

Raw monetary/account quantities remain in Physics.

### 4. Central Brain policy

Implement the minimal stochastic policy required by `R12_MINIMAL_LEARNER_CONTRACT.md`:

- categorical direction head: SHORT / FLAT / LONG;
- direction-conditioned Beta requested-risk head for non-FLAT actions;
- scalar state-value baseline.

Do not add Transformer blocks, recurrence, explicit StrategyMemory, Teacher targets, or extra prediction heads.

### 5. Action / Permission / Physics

Implement target-exposure semantics and the timing contract:

```text
observe through close_t
-> decide
-> execute at open_{t+1}
-> mark at close_{t+1}
```

Implement:

- nominal target exposure;
- cost-aware Permission projection;
- continuous-notional price-taker execution;
- fixed proportional transaction friction;
- actual post-cost account truth;
- account continuity;
- terminal/truncation separation.

No order book, market impact, lot size, venue minimum, exchange API, or complex liquidation engine.

### 6. Minimal learner

Implement strict on-policy Monte Carlo policy gradient with learned value baseline.

For one frozen policy generation:

```text
collect complete controlled trajectories
-> compute log-equity rewards
-> compute undiscounted return-to-go
-> update actor
-> update critic
-> commit next policy generation
```

No replay and no mixing old generations into the baseline update.

## Required controlled science

### Task A — Account-conditioned action

Construct a synthetic known-answer environment where:

- market representation is identical;
- two reachable AccountStates differ;
- their economically correct target actions differ.

The frozen post-training policy must distinguish the two account states in the preregistered expected direction.

### Task B — Delayed consequence credit

Construct a synthetic environment where:

- an early action affects a later account consequence;
- immediate reward is insufficient to identify the correct early action;
- no realized-winner supervised action label is supplied.

The frozen post-training policy must improve the early action from later economic return.

### Controls

Run at minimum:

1. matched no-signal / zero-relation control;
2. shuffled action-to-consequence control for Task B.

The positive task must separate from its relevant control under the frozen gate.

## Gate preregistration

Before executing qualification runs, write one compact machine-readable experiment spec containing:

- random seeds;
- training budget;
- trajectory count/length;
- learning rate(s);
- Central Brain dimensions;
- nominal exposure budget;
- transaction friction;
- exact positive-task success metric;
- exact control comparison;
- failure classification rule.

Do not choose thresholds after viewing results.

## Required semantic tests

At minimum test:

- no-lookahead market state;
- price/account scale invariance where applicable;
- frozen sensory weights;
- FLAT canonical semantics;
- requested_risk bounds and finite Beta parameters;
- Permission cannot create risk from FLAT;
- cost-aware permitted target remains feasible after cost;
- risk-reducing action remains available when new-risk capacity is exhausted;
- reversal pays full turnover cost;
- actual exposure is recomputed after cost;
- account continuity across trajectory chunks/checkpoint operations used by the task;
- terminal != truncation;
- return-to-go telescopes to log terminal-equity ratio;
- old policy trajectories are not silently replayed after generation update.

## Required evidence

Produce:

```text
RESULT.json
REPORT.md
experiment_spec.json
```

`RESULT.json` must include, for Task A, Task B, and each control:

- frozen configuration identity;
- seed-level behavioral metric(s);
- aggregate gate result;
- PASS / SCIENTIFIC_FAIL / EXECUTION_BLOCKED / HARDWARE_LIMIT / EVIDENCE_INSUFFICIENT / CONTRACT_MISMATCH;
- relevant test-suite result.

`REPORT.md` must distinguish:

```text
implementation correctness
controlled learnability evidence
open scientific limitations
```

Loss curves may be reported but cannot determine PASS.

## Allowed scope

- small local Python package/modules;
- CPU-only PyTorch or equivalent minimal tensor runtime;
- deterministic synthetic environments;
- targeted unit/integration tests;
- compact experiment artifacts;
- one coherent task PR.

## Do not change

Do not:

- open final holdout;
- use historical-market profitability as the task gate;
- run normalization Issue #1;
- add PPO, GAE, SAC, DDPG, V-trace, replay, Teacher, distributional targets, or multi-objective training;
- add databases, queues, distributed workers, GPU requirements, leases/fencing, or generalized orchestration;
- change canonical scientific semantics to rescue a failed run;
- optimize throughput beyond fixing a concrete task blocker.

## Done when

The task is done when:

1. the closed loop executes end-to-end on CPU;
2. semantic/invariant tests pass;
3. Task A and Task B are run exactly under the preregistered protocol;
4. matched controls are run;
5. evidence artifacts are produced;
6. the Builder posts a concise `BUILD_REPORT`;
7. the result is ready for Chat-SOL review without requiring hidden local context.
