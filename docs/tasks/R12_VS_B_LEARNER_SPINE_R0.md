# R12 VS-B — Frozen Sensory + Policy/Critic + On-Policy Learner Spine

Status: **READY TO BUILD — IMPLEMENTATION CORRECTNESS ONLY**

This task implements the smallest CPU-only learner spine that sits on top of the already-merged VS-A market/account/permission/physics layer.

It does **not** run or qualify Task A / Task B scientific learnability yet. Those experiments belong to the next VS stage after this implementation is reviewed.

## Objective

Implement exactly this boundary:

```text
canonical normalized market tensor [L,5]
-> frozen sensory representation Z_t
+ AccountState [signed_exposure, survival_cushion, new_risk_capacity]
-> Actor
   -> Categorical(SHORT, FLAT, LONG)
   -> direction-conditioned Beta(requested_risk) for non-FLAT
-> Critic V(s)
-> complete trajectory records
-> undiscounted Monte-Carlo return-to-go
-> one strict on-policy actor update
-> one critic update
-> next generation id
```

The task is complete when the above can be exercised deterministically on CPU and the semantic regression tests below pass.

## Existing authority / interfaces

Use the current `main` implementations and forward-looking contracts literally:

- `science/cb16_science/vslice/contracts.py`
- `science/cb16_science/vslice/market.py`
- `science/cb16_science/vslice/physics.py`
- `docs/R12_MINIMAL_LEARNER_CONTRACT.md`
- `docs/R12_FIRST_VERTICAL_SLICE_TASK.md`
- `docs/TASK_REVIEW_PROTOCOL.md`
- `AGENTS.md`

Do not modify VS-A formulas or reinterpret its semantics.

## Required semantics

### 1. Runtime state

The learner state is

```text
s_t = (Z_t, A_t)
```

for this controlled slice. `X_t` is omitted because execution economics are fixed by `PhysicsConfig`.

`A_t` is exactly the existing 3-vector, in this order:

```text
[signed_exposure, survival_cushion, new_risk_capacity]
```

Do not add raw equity, raw price, position quantity, symbol, asset identity, entry price, PnL history, holding time, previous action, drawdown, trade count, or any other account/history feature.

The Actor/Critic must never receive raw OHLCV in parallel with normalized market state.

### 2. Frozen sensory boundary

Implement one small deterministic `FrozenSensory` module whose sole input is the canonical normalized market tensor with shape `[B, L, 5]` or a single `[L, 5]` sample.

For VS-B use a deterministic fixed random projection, not a trainable organ:

- flatten the normalized market context to `L*5`;
- project to `z_dim = 32`;
- projection matrix is initialized deterministically from a local fixed seed `12012`;
- scale entries so projection variance is numerically reasonable (for example by `1/sqrt(L*5)`);
- register the projection as a buffer or otherwise make it non-trainable;
- sensory exposes zero trainable parameters;
- sensory output is deterministic for identical input;
- sensory output must be finite or fail explicitly.

A simple bounded pointwise activation such as `tanh` after projection is allowed and should be frozen as part of this module.

The fixed random projection is only a controlled placeholder for the frozen sensory role. Do not claim it is a pretrained market model.

### 3. Central Brain Actor

Implement a small CPU MLP Actor with experiment-scoped default hidden width `64` and two hidden layers.

Input dimension is exactly:

```text
z_dim + 3
```

Use a simple stable activation such as `tanh`.

Actor outputs:

1. `direction_logits` with shape `[..., 3]` in the fixed semantic order:

```text
index 0 -> SHORT
index 1 -> FLAT
index 2 -> LONG
```

2. Direction-conditioned Beta parameters for SHORT and LONG only.

A convenient output layout is four raw scalars per state:

```text
short_alpha_raw
short_beta_raw
long_alpha_raw
long_beta_raw
```

Transform them with a numerically stable positive transform:

```text
alpha = softplus(raw) + beta_floor
beta  = softplus(raw) + beta_floor
```

For VS-B freeze `beta_floor = 1.0` as an implementation constant. It is experiment-scoped, not permanent architecture authority.

All logits and Beta parameters must be finite. Beta parameters must be strictly positive.

Do not add entropy heads, confidence heads, return heads, Teacher heads, predictive heads, recurrence, Transformer blocks, or StrategyMemory.

### 4. Stochastic action semantics

The semantic action remains the existing `NominalAction(direction, requested_risk)`.

Use:

```text
d ~ Categorical(direction_logits)
```

For `FLAT`:

```text
requested_risk = 0 exactly
log_prob = categorical_log_prob only
```

For SHORT/LONG:

```text
requested_risk ~ Beta(alpha_d, beta_d)
log_prob = categorical_log_prob + beta_log_prob
```

Do not generate an unbounded Gaussian and clip it into `[0,1]`.

Do not silently clip sampled Beta values. If the tensor runtime returns a non-finite or out-of-domain value, fail explicitly.

Expose one typed action sample record containing at minimum:

```text
direction
requested_risk
log_prob
```

and, for diagnostics, the selected Beta alpha/beta when direction is non-FLAT.

### 5. Deterministic frozen-policy adapter

Implement the preregistered deterministic evaluation adapter from the learner contract:

```text
direction = argmax(direction_logits)
FLAT -> requested_risk = 0
SHORT/LONG -> requested_risk = alpha / (alpha + beta)
```

Tie behavior must be deterministic under the tensor runtime's ordinary first-index argmax rule. Do not add random tie-breaking.

This adapter is evaluation-only. Do not use it as the stochastic training behavior policy.

### 6. Critic

Implement a separate scalar `ValueCritic` MLP with the same state input `Z_t + AccountState`.

Use separate parameters from the Actor; no actor/critic parameter sharing in VS-B.

Output exactly one finite scalar per state:

```text
V_phi(s_t)
```

The critic is a variance-reduction baseline only. It is not economic truth or promotion authority.

### 7. Trajectory records

Implement a minimal complete-trajectory representation.

Each step must retain enough information to recompute the Actor log-probability under the same frozen generation before the update. At minimum retain:

```text
state tensor (or lossless inputs required to reconstruct it)
semantic action: direction + requested_risk
reward
terminal
truncated
generation_id
```

Do not treat a crash, timeout, logging cutoff, or arbitrary chunk end as terminal.

A trajectory accepted by the Monte-Carlo update must be:

- complete at its declared scientific endpoint or true terminal;
- `truncated == false` for every accepted terminal endpoint in VS-B;
- internally from exactly one `generation_id`.

Do not store replay priorities, behavior-policy correction weights, Q targets, Teacher labels, or realized-winner labels.

### 8. Return-to-go

For a complete trajectory compute:

```text
G_t = sum_{k=t}^{T-1} reward_k
```

with no discount factor.

Provide a tested helper using reverse cumulative sum.

For rewards produced by VS-A Physics, verify in a known-answer test that the cumulative return telescopes to:

```text
sum_t r_t == log(W_T / W_0)
```

within normal floating-point tolerance.

### 9. Strict on-policy generation boundary

Implement one learner object with integer `generation_id`, starting at `0`.

The required lifecycle is:

```text
freeze actor theta_g
collect complete trajectories carrying generation_id = g
finish collection
recompute log_prob under unchanged theta_g
compute G_t
compute detached advantage = G_t - stopgrad(V_phi(s_t))
actor optimizer step
critic optimizer step
mark consumed batch used
increment generation_id to g+1
```

The learner must reject, with an explicit contract error:

- a trajectory/batch whose generation id differs from current learner generation;
- mixed-generation trajectories in one update;
- a batch that has already been consumed;
- a truncated/incomplete trajectory;
- an empty update batch;
- non-finite rewards, returns, values, log-probabilities, actor loss, or critic loss.

There is no replay. Once a batch is consumed it cannot be used again.

Do not update the Actor midway through a trajectory declared to belong to generation `g`.

### 10. Losses

Actor advantage:

```text
A_hat_t = G_t - stopgrad(V_phi(s_t))
```

Actor loss:

```text
L_actor = -mean(log_pi(a_t|s_t) * A_hat_t)
```

Critic loss:

```text
L_value = mean((V_phi(s_t) - G_t)^2)
```

Actor and Critic must use separate optimizer instances and separate optimizer steps.

No arbitrary actor/critic weighted scalarization is allowed.

No entropy bonus, PPO clipping, GAE, n-step bootstrap, target network, replay, V-trace, importance sampling, SAC, DDPG, Q-learning, or Teacher target.

### 11. Optimizer defaults

For implementation tests, use CPU `torch.optim.Adam` by default.

Freeze experiment-scoped defaults:

```text
actor_lr = 3e-4
critic_lr = 1e-3
```

These are not scientific qualification hyperparameters for Task A/B; the next stage will preregister those experiments explicitly.

Do not add schedulers or adaptive rescue logic.

## Dependency / package rule

Use CPU PyTorch.

If the repository still lacks project dependency metadata, add the smallest standard `pyproject.toml` needed for this science package and a reproducible `uv.lock` if the current UV tool can generate it normally.

Required direct runtime dependencies are only what this slice actually needs (currently NumPy + PyTorch). Do not add RL frameworks.

Use the existing Builder UV/cache path exposed by the execution environment. Ordinary package installation is a Builder task, not an Infra blocker.

Do not modify Builder Orchestrator, dispatcher, workflows, runner configuration, session registry, sandbox policy, or credential plumbing.

## Allowed implementation files

Prefer additions under:

```text
science/cb16_science/vslice/
```

Suggested modules:

```text
sensory.py
policy.py
trajectory.py
learner.py
```

Small package exports may be updated as required.

Add focused tests under `tests/`.

Dependency metadata at repository root is allowed only as required above.

## Do not change

Do not modify:

- VS-A normalization formulas;
- VS-A Permission feasibility predicate;
- VS-A bisection semantics;
- VS-A account transition formulas;
- VS-A reward formula;
- Builder Orchestrator / dispatcher / GitHub Actions workflows;
- historical data or final holdout logic;
- normalization ablation tasks;
- Task A / Task B scientific experiment thresholds;
- any scientific result artifact claiming learnability.

Do not implement synthetic Task A/B environments in this task beyond tiny unit-test fixtures needed to exercise the learner API.

## Required tests

At minimum add tests proving all of the following.

### Frozen sensory

1. Same normalized input -> bitwise-identical sensory output on CPU.
2. Shape `[L,5]` and `[B,L,5]` are handled explicitly; malformed shapes fail.
3. Sensory has zero trainable parameters.
4. Sensory projection buffer is unchanged after a valid Actor/Critic learner update.
5. Non-finite normalized inputs or sensory output fail explicitly.

### State boundary

6. Actor/Critic input dimension is exactly `z_dim + 3`.
7. Changing AccountState with identical market representation changes the state vector only in the final 3 account coordinates.
8. No raw OHLCV/account monetary fields are present in the public learner state API.

### Actor / policy distribution

9. Direction semantic order is SHORT, FLAT, LONG.
10. Direction logits and all Beta parameters are finite.
11. All Beta alpha/beta values are > 0 and obey the frozen floor.
12. FLAT sample returns requested_risk exactly 0 and excludes Beta density from log_prob.
13. Non-FLAT sampled requested_risk lies in `[0,1]` and total log_prob equals categorical + selected Beta density.
14. Deterministic adapter uses argmax direction and Beta mean exactly.
15. Invalid/non-finite distribution outputs fail closed.

### Critic

16. Critic returns exactly one finite scalar per state.
17. Actor and Critic share no parameter objects.

### Trajectory / returns

18. Reverse cumulative return-to-go has a hand-computed known answer.
19. VS-A log-equity rewards telescope to `log(W_T/W_0)` in a multi-step known-answer trajectory.
20. Terminal and truncation remain distinct.
21. Empty, incomplete, truncated, mixed-generation, or non-finite trajectories are rejected.

### Generation boundary / update

22. Learner generation begins at 0.
23. During collection there is no Actor optimizer step and Actor parameters remain unchanged.
24. One valid completed current-generation batch can update Actor and Critic.
25. The consumed batch is marked used and cannot be updated twice.
26. After the valid update `generation_id` increments exactly once.
27. Old-generation trajectories are rejected after increment.
28. Sensory buffer remains unchanged through the update.
29. Actor advantage uses a detached Critic baseline: Actor backward must not populate Critic gradients from `L_actor`.
30. Critic regression target is realized undiscounted return-to-go.
31. No replay buffer or off-policy correction object is introduced.

### Existing-suite regression

32. All existing VS-A and repository tests remain green.

## Required evidence

Builder PR report must include:

```text
files changed
PyTorch/NumPy/UV versions actually used
focused test command + result
full test command + result
CPU-only confirmation
whether dependency metadata was added
known unresolved implementation issue, if any
```

No `RESULT.json`, `REPORT.md`, or scientific PASS claim is required in VS-B because no controlled learnability experiment is authorized here.

## Done when

VS-B is done only when:

1. the frozen sensory / Actor / Critic / trajectory / learner spine exists on CPU;
2. all 32 semantic test requirements are represented by concrete regression tests;
3. the strict generation boundary rejects replay/stale/truncated data;
4. Actor/Critic can perform one valid update without modifying sensory;
5. VS-A regression suite remains green;
6. Builder posts a concise `BUILD_REPORT` for Chat-SOL review.
