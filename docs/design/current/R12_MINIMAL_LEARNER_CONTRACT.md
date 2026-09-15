# R12 Minimal Learner Contract

Status: **MASTER-ALIGNED V1 CANDIDATE — FOR CONTROLLED LEARNABILITY FIRST**

Authority class: `CURRENT_DESIGN_CANDIDATE`; subordinate to `docs/authority/`.

Purpose: prove that the R12 closed loop can learn account-conditioned sequential behavior from economic consequences before adding replay, off-policy correction, PPO, distributed training, or a more elaborate critic.

This learner is a minimal scientific baseline, not a claim that it is the final historical/production learner.

## 1. Learning method

Use an **on-policy Monte Carlo policy gradient with a learned state-value baseline** (`REINFORCE + V(s) baseline`).

Do not use by default:

- replay buffers;
- importance sampling / V-trace;
- PPO clipping;
- target networks;
- Q-learning;
- SAC/DDPG;
- Teacher targets;
- hindsight action labels.

A policy generation collects a bounded batch of complete trajectories. The policy remains frozen while that batch is collected. The batch is used once for the update and is not replayed under later generations.

## 2. Policy state

The learner consumes the canonical runtime state:

\[
s_t=(Z_t,A_t,X_t),
\]

with \(X_t\) omitted when execution economics are constant.

The learner does not receive symbol identity, raw nominal price, future market information, realized future winner labels, or post-action account truth when choosing the action.

## 3. Hybrid action distribution

The semantic action remains:

\[
a_t=(d_t,r_t),
\]

where \(d_t\in\{SHORT,FLAT,LONG\}\) and \(r_t\in[0,1]\).

Factor the stochastic policy as:

\[
\pi_\theta(a_t\mid s_t)
=
\pi^d_\theta(d_t\mid s_t)
\pi^r_\theta(r_t\mid s_t,d_t)
\]

for non-FLAT directions.

### Direction head

Use a 3-way categorical distribution:

\[
d_t\sim\operatorname{Categorical}(p^{short}_t,p^{flat}_t,p^{long}_t).
\]

### Requested-risk head

For `LONG` and `SHORT`, use a Beta distribution with state- and direction-conditioned positive parameters:

\[
r_t\sim\operatorname{Beta}(\alpha_{t,d},\beta_{t,d}).
\]

For `FLAT`:

\[
r_t=0
\]

deterministically and no risk-density term is included.

The Beta support matches the bounded risk request and avoids generating an unbounded action that must later be clipped into `[0,1]`.

Policy parameters that define \(\alpha,\beta\) must use a numerically safe positive transform. Exact lower numerical floors are experiment constants.

The action log-probability is:

\[
\log\pi_\theta(a_t\mid s_t)
=
\log\pi^d_\theta(d_t\mid s_t)
+
\mathbf 1[d_t\neq FLAT]\log\pi^r_\theta(r_t\mid s_t,d_t).
\]

## 4. Economic return-to-go

Use the true net log-equity reward from Physics:

\[
r_t=\log\frac{W_{t+1}}{W_t}.
\]

For a completed controlled trajectory ending at \(T\), use the undiscounted return-to-go:

\[
G_t=\sum_{k=t}^{T-1}r_k.
\]

Because log-equity reward telescopes:

\[
G_t=\log\frac{W_T}{W_t}.
\]

The first controlled learner therefore does not require an arbitrary discount factor to define its economic objective.

## 5. Value baseline

A scalar critic estimates:

\[
V_\phi(s_t)\approx\mathbb E[G_t\mid s_t].
\]

Use the detached advantage estimate:

\[
\hat A_t=G_t-\operatorname{stopgrad}(V_\phi(s_t)).
\]

The value network is a variance-reduction baseline. It is not a promotion authority and does not define economic truth.

Critic regression target is the realized return-to-go:

\[
L_V=\frac{1}{N}\sum_t(V_\phi(s_t)-G_t)^2.
\]

## 6. Actor update

Use the policy-gradient objective:

\[
L_\pi
=-\frac{1}{N}\sum_t
\log\pi_\theta(a_t\mid s_t)\,\hat A_t.
\]

Actor and critic may use separate optimizer steps so that no arbitrary actor/critic scalarization weight is needed.

No entropy bonus is required in the baseline. If premature policy collapse is later demonstrated, exploration regularization becomes a new explicit experimental change rather than a hidden rescue.

## 7. On-policy generation boundary

For policy generation \(g\):

```text
freeze theta_g
-> collect complete controlled trajectories
-> compute rewards and return-to-go
-> update actor/critic
-> commit theta_{g+1}
-> future collection uses theta_{g+1}
```

Do not update actor parameters halfway through a trajectory that is declared to belong to one behavior generation.

Do not mix trajectories from older policy generations into the baseline actor update.

## 8. Terminal and truncation semantics

The Monte Carlo target requires the declared controlled trajectory to reach its scientific endpoint or true economic terminal state.

A process timeout, crash, logging cutoff, BPTT boundary, or arbitrary compute chunk is not a terminal reward event.

Incomplete trajectories must be resumed or excluded according to the preregistered run contract; they must not be silently converted into terminal samples.

The first learnability tasks should use bounded trajectories with known task endpoints so that full return-to-go can be computed without bootstrapped truncation logic.

## 9. Frozen-policy evaluation

Training diagnostics are not qualification.

After a declared training budget, freeze the candidate policy and evaluate under a preregistered action adapter.

The adapter must be chosen before results are inspected. A simple deterministic adapter candidate is:

- direction = highest-probability categorical direction;
- for non-FLAT direction, `requested_risk` = Beta mean \(\alpha/(\alpha+\beta)\);
- FLAT implies zero requested risk.

If stochastic deployment behavior is being studied instead, evaluate the frozen stochastic policy with a frozen seed/evidence protocol. Do not switch between stochastic and deterministic evaluation after seeing results.

## 10. First controlled learnability claims

The learner must first pass two positive tasks.

### A. Account-conditioned action

Hold market representation fixed while presenting distinct reachable account states for which the correct economic action differs.

Success requires the frozen policy to change behavior in the expected direction as account state changes.

### B. Delayed consequence credit

Construct a task where an early action changes a later account consequence and immediate outcome is insufficient to identify the correct early action.

Success requires the frozen policy to improve the early decision from later economic return, without realized-winner labels.

## 11. Required controls

At minimum include:

- no-signal / zero-relation control;
- shuffled action-to-consequence control for delayed credit where applicable.

A positive-task result that does not separate from the relevant control is not controlled learnability.

## 12. Evidence and failure semantics

The following are diagnostics only:

- lower actor loss;
- lower critic loss;
- non-zero gradients;
- parameter movement;
- checkpoint creation.

Scientific qualification requires preregistered frozen-policy behavioral evidence. Actor/critic loss is not required to decrease monotonically across on-policy generations: the policy and visited state/action distribution are changing, so a monotonic cross-generation loss curve is not a generic correctness criterion unless an experiment explicitly defines one.

For a valid implementation, `SCIENTIFIC_FAIL` means the frozen qualification gate was not met. The tested learner/configuration does not earn that qualification under the declared scope. This is not permission to rescue the same run, but neither is it automatically a falsification of the whole learner family or a diagnosis of the failing mechanism. Record qualification, inference, attribution, and promotion separately under `docs/authority/R12_EVIDENCE_SCOPE_AND_CLAIM_AUTHORITY.md`.

This minimal learner is an intentionally reduced candidate component. Its job is to provide scoped evidence about one simple learning spine and expose measurable capability gaps before richer composition. A PASS does not prove future composition; a FAIL blocks only the promotion that depends on the missed qualification unless a broader logical dependency is separately established.

## 13. Escalation rule

Only consider a more complex learner after the minimal baseline exposes a **measured capability gap, stability problem, or explicit interaction hypothesis** that the added mechanism is intended to address. A uniquely proven root cause is not required before research can continue, but generic complexity without a falsifiable reason is not authorized.

Examples:

- excessive Monte Carlo variance prevents controlled learnability -> test bootstrapped n-step/GAE as a new method;
- unstable large policy updates after controlled learnability is already established -> test PPO/trust-region stabilization as a new method;
- sample reuse becomes necessary and materially off-policy -> introduce replay plus the required correction mechanism;
- bounded-risk distribution itself is inadequate -> test a different action parameterization.

Complexity is added to solve measured failure, not because it is standard RL practice.

## 14. Still experiment-scoped

Do not silently freeze:

- Central Brain hidden width/depth;
- optimizer type and learning rate;
- batch trajectory count;
- trajectory length for each controlled task;
- Beta numerical parameter floor;
- actor/critic parameter sharing;
- deterministic versus stochastic evaluation adapter beyond the declared experiment;
- later historical learner architecture.
