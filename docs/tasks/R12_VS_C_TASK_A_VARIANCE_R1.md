# R12 VS-C — Task-A Batch-Variance Attribution R1

Status: **PREREGISTERED — IMPLEMENT, THEN RUN IMMUTABLE SCIENCE**

Machine-readable authority:

`config/experiments/r12_vs_c_task_a_variance_r1.json`

This experiment asks one narrow question:

> Does reducing Monte-Carlo collection variance, without changing the learning algorithm or economic semantics, materially stabilize Task-A account-conditioned maintenance learning?

It is not a profitability test and it does not authorize historical-market execution or the final holdout.

## Objective

Compare two otherwise identical Task-A training arms:

```text
baseline_b90:    90 trajectories / generation
high_batch_b900: 900 trajectories / generation
```

Everything except per-generation collection count is frozen between the two arms.

The experiment is an attribution test, not a rescue of any earlier run. Its result belongs only to the new R1 identity above.

## Frozen semantics

Both arms use exactly:

- canonical N0 normalization;
- the existing frozen sensory projection;
- `AccountState = [signed_exposure, survival_cushion, new_risk_capacity]`;
- the existing Actor and ValueCritic architecture;
- categorical `SHORT / FLAT / LONG` plus direction-conditioned Beta risk;
- strict on-policy REINFORCE with learned state-value baseline;
- Adam actor LR `0.001` and critic LR `0.001`;
- undiscounted Monte-Carlo return-to-go;
- zero entropy bonus;
- no replay, PPO, GAE, Teacher, target network, scheduler or early stopping;
- 256 generations;
- the eight paired seeds `[1201..1208]` listed in the JSON;
- CPU-only deterministic execution;
- at most four seed workers, each with one Torch thread.

Do not change any of these after viewing results under this experiment identity.

## Task-A environment

Market input is one identical flat K-line window for every account case:

```text
one predecessor + 64 represented bars
OHLC = 100.0
volume = 10.0
```

It must pass through canonical N0 and FrozenSensory.

The three actual account truths are:

```text
SHORT_HALF: equity=1000, quantity=-5, mark=100 -> desired target=-0.5
FLAT:       equity=1000, quantity= 0, mark=100 -> desired target= 0.0
LONG_HALF:  equity=1000, quantity=+5, mark=100 -> desired target=+0.5
```

Physics is fixed:

```text
open_next = 100
close_next = 100
kappa = 0.10
permission_bisection_iterations = 24
nominal exposure budget = 1
hard exposure limit = 1
```

There is no price edge. Reward comes only from the economic consequence of changing exposure. Maintaining the current target has zero trading cost; unnecessary movement loses equity.

The desired targets are evaluation authority only. They are never supplied as learner labels.

## Arm construction

### baseline_b90

Positive generation:

```text
30 SHORT_HALF observed as SHORT_HALF
30 FLAT       observed as FLAT
30 LONG_HALF  observed as LONG_HALF
```

Matched zero-relation control generation:

```text
all 3 x 3 actual-account × observed-account pairs
10 trajectories per pair
```

### high_batch_b900

Positive generation:

```text
300 SHORT_HALF observed as SHORT_HALF
300 FLAT       observed as FLAT
300 LONG_HALF  observed as LONG_HALF
```

Matched zero-relation control generation:

```text
all 3 x 3 actual-account × observed-account pairs
100 trajectories per pair
```

The control preserves both account marginals while making actual account and policy-observed account exactly independent. Reward and Physics always use the actual AccountTruth.

## Why batch size is the only manipulated variable

The Actor loss is a mean over collected steps. With optimizer, learning rate, state distribution and objective frozen, increasing the balanced batch from 90 to 900 does not intentionally rescale the expected policy-gradient objective. It primarily reduces sampling noise in the stochastic action/reward estimate.

Therefore this experiment may support a variance-limited explanation only if the high-batch arm produces a material, controlled increase in stable known-answer behavior.

No claim is made that batch 900 should become the production training configuration.

## Evaluation

For each seed, arm and positive/control learner, record PRE and POST deterministic-adapter behavior on the true account states:

- direction for all three cases;
- requested risk;
- nominal target exposure;
- target absolute error;
- `direction_correct_count`;
- `target_exposure_mae`;
- mean one-step log growth;
- `all_flat_post`, true only when all three POST deterministic directions are FLAT.

A seed passes using the unchanged known-answer gate:

```text
direction_correct_count >= 3
target_exposure_mae <= 0.15
```

Losses and gradient norms may be recorded as diagnostics but are not scientific gates.

## Attribution gate

Compute mechanically from the committed JSON.

Variance support requires all of:

```text
baseline control pass count <= 2
high-batch positive pass count >= 6
high-batch control pass count <= 2
high-batch positive pass count - baseline positive pass count >= 3
```

Interpretation:

- **VARIANCE_SUPPORTED** — all conditions hold. Larger balanced collection materially stabilizes the same learner while controls remain negative.
- **CONTROL_INVALID** — either control exceeds its preregistered maximum. No variance attribution may be made.
- **VARIANCE_NOT_SUFFICIENT** — controls remain valid but one or more support conditions fail. Collection variance alone is not a sufficient explanation at the tested 10× batch.

`all_flat_post` is a mechanism diagnostic only; it is not part of the gate.

A complete experiment whose variance-support gate is not met is `SCIENTIFIC_FAIL` for this **L2 optimization-mechanism attribution hypothesis**: the preregistered claim that the tested batch-size variance change is sufficient is not supported. It does not mean the learner, Task A relation, or broader scientific program is impossible. It is not permission to change thresholds under the same identity.

## Implementation requirements

Use the merged vectorized controlled-task execution engine. Do not reconnect the old scalar per-trajectory collection loop for the formal R1 workload.

The formal implementation must:

1. validate the committed JSON and reject altered authority;
2. construct each arm only by replacing Task-A balanced batch counts with the preregistered values;
3. call `torch.manual_seed(seed)` before every learner construction so positive/control and B90/B900 learners start from the same seeded initialization;
4. preserve the existing deterministic collection-stream derivation from seed, task, arm and generation;
5. run independent seeds with at most four one-thread workers;
6. preserve returned evidence in preregistered seed order;
7. run the repository implementation test suite against an exact-content copy before producing a scientific verdict;
8. write only `experiment_spec.json`, `RESULT.json`, and `REPORT.md` under `CB16_RESULT_DIR`.

## Do not change

Do not change:

- AccountState fields;
- N0;
- FrozenSensory seed or width;
- Actor/Critic architecture;
- Beta semantics;
- deterministic evaluation adapter;
- reward or Permission/Physics formulas;
- friction;
- seeds;
- generation count;
- optimizer or learning rates;
- entropy bonus;
- controls;
- seed gate;
- attribution gate.

Do not add entropy regularization, behavior-policy floors, supervised targets, action-contrast targets or direct cost-sensitive objectives to R1. Those belong to a separate later experiment if variance is not sufficient.

## Failure taxonomy

- implementation or spec mismatch: `CONTRACT_MISMATCH`
- environment/runtime cannot complete the frozen run: `EXECUTION_BLOCKED`
- hardware cannot support the frozen run: `HARDWARE_LIMIT`
- complete valid run but variance-support gate misses: `SCIENTIFIC_FAIL` scoped to **L2 / the frozen variance-sufficiency hypothesis**

No same-identity rescue. Apply the failure-locality rule from `docs/R12_FAILURE_LOCALITY_AND_SCIENTIFIC_LAYERS.md`; this outcome does not erase upstream contract evidence or decide later historical/economic layers.
