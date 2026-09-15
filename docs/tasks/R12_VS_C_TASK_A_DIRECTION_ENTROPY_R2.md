# R12 VS-C — Task-A Direction-Exploration Attribution R2

Status: **PREREGISTERED — DO NOT RUN BEFORE IMPLEMENTATION MERGE**

Machine-readable authority:

`config/experiments/r12_vs_c_task_a_direction_entropy_r2.json`

## Objective

Test one mechanism only:

> Does a small entropy bonus on the three-way **direction categorical** materially prevent premature collapse to FLAT and stabilize Task-A account-conditioned behavior?

R1 showed that increasing balanced collection from 90 to 900 trajectories per generation reduced instability but was not sufficient under the preregistered gate. R2 therefore keeps the high-batch setting and changes only categorical direction exploration.

This is not a historical-profitability experiment and not a production hyperparameter search.

## Frozen common surface

Both R2 arms use exactly:

- the existing N0 market boundary;
- the existing fixed FrozenSensory seed and representation width;
- the existing Actor/Critic architecture;
- Adam actor LR `0.001` and critic LR `0.001`;
- undiscounted complete-trajectory return-to-go;
- the existing hybrid action log-probability semantics;
- **no entropy term on the Beta risk distribution**;
- no replay, PPO, GAE, target network, Teacher, scheduler or supervised target;
- Task-A batch size `900` per generation;
- `256` generations;
- paired seeds `1201..1208`;
- the same AccountTruths, AccountState, Permission and Physics;
- the same positive observation relation and zero-relation control;
- the same deterministic evaluation adapter and seed pass gate.

Within a seed, baseline and entropy arms and their matched controls must start from bitwise-identical Actor and Critic parameters. Collection RNG streams must use the same deterministic seed schedule across matched arms.

## Arms

### Baseline

```text
baseline_no_entropy
categorical direction entropy coefficient = 0.0
Beta risk entropy coefficient = 0.0
```

### Direction-entropy arm

```text
direction_entropy_005
categorical direction entropy coefficient = 0.005
Beta risk entropy coefficient = 0.0
```

Only the Actor objective changes:

```text
L_actor = -mean(log_pi(a_t | s_t) * A_hat_t)
          - lambda_dir * mean(H[pi_direction(. | s_t)])
```

where `lambda_dir` is `0` or `0.005` according to the arm.

The hybrid behavior action and `log_pi(a_t|s_t)` remain unchanged. FLAT still has deterministic risk zero and still carries no Beta log-density term.

## Why lambda = 0.005

This value is frozen before R2 implementation/results and is tied to the controlled Task-A reward scale rather than selected by a sweep.

For three directions, maximum categorical entropy is:

```text
log(3) = 1.0986122886681098
```

so the maximum entropy contribution is:

```text
0.005 * log(3) = 0.005493061443340549
```

Flattening an existing half-exposure under Task-A friction produces the known economic loss:

```text
-log(0.95) = 0.05129329438755058
```

Thus maximum entropy pressure is about `10.7%` of that explicit one-sided economic error. It is intended to resist premature categorical collapse without becoming the dominant objective.

No coefficient sweep is authorized under R2.

## Task A and control

Reuse the exact Task-A construction from current authority:

```text
SHORT_HALF -> desired target -0.5
FLAT       -> desired target  0.0
LONG_HALF  -> desired target +0.5
```

Flat market, price `100`, volume `10`, `kappa=0.10`, hard exposure limit `1`, nominal budget `1`.

Each generation contains `900` trajectories.

Positive:

```text
300 SHORT actual observed as SHORT
300 FLAT actual observed as FLAT
300 LONG actual observed as LONG
```

Control:

```text
all 3 x 3 actual-account x observed-account pairs
100 trajectories per pair
```

Reward always uses actual AccountTruth.

## Evaluation and seed gate

Evaluate PRE and POST using true AccountState and the unchanged deterministic adapter.

One seed passes iff both hold:

```text
direction_correct_count >= 3
target_exposure_mae <= 0.15
```

`all_flat_post` remains a mechanism diagnostic only.

## Aggregate attribution gate

Direction-exploration attribution is supported only if all hold:

```text
baseline control pass count <= 2
entropy positive pass count >= 6
entropy control pass count <= 2
entropy positive pass count - baseline positive pass count >= 3
```

Outcomes:

- `DIRECTION_EXPLORATION_SUPPORTED`
- `CONTROL_INVALID`
- `DIRECTION_ENTROPY_NOT_SUFFICIENT`

A gate miss is `SCIENTIFIC_FAIL` scoped to **L2 — optimization-mechanism attribution** for the frozen direction-exploration hypothesis. It says the tested categorical-direction entropy intervention did not support that mechanism strongly enough; it does not establish that controlled learnability or the broader learner family is impossible. It is not permission to tune entropy or increase training under the same identity.

## Implementation constraints

Prefer a minimal extension of the existing learner/policy surface:

- default learner semantics must remain entropy coefficient `0.0`;
- expose direction categorical entropy without changing sampled-action semantics;
- apply the bonus only to the direction distribution;
- Beta risk entropy must remain absent;
- generation snapshot, replay rejection, critic stop-gradient and complete-trajectory rules must remain unchanged;
- reuse the vectorized Task-A engine and max four one-thread seed workers.

Before formal Science, tests must establish:

1. coefficient `0.0` reproduces the existing actor objective exactly;
2. positive coefficient changes only the actor loss by `-lambda * mean(direction_entropy)`;
3. critic objective/gradient path is unchanged;
4. Beta risk parameters receive no direct entropy gradient when advantage is zero;
5. direction logits receive entropy gradient when the distribution is non-uniform;
6. all four learners for a paired seed begin bitwise-identical;
7. Task-A schedules remain exactly balanced at 900;
8. attribution truth table and failure taxonomy are mechanical;
9. the preregistered JSON is content-addressed by a fixed test hash.

## Formal runtime

Use the immutable Science lane, locked `uv.lock`, CPU only, one Torch thread per worker, maximum four seed workers. Formal implementation tests run first. No DSH is involved in the Science run.

Required artifacts:

- `experiment_spec.json`
- `RESULT.json`
- `REPORT.md`
