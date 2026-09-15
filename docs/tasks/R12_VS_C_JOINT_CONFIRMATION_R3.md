# R12 VS-C — Joint Learnability Confirmation R3

Status: **PREREGISTERED — DO NOT RUN BEFORE IMPLEMENTATION MERGE**

Machine-readable authority: `config/experiments/r12_vs_c_joint_confirmation_r3.json`

## Objective

Confirm on a fresh seed set that one fixed learner configuration can satisfy both previously isolated synthetic mechanisms at once:

1. **Task A — account-conditioned maintenance:** identical market, different AccountState, correct action must depend on the account.
2. **Task B — delayed consequence credit:** action at step 0 has zero immediate reward and must receive credit from the later gap consequence.

This is a confirmation experiment, not a new hyperparameter search and not a historical-profitability experiment.

## Frozen learner configuration

R3 uses one configuration for both tasks:

- Adam actor LR `0.001`, critic LR `0.001`;
- categorical-direction entropy coefficient `0.005`;
- **Beta-risk entropy coefficient `0.0`**;
- undiscounted complete-trajectory return-to-go;
- no replay, PPO, GAE, Teacher, target network or scheduler;
- same N0 normalization, FrozenSensory, Actor/Critic architecture, AccountState, Physics and deterministic evaluation adapter as prior VS-C authority;
- CPU only, deterministic algorithms, one Torch thread per worker, maximum four seed workers.

R2 already established attribution for `lambda_dir=0.005`; R3 does not compare coefficients and does not authorize a sweep.

## Fresh seeds

Formal seeds are frozen to:

`2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208`

They must not overlap the `1201..1208` seeds used by R0/R1/R2.

## Task A

Reuse the R2 B900 Task-A construction exactly, including the same positive and account-observation-zero-relation control, 256 generations, Physics, evaluation adapter and per-seed gate:

```text
direction_correct_count >= 3
target_exposure_mae <= 0.15
```

Confirmation requires all:

```text
positive seed pass count >= 7
control seed pass count <= 1
positive target MAE strictly better than matched control in >= 7 seeds
median PRE->POST target MAE improvement >= 0.15
```

## Task B

Reuse the R0 delayed-consequence Task-B construction exactly: B128, 256 generations, zero immediate step-0 reward, +/-10% delayed gap, and deterministic non-identity shuffled action-to-consequence control.

Per-seed gate remains:

```text
UP   -> LONG with requested_risk >= 0.5
DOWN -> SHORT with requested_risk >= 0.5
mean true delayed log growth >= 0.03
```

Confirmation requires all:

```text
positive seed pass count >= 7
control seed pass count <= 1
positive growth exceeds matched control by >= 0.02 in >= 7 seeds
median PRE->POST true delayed log-growth improvement >= 0.02
```

## Global outcome

`JOINT_LEARNABILITY_CONFIRMED` only if both Task A and Task B confirmation gates pass and both controls remain valid.

If either control exceeds its preregistered maximum, outcome is `CONTROL_INVALID`.

If controls remain valid but any confirmation condition fails, outcome is `JOINT_CONFIRMATION_FAILED` and top-level classification is `SCIENTIFIC_FAIL` scoped to **L3 — Controlled learnability confirmation**. This would reject only the fresh-seed joint-confirmation claim for the frozen learner configuration; it would not by itself decide historical-information or profitability layers. No rescue, threshold change, extra generations or coefficient tuning is allowed under R3.

## Formal runtime

Implementation tests must run first on the exact merge commit. Formal Science emits only:

- `experiment_spec.json`
- `RESULT.json`
- `REPORT.md`
