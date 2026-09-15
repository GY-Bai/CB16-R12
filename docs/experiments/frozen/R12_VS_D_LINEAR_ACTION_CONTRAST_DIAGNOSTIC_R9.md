# R12 VS-D — Linear Action-Contrast Diagnostic R9

Status: **PREREGISTERED — IMPLEMENTATION AND FORMAL RUN FORBIDDEN UNTIL THIS SPEC MERGES**

Authority class: `CURRENT_TASK_SCOPE_ONLY`.
Experiment role: `DIAGNOSTIC`.
Machine-readable authority: `config/experiments/r12_vs_d_linear_action_contrast_diagnostic_r9.json`.
Frozen preregistration SHA256: `0da7b886a53fcab1b4f0c3b80dfe66d3f9705d550b80d2bdbbb478ecc678a2ae`.

## Question

R3-R8 repeatedly produced weak positive return-correlation point evidence without robust promotion. Does the unchanged 64h N0 ridge(lambda=1) score nevertheless contain a decision-relevant **directional action contrast** when mapped only by its sign?

R9 changes the estimand, not the representation or probe family.

## Frozen construction

Use only BTCUSDT January-February 2020:

- January fit: 615 decisions, exactly the R6/R8 common fit window;
- February evaluation: 672 decisions, exactly the R6/R8 common validation window;
- representation: flattened 64h N0;
- score probe: closed-form ridge lambda `1.0`;
- target used for January fitting: one-hour arithmetic return;
- deterministic action: LONG when score > 0, SHORT when score < 0, FLAT only when score is exactly zero;
- per-step diagnostic contrast: `direction * realized_one_hour_return`, relative to FLAT = 0.

Lambda `0.1` from R8 is **not** adopted because it did not receive promotion. March and final holdout are forbidden.

## Controls and uncertainty

Fit eight matched controls using no-fixed-point shuffled January targets, seeds `23001..23008`, with identical features and lambda.

Use one paired circular 48h moving-block bootstrap over the 672 February hours:

- 4096 replicates;
- seed `24001`;
- one-sided lower-bound order index `204`;
- identical resamples for true and all controls.

## Gate

R9 supports the diagnostic claim only if both lower bounds are strictly positive:

1. mean true unit-direction action contrast;
2. true mean action contrast minus the median shuffled-control mean action contrast.

PASS may only promote `PROMOTE_LINEAR_ACTION_CONTRAST_CANDIDATE_HYPOTHESIS` to a new experiment.

This is not profitability: there are no transaction costs, funding, requested-risk sizing, holding-path/account effects, or learned FLAT threshold. February is adaptively reused and not confirmation.

## No rescue

Do not change chronology, context, lambda, sign mapping, introduce a deadband, change controls/bootstrap, add cost/account semantics, open March, or open final holdout under R9.
