# R12 VS-D — Objective-Balance / Entropy-Gradient Diagnostic R2

Status: **PREREGISTERED — IMPLEMENTATION AND FORMAL RUN FORBIDDEN UNTIL THIS SPEC MERGES**

Authority class: `CURRENT_TASK_SCOPE_ONLY`.

Machine-readable authority:
`config/experiments/r12_vs_d_objective_balance_entropy_gradient_r2.json`

Frozen preregistration SHA256:
`fc0ccf1040e8d6467b3ee520a425725f905286b66208a049f5af88193f8ec597`

## Question

Under the frozen VS-D R0 historical training objective, does the lambda-weighted categorical direction-entropy term systematically dominate the reward-driven categorical-direction policy-gradient term?

This is a **training-only optimization diagnostic**. It is not a lambda intervention, not a rescue of R0/R1, and not historical-information qualification.

## Data boundary

Read only the already-used BTCUSDT January-February 2020 training archives. March must not be opened by R2. Final holdout remains unopened.

Reuse the exact R0 N0 windows, flat-reset account semantics, one-hour consequence, true-context arm, shuffled-consequence arm, seeds `3101..3108`, and 256 generations.
## Frozen objective and measurement

Do not change learner structure, Adam `0.001/0.001`, direction entropy coefficient `0.005`, reward, sampling, generation count, or update semantics.

Before every original combined actor update, on the exact current `theta_g`, critic and sampled batch, measure without mutation:

- reward-driven categorical-direction PG loss and gradient;
- lambda-weighted categorical direction-entropy loss and gradient;
- full hybrid-action PG gradient as a secondary diagnostic;
- gradient-norm ratio and cosine;
- reward, advantage and direction-entropy scale.

The primary ratio is:

`weighted_entropy_grad_l2 / max(direction_pg_grad_l2, 1e-12)`.

Both primary terms are direction-only objectives, so their gradients are compared in the same actor parameter space; Beta-risk-only output rows receive no direct gradient from either primary term.

The diagnostic autograd calls must consume no RNG, must not leave persistent `.grad`, and must not alter the original learner update.
## Gate

For each positive-arm seed, compute all 256 generation log10 ratios, sort ascending, and take zero-based index `63`.

The seed gate passes only when that value is strictly greater than zero. This means entropy-gradient norm exceeds reward-driven categorical-direction PG norm in at least 193/256 generations.

Aggregate qualification requires:

- at least `6/8` positive-arm seeds pass; and
- the median across eight seed-level index-63 values is strictly greater than zero.

The shuffled arm is measured identically for characterization but does not enter the primary claim gate.

## Authority

PASS may promote only `PROMOTE_ENTROPY_SCALE_INTERVENTION_HYPOTHESIS` for a new experiment identity. It does **not** prove entropy caused R0/R1 failure.

FAIL means this particular systematic-gradient-dominance hypothesis did not earn qualification. Do not rescue by changing lambda, thresholds, seeds, generations, reward scaling, representation, or data under R2.

No outcome from R2 is historical-market qualification, profitability evidence, or final-holdout authority.