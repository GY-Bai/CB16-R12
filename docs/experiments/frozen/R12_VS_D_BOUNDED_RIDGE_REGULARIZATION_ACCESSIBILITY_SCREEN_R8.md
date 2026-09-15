# R12 VS-D — Bounded Ridge Regularization Accessibility Screen R8

Status: **PREREGISTERED — IMPLEMENTATION AND FORMAL RUN FORBIDDEN UNTIL THIS SPEC MERGES**

Authority class: `CURRENT_TASK_SCOPE_ONLY`.
Experiment role: `SCREENING`.
Machine-readable authority: `config/experiments/r12_vs_d_bounded_ridge_regularization_accessibility_screen_r8.json`.
Frozen preregistration SHA256: `8e6b51a30b4f12ebeaf6ec8b6d4ad0092ada445e9cb1c1721d139e3fb12f6d50`.

## Question

R6 showed that 64h remained the strongest bounded context, while the exact lambda=1 linear probe still produced only a weak positive point relation. Before adding representation or learner complexity, does a bounded change in ridge regularization materially improve that same 64h/1h linear relation on adaptively reused February development?

R8 is a **screen**, not qualification and not a rescue of R3/R6/R7.

## Frozen comparison

Use exactly:

- BTCUSDT January-February 2020 only;
- January fit: 615 decisions, 2020-01-06 09:00 through 2020-01-31 23:00 UTC;
- February validation: 672 decisions, 2020-02-01 00:00 through 2020-02-28 23:00 UTC;
- 64h flattened N0 state, 320 features before constant-feature drop;
- one-hour arithmetic consequence target;
- January-only feature standardization;
- closed-form ridge probes at lambda `0.1, 1, 10, 100, 1000`.

Lambda `1` is the fixed reference. March and final holdout are forbidden.

## Controls and uncertainty

For every lambda use the same eight no-fixed-point January-target permutations, seeds `21001..21008`. February targets and features are never shuffled or fitted.

Use one shared paired circular 48h moving-block bootstrap across every lambda and control:

- 672 February hours;
- 4096 replicates;
- bootstrap seed `22001`;
- one-sided lower-bound order index `204`.

The bootstrap draws are uncertainty draws, not independent scientific replicates.

## Candidate rule

A non-reference lambda is screen-eligible only when all three paired lower bounds are strictly positive:

1. true February correlation;
2. true correlation minus median shuffled-control correlation;
3. true correlation minus the lambda=1 true correlation.

If multiple lambdas qualify, choose the highest point true correlation; exact ties prefer the lambda closest to 1 in `log10` distance, then the smaller lambda.

Any promotion is only a **ridge-lambda candidate hypothesis** for a new robustness/transfer experiment. It does not qualify historical market information, make February confirmatory, establish optimal regularization, or identify regularization as a unique root cause.

## No rescue

Do not change the lambda grid, Jan/Feb windows, context, target, control seeds, bootstrap, candidate rule, March boundary, or final-holdout boundary under R8.
