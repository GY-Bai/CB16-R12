# R12 VS-D — Bounded Horizon Accessibility Screen R5

Status: **PREREGISTERED — IMPLEMENTATION AND FORMAL RUN FORBIDDEN UNTIL THIS SPEC MERGES**

Authority class: `CURRENT_TASK_SCOPE_ONLY`.
Experiment role: `SCREENING`.
Machine-readable authority: `config/experiments/r12_vs_d_bounded_horizon_accessibility_screen_r5.json`.
Frozen preregistration SHA256: `f85afb41c66ec4b452185418626dabe51d0abadeb59dddc37cc89cb6c22e935e`.

## Question

After R3/R4 failed to qualify robust one-hour accessibility, is the fixed 64h N0 state more strongly aligned with one of three bounded longer forward-return horizons: 4h, 12h, or 24h?

This is a **screen**, not a qualification experiment. It may prioritize one candidate horizon for a later new experiment; it cannot qualify market information, profitability, or a final Trader.

## Frozen comparison

Use only flattened N0 and the exact R3 closed-form ridge probe (`lambda=1`). No MLP, actor, critic, FrozenSensory change, or RL training is part of R5.

All horizons use the same common chronology:

- January common fit: 655 first-consequence hours, 2020-01-03 17:00 through 2020-01-30 23:00 UTC;
- February common validation: 672 first-consequence hours, 2020-02-01 00:00 through 2020-02-28 23:00 UTC;
- March is forbidden; final holdout is forbidden.

For horizon `h`, target is:

`close[first_consequence_hour + h - 1] / open[first_consequence_hour] - 1`.

The N0 state uses only the 64 represented hours strictly before `first_consequence_hour`. Thus `h=1` reproduces the earlier one-hour target semantics while every horizon shares identical state timestamps.

## Horizons and controls

Screen exactly `1h, 4h, 12h, 24h`; `1h` is the fixed reference and cannot be selected as a new candidate.

For each horizon, fit the ridge probe on the true January target and eight no-fixed-point shuffled-January-target controls using seeds `16001..16008`. February targets and features are never shuffled or fitted.

## Dependence-aware uncertainty

Use one shared paired circular moving-block bootstrap across every horizon and control:

- February sample length: 672 hours;
- circular block length: 48 hours;
- 4096 replicates;
- bootstrap seed `17001`;
- one-sided lower-bound order index `204`.

The 4096 draws are uncertainty draws, not scientific replicates.

## Candidate rule

A longer horizon is **screen-eligible** only when all three paired bootstrap lower bounds are strictly greater than zero:

1. true correlation;
2. true correlation minus median shuffled-control correlation;
3. true correlation minus the 1h-reference true correlation.

If multiple horizons are eligible, select the one with the largest point true correlation; exact ties select the shorter horizon.

The only allowed promotions are `PROMOTE_H4_CANDIDATE_HYPOTHESIS`, `PROMOTE_H12_CANDIDATE_HYPOTHESIS`, or `PROMOTE_H24_CANDIDATE_HYPOTHESIS`. The promoted candidate still requires a new experiment before any qualification claim.

## Authority boundary

R5 uses adaptively reused Jan-Feb development after R3/R4 inspection. It is not fresh confirmation. A positive screen does not establish that the selected horizon is generally optimal or economically useful. A negative screen does not falsify unscreened horizons, nonlinear representations, market information, or full CB16.

No rescue: do not change horizon set, target formula, common windows, ridge lambda, controls, bootstrap, candidate rule, March boundary, or final-holdout boundary under R5.
