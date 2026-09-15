# R12 VS-D — Bounded Context-Length Accessibility Screen R6

Status: **PREREGISTERED — IMPLEMENTATION AND FORMAL RUN FORBIDDEN UNTIL THIS SPEC MERGES**

Authority class: `CURRENT_TASK_SCOPE_ONLY`.
Experiment role: `SCREENING`.
Machine-readable authority: `config/experiments/r12_vs_d_bounded_context_length_accessibility_screen_r6.json`.
Frozen preregistration SHA256: `3e4706036bb7227aff885ec768d1c6a1db32e17e5d4f4aed997c42c091d3c8dd`.

## Question

R5 found no 4h/12h/24h target-horizon candidate and left 1h as the strongest point estimate. Before adding learner or representation machinery, R6 asks whether the **64h N0 state window itself** is a poor bounded MVP timescale compared with 16h, 32h, or 128h.

This is a screen, not a qualification experiment.

## Frozen comparison

Use the same one-hour consequence target and one fixed closed-form ridge family (`lambda=1`). The only scientific factor varied is N0 represented context length:

- 16h -> raw window 17h -> 80 pre-drop features;
- 32h -> raw window 33h -> 160 pre-drop features;
- 64h -> raw window 65h -> 320 pre-drop features (**reference**);
- 128h -> raw window 129h -> 640 pre-drop features.

All states end strictly before the one-hour consequence hour.

## Common chronology

To prevent sample-window differences from masquerading as context effects, every context uses identical consequence timestamps:

- January fit: `2020-01-06 09:00` through `2020-01-31 23:00` UTC, 615 decisions;
- February validation: `2020-02-01 00:00` through `2020-02-28 23:00` UTC, 672 decisions;
- March is forbidden;
- final holdout is forbidden.

January supplies context-specific feature mean/population-std and probe fitting. February is never fitted.

## Controls

For every context, fit eight matched no-fixed-point shuffled-January-target controls using seeds `18001..18008`. For a given seed, the **same target permutation** is used across all four contexts.

This controls one source of fitting noise but does not solve dimension-dependent regularization; that remains an explicit unresolved alternative because 16h/32h/64h/128h have different feature widths.

## Dependence-aware uncertainty

Use one shared paired circular moving-block bootstrap across all contexts and controls:

- sample length: 672 February hours;
- circular block length: 48 hours;
- 4096 replicates;
- bootstrap seed `19001`;
- one-sided lower-bound order index `204`.

Bootstrap draws are uncertainty draws, not independent scientific replicates.

## Candidate rule

A non-reference context (16h, 32h, or 128h) is screen-eligible only if all three preregistered lower bounds are strictly positive:

1. its true February correlation;
2. true minus median shuffled-control correlation;
3. true minus 64h-reference correlation under the same resamples.

If multiple contexts are eligible, select the one with the highest point true correlation; exact ties select the shorter context.

Allowed promotions are only:

- `PROMOTE_CONTEXT16_CANDIDATE_HYPOTHESIS`;
- `PROMOTE_CONTEXT32_CANDIDATE_HYPOTHESIS`;
- `PROMOTE_CONTEXT128_CANDIDATE_HYPOTHESIS`.

Any promoted candidate requires a new experiment before qualification.

## Authority boundary

R6 reuses Jan-Feb after extensive prior inspection and is therefore not fresh confirmation. PASS would not prove the selected context is globally optimal, would not prove context length is the unique bottleneck, and would not qualify historical-market information or profitability. FAIL would not falsify unscreened context lengths, nonlinear representations, richer final sensory composition, or full CB16.

No rescue: do not change context set, one-hour target, common windows, ridge lambda, controls, bootstrap, selection rule, March boundary, or final-holdout boundary under R6.
