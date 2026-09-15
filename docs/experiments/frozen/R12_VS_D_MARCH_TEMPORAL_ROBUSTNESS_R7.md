# R12 VS-D — March Temporal Robustness of the 64h Linear Hint R7

Status: **PREREGISTERED — IMPLEMENTATION AND FORMAL RUN FORBIDDEN UNTIL THIS SPEC MERGES**

Authority class: `CURRENT_TASK_SCOPE_ONLY`.
Experiment role: `ROBUSTNESS`.
Machine-readable authority: `config/experiments/r12_vs_d_march_temporal_robustness_r7.json`.
Frozen preregistration SHA256: `31862bd6a50fcaa0ab9dc4379073d10e5ec55245a928fea7481d3e62bc4c8416`.

## Question

R3–R6 repeatedly left a weak positive one-hour linear point estimate on the 64h N0 surface without robust qualification. R7 asks a narrower question:

> Does the exact simple January-fitted 64h N0 ridge relation retain positive true/control separation when evaluated on **March 2020**, with no refit or method change?

March was already inspected in R0/R1. Therefore R7 is **robustness on adaptively reused development**, not confirmation and not qualification.

## Frozen probe

Fit only on the R6-style January window:

- consequences: `2020-01-06 09:00` through `2020-01-31 23:00 UTC`;
- 615 decisions;
- N0 represented context = 64h (65h raw window);
- flattened input width before constant drop = 320;
- January feature mean/population-std only;
- closed-form ridge `lambda=1`;
- target = same-hour `close/open - 1` for the consequence hour.

Fit eight matched controls by shuffling only the January target with no-fixed-point seeds `18001..18008`. No February or March target enters fitting.

## March evaluation

Evaluate the frozen true and control probes on all March 2020 consequence hours:

- `2020-03-01 00:00` through `2020-03-31 23:00 UTC`;
- 744 decisions;
- 31 complete UTC day blocks;
- early-March states may use trailing February bars strictly before the consequence hour;
- March targets are evaluation-only.

The allowed source archives are January, February, and March 2020 BTCUSDT only. Final holdout remains forbidden.

## Robustness gate

Use one shared paired bootstrap of the 31 UTC March day blocks:

- 4096 replicates;
- seed `20001`;
- one-sided lower-bound order index `204`.

The robustness gate passes only if both are strictly positive:

1. March true-probe Pearson-correlation LCB;
2. March true minus median shuffled-control correlation LCB.

A PASS may only promote `PROMOTE_TEMPORAL_ROBUSTNESS_CANDIDATE_HYPOTHESIS` for a later independent design. It does **not** make March fresh, qualify historical-market information, establish profitability, or authorize final holdout.

A miss means temporal robustness was not established for this exact January-fitted probe. It does not prove a regime shift, absence of market information, or general temporal instability.

## Exposure history

March has already been used as development validation in R0 and re-used in R1. R7 was designed after those March results and after R3–R6 Jan-Feb diagnostics were inspected. This adaptation history is part of the result authority and cannot be reset by the R7 experiment ID.

No rescue: do not change the January fit, March window, context, target, ridge lambda, control seeds, bootstrap, gate, or final-holdout boundary under R7.
