# R12 VS-D — Evaluation Adapter Transfer Diagnostic R1

Status: **PREREGISTERED — FORMAL SCIENCE FORBIDDEN UNTIL IMPLEMENTATION MERGE**

Authority class: `CURRENT_TASK_SCOPE_ONLY`.

Machine-readable scientific authority:
`config/experiments/r12_vs_d_evaluation_adapter_transfer_r1.json`

Frozen preregistration SHA256:
`767e70bc90003bddf60fa085d15cf6bcbf00f27db1ccadeaeb615504fe3bb29a`

## Objective

Diagnose one transfer assumption left unresolved by VS-D R0: the learner is trained as a stochastic policy, while R0 qualified it through a deterministic `argmax(direction) + Beta mean` adapter.

Keep the full R0 training program frozen and compare that original adapter against a deterministic evaluator of expected one-step log-growth under the complete stochastic policy distribution.

This is **diagnostic**, not a rescue or confirmation run.
## Frozen axis

Do not change from parent R0:

- BTCUSDT Jan-Feb training / March reused development validation;
- N0 64h representation and `FrozenSensory(z=32, seed=12012)`;
- flat-reset account semantics, zero friction, funding off, one-hour consequence;
- actor/critic architecture, Adam LR `0.001/0.001`, direction entropy `0.005`;
- 256 generations, seeds `3101..3108`, true-context and shuffled-consequence arms;
- 31 UTC-day block bootstrap with 4096 replicates;
- final holdout remains unopened.

The **only scientific axis changed** is evaluation of the already-trained policy distribution.

## Diagnostic evaluator

For each validation state, compute direction probabilities from the categorical head. For SHORT/LONG, integrate one-step log-growth over the corresponding Beta risk distribution with deterministic 64-node Gauss-Legendre quadrature on `[0,1]`; FLAT contributes zero.

No validation action sampling is permitted. Quadrature is a deterministic numerical evaluator, not a new policy.
## Claims and gates

`C_ADAPTER_TRANSFER` asks whether policy-distribution evaluation improves the true-context advantage relative to both shuffled-control and PRE compared with the original argmax adapter. Its seed gate requires both paired day-block LCB deltas to exceed zero; aggregate support requires at least `6/8` seeds plus positive cross-seed medians.

`C_DISTRIBUTION_LEVEL_RELATION` asks whether the positive trained policy distribution itself shows a reused-development candidate relation beyond shuffled-control and PRE. This is explicitly **not historical-information qualification** because March has already been inspected by the parent series.

A gate miss is `GATE_NOT_MET / NOT_QUALIFIED`, not automatic falsification and not root-cause resolution.

## Promotion rules

If `C_ADAPTER_TRANSFER` passes, only promote a distribution-aligned evaluation hypothesis to a later fresh authorized validation design.

If it does not pass, do not rescue R1. The next diagnostic may isolate representation, objective, or horizon under a new preregistered identity.

Even if `C_DISTRIBUTION_LEVEL_RELATION` passes on reused March development, do not claim historical qualification, profitability, final-holdout success, or production authority.
## Implementation contract

The implementation PR may add:

- a v2 claim-authority validator used by new experiments;
- one R1 runner and its tests;
- one allowlisted Science entrypoint;
- deterministic policy-distribution evaluation and diagnostics;
- result/report rendering that emits `cb16.result.v2` fields required by canonical authority.

It must not modify the preregistered JSON, parent R0 runner, parent historical artifacts, frozen historical contracts, data manifest, learner semantics, or final holdout boundary.

Before formal Science execution, the implementation must pass the exact repository suite on the exact commit. Formal execution may begin only after implementation merge.

## Result authority

`RESULT.json` must separately report execution/validity, individual gate results, claim assessments, prior-evidence assessment, attribution status, and promotion decision. Unknown mechanism remains `UNRESOLVED`.
