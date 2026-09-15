# R12 VS-D — Bounded Nonlinear Accessibility Diagnostic R4

Status: **PREREGISTERED — IMPLEMENTATION AND FORMAL RUN FORBIDDEN UNTIL THIS SPEC MERGES**

Authority class: `CURRENT_TASK_SCOPE_ONLY`.

Machine-readable authority:
`config/experiments/r12_vs_d_bounded_nonlinear_accessibility_r4.json`

Frozen preregistration SHA256:
`68e3c09f605625fd89e037aac4ad1bdf9a876e330ed984b490240dd12a472659`

## Question

R3 did not qualify fixed linear accessibility from N0, although its N0 point correlation was positive. R4 asks a narrower successor question: can one fixed, bounded nonlinear supervised probe extract a robust next-hour relation from the same N0 surface, and does it robustly improve over the exact R3 ridge baseline?

R4 does not modify R3. It is a new, adaptation-aware development diagnostic.

## Frozen data boundary

Reuse exactly the R3 chronology:

- January probe fit: `679` decisions;
- February reused-development validation: `696 = 29 x 24` decisions;
- March: forbidden;
- final holdout: forbidden.

## Fixed nonlinear probe

Input is flattened N0 only. Fit feature mean/std on January and freeze for February. Standardize the January target using January mean/std; inverse-transform predictions before evaluation.

The only nonlinear probe is:

`active_features -> Linear(64) -> Tanh -> Linear(64) -> Tanh -> Linear(1)`.

Training is deterministic CPU full-batch MSE for exactly `256` epochs with Adam `lr=0.001`, default betas `0.9/0.999`, `eps=1e-8`, zero weight decay, zero dropout, no early stopping and no validation monitoring.

Model seeds are `4501..4508`.

## Matched negative control

For every model seed, construct a control model from bitwise-identical initial parameters. Train it with the same schedule after a fixed-point-free permutation of January targets only. Control permutation seeds are `14501..14508`.

February targets are never shuffled. Validation is never used for model selection.

## Linear baseline and gates

Reproduce the exact R3 N0 ridge baseline (`lambda=1.0`) on the same January/February split.

For each MLP seed, use a paired 4096-replicate February UTC-day bootstrap. The nonlinear seed gate requires both the MLP true-correlation LCB and MLP-minus-matched-control correlation LCB to be strictly positive.

The nonlinear-over-linear seed gate requires the paired MLP-minus-R3-linear correlation LCB to be strictly positive.

Aggregate qualification requires at least `6/8` seed passes for the relevant claim plus the preregistered positive cross-seed median point estimates.

## Authority

If nonlinear accessibility qualifies but nonlinear-over-linear gain does not, only promote `PROMOTE_BOUNDED_NONLINEAR_ACCESSIBILITY_HYPOTHESIS`.

Only if both claims qualify may R4 promote `PROMOTE_SUPERVISED_SENSORY_CANDIDATE_HYPOTHESIS` for a separate future experiment. R4 does not integrate this MLP into the RL learner.

A FAIL does not establish general nonlinear inaccessibility or absence of market information. A PASS does not qualify historical market information, profitability, or production.

No architecture, epochs, learning rate, seeds, split, target, bootstrap, or gate may change after results.
