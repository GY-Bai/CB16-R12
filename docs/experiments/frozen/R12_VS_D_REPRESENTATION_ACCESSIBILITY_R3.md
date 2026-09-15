# R12 VS-D — Representation Accessibility Diagnostic R3

Status: **PREREGISTERED — IMPLEMENTATION AND FORMAL RUN FORBIDDEN UNTIL THIS SPEC MERGES**

Authority class: `CURRENT_TASK_SCOPE_ONLY`.

Machine-readable authority:
`config/experiments/r12_vs_d_representation_accessibility_r3.json`

Frozen preregistration SHA256:
`04a2151aa822c685abbf744c854cd0ed8a87df48b633367f2043e7ed217f4111`

## Question

Before changing the RL learner, test whether the frozen one-hour consequence relation is linearly accessible from either of two existing representation surfaces: raw flattened N0 or the exact `FrozenSensory(z=32, seed=12012)` output.

This is a representation-accessibility diagnostic. It does not train a trading policy and cannot qualify market information, profitability, or production behavior.

## Chronology and data boundary

Use only BTCUSDT January-February 2020 archives already present in the research series. March and final holdout are forbidden reads.

- probe fit: `2020-01-03 17:00 UTC` through `2020-01-31 23:00 UTC` — exactly `679` decisions;
- diagnostic validation: all February 2020 — exactly `696 = 29 x 24` decisions.

February is reused development material, not fresh confirmation. Preprocessing parameters are learned from January only and then frozen.

## Target and surfaces

Target is the next-hour arithmetic return:

`close_next / open_next - 1`.

The two surfaces are:

1. N0 endpoint-anchored normalized tensor flattened from `64 x 5` to width `320`;
2. the existing deterministic `FrozenSensory` transform, width `32`, seed `12012`.

No learned sensory model is introduced. No raw symbol identity, account history, realized winner label, March data, or final holdout enters this task.

## Probe

Use one deterministic closed-form ridge family on both surfaces.

For each surface, fit January feature mean and population standard deviation; drop features with training std `<= 1e-12`; freeze that transform for February. Center the January target by its January mean. Solve:

`w = solve(X'X + I, X'(y - mean(y)))`.

The ridge coefficient is exactly `1.0`. Validation refitting, feature selection, nonlinear probes, and hyperparameter search are forbidden.

## Matched negative controls

Create eight deterministic controls by applying `controlled_tasks.no_fixed_point_permutation` to the January training target only, with seeds `4401..4408` and generation argument `0`.

The feature chronology and February true targets remain unchanged. The same control permutations are used for both representation surfaces. Controls are not independent scientific replicates; they are matched negative controls.

## Uncertainty and gates

Use 4096 paired bootstrap resamples of the 29 February UTC day blocks, with fixed bootstrap seed `53031` and zero-based 5% order statistic index `204`.

For each surface, qualification requires both:

- lower bound of true-probe February Pearson correlation `> 0`;
- lower bound of true-probe correlation minus the median of the eight shuffled-target probe correlations `> 0`.

`C_SENSORY_RETENTION_GAP` additionally requires N0 accessibility to qualify and the paired lower bound of `corr(N0) - corr(FrozenSensory)` to be strictly positive.

## Interpretation authority

If FrozenSensory accessibility qualifies, R3 may promote only a **post-sensory diagnostic route**; it does not prove that any downstream component is defective.

If N0 accessibility qualifies and the retention-gap claim also qualifies while FrozenSensory does not, R3 may promote only a **sensory-retention-gap hypothesis** for a new preregistered experiment.

A retention-gap result is surface-level evidence only. Because the compared surfaces are 320D and 32D, R3 cannot claim that the random projection or `tanh` uniquely destroyed information.

If neither accessibility gate qualifies, do not conclude that the market contains no information. The only justified statement is that this fixed linear ridge probe did not qualify accessibility under this reused Jan→Feb development split.

## No rescue

Do not change ridge lambda, split, target, shuffle seeds, bootstrap protocol, representation definitions, or gate thresholds after observing results. Any nonlinear probe, alternate horizon, representation replacement, or fresh-validation claim requires a new experiment identity.
