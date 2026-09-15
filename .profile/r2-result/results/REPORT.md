# R12 VS-C Task-A Direction-Entropy Attribution R2 — REPORT

- classification: **PASS**
- scientific outcome: **DIRECTION_EXPLORATION_SUPPORTED**
- commit: `eafac84e31de7c1fbc9a5fa09f455272a229c14e`
- engine: `vectorized-batched-v1`
- paired seeds: [1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208]

## Per-seed behavior

| seed | arm | lambda_dir | positive POST dirs | positive POST MAE | pos gate | control POST MAE | ctl gate | all-flat positive |
| --- | --- | ---: | --- | ---: | --- | ---: | --- | --- |
| 1201 | baseline_no_entropy | 0.000 | FLAT/FLAT/LONG | 0.170682 | FAIL | 0.333333 | FAIL | False |
| 1201 | direction_entropy_005 | 0.005 | SHORT/FLAT/LONG | 0.001650 | PASS | 0.333333 | FAIL | False |
| 1202 | baseline_no_entropy | 0.000 | SHORT/FLAT/FLAT | 0.168151 | FAIL | 0.333333 | FAIL | False |
| 1202 | direction_entropy_005 | 0.005 | SHORT/FLAT/LONG | 0.003520 | PASS | 0.333333 | FAIL | False |
| 1203 | baseline_no_entropy | 0.000 | SHORT/FLAT/LONG | 0.004747 | PASS | 0.333333 | FAIL | False |
| 1203 | direction_entropy_005 | 0.005 | SHORT/FLAT/LONG | 0.000093 | PASS | 0.333333 | FAIL | False |
| 1204 | baseline_no_entropy | 0.000 | SHORT/FLAT/LONG | 0.001090 | PASS | 0.333333 | FAIL | False |
| 1204 | direction_entropy_005 | 0.005 | SHORT/FLAT/LONG | 0.002633 | PASS | 0.333333 | FAIL | False |
| 1205 | baseline_no_entropy | 0.000 | FLAT/FLAT/FLAT | 0.333333 | FAIL | 0.333333 | FAIL | True |
| 1205 | direction_entropy_005 | 0.005 | SHORT/FLAT/LONG | 0.001105 | PASS | 0.333333 | FAIL | False |
| 1206 | baseline_no_entropy | 0.000 | FLAT/FLAT/LONG | 0.168014 | FAIL | 0.333333 | FAIL | False |
| 1206 | direction_entropy_005 | 0.005 | SHORT/FLAT/LONG | 0.002901 | PASS | 0.333333 | FAIL | False |
| 1207 | baseline_no_entropy | 0.000 | SHORT/FLAT/LONG | 0.010454 | PASS | 0.333333 | FAIL | False |
| 1207 | direction_entropy_005 | 0.005 | SHORT/FLAT/LONG | 0.002200 | PASS | 0.333333 | FAIL | False |
| 1208 | baseline_no_entropy | 0.000 | FLAT/FLAT/FLAT | 0.333333 | FAIL | 0.333333 | FAIL | True |
| 1208 | direction_entropy_005 | 0.005 | SHORT/FLAT/LONG | 0.001613 | PASS | 0.333333 | FAIL | False |

## Aggregate arms

### baseline_no_entropy

- positive seed passes: `3`
- control seed passes: `0`
- positive all-FLAT POST count: `2`
- positive median POST MAE: `0.168083`
- positive median PRE→POST MAE improvement: `0.333088`

### direction_entropy_005

- positive seed passes: `8`
- control seed passes: `0`
- positive all-FLAT POST count: `0`
- positive median POST MAE: `0.001925`
- positive median PRE→POST MAE improvement: `0.496000`

## Attribution gate

| component | value | condition |
| --- | ---: | --- |
| baseline_control_seed_pass_count | 0 | ok |
| entropy_positive_seed_pass_count | 8 | ok |
| entropy_control_seed_pass_count | 0 | ok |
| entropy_minus_baseline_positive_pass_count | 5 | ok |

Outcome: **DIRECTION_EXPLORATION_SUPPORTED**

Only categorical direction entropy differs between arms; Beta-risk entropy remains zero.
`all_flat_post` is diagnostic only and is not part of the gate.

## Limitations

- This is a controlled synthetic mechanism experiment, not a historical-profitability claim.
- A supported attribution would not promote lambda=0.005 to production authority.
- A gate miss does not authorize an entropy sweep under this experiment identity.
