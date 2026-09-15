# R12 VS-C Joint Learnability Confirmation R3 — REPORT

- classification: **PASS**
- scientific outcome: **JOINT_LEARNABILITY_CONFIRMED**
- commit: `465c8c916d0f67472b1575e177e8e24b1e29a4cf`
- fresh seeds: [2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208]

## Per-seed

| seed | A pos/ctl | A POST MAE pos/ctl | B pos/ctl | B POST growth pos/ctl |
| --- | --- | --- | --- | --- |
| 2201 | PASS/FAIL | 0.003346/0.333333 | PASS/FAIL | 0.087021/-0.000975 |
| 2202 | PASS/FAIL | 0.002141/0.333333 | PASS/FAIL | 0.086943/-0.001600 |
| 2203 | PASS/FAIL | 0.000759/0.333333 | PASS/FAIL | 0.086850/-0.001295 |
| 2204 | PASS/FAIL | 0.002192/0.333333 | PASS/FAIL | 0.087270/0.000000 |
| 2205 | PASS/FAIL | 0.003332/0.333333 | PASS/FAIL | 0.086913/-0.001247 |
| 2206 | PASS/FAIL | 0.002800/0.333333 | PASS/FAIL | 0.087342/-0.000903 |
| 2207 | PASS/FAIL | 0.001539/0.333333 | PASS/FAIL | 0.087085/-0.001140 |
| 2208 | PASS/FAIL | 0.000635/0.333333 | PASS/FAIL | 0.087226/-0.001292 |

## Joint gate

### task_a

- positive_seed_pass_count: `8`
- control_seed_pass_count: `0`
- positive_median_pre_to_post_target_mae_improvement: `0.4982339726828743`
- paired_positive_target_mae_strictly_better_than_control_count: `8`
- passed: **True**

### task_b

- positive_seed_pass_count: `8`
- control_seed_pass_count: `0`
- positive_median_pre_to_post_true_delayed_log_growth_improvement: `0.08816077539895895`
- paired_positive_growth_minus_control_count: `8`
- paired_positive_growth_minus_control_min_observed: `0.08727025990372532`
- passed: **True**

Controls valid: **True**

Outcome: **JOINT_LEARNABILITY_CONFIRMED**

## Limitations

- Synthetic confirmation only; not a historical-profitability claim.
- R3 does not promote lambda=0.005 to production authority.
- No coefficient sweep, threshold rescue or extra generations are authorized under R3.
