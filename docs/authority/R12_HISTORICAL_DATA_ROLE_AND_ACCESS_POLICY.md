# R12 Historical Data Role and Access Policy

Status: **CURRENT_CANONICAL — HISTORICAL DATA ROLE / ACCESS GOVERNANCE**

Authority class: `CURRENT_CANONICAL`.
Machine-readable policy: `config/r12_historical_data_role_policy_v1.json`.

## 1. Purpose

Raw historical-data availability is not scientific access authority. R12 must know a dataset/split's role **before** a formal protocol may read it.

This policy closes a governance gap without inventing a new holdout boundary.

## 2. Current explicit assignments

For the VS-D research series, only these BTCUSDT monthly archives have an assigned scientific role:

| Month | Current role | Exposure status |
| --- | --- | --- |
| 2020-01 | `EXPOSED_DEVELOPMENT` | repeatedly used for training/diagnostics |
| 2020-02 | `EXPOSED_DEVELOPMENT` | repeatedly used for training/diagnostics/development evaluation |
| 2020-03 | `EXPOSED_DEVELOPMENT` | inspected in R0/R1 and reused for R7 robustness |

These months are not fresh evidence merely because a later experiment has a new identity.

## 3. Default-deny rule

Any historical market slice not explicitly assigned by current canonical authority is `UNASSIGNED_PROTECTED`.

`UNASSIGNED_PROTECTED` means **do not read it in formal science yet**. It does **not** mean the slice is the final holdout, statistically independent confirmation, or permanently reserved.

A new experiment spec, branch, issue, commit, or agent decision cannot self-assign a data role. A separate reviewed canonical policy change must assign the role before protocol preregistration.

## 4. Final holdout

This repository currently does **not** assign a calendar range to `FINAL_HOLDOUT` in current canonical authority. Do not infer one from chronology, the raw download range, old filenames, or data availability.

Opening a final holdout requires later explicit canonical authority. Until then, the existing invariant remains: final-holdout access is unauthorized.

## 5. Role changes and exposure history

Assigning a previously unassigned slice to development or transfer use does not erase any prior human/model exposure. Adaptive exposure history remains part of the evidence record.

Likewise, already exposed January-March 2020 data cannot later be relabeled as fresh confirmation or final holdout.

## 6. Enforcement

Repository tests check that current formal experiment specs with explicit Binance monthly `allowed_archives` use only slices assigned by this policy. Future access to a new month therefore requires a reviewed policy update before the experiment protocol is frozen.

This is an access/governance guardrail, not evidence that any unassigned slice is statistically suitable for a future role.
