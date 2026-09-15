# Scientific Experiment Protocol Template

Authority class: `TEMPLATE_ONLY` — copy/adapt for a new formal experiment; this file itself authorizes nothing.

Use the smallest subset that is scientifically material, but do not omit a field merely because the likely result seems obvious. The frozen machine-readable experiment spec remains the execution authority when one exists.

## Identity and role

- `experiment_id`:
- `parent_experiment_ids`:
- `experiment_role`: `QUALIFICATION | FALSIFICATION | DIAGNOSTIC | SCREENING | EXPLORATORY | ROBUSTNESS | TRANSFER`
- `claim_kind`: `VERIFICATION | VALIDATION` (split into multiple claims when both are needed; `VALIDATION` here is a claim kind, not a validation-data split)
- `tested_object`:

## Scientific question and claims

State the question in one sentence. List the exact claim(s), domain, and scope. For a high-consequence or mechanistically ambiguous claim, add one short `supporting_argument`: why the planned evidence would support that claim if the validity prerequisites hold.

## Estimand and estimator

- `estimand`: what quantity would answer the question?
- `estimator`: what statistic/procedure estimates it?
- `replication_unit`: what is independent, paired, clustered, or dependence-aware?
- `uncertainty_protocol`:

Do not substitute a convenient metric for the intended scientific quantity without stating the resulting limitation.

## Scope and data exposure

Record asset/time/environment/account/cost/horizon/adapter/version boundaries. For every development/validation split used here:

- dataset/split identity;
- roles in which it was previously seen;
- prior experiment IDs;
- whether results were inspected before the current design was chosen;
- adaptations made after prior exposure;
- freshness class: `FRESH | PREVIOUSLY_SEEN | ADAPTIVELY_REUSED`.

## Validity prerequisites and protocol deviations

List conditions that must hold for the experiment to adjudicate the claim. Predeclare which deviations make the result `INVALID_FOR_CLAIM`, which merely narrow scope, and which are operationally irrelevant.

## Controls and credible alternatives

List the material alternative explanations. For each, state the matched control, measurement, or why it deliberately remains unresolved. Negative/random/shuffle controls are required when they distinguish leakage, credit, or relation learning from a simpler explanation.

## Frozen method, budget, and stopping rule

Freeze the inputs/transforms, model/configuration, seeds, optimizer/objective where applicable, budget, stopping rule, evaluation adapter, and no-rescue boundary. A changed method after result inspection gets a new experiment identity.

## Gate and inference authority

- preregistered gate(s):
- `authority_scope`: what may this result support?
- `promotion_scope`: what exact next use may a PASS authorize?
- `invalid_inferences`: what must not be claimed from either PASS or FAIL?

A `DIAGNOSTIC`, `SCREENING`, or `EXPLORATORY` result does not automatically qualify a component. A `QUALIFICATION` miss does not automatically falsify a proposition or identify root cause.

## Result adjudication

The result review should report separately:

- execution status;
- validity status and protocol deviations;
- individual gate observations;
- qualification status;
- inference status;
- attribution status;
- prior-evidence/transfer assessment;
- data-exposure assessment;
- exact promotion decision.

## Escalation or complexity trigger

If the experiment is intended to justify a more complex learner, replay path, service, database, GPU/distributed dependency, or other permanent mechanism, state the measured gap/requirement it addresses and what follow-up evidence would justify retaining that complexity.
