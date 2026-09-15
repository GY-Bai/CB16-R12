# R12 Evidence Scope and Claim Authority

Status: **CANONICAL SCIENTIFIC-INTERPRETATION AND PROMOTION RULE**

This document governs how CB16-R12 interprets evidence, PASS/FAIL, component qualification, composition, and roadmap consequences. It does not change any frozen historical gate or retroactively convert a prior result.

## 1. Core principle

Every conclusion must match the object, claim, conditions, estimand, controls, and evidence that were actually and validly tested.

A formal gate result has **claim-local authority**. Neither PASS nor FAIL may be widened beyond the scope supported by the experiment.

A gate miss means first:

> **The frozen qualification gate was not met; the specified claim did not earn the requested qualification.**

It does **not** automatically mean the proposition was falsified, the root cause was identified, a component is intrinsically defective, or the final CB16 architecture is impossible.

Likewise, a PASS means only that the declared qualification was earned under its stated scope. It does not automatically transfer to a different configuration, composition, dataset, regime, or production setting.

The project objective is not to protect CB16 from falsification. It is to make support, negative evidence, qualification, promotion, and engineering decisions no broader than the evidence can carry.
## 2. Separate execution, qualification, inference, attribution, and promotion

These judgments are different and must not be collapsed:

| Judgment | Meaning |
|---|---|
| `execution_status` | Did the intended code/data/runtime/spec execute validly? |
| `gate_result` | Did the frozen numerical/behavioral rule pass? |
| `qualification_status` | Did the tested object earn the declared qualification? |
| `inference_status` | What positive or negative proposition is scientifically supported by the evidence? |
| `attribution_status` | Is a failure or success mechanism actually identified, or still unresolved? |
| `promotion_decision` | What specific downstream use is allowed or blocked? |

Use the following distinctions explicitly:

- `GATE_NOT_MET`: the preregistered decision rule failed.
- `NOT_QUALIFIED`: the requested qualification was not earned.
- `EVIDENCE_AGAINST_CLAIM`: the design and evidence support a scoped negative inference about a proposition.
- `INVALID_FOR_CLAIM`: validity prerequisites failed, so the experiment cannot adjudicate that claim.
- `NOT_TESTED`: the claim was outside the experiment.
- `BLOCKED_BY_DEPENDENCY`: a later promotion lacks a required prerequisite.

`GATE_NOT_MET` or `NOT_QUALIFIED` must not be silently upgraded to `EVIDENCE_AGAINST_CLAIM`. A confidence lower bound failing to exceed zero, for example, does not by itself prove the underlying effect is non-positive.
## 3. Authoritative model: typed claim graph, not a universal linear ladder

CB16's scientific authority is a graph of scoped claims and typed dependencies. The old L0-L6 sequence may remain as a reading aid or execution order, but layer numbers do not automatically create logical implication, evidence inheritance, or promotion authority.

Each formal experiment should declare one `experiment_role`: `QUALIFICATION`, `FALSIFICATION`, `DIAGNOSTIC`, `SCREENING`, `EXPLORATORY`, `ROBUSTNESS`, or `TRANSFER`. Each claim inside it should identify:

- `claim_id`: stable identifier;
- `claim_kind`: `VERIFICATION` when the question is whether an implementation satisfies a declared contract, or `VALIDATION` when the question is whether the system has the intended scientific/economic capability in a declared use context;
- `tested_object`: component, interface, learner, configuration, integration, or composed system;
- `claim_domain`: semantics, optimization, controlled behavior, decision value, economics, robustness, generalization, or another declared domain;
- `claim_scope`: data, asset, time, environment, account semantics, costs, budget, horizon, evaluation adapter, and version boundaries;
- `estimand`: the scientific quantity whose value would answer the declared question;
- `estimator`: the statistic/procedure used to estimate the estimand;
- `replication_unit`: the independent or dependence-aware unit used for uncertainty;
- `validity_prerequisites`: conditions that must hold before the result can adjudicate the claim;
- `authority_scope`: conclusions the experiment may support;
- `promotion_scope`: concrete downstream use that may be granted;
- `invalid_inferences`: conclusions the experiment is not authorized to make;
- `credible_alternatives`: material alternative explanations that the design controls, measures, or explicitly leaves unresolved.

Dependencies must be typed rather than called generically "upstream":

| Dependency | Meaning |
|---|---|
| `execution_prerequisite` | Required for the experiment to run validly. |
| `validity_dependency` | If invalid, the scientific interpretation is compromised. |
| `promotion_prerequisite` | Project-governance evidence required before a later use is authorized. |
| `logical_necessity` | A proposition that must be true for the dependent claim to hold in the same scope. |
| `transfer_assumption` | Additional assumption required to reuse evidence in a changed configuration or context. |

A synthetic prerequisite imposed by project process is not automatically a logical necessity for every possible historical-market capability.
## 4. Object granularity and claim domain are separate axes

The object under test may be a component, interface, configuration, integration, composition, or broader system. The claim may concern semantics, optimization response, controlled behavior, decision value, economics, robustness, or generalization.

Do not infer object-level root cause from a configuration-level gate miss. A configuration such as:

`representation × learner × objective × budget × adapter × horizon × environment`

may fail jointly without identifying which element caused the result.

Likewise, an intervention that changes performance is not automatically a proof of the intervention's causal mechanism. If a batch-size or entropy intervention changes outcomes while the proposed mediator is not itself identified by the frozen design, record the intervention effect and keep mechanism attribution `UNRESOLVED`.

## 5. Navigation view only: current R12 evidence domains

For readability, current work may still be described using the following approximate sequence:

- execution / infra integrity;
- economic and reward-loop semantics;
- optimization diagnostics and intervention response;
- controlled synthetic learnability;
- historical-development decision qualification;
- cost-aware / continuing-account economic robustness;
- protected-holdout / generalization / deployment authority.

These are **not** a universal linear scientific hierarchy. A task may touch several domains, and a later experiment does not inherit every earlier result merely because its label appears later in this list.

## 6. Prior evidence: preserved validity is not automatic transfer

A new gate miss does not, by itself, revoke prior evidence that remains valid and non-conflicting. But prior evidence is not permanent or automatically transferable.

For each reused claim, record whether it is:

- `UNAFFECTED_IN_ORIGINAL_SCOPE`;
- `TRANSFER_APPLICABILITY_UNRESOLVED`;
- `REVIEW_REQUIRED` because a shared implementation, oracle, reward, leakage, provenance, or other validity dependency may be compromised;
- `QUALIFICATION_REVOKED` when evidence justifies that governance action.

A prior synthetic PASS can remain valid in its own task while providing no guarantee that a changed historical configuration has the same optimization, representation, credit, or retention behavior.
## 7. PASS and FAIL have symmetric scope limits

PASS and FAIL are both bounded by the frozen claim and evidence.

A PASS may authorize only the explicit next use named in `promotion_scope`. It does not grant universal downstream trust. Composition requires its own tests because interface shape/type compatibility does not establish semantic, distributional, temporal, optimization, or feedback compatibility after modules interact.

A FAIL blocks the qualification or promotion that depended on the failed gate. It does not automatically block unrelated research branches or alternate implementations.

Unqualified candidates may still enter explicitly diagnostic or exploratory experiments when the purpose is to understand interactions. They must not be represented as already-qualified dependencies for stronger scientific or production claims.

## 8. Architecture authority and falsification authority

Reduced MVPs do not define the full CB16 target architecture, but the full architecture is not scientifically exempt.

The owner/design authority defines goals, required capabilities, and candidate architecture. Evidence governs whether concrete technical claims and candidate implementations have earned their stated qualifications.

Therefore:

- one candidate steel plate failing does not prove that no aircraft carrier can be built;
- a specific carrier design that logically requires that exact failed plate specification cannot proceed unchanged if the required property has been validly contradicted in the same scope;
- replacing the candidate implementation does not delete the underlying system requirement;
- preserving the requirement does not oblige the project to continue funding a failed implementation forever.

Conclusions may propagate to a broader architecture only through an explicit, scope-matched `logical_necessity` or through a separately documented synthesis of multiple pieces of evidence. A small experiment never receives broader authority merely from its position in the roadmap.

## 9. MVP and vertical-slice role

R12 vertical slices are reduced qualification scaffolds, not miniature final-system verdicts.

Their purpose is to generate reusable but conditional evidence about components, interfaces, mechanisms, and compositions needed by the larger system.

The discipline is:

`verify scoped contract -> grant scoped experimental admission -> compose -> test interaction -> qualify stronger claim`

Use **component qualification evidence**, not "component proof", for empirical claims. A passed component may be admitted to a named next experiment; it is not proven correct in every future composition.
## 10. No-rescue, adaptation history, and research-series governance

No-rescue remains in force for a frozen formal run. Do not change thresholds, seeds, budgets, controls, objectives, or data after seeing results and reuse the same run identity.

A new experiment ID is necessary for a changed method, but it does not make reused development data statistically fresh.

Each research series should therefore record:

- parent experiments and observed prior results;
- a compact data-exposure record: dataset/split identity, the role in which it was seen, prior experiment IDs, and whether research/model-design decisions were adapted after inspecting it;
- what changed and why that change is hypothesized to matter;
- preregistered budget and stopping/escalation rules;
- the next experiment's declared role (`QUALIFICATION`, `FALSIFICATION`, `DIAGNOSTIC`, `SCREENING`, `EXPLORATORY`, `ROBUSTNESS`, or `TRANSFER`).

This record may live inside the experiment spec or research-series metadata. R12 does not require a separate database or service merely to track exposure.

Repeatedly adapting to the same development period can create selection bias even when every run has a new ID. The project must not treat IDs as resetting the evidence history.

## 11. Reporting and compatibility vocabulary

Dispatcher compatibility may retain the coarse top-level values:

```text
PASS
SCIENTIFIC_FAIL
EXECUTION_BLOCKED
HARDWARE_LIMIT
EVIDENCE_INSUFFICIENT
CONTRACT_MISMATCH
```

For a scientifically valid run whose frozen gate is missed, `SCIENTIFIC_FAIL` means only that the run produced a negative qualification outcome under its frozen rule. It is **not** itself a falsification label or a root-cause diagnosis.

Future result schemas should additionally carry, at minimum:

```text
execution_status
validity_status
gate_results[]
claim_assessments[]
prior_evidence_assessment[]
promotion_decision
```

Historical result enums remain part of provenance. Interpretation supplements may narrow their meaning, but must not rewrite the original artifact.
## 12. Minimum structure for future formal experiment specs

For new formal scientific work, `experiment_spec.json` should progressively adopt:

```text
experiment_id
parent_experiment_ids
experiment_role
tested_object
claims[]
claim_kind
claim_scope
estimand
estimator
replication_unit
validity_prerequisites
controls
credible_alternatives[]
gate_mapping
uncertainty_protocol
authority_scope
promotion_scope
invalid_inferences
composition_dependencies
data_exposure
research_series_policy
```

`RESULT.json` should progressively adopt:

```text
execution_status
validity_status
protocol_deviations[]
gate_results[]
claim_assessments[]
prior_evidence_assessment[]
data_exposure_assessment
promotion_decision
provenance
```

Mechanical rules:

1. RESULT must not widen the object or claim scope declared in the spec.
2. An aggregate gate FAIL must preserve component gate observations rather than erase them.
3. PASS must not create a promotion not authorized by the spec.
4. Failed controls or validity prerequisites block affected scientific interpretation.
5. Unknown root cause must remain `attribution_status: UNRESOLVED`.
6. Changed versions/configurations do not inherit qualification without an explicit transfer assessment.
7. A `DIAGNOSTIC`, `SCREENING`, or `EXPLORATORY` result does not silently become a qualification or falsification result.
8. A verification PASS does not establish validation unless the validation claim was separately designed and tested.
9. Protocol deviations are reported as evidence about validity, not silently repaired in the result narrative.

These fields are a governance target for new experiment schemas. Existing `cb16.result.v1` artifacts are not retroactively invalid because they do not contain them; their interpretation must instead follow this canonical document until structured enforcement is implemented.
## 13. Temporal authority and supersession

CB16 is a long-running evolving research program. Documents must not be read as a flat set of equally current rules.

Every reviewer, Builder, and handoff agent should reconstruct authority in temporal order:

1. current canonical authority and explicit supersession rules;
2. frozen experiment/task specifications that remain valid for their historical run identities;
3. later interpretation supplements that narrow or clarify what those historical results mean;
4. older planning, archaeology, or superseded rules only as provenance and context.

A newer canonical interpretation may change how an old result is **described or propagated**, but it must not rewrite the old experiment's frozen gate, observed result, or artifact.

Conversely, an old task document must not silently override a newer canonical authority merely because the old wording is more specific or appears in more files.

Handoff material should therefore distinguish:

- `CURRENT_CANONICAL`: governing rule now;
- `FROZEN_HISTORICAL_AUTHORITY`: still authoritative for a specific completed experiment/run;
- `SUPERSEDED`: retained for provenance, not active governance;
- `INTERPRETATION_SUPPLEMENT`: later scope clarification that does not mutate the historical record.

This prevents rule accumulation from becoming rule ambiguity. Chronology is part of correct interpretation, not optional background reading.

## 14. Physical documentation authority separation

Authority class must be visible in the repository layout, not inferred from prose buried inside files.

- `docs/authority/` — `CURRENT_CANONICAL`; default governance context.
- `docs/design/current/` — `CURRENT_DESIGN_CANDIDATE`; active candidate design, subordinate to canonical authority.
- `docs/operations/current/` — `CURRENT_OPERATIONAL`; execution/infra semantics, not product-science truth.
- `docs/tasks/active/` — `CURRENT_TASK_SCOPE_ONLY`; bounded current work only.
- `docs/experiments/frozen/` — `FROZEN_HISTORICAL_AUTHORITY`; authoritative only for the named historical contract/run.
- `docs/archive/` — `SUPERSEDED / ARCHAEOLOGY`; excluded from default agent context.
- `docs/templates/` — `TEMPLATE_ONLY`; never executable authority.

Agents must not recursively load `docs/` and flatten these classes into one context. Start from `docs/README.md` and `docs/authority/README.md`, then follow only dependencies relevant to the current task.

Completed one-time prompts or bootstrap instructions that add no durable provenance value should be deleted from the current tree; Git history remains the archive of record. Historical material retained for provenance belongs outside current-authority directories.
## 15. Current interpretation of R1/R2/R3 and VS-D R0

Historical experiment labels and artifacts remain unchanged. Present interpretation is:

- **R1 batch intervention:** evidence concerns whether the frozen 10x collection intervention was sufficient under its gate. It does not uniquely identify Monte-Carlo variance as the root cause; mechanism attribution may remain unresolved.
- **R2 direction-entropy intervention:** the intervention produced the preregistered controlled improvement, but that does not prove one unique mediator, make `0.005` a production authority, or establish historical-market value.
- **R3 `JOINT_LEARNABILITY_CONFIRMED`:** `joint` means joint adjudication of two synthetic tasks using the same fixed learner configuration across separate task-specific training instances. It does **not** establish one checkpoint simultaneously holding both abilities, multitask coexistence, A→B retention, or continuing-account composition.
- **VS-D R0:** preserve the historical outcome string `HISTORICAL_MARKET_INFORMATION_NOT_QUALIFIED`, but the current scoped assessment is `TESTED_CONFIGURATION_DEV_DECISION_GATE_NOT_MET`.

For VS-D, the tested object is the frozen N0 + sensory + learner/training program + objective + budget + deterministic evaluation adapter + one-hour horizon under the specified BTCUSDT Jan-Feb training / March development-validation, zero-friction, funding-off, flat-reset environment. The joint qualification gate was not met, so that configuration did not earn the requested development qualification.

VS-D did not identify a unique failing component or mechanism, did not test all historical relations, and did not adjudicate continuing-account economics, cost survival, final holdout, or the feasibility of the full CB16 design. Component gate observations and favorable point estimates remain evidence in the artifact even though they cannot rescue the frozen aggregate FAIL.

March is development material already inspected by this research series. Future configuration changes evaluated on March must record that adaptive reuse and must not describe March as statistically fresh confirmation.

## 16. Experiment roles and inference authority

Every new formal scientific experiment declares its role before results are observed. The role constrains what kind of conclusion the experiment may support:

| Role | Primary purpose | Typical authority |
|---|---|---|
| `QUALIFICATION` | Decide whether a scoped object earns a preregistered capability/use qualification. | Grant or deny the named qualification/promotion only. |
| `FALSIFICATION` | Deliberately test a proposition with a design capable of producing scoped evidence against it. | Support or weaken the stated proposition; a generic gate miss is not enough. |
| `DIAGNOSTIC` | Measure a suspected mechanism or localize why a prior configuration behaved as observed. | Mechanism evidence within the diagnostic design; no automatic qualification. |
| `SCREENING` | Compare several plausible factors/interventions efficiently to decide what deserves a stronger follow-up. | Prioritize hypotheses; usually not confirmatory. |
| `EXPLORATORY` | Generate hypotheses or characterize an unknown surface. | Hypothesis generation only unless separately preregistered. |
| `ROBUSTNESS` | Test sensitivity across declared perturbations, regimes, seeds, costs, or implementation choices. | Bound stability within the tested perturbation set. |
| `TRANSFER` | Test whether earlier evidence remains applicable after a declared change of data, component, composition, or context. | Grant or deny the named evidence transfer. |

A new run identity does not change an experiment's role. A result may be interesting outside its role, but stronger use requires a new appropriately designed experiment.

## 17. Verification and validation are different claims

R12 uses the systems-engineering distinction explicitly:

- **Verification** asks whether the implementation satisfies the declared contract: formulas, invariants, interfaces, determinism, accounting, masks, timing, provenance, and other specified behavior.
- **Validation** asks whether the verified system has the intended scientific/economic capability in the declared use context: learns the relation, improves the decision, transfers, survives costs, or supports deployment.

Verification is often a prerequisite for validation, but it is not validation evidence by itself. Conversely, an observed validation improvement does not excuse a contract-invalid implementation. Reviews should state which kind of claim each gate supports. Here `VALIDATION` is a systems-engineering claim kind; it is not the same thing as a dataset role named development/validation.

## 18. Claim argument and credible alternatives

High-consequence or mechanistically ambiguous claims should record the smallest useful argument structure:

```text
claim
  -> why the declared evidence would support it
  -> validity prerequisites
  -> credible alternative explanations
  -> control/measurement for each addressed alternative
  -> alternatives intentionally left unresolved
```

This is an argument record, not a new graph service. Evidence-file count, green CI, or a verbose report does not strengthen a claim unless the evidence closes a declared part of the argument. Counterevidence and unresolved alternatives remain visible.

## 19. Adaptive data exposure

Development data have an exposure history. Repeated inspection and adaptation can make later results less independent even when code SHA, seed, branch, or experiment ID changes. For any repeatedly used development period, new formal specs should record at least:

```text
dataset_or_split_id
roles_seen[]                 # training / development validation / diagnostic / other
prior_experiment_ids[]
human_or_model_result_exposure
adaptations_made_after_exposure[]
freshness_class             # fresh / previously seen / adaptively reused
```

This record governs interpretation; it does not ban iterative research. Adaptively reused development data remain useful for diagnosis and engineering, but must not be represented as fresh confirmation or protected-holdout evidence.

## 20. Enforcement status

This document is canonical interpretation/governance now. Not every rule is yet machine-enforced by `experiment_spec.json`, `RESULT.json`, or the dispatcher.

Until a newer result schema and validator explicitly implement the claim-authority fields, reviewers must apply them during preregistration and result review. A green repository test suite proves implementation regression checks passed; it does not prove that every claim-scope boundary is mechanically enforced.

Future schema/compiler work may enforce these rules, but must not retroactively mutate historical artifacts. The transition from interpretation-level governance to machine enforcement requires its own reviewed implementation change.
