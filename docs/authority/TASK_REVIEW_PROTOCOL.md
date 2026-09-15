# Task and Review Protocol

Status: **CURRENT_CANONICAL — TASK / SCIENCE REVIEW GOVERNANCE**

Authority class: `CURRENT_CANONICAL`.

## Task contract
A task should be executable by a fresh Builder without prior chat history.

Required sections:

```text
# Objective
# Existing interface / files
# Required semantics
# Allowed scope
# Do not change
# Required tests
# Done when
```

Task contracts should be precise, not long.

## High-risk tasks
Chat-SOL should specify rather than delegate invention for:

- RL losses and targets;
- probability/log-density math;
- terminal vs truncation semantics;
- account transitions and economics;
- nominal vs executed actions;
- replay / behavior-policy semantics;
- checkpoint/generation continuity;
- qualification / promotion logic;
- masks, NaN/Inf, and zero-denominator behavior.

## Scientific protocol review

Formal scientific work has two distinct review phases.

**Stage 1 — protocol review, before results are known.** The reviewer checks that the experiment is capable of answering its declared question and freezes, as applicable:

- `experiment_role` and `claim_kind` (`VERIFICATION` or `VALIDATION`);
- scientific question, tested object, claim/domain/scope;
- estimand, estimator, replication/dependence unit, and uncertainty procedure;
- validity prerequisites and conditions that would make the result invalid for the claim;
- dataset/split identity plus prior/adaptive exposure;
- controls and material credible alternative explanations;
- budget, stopping rule, seeds, transforms, objective, and gate;
- allowed/invalid inference and exact promotion authority.

Protocol acceptance authorizes execution of that experiment. It does not predict or pre-approve the result.

**Stage 2 — result adjudication, after execution.** The reviewer checks protocol adherence, deviations, execution/validity, individual gates, claim-local inference, unresolved alternatives, prior-evidence transfer, and promotion. Stage 2 does not redesign the frozen protocol to obtain a preferred answer.

## Scientific execution
A formal scientific run freezes its question, data scope, objective, controls, budget, and qualification rule **before** inspecting the result.

Once qualification begins:

- do not raise budget because a result failed;
- do not remove bad seeds/windows;
- do not change reward, thresholds, or controls to rescue the same run;
- do not tune on qualification results and reuse the same run identity;
- preserve negative/null results.

A changed scientific question or method gets a new run identity. A new identity does **not** make reused development data statistically fresh. The research series must retain parent experiments, already-inspected data, the change rationale, and any preregistered budget/stopping rule.

Training loss, non-zero gradients, changed parameters, throughput, or a written checkpoint are diagnostics. They do not by themselves prove controlled learning or economic improvement.

Use explicit negative/random/shuffle controls when they materially test leakage or credit assignment.

Every formal experiment also declares one role before execution: `QUALIFICATION`, `FALSIFICATION`, `DIAGNOSTIC`, `SCREENING`, `EXPLORATORY`, `ROBUSTNESS`, or `TRANSFER`. Diagnostic, screening, and exploratory work may guide the next hypothesis but do not silently become qualification evidence. A falsification experiment must be designed so the evidence can actually support the scoped negative proposition; a qualification gate miss alone is insufficient.

For claims that cross implementation correctness and scientific capability, state them separately. **Verification** asks whether the declared system was built correctly; **Validation** asks whether that verified system does the scientifically/economically intended job in the declared context. Neither substitutes for the other.

The top-level dispatcher/run vocabulary remains:

```text
SCIENTIFIC_FAIL
EXECUTION_BLOCKED
HARDWARE_LIMIT
EVIDENCE_INSUFFICIENT
CONTRACT_MISMATCH
```

`SCIENTIFIC_FAIL` is a compatibility classification for a scientifically valid run whose frozen qualification gate was not met. It is **not** by itself a falsification label or a root-cause diagnosis. Review each formal result separately along these dimensions:

```text
execution_status
validity_status
gate_result
qualification_status
inference_status
attribution_status
promotion_decision
```

PASS and FAIL have symmetric scope limits. A result may not widen the object, claim, conditions, estimand, or promotion authority declared by the experiment. A gate miss normally means `GATE_NOT_MET / NOT_QUALIFIED`; use `EVIDENCE_AGAINST_CLAIM` only when the design and evidence justify that stronger inference. Keep root cause `UNRESOLVED` unless it was actually identified.

Review should name the tested object, claim/domain/scope, individual gate observations, invalid inferences, relevant prior-evidence assessment, and the exact promotion allowed or blocked. Avoid generic "upstream/downstream" propagation when the real dependency type is execution, validity, promotion, logical necessity, or transfer. See `docs/authority/R12_EVIDENCE_SCOPE_AND_CLAIM_AUTHORITY.md`.

Do not rescue a frozen result by silently changing the method or gate. Equally, do not use the size of the final CB16 vision to excuse an unqualified required dependency. A reduced MVP result may affect a broader architecture only through an explicit scope-matched dependency or evidence synthesis; otherwise its authority remains local.

## Diagnostic and screening work after a miss

When a joint configuration misses a qualification and attribution is unresolved, do not serially patch one guessed cause at a time under the failed experiment identity. Freeze the miss first. Then, if useful, open a new `DIAGNOSTIC` or `SCREENING` experiment that compares multiple plausible factors or mechanisms under an explicit design.

A screening experiment is allowed to be cheaper and broader than a confirmatory qualification. When several factors are plausible, prefer a bounded comparative or factorial/fractional-factorial screen over a long sequence of one-factor-at-a-time rescue attempts when the design is practical. Its output is prioritization/effect evidence, not automatic proof of a mechanism. If screening identifies a promising intervention, the stronger claim still requires a new appropriately scoped qualification/falsification/transfer experiment.

## Complexity admission

New machinery is a scientific-engineering cost, not free capability. A new learner mechanism, replay path, database, queue, service, cache, distributed executor, GPU dependency, authority subsystem, compatibility layer, or permanent state surface should enter R12 only when at least one of the following is explicit:

1. an R12 semantic requirement cannot be met without it;
2. a measured capability/reliability/performance gap motivates it;
3. a preregistered interaction hypothesis specifically requires it;
4. an external deployment/safety constraint requires it.

The proposing task should state the observed gap/requirement, the smallest mechanism intended to address it, and what evidence would justify keeping the added complexity. Do not create a registry/compiler/service merely because a governance concept can be represented in software.

## Permanent guardrails vs experiment parameters

Promote only durable semantic invariants into permanent repository/CI guardrails: causality/no-lookahead, authority boundaries, accounting identities, terminal/truncation separation, protected-holdout access, or other rules intended to remain true across method changes. Keep optimizer choice, model width, budget, seed set, threshold, horizon, and similar method parameters scoped to their experiment/version unless a later authority explicitly elevates them.

This prevents yesterday's successful configuration from becoming tomorrow's accidental architecture law.

## BUILD_REPORT
Every Builder execution should end with a concise GitHub-visible report:

```text
BUILD_REPORT
Task:
PR:
Changed:
Targeted tests:
Affected suite:
Runtime/errors:
Known unresolved:
```

Large logs may remain in Actions/artifacts. The summary should not require artifact archaeology.

## Review loop

Implementation tasks use:

```text
Task contract
  -> Builder implementation
  -> Draft PR
  -> tests + BUILD_REPORT
  -> SOL review
       -> APPROVE
       or
       -> CHANGES_REQUIRED
             -> same PR / same writer
             -> tests
             -> re-review
```

Formal science uses:

```text
Protocol draft
  -> Stage-1 protocol review
  -> freeze experiment identity/spec
  -> authorized execution
  -> immutable result/evidence
  -> Stage-2 result adjudication
  -> scoped inference + promotion decision
```

A normal review bug does not require a new task or branch. A scientific protocol change after results are inspected requires a new experiment identity and retains the prior result in the research series.

## Probe protocol
When GitHub-visible evidence is insufficient, SOL may request a bounded probe: a specific test, selected runtime values, one failure reproduction, a measured runtime quantity, or a bounded benchmark. Do not turn GitHub comments into an arbitrary shell execution interface.

## Status vocabulary
Prefer GitHub native states. Suggested custom command labels are `ds:run`, `sol:review`, and `blocked`.
