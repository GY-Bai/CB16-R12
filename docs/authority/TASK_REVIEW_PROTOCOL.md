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

A normal review bug does not require a new task or branch.

## Probe protocol
When GitHub-visible evidence is insufficient, SOL may request a bounded probe: a specific test, selected runtime values, one failure reproduction, a measured runtime quantity, or a bounded benchmark. Do not turn GitHub comments into an arbitrary shell execution interface.

## Status vocabulary
Prefer GitHub native states. Suggested custom command labels are `ds:run`, `sol:review`, and `blocked`.
