# R12 Failure Locality and Scientific-Layer Taxonomy

Status: **CANONICAL INTERPRETATION RULE**

This document defines how CB16-R12 interprets failure evidence. It does not change any already-frozen experiment gate or retroactively convert a FAIL into a PASS.

## 1. Core rule: failure is local to the claim that was tested

A formal gate miss rejects the **smallest preregistered claim owned by that gate**. It does not automatically reject the learner, the architecture, the research direction, or the entire CB16 scientific program.

`SCIENTIFIC_FAIL` remains a useful top-level execution classification meaning: **the formal experiment executed validly and its frozen scientific gate did not pass**.

It must never be read as shorthand for `THE_SCIENTIFIC_PROGRAM_IS_FALSE`.

Every scientific FAIL must therefore be reported with:

- `scientific_layer`: the layer whose claim was tested;
- `failed_claim`: the narrow preregistered proposition that did not qualify;
- `upstream_evidence_preserved`: previously established layers that remain valid;
- `downstream_claims_not_reached`: stronger claims that were not tested and therefore are neither PASS nor FAIL.

The no-rescue rule still applies. Failure locality changes interpretation, not the frozen result.

## 2. Nested scientific layers

CB16 evidence is nested like an onion. A stronger inner claim is meaningful only when its required outer layers are intact.

| Layer | Question | Typical evidence | Meaning of failure |
|---|---|---|---|
| L0 — Execution / Infra integrity | Did the intended code, data, runtime and frozen contract actually execute? | tests, exact commit, hashes, deterministic runtime, artifact integrity | No scientific inference. Use `EXECUTION_BLOCKED`, `HARDWARE_LIMIT`, `EVIDENCE_INSUFFICIENT`, or `CONTRACT_MISMATCH`. |
| L1 — Economic / reward-loop integrity | Do state → action → permission → execution → account consequence → reward and credit semantics represent the intended economic loop? | Physics invariants, reward identities, causal timing, negative controls, return-to-go semantics | The learning loop is not scientifically trustworthy yet. This does not imply market information is absent. |
| L2 — Optimization response | Does the implemented learner actually respond to its objective without pathological collapse? | finite/non-zero gradients, parameter movement, loss/value diagnostics, entropy/collapse diagnostics | Current optimization dynamics are not qualified. This does not establish that the task is unlearnable or information-free. |
| L3 — Controlled learnability | On known-answer synthetic tasks, does training create the intended behavioral improvement beyond matched controls? | positive vs zero/shuffle controls, account-conditioned behavior, delayed-credit tasks | Controlled learnability of the tested mechanism/configuration is not established. Historical/economic claims are not reached. |
| L4 — Historical information extraction | On frozen out-of-sample historical data, does the trained policy extract a context→consequence relation beyond PRE and matched shuffle/null learners? | development validation, block bootstrap, fresh seeds, matched controls | The tested representation/objective/horizon/configuration did not qualify historical information on that scope. It is not evidence that markets contain no learnable information. |
| L5 — Economic robustness | Does qualified information survive costs, funding, continuing-account dynamics and path/survival risk, and beat relevant benchmarks? | net log-growth, Buy & Hold, drawdown/survival metrics, confidence-aware Pareto qualification | The tested strategy economics are not qualified. Upstream learnability or information evidence can remain valid. |
| L6 — Generalization / deployment authority | Does the system survive protected holdout, broader regimes/assets and operational deployment constraints? | final holdout, multi-regime/multi-asset evidence, deployment qualification | Production/generalization authority is not established. It does not erase lower-layer scientific results. |

## 3. Non-implication rules

The following inferences are forbidden unless a dedicated experiment supports them:

- L4 historical FAIL -> “the learner cannot learn”; forbidden if L3 controlled learnability already passed.
- L4 historical FAIL -> “the market has no information”; forbidden because the result is conditional on the tested representation, objective, horizon, data slice and model capacity.
- L5 economic FAIL -> “no predictive relation exists”; forbidden because costs/path risk may destroy an otherwise real signal.
- lower training loss -> L3/L4/L5 PASS; forbidden because optimization diagnostics are not behavioral or economic qualification.
- one successful development slice -> production authority; forbidden because L6 remains separate.

A later-layer failure may motivate a new hypothesis about an earlier design choice, but that hypothesis must receive a new preregistered experiment identity.

## 4. Reporting vocabulary

Top-level run classifications remain stable for dispatcher compatibility:

```text
PASS
SCIENTIFIC_FAIL
EXECUTION_BLOCKED
HARDWARE_LIMIT
EVIDENCE_INSUFFICIENT
CONTRACT_MISMATCH
```

For `SCIENTIFIC_FAIL`, reports should additionally name a scoped outcome such as `CONTROLLED_LEARNABILITY_NOT_CONFIRMED`, `HISTORICAL_MARKET_INFORMATION_NOT_QUALIFIED`, or `ECONOMIC_ROBUSTNESS_NOT_QUALIFIED`.

A complete report should make the scope explicit:

```text
classification: SCIENTIFIC_FAIL
scientific_layer: L4_HISTORICAL_INFORMATION_EXTRACTION
failed_claim: <exact preregistered proposition>
upstream_evidence_preserved:
  - L0 execution/contract integrity
  - L1 reward-loop semantics already qualified by their own evidence
  - L3 controlled learnability already qualified by R3

downstream_claims_not_reached:
  - L5 cost-aware / continuing-account economic robustness
  - L6 final-holdout / deployment authority
```

`upstream_evidence_preserved` does not mean every possible claim inside an earlier layer is permanently proven. It means the specific earlier results are not logically invalidated merely because a stronger downstream claim failed.

## 5. Diagnostics versus gates

Loss, gradient norm, entropy, parameter movement and value-function error belong primarily to L2 diagnostics unless explicitly preregistered as a gate.

They answer whether optimization is behaving, not whether historical information or profitability exists. Conversely, a profitability gate miss cannot by itself diagnose the optimizer. Attribution requires a separately designed experiment.

This prevents two opposite errors:

1. declaring success because loss decreased;
2. declaring the whole learner scientifically invalid because a downstream economic gate failed.

## 6. Application to the current R12 evidence

The current evidence should be read as follows:

- VS-A / VS-B and their regression evidence establish the implemented Physics / learner-spine contracts needed for later experiments; downstream results do not retroactively turn passing implementation contracts into failures.
- VS-C R3 established `JOINT_LEARNABILITY_CONFIRMED` on fresh controlled synthetic tasks. That L3 result remains evidence that the minimal learner can change behavior appropriately under known-answer economic relations.
- VS-D R0 returned `SCIENTIFIC_FAIL / HISTORICAL_MARKET_INFORMATION_NOT_QUALIFIED`. Its failure scope is L4: the frozen N0 + frozen sensory + current learner/objective/horizon/configuration did not qualify the preregistered BTCUSDT Jan-Feb → March historical-information claim under the block-bootstrap gates.
- VS-D R0 did **not** test L5 transaction-cost survival, funding, continuing-account profitability, Buy & Hold superiority, or path-risk robustness. Those claims are `NOT_REACHED`, not failed.
- VS-D R0 also does not negate the existence of learnable market structure in other representations, horizons, objectives, regimes or model classes. Those are separate hypotheses.

The correct next action after a scoped FAIL is diagnosis and a newly preregistered hypothesis, not semantic overreach and not post-hoc rescue of the failed run.

## 7. Historical humility

Scientific programs often progress through local failures. A failed present configuration is evidence about that configuration and claim, not a proof that future model classes or better representations cannot cross the boundary.

CB16 should therefore be skeptical at the level of each claim and conservative about promotion, while remaining equally skeptical of over-broad negative conclusions.

The project standard is:

> **Reject exactly what the evidence rejects; preserve exactly what prior evidence still supports; leave stronger or different hypotheses open until tested.**
## 8. Minimal vertical slices are component proofs, not miniature final-system verdicts

R12 begins with deliberately reduced vertical slices because the full CB16 design is too large to debug scientifically as one undifferentiated system.

A vertical slice or MVP is therefore **verification scaffolding**. Its job is to isolate one necessary capability, interface, mechanism, or causal link and produce evidence about that local object.

Failure of an MVP component means one of the following must happen before composition: repair the component, replace the mechanism, change the representation, or formulate a new scoped hypothesis. It does **not** authorize the inference that the intended full CB16 system cannot exist.

Likewise, PASS of a small component does not prove the final system will work. It only licenses that component to participate in the next composition stage under its stated contract.

The final architecture may contain richer sensory organs, memory, routing, multiple objectives, heterogeneous modules, more realistic account dynamics, broader data, and additional coordination mechanisms that are intentionally absent from an early vertical slice.

Therefore every reduced experiment must distinguish:

- `component_under_test`: the minimal module/mechanism being isolated;
- `composition_role`: why that component is necessary for later CB16 assembly;
- `qualified_interface`: what downstream stages may rely on if it passes;
- `non_claims`: final-system capabilities that the reduced experiment is not authorized to judge.

A local FAIL blocks promotion of that component **as currently instantiated**. It does not define the feasibility boundary of the final custom system.
## 9. Authority boundary: experiments qualify components; they do not redefine the final design

The intended full CB16 architecture is defined by its design authorities and owner decisions, not by accidental limitations of an early MVP.

A reduced experiment has authority to answer only:

1. did this component execute under its contract;
2. did this mechanism satisfy its preregistered local scientific claim;
3. may downstream composition rely on that qualified interface;
4. if not, what local uncertainty must be diagnosed next.

It does **not** have authority to delete deferred final-system capabilities, collapse the target architecture to the MVP, or declare that a richer design is unnecessary/impossible merely because the reduced implementation omitted those capabilities.

Conversely, the final design vision does not excuse a failing component. Every necessary building block must still earn enough evidence before it is trusted in composition.

R12 therefore follows a compositional discipline:

`prove local contract -> promote interface -> compose -> test interaction -> qualify stronger system claim`.

At each step, the experiment owns the quality of the building block in front of it, not the definition of the eventual finished system.
