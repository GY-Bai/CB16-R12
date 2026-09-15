"""Claim-local evidence governance for new R12 formal experiments.

Historical ``cb16.result.v1`` artifacts remain valid provenance.  New formal
experiments may opt into the v2 contract here so result structure cannot silently
widen the preregistered claim or promotion authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class EvidenceScopeError(ValueError):
    """A claim-authority spec/result violates the v2 governance contract."""


EXPERIMENT_SCHEMA = "cb16.experiment.v2"
RESULT_SCHEMA = "cb16.result.v2"

DEPENDENCY_TYPES = {
    "execution_prerequisite",
    "validity_dependency",
    "promotion_prerequisite",
    "logical_necessity",
    "transfer_assumption",
}

INFERENCE_STATUSES = {
    "SUPPORTED",
    "EVIDENCE_AGAINST_CLAIM",
    "GATE_NOT_MET",
    "INVALID_FOR_CLAIM",
    "NOT_TESTED",
    "BLOCKED_BY_DEPENDENCY",
}

ATTRIBUTION_STATUSES = {"RESOLVED", "PARTIAL", "UNRESOLVED", "NOT_APPLICABLE"}
QUALIFICATION_STATUSES = {"QUALIFIED", "NOT_QUALIFIED", "NOT_APPLICABLE"}

def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceScopeError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvidenceScopeError(f"{name} must be a sequence")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceScopeError(f"{name} must be a non-empty string")
    return value


def _string_list(value: Any, name: str, *, nonempty: bool = True) -> tuple[str, ...]:
    seq = _sequence(value, name)
    items = tuple(_string(item, f"{name}[]") for item in seq)
    if nonempty and not items:
        raise EvidenceScopeError(f"{name} must not be empty")
    if len(items) != len(set(items)):
        raise EvidenceScopeError(f"{name} must not contain duplicates")
    return items


def _claim_map(spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    claims = _sequence(spec.get("claims"), "claims")
    out: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(claims):
        claim = _mapping(raw, f"claims[{index}]")
        claim_id = _string(claim.get("claim_id"), f"claims[{index}].claim_id")
        if claim_id in out:
            raise EvidenceScopeError(f"duplicate claim_id {claim_id!r}")
        out[claim_id] = claim
    if not out:
        raise EvidenceScopeError("claims must not be empty")
    return out

def validate_experiment_spec(spec: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    spec = _mapping(spec, "experiment spec")
    if spec.get("schema") != EXPERIMENT_SCHEMA:
        raise EvidenceScopeError(f"experiment schema must be {EXPERIMENT_SCHEMA}")
    _string(spec.get("experiment_id"), "experiment_id")
    _string_list(spec.get("parent_experiment_ids"), "parent_experiment_ids", nonempty=False)
    _string(spec.get("tested_object"), "tested_object")
    _mapping(spec.get("claim_scope"), "claim_scope")
    _mapping(spec.get("research_series_policy"), "research_series_policy")

    claim_map = _claim_map(spec)
    for claim_id, claim in claim_map.items():
        prefix = f"claim {claim_id}"
        _string(claim.get("tested_object"), f"{prefix}.tested_object")
        _string(claim.get("claim_domain"), f"{prefix}.claim_domain")
        _mapping(claim.get("claim_scope"), f"{prefix}.claim_scope")
        _mapping(claim.get("estimand"), f"{prefix}.estimand")
        _string_list(claim.get("authority_scope"), f"{prefix}.authority_scope")
        _string_list(claim.get("promotion_scope"), f"{prefix}.promotion_scope")
        _string_list(claim.get("invalid_inferences"), f"{prefix}.invalid_inferences")
        deps = _sequence(claim.get("composition_dependencies"), f"{prefix}.composition_dependencies")
        for dep_index, raw_dep in enumerate(deps):
            dep = _mapping(raw_dep, f"{prefix}.composition_dependencies[{dep_index}]")
            dep_type = _string(dep.get("type"), f"{prefix}.dependency.type")
            if dep_type not in DEPENDENCY_TYPES:
                raise EvidenceScopeError(f"{prefix} uses unknown dependency type {dep_type!r}")
            _string(dep.get("ref"), f"{prefix}.dependency.ref")

    gate_mapping = _mapping(spec.get("gate_mapping"), "gate_mapping")
    if set(gate_mapping) != set(claim_map):
        raise EvidenceScopeError("gate_mapping keys must exactly match claim_ids")
    for claim_id, gate_ids in gate_mapping.items():
        _string_list(gate_ids, f"gate_mapping.{claim_id}")

    allowed_promotions = _string_list(spec.get("allowed_promotion_decisions"), "allowed_promotion_decisions")
    if "NO_PROMOTION" not in allowed_promotions:
        raise EvidenceScopeError("allowed_promotion_decisions must include NO_PROMOTION")
    return claim_map

def validate_result_against_spec(result: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    claim_map = validate_experiment_spec(spec)
    result = _mapping(result, "result")
    if result.get("schema") != RESULT_SCHEMA:
        raise EvidenceScopeError(f"result schema must be {RESULT_SCHEMA}")
    if result.get("experiment_id") != spec.get("experiment_id"):
        raise EvidenceScopeError("result experiment_id must match the frozen spec")
    _string(result.get("execution_status"), "execution_status")
    _string(result.get("validity_status"), "validity_status")

    gate_results = _sequence(result.get("gate_results"), "gate_results")
    gate_ids: set[str] = set()
    for index, raw in enumerate(gate_results):
        gate = _mapping(raw, f"gate_results[{index}]")
        gate_id = _string(gate.get("gate_id"), f"gate_results[{index}].gate_id")
        if gate_id in gate_ids:
            raise EvidenceScopeError(f"duplicate gate_id {gate_id!r}")
        gate_ids.add(gate_id)
        if not isinstance(gate.get("passed"), bool):
            raise EvidenceScopeError(f"gate_results[{index}].passed must be bool")

    declared_gate_ids = {
        gate_id
        for ids in _mapping(spec["gate_mapping"], "gate_mapping").values()
        for gate_id in _string_list(ids, "gate_mapping[]")
    }
    if not gate_ids.issubset(declared_gate_ids):
        raise EvidenceScopeError("result contains gate_ids not declared by the spec")

    assessments = _sequence(result.get("claim_assessments"), "claim_assessments")
    assessment_ids: set[str] = set()
    for index, raw in enumerate(assessments):
        assessment = _mapping(raw, f"claim_assessments[{index}]")
        claim_id = _string(assessment.get("claim_id"), f"claim_assessments[{index}].claim_id")
        if claim_id not in claim_map:
            raise EvidenceScopeError(f"result assesses undeclared claim {claim_id!r}")
        if claim_id in assessment_ids:
            raise EvidenceScopeError(f"duplicate claim assessment {claim_id!r}")
        assessment_ids.add(claim_id)
        if assessment.get("qualification_status") not in QUALIFICATION_STATUSES:
            raise EvidenceScopeError(f"invalid qualification_status for {claim_id}")
        if assessment.get("inference_status") not in INFERENCE_STATUSES:
            raise EvidenceScopeError(f"invalid inference_status for {claim_id}")
        if assessment.get("attribution_status") not in ATTRIBUTION_STATUSES:
            raise EvidenceScopeError(f"invalid attribution_status for {claim_id}")

    if assessment_ids != set(claim_map):
        raise EvidenceScopeError("result must assess every preregistered claim exactly once")

    _sequence(result.get("prior_evidence_assessment"), "prior_evidence_assessment")
    promotion = _mapping(result.get("promotion_decision"), "promotion_decision")
    decision = _string(promotion.get("decision"), "promotion_decision.decision")
    allowed = set(_string_list(spec.get("allowed_promotion_decisions"), "allowed_promotion_decisions"))
    if decision not in allowed:
        raise EvidenceScopeError(f"promotion decision {decision!r} is not allowed by the spec")
    _mapping(result.get("provenance"), "provenance")


__all__ = [
    "ATTRIBUTION_STATUSES",
    "DEPENDENCY_TYPES",
    "EXPERIMENT_SCHEMA",
    "EvidenceScopeError",
    "INFERENCE_STATUSES",
    "QUALIFICATION_STATUSES",
    "RESULT_SCHEMA",
    "validate_experiment_spec",
    "validate_result_against_spec",
]
