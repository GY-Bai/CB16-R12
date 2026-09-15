from __future__ import annotations

import copy
import unittest

from cb16_science.evidence_scope import (
    EvidenceScopeError,
    validate_experiment_spec,
    validate_result_against_spec,
)


def valid_spec() -> dict:
    return {
        "schema": "cb16.experiment.v2",
        "experiment_id": "exp.test.r0",
        "parent_experiment_ids": ["exp.parent.r0"],
        "tested_object": "one tested configuration",
        "claim_scope": {"data": "dev"},
        "research_series_policy": {"freshness": "reused_dev"},
        "claims": [
            {
                "claim_id": "C1",
                "tested_object": "adapter transfer",
                "claim_domain": "decision_value",
                "claim_scope": {"adapter": "A"},
                "estimand": {"name": "delta", "unit": "day_block"},
                "authority_scope": ["adapter candidate only"],
                "promotion_scope": ["PROMOTE_ADAPTER_HYPOTHESIS"],
                "invalid_inferences": ["not production"],
                "composition_dependencies": [
                    {"type": "transfer_assumption", "ref": "exp.parent.r0"}
                ],
            }
        ],
        "gate_mapping": {"C1": ["G1"]},
        "allowed_promotion_decisions": ["NO_PROMOTION", "PROMOTE_ADAPTER_HYPOTHESIS"],
    }


def valid_result() -> dict:
    return {
        "schema": "cb16.result.v2",
        "experiment_id": "exp.test.r0",
        "execution_status": "EXECUTED",
        "validity_status": "VALID",
        "gate_results": [{"gate_id": "G1", "passed": True}],
        "claim_assessments": [
            {
                "claim_id": "C1",
                "qualification_status": "QUALIFIED",
                "inference_status": "SUPPORTED",
                "attribution_status": "PARTIAL",
            }
        ],
        "prior_evidence_assessment": [
            {"claim_ref": "exp.parent.r0", "status": "UNAFFECTED_IN_ORIGINAL_SCOPE"}
        ],
        "promotion_decision": {"decision": "PROMOTE_ADAPTER_HYPOTHESIS"},
        "provenance": {"commit_sha": "abc"},
    }


class EvidenceScopeTests(unittest.TestCase):
    def test_valid_spec_and_result(self) -> None:
        spec = valid_spec()
        self.assertEqual(set(validate_experiment_spec(spec)), {"C1"})
        validate_result_against_spec(valid_result(), spec)

    def test_spec_rejects_unknown_dependency_type(self) -> None:
        spec = valid_spec()
        spec["claims"][0]["composition_dependencies"][0]["type"] = "upstream"
        with self.assertRaises(EvidenceScopeError):
            validate_experiment_spec(spec)

    def test_spec_requires_gate_mapping_for_every_claim(self) -> None:
        spec = valid_spec()
        spec["gate_mapping"] = {}
        with self.assertRaises(EvidenceScopeError):
            validate_experiment_spec(spec)

    def test_result_cannot_assess_undeclared_claim(self) -> None:
        result = valid_result()
        result["claim_assessments"][0]["claim_id"] = "C2"
        with self.assertRaises(EvidenceScopeError):
            validate_result_against_spec(result, valid_spec())

    def test_result_cannot_emit_undeclared_gate(self) -> None:
        result = valid_result()
        result["gate_results"][0]["gate_id"] = "G9"
        with self.assertRaises(EvidenceScopeError):
            validate_result_against_spec(result, valid_spec())

    def test_result_cannot_promote_beyond_spec(self) -> None:
        result = valid_result()
        result["promotion_decision"]["decision"] = "PRODUCTION"
        with self.assertRaises(EvidenceScopeError):
            validate_result_against_spec(result, valid_spec())

    def test_result_must_assess_every_claim(self) -> None:
        spec = valid_spec()
        extra = copy.deepcopy(spec["claims"][0])
        extra["claim_id"] = "C2"
        spec["claims"].append(extra)
        spec["gate_mapping"]["C2"] = ["G2"]
        with self.assertRaises(EvidenceScopeError):
            validate_result_against_spec(valid_result(), spec)


if __name__ == "__main__":
    unittest.main()


class AdapterTransferPreregistrationTests(unittest.TestCase):
    def test_r1_preregistered_spec_is_v2_and_hash_frozen(self) -> None:
        import hashlib
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        path = root / "config" / "experiments" / "r12_vs_d_evaluation_adapter_transfer_r1.json"
        raw = path.read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "767e70bc90003bddf60fa085d15cf6bcbf00f27db1ccadeaeb615504fe3bb29a",
        )
        spec = json.loads(raw)
        claims = validate_experiment_spec(spec)
        self.assertEqual(set(claims), {"C_ADAPTER_TRANSFER", "C_DISTRIBUTION_LEVEL_RELATION"})
        self.assertEqual(spec["claim_scope"]["changed_axis"], "evaluation_adapter_only")
        self.assertEqual(spec["research_series_policy"]["freshness_status"], "NOT_FRESH_CONFIRMATION")
