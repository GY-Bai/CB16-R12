"""Preregistration invariants for VS-D R2 objective-balance diagnostic."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from cb16_science.evidence_scope import validate_experiment_spec


ROOT = Path(__file__).resolve().parents[1]
R2 = ROOT / "config" / "experiments" / "r12_vs_d_objective_balance_entropy_gradient_r2.json"
R0 = ROOT / "config" / "experiments" / "r12_vs_d_historical_market_canary_r0.json"


class ObjectiveBalanceR2SpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = json.loads(R2.read_text())
        self.r0 = json.loads(R0.read_text())

    def test_v2_claim_authority_contract(self) -> None:
        claims = validate_experiment_spec(self.spec)
        self.assertEqual(set(claims), {"C_DIRECTION_ENTROPY_GRADIENT_DOMINANCE"})

    def test_parent_training_configuration_is_frozen(self) -> None:
        self.assertEqual(self.spec["authority"], self.r0["authority"])
        self.assertEqual(self.spec["learner"], self.r0["learner"])
        self.assertEqual(self.spec["model_seeds"], self.r0["model_seeds"])
        self.assertEqual(self.spec["training_positive"], self.r0["training_positive"])
        self.assertEqual(self.spec["training_control"], self.r0["training_control"])

    def test_march_is_forbidden_and_only_parent_training_archives_are_allowed(self) -> None:
        self.assertTrue(self.spec["claim_scope"]["march_read_forbidden"])
        self.assertFalse(self.spec["claim_scope"]["validation_data_read"])
        self.assertEqual(
            [x["name"] for x in self.spec["data"]["allowed_archives"]],
            ["BTCUSDT-1m-2020-01.zip", "BTCUSDT-1m-2020-02.zip"],
        )
        self.assertIn("BTCUSDT-1m-2020-03.zip", self.spec["data"]["forbidden_reads"])

    def test_primary_gate_is_seed_level_not_generation_as_independent_replicates(self) -> None:
        estimand = self.spec["claims"][0]["estimand"]
        self.assertEqual(estimand["unit"], "model_seed")
        self.assertIn("index 63", estimand["within_seed_generation_summary"])
        self.assertEqual(self.spec["aggregate_gate"]["seed_pass_count_min"], 6)

    def test_r2_cannot_claim_root_cause_or_historical_qualification(self) -> None:
        invalid = set(self.spec["claims"][0]["invalid_inferences"])
        self.assertIn("does_not_prove_entropy_caused_parent_R0_gate_miss", invalid)
        self.assertIn("does_not_qualify_historical_market_information", invalid)
        self.assertEqual(
            self.spec["allowed_promotion_decisions"],
            ["NO_PROMOTION", "PROMOTE_ENTROPY_SCALE_INTERVENTION_HYPOTHESIS"],
        )


if __name__ == "__main__":
    unittest.main()
