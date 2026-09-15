"""Preregistration invariants for VS-D R3 representation accessibility."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from cb16_science.evidence_scope import validate_experiment_spec


ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "config" / "experiments" / "r12_vs_d_representation_accessibility_r3.json"


class RepresentationAccessibilityR3SpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = json.loads(R3.read_text())

    def test_v2_claim_authority_contract(self) -> None:
        claims = validate_experiment_spec(self.spec)
        self.assertEqual(
            set(claims),
            {
                "C_N0_LINEAR_ACCESSIBILITY",
                "C_FROZEN_SENSORY_LINEAR_ACCESSIBILITY",
                "C_SENSORY_RETENTION_GAP",
            },
        )

    def test_chronology_is_january_fit_february_validation(self) -> None:
        data = self.spec["data"]
        self.assertEqual(data["probe_fit"]["expected_decisions"], 679)
        self.assertEqual(data["probe_validation"]["expected_decisions"], 696)
        self.assertEqual(data["probe_validation"]["expected_day_blocks"], 29)
        self.assertEqual(data["probe_validation"]["hours_per_day"], 24)
        self.assertTrue(self.spec["claim_scope"]["march_read_forbidden"])
        self.assertIn("BTCUSDT-1m-2020-03.zip", data["forbidden_reads"])

    def test_probe_is_fixed_before_results(self) -> None:
        probe = self.spec["probe"]
        self.assertEqual(probe["ridge_lambda"], 1.0)
        self.assertTrue(probe["validation_refit_forbidden"])
        self.assertTrue(probe["hyperparameter_search_forbidden"])
        self.assertTrue(probe["same_probe_family_across_surfaces"])
        self.assertEqual(self.spec["negative_controls"]["control_seeds"], list(range(4401, 4409)))

    def test_bootstrap_and_gates_are_frozen(self) -> None:
        uncertainty = self.spec["uncertainty_protocol"]
        self.assertEqual(uncertainty["bootstrap_replicates"], 4096)
        self.assertEqual(uncertainty["lower_bound_order_index_zero_based"], 204)
        self.assertTrue(uncertainty["paired_resamples_across_surfaces_and_controls"])
        self.assertEqual(set(self.spec["gates"]), {
            "G_N0_ACCESSIBILITY", "G_FROZEN_SENSORY_ACCESSIBILITY", "G_SENSORY_RETENTION_GAP"
        })

    def test_retention_gap_cannot_be_unique_causal_attribution(self) -> None:
        claim = next(c for c in self.spec["claims"] if c["claim_id"] == "C_SENSORY_RETENTION_GAP")
        invalid = set(claim["invalid_inferences"])
        self.assertIn("does_not_prove_random_projection_or_tanh_uniquely_destroyed_information", invalid)
        self.assertIn("does_not_equalize_effective_capacity_between_320D_and_32D_surfaces", invalid)
        self.assertFalse(claim["claim_scope"]["causal_attribution"])

    def test_no_rescue_surface_is_explicit(self) -> None:
        blocked = set(self.spec["no_rescue"])
        for item in (
            "no_probe_lambda_change",
            "no_split_change",
            "no_target_change",
            "no_nonlinear_probe_inside_R3",
            "no_March_read",
            "no_final_holdout_read",
        ):
            self.assertIn(item, blocked)


if __name__ == "__main__":
    unittest.main()
