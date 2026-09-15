"""Preregistration invariants for VS-D R4 bounded nonlinear accessibility."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from cb16_science.evidence_scope import validate_experiment_spec


ROOT = Path(__file__).resolve().parents[1]
R4 = ROOT / "config" / "experiments" / "r12_vs_d_bounded_nonlinear_accessibility_r4.json"
R3 = ROOT / "config" / "experiments" / "r12_vs_d_representation_accessibility_r3.json"


class BoundedNonlinearAccessibilityR4SpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = json.loads(R4.read_text())
        self.r3 = json.loads(R3.read_text())

    def test_v2_claim_authority_contract(self) -> None:
        claims = validate_experiment_spec(self.spec)
        self.assertEqual(set(claims), {
            "C_N0_BOUNDED_NONLINEAR_ACCESSIBILITY",
            "C_NONLINEAR_OVER_LINEAR_GAIN",
        })

    def test_r4_reuses_exact_r3_chronology(self) -> None:
        for key in ("allowed_archives", "forbidden_reads", "context", "probe_fit", "probe_validation"):
            self.assertEqual(self.spec["data"][key], self.r3["data"][key])
        self.assertTrue(self.spec["claim_scope"]["march_read_forbidden"])

    def test_nonlinear_probe_is_frozen(self) -> None:
        probe = self.spec["nonlinear_probe"]
        self.assertEqual(probe["architecture"], [
            "Linear(active_features,64)", "Tanh", "Linear(64,64)", "Tanh", "Linear(64,1)"
        ])
        self.assertEqual(probe["learning_rate"], 0.001)
        self.assertEqual(probe["epochs"], 256)
        self.assertEqual(probe["model_seeds"], list(range(4501, 4509)))
        self.assertFalse(probe["early_stopping"])
        self.assertFalse(probe["validation_monitoring_during_training"])

    def test_matched_controls_are_frozen(self) -> None:
        control = self.spec["negative_control"]
        self.assertEqual(control["control_permutation_seeds"], list(range(14501, 14509)))
        self.assertTrue(control["matched_model_initialization"])
        self.assertTrue(control["same_training_schedule"])
        self.assertTrue(control["February_true_target_unchanged"])

    def test_bootstrap_and_seed_gates_are_fixed(self) -> None:
        uncertainty = self.spec["uncertainty_protocol"]
        self.assertEqual(uncertainty["bootstrap_replicates"], 4096)
        self.assertEqual(uncertainty["bootstrap_seeds"], list(range(15401, 15409)))
        self.assertEqual(uncertainty["lower_bound_order_index_zero_based"], 204)
        self.assertEqual(self.spec["aggregate_gates"]["G_NONLINEAR_AGGREGATE"]["seed_pass_count_min"], 6)
        self.assertEqual(self.spec["aggregate_gates"]["G_GAIN_AGGREGATE"]["seed_pass_count_min"], 6)

    def test_promotion_requires_new_experiment_identity(self) -> None:
        self.assertEqual(
            self.spec["promotion_policy"]["if_both_claims_qualified"],
            "PROMOTE_SUPERVISED_SENSORY_CANDIDATE_HYPOTHESIS",
        )
        invalid = set(self.spec["claims"][1]["invalid_inferences"])
        self.assertIn("does_not_authorize_integration_into_R4", invalid)

    def test_no_rescue_is_explicit(self) -> None:
        blocked = set(self.spec["no_rescue"])
        for key in ("no_architecture_change", "no_epoch_change", "no_learning_rate_change", "no_model_seed_change", "no_split_change", "no_March_read"):
            self.assertIn(key, blocked)


if __name__ == "__main__":
    unittest.main()
