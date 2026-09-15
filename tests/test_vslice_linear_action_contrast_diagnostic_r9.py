"""Implementation tests for VS-D R9 linear action-contrast diagnostic."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from cb16_science.evidence_scope import validate_result_against_spec
from cb16_science.vslice import linear_action_contrast_diagnostic_r9 as r9

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "config/experiments/r12_vs_d_linear_action_contrast_diagnostic_r9.json").read_text())


class R9ImplementationTests(unittest.TestCase):
    def test_spec_sha_and_contract(self):
        self.assertEqual(r9.load_spec(), SPEC)
        r9.validate_spec(SPEC)

    def test_control_orders_are_deterministic_derangements(self):
        a = r9.control_orders(615, [23001, 23002])
        b = r9.control_orders(615, [23001, 23002])
        for seed in a:
            self.assertTrue(np.array_equal(a[seed], b[seed]))
            self.assertFalse(np.any(a[seed] == np.arange(615)))

    def test_direction_mapping_has_exact_zero_flat_only(self):
        score = np.array([-2.0, -1e-300, 0.0, 1e-300, 3.0])
        self.assertTrue(np.array_equal(r9.direction_from_score(score), np.array([-1, -1, 0, 1, 1], dtype=np.int8)))

    def test_per_step_contrast_is_direction_times_return(self):
        score = np.array([-1.0, 0.0, 2.0])
        realized = np.array([0.02, -0.50, -0.03])
        self.assertTrue(np.allclose(r9.per_step_contrast(score, realized), np.array([-0.02, 0.0, -0.03])))

    def test_fit_scores_uses_one_true_and_all_controls(self):
        rng = np.random.default_rng(91)
        fit = rng.normal(size=(615, 8))
        evaluation = rng.normal(size=(672, 8))
        target = rng.normal(size=615)
        ds = r9.ActionContrastDataset(fit, evaluation, target, np.zeros(672), {})
        spec = json.loads(json.dumps(SPEC))
        spec["probe"]["control_seeds"] = [23001, 23002]
        out = r9.fit_scores(ds, spec)
        self.assertEqual(out["true_score"].shape, (672,))
        self.assertEqual(len(out["control_scores"]), 2)
        self.assertEqual(out["input_feature_count"], 8)

    def test_synthetic_strong_direction_signal_passes_gate_metrics(self):
        rng = np.random.default_rng(92)
        target = rng.normal(loc=0.0, scale=0.01, size=672)
        true_score = target.copy()
        controls = tuple(rng.normal(size=672) for _ in range(8))
        ds = r9.ActionContrastDataset(np.zeros((615, 2)), np.zeros((672, 2)), np.zeros(615), target, {})
        scores = {"true_score": true_score, "control_scores": controls, "active_feature_count": 2, "input_feature_count": 2}
        out = r9.evaluate_action_contrast(ds, scores, SPEC)
        self.assertGreater(out["bootstrap"]["true_mean_unit_direction_return_contrast_lcb"], 0.0)
        self.assertGreater(out["bootstrap"]["true_minus_median_shuffle_mean_direction_return_contrast_lcb"], 0.0)
        self.assertEqual(sum(out["direction_counts"].values()), 672)

    def test_failure_result_is_invalid_for_claim(self):
        result = r9.failure_result(SPEC, "CONTRACT_MISMATCH", "x")
        validate_result_against_spec(result, SPEC)
        self.assertEqual(result["validity_status"], "INVALID")
        self.assertEqual(result["claim_assessments"][0]["inference_status"], "INVALID_FOR_CLAIM")

    def test_allowlist_entry(self):
        entry = json.loads((ROOT / "config/cb16_science_allowlist.json").read_text())["entrypoints"][r9.RESULT_COMMAND]
        self.assertIn("linear_action_contrast_diagnostic_r9", " ".join(entry["argv"]))
        self.assertEqual(set(entry["produces"]), {"experiment_spec.json", "RESULT.json", "REPORT.md"})


if __name__ == "__main__":
    unittest.main()
