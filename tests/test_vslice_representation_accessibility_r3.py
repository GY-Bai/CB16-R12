"""Implementation tests for VS-D R3 representation accessibility."""

from __future__ import annotations

import json
import math
import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cb16_science.evidence_scope import validate_result_against_spec
from cb16_science.vslice import representation_accessibility_r3 as r3


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "config" / "experiments" / "r12_vs_d_representation_accessibility_r3.json"


class RepresentationAccessibilityR3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = json.loads(SPEC.read_text())

    def test_committed_spec_sha_is_frozen(self) -> None:
        self.assertEqual(r3.load_spec(), self.spec)
        self.assertEqual(r3.SPEC_FILE_SHA256, "04a2151aa822c685abbf744c854cd0ed8a87df48b633367f2043e7ed217f4111")

    def test_standardizer_is_fit_only_from_training_rows(self) -> None:
        train = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
        validation = np.array([[1000.0, -900.0]])
        scaler = r3.fit_standardizer(train)
        self.assertTrue(np.allclose(scaler.mean, [3.0, 14.0]))
        transformed = scaler.transform(validation)
        self.assertGreater(abs(float(transformed[0, 0])), 100.0)

    def test_ridge_recovers_a_simple_linear_relation(self) -> None:
        rng = np.random.default_rng(12)
        x = rng.normal(size=(200, 3))
        y = 0.4 + 2.0 * x[:, 0] - 0.5 * x[:, 1]
        probe = r3.fit_ridge(x, y, ridge_lambda=1.0)
        prediction = probe.predict(x)
        self.assertGreater(r3.pearson_corr(prediction, y), 0.999)
        self.assertLess(r3.point_metrics(prediction, y)["rmse"], 0.03)

    def test_constant_training_feature_is_dropped_without_validation_peeking(self) -> None:
        x = np.column_stack((np.arange(20, dtype=float), np.ones(20)))
        scaler = r3.fit_standardizer(x)
        self.assertEqual(scaler.active.tolist(), [True, False])
        transformed = scaler.transform(np.array([[100.0, 999.0]]))
        self.assertEqual(transformed.shape, (1, 1))

    def test_paired_day_bootstrap_separates_signal_from_controls(self) -> None:
        rng = np.random.default_rng(42)
        target = rng.normal(size=696)
        true_prediction = target + 0.15 * rng.normal(size=696)
        controls = tuple(rng.normal(size=696) for _ in range(8))
        day_index = np.repeat(np.arange(29), 24)
        samples = r3.bootstrap_indices(day_index, replicates=256, seed=53031)
        summary = r3.bootstrap_surface(
            target,
            true_prediction,
            controls,
            samples,
            order_index=12,
        )
        self.assertGreater(summary["true_corr_lcb"], 0.9)
        self.assertGreater(summary["true_minus_median_shuffle_corr_lcb"], 0.8)

    def test_gap_bootstrap_is_paired_on_identical_resamples(self) -> None:
        rng = np.random.default_rng(7)
        target = rng.normal(size=696)
        better = target + 0.1 * rng.normal(size=696)
        worse = 0.25 * target + rng.normal(size=696)
        samples = r3.bootstrap_indices(np.repeat(np.arange(29), 24), replicates=256, seed=53031)
        lcb = r3.bootstrap_gap(target, better, worse, samples, order_index=12)
        self.assertGreater(lcb, 0.4)

    def test_representation_surfaces_have_exact_frozen_widths(self) -> None:
        rng = np.random.default_rng(19)
        dataset = r3.ProbeDataset(
            fit_market=rng.normal(size=(679, 64, 5)),
            fit_target=rng.normal(size=679),
            validation_market=rng.normal(size=(696, 64, 5)),
            validation_target=rng.normal(size=696),
            validation_day_index=np.repeat(np.arange(29), 24),
            evidence={},
        )
        surfaces = r3.representation_surfaces(dataset, self.spec)
        self.assertEqual(surfaces["N0_FLATTENED_64x5"][0].shape, (679, 320))
        self.assertEqual(surfaces["N0_FLATTENED_64x5"][1].shape, (696, 320))
        self.assertEqual(surfaces["FROZEN_SENSORY_Z32_SEED12012"][0].shape, (679, 32))
        self.assertEqual(surfaces["FROZEN_SENSORY_Z32_SEED12012"][1].shape, (696, 32))

    def test_validate_spec_rejects_march_as_allowed_archive(self) -> None:
        altered = json.loads(json.dumps(self.spec))
        altered["data"]["allowed_archives"].append({
            "name": "BTCUSDT-1m-2020-03.zip", "sha256": "x", "expected_rows": 1
        })
        with self.assertRaises(r3.ContractError):
            r3.validate_spec(altered)

    def test_failure_result_is_claim_local_v2(self) -> None:
        with mock.patch.object(r3.q, "commit_sha", return_value="a" * 40), mock.patch.object(
            r3, "runtime_record", return_value={}
        ):
            result = r3.failure_result(self.spec, "CONTRACT_MISMATCH", "fixture")
        validate_result_against_spec(result, self.spec)
        self.assertEqual(len(result["claim_assessments"]), 3)
        self.assertTrue(all(x["inference_status"] == "INVALID_FOR_CLAIM" for x in result["claim_assessments"]))

    def test_allowlist_entry_is_exact(self) -> None:
        allowlist = json.loads((ROOT / "config" / "cb16_science_allowlist.json").read_text())
        entry = allowlist["entrypoints"][r3.RESULT_COMMAND]
        self.assertEqual(entry["argv"][-2:], ["-m", "cb16_science.vslice.representation_accessibility_r3"])
        self.assertEqual(set(entry["produces"]), {"experiment_spec.json", "RESULT.json", "REPORT.md"})


if __name__ == "__main__":
    unittest.main()
