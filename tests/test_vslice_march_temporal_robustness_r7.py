"""Implementation tests for VS-D R7 March temporal robustness."""
from __future__ import annotations
import json, unittest
from pathlib import Path
import numpy as np
from cb16_science.evidence_scope import validate_result_against_spec
from cb16_science.vslice import march_temporal_robustness_r7 as r7

ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'config/experiments/r12_vs_d_march_temporal_robustness_r7.json').read_text())

class R7ImplementationTests(unittest.TestCase):
    def test_spec_sha_and_contract(self):
        self.assertEqual(r7.load_spec(),SPEC); r7.validate_spec(SPEC)

    def test_control_orders_are_no_fixed_point_and_deterministic(self):
        a=r7.control_orders(615,[18001,18002]); b=r7.control_orders(615,[18001,18002])
        for seed in a:
            self.assertTrue(np.array_equal(a[seed],b[seed]))
            self.assertFalse(np.any(a[seed]==np.arange(615)))
            self.assertTrue(np.array_equal(np.sort(a[seed]),np.arange(615)))

    def test_march_bootstrap_is_31_paired_day_blocks(self):
        days=np.repeat(np.arange(31),24)
        a=r7.march_bootstrap_indices(days,replicates=4,seed=20001)
        b=r7.march_bootstrap_indices(days,replicates=4,seed=20001)
        self.assertEqual(len(a),4)
        for x,y in zip(a,b):
            self.assertTrue(np.array_equal(x,y)); self.assertEqual(x.shape,(744,))
            # each sampled 24-row chunk must be one complete original UTC day block
            for k in range(31):
                chunk=x[k*24:(k+1)*24]
                self.assertEqual(len(np.unique(days[chunk])),1)

    def test_fit_predictions_uses_only_fit_target_and_march_features(self):
        rng=np.random.default_rng(71)
        fit_x=rng.normal(size=(615,8)); march_x=rng.normal(size=(744,8)); y=rng.normal(size=615)
        ds=r7.RobustnessDataset(fit_x,march_x,y,np.zeros(744),np.repeat(np.arange(31),24),{})
        spec=json.loads(json.dumps(SPEC)); spec['probe']['control_seeds']=[18001]
        out=r7.fit_predictions(ds,spec)
        self.assertEqual(out['true_prediction'].shape,(744,))
        self.assertEqual(len(out['control_predictions']),1)
        self.assertEqual(out['control_predictions'][0].shape,(744,))

    def test_evaluate_robustness_detects_strong_signal(self):
        rng=np.random.default_rng(72); y=rng.normal(size=744)
        true=y+0.05*rng.normal(size=744)
        ctrls=tuple(rng.normal(size=744) for _ in range(8))
        ds=r7.RobustnessDataset(np.zeros((615,3)),np.zeros((744,3)),np.zeros(615),y,np.repeat(np.arange(31),24),{})
        out=r7.evaluate_robustness(ds,{'true_prediction':true,'control_predictions':ctrls,'active_feature_count':3,'input_feature_count':3},SPEC)
        self.assertGreater(out['bootstrap']['true_corr_lcb'],0)
        self.assertGreater(out['bootstrap']['true_minus_median_shuffle_corr_lcb'],0)

    def test_failure_result_is_invalid_for_claim(self):
        result=r7.failure_result(SPEC,'CONTRACT_MISMATCH','x')
        validate_result_against_spec(result,SPEC)
        self.assertEqual(result['validity_status'],'INVALID')
        self.assertEqual(result['claim_assessments'][0]['inference_status'],'INVALID_FOR_CLAIM')
        self.assertEqual(result['claim_assessments'][0]['qualification_status'],'NOT_APPLICABLE')

    def test_allowlist_entry(self):
        entry=json.loads((ROOT/'config/cb16_science_allowlist.json').read_text())['entrypoints'][r7.RESULT_COMMAND]
        self.assertIn('march_temporal_robustness_r7',' '.join(entry['argv']))
        self.assertEqual(set(entry['produces']),{'experiment_spec.json','RESULT.json','REPORT.md'})

if __name__=='__main__': unittest.main()
