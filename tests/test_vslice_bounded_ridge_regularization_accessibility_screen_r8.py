"""Implementation tests for VS-D R8 ridge regularization screen."""
from __future__ import annotations
import json, unittest
from pathlib import Path
import numpy as np
from cb16_science.evidence_scope import validate_result_against_spec
from cb16_science.vslice import bounded_ridge_regularization_accessibility_screen_r8 as r8

ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'config/experiments/r12_vs_d_bounded_ridge_regularization_accessibility_screen_r8.json').read_text())

class R8ImplementationTests(unittest.TestCase):
    def test_spec_sha_and_contract(self):
        self.assertEqual(r8.load_spec(),SPEC); r8.validate_spec(SPEC)
    def test_shared_control_orders_are_deterministic_no_fixed_point(self):
        a=r8.control_orders(615,[21001,21002]); b=r8.control_orders(615,[21001,21002])
        for seed in a:
            self.assertTrue(np.array_equal(a[seed],b[seed])); self.assertFalse(np.any(a[seed]==np.arange(615)))
    def test_fit_predictions_has_all_lambdas_and_shared_dimensions(self):
        rng=np.random.default_rng(81); fit=rng.normal(size=(615,8)); val=rng.normal(size=(672,8)); y=rng.normal(size=615)
        ds=r8.RidgeDataset(fit,val,y,np.zeros(672),{})
        spec=json.loads(json.dumps(SPEC)); spec['screen']['control_seeds']=[21001]
        out=r8.fit_predictions(ds,spec)
        self.assertEqual(set(out),{0.1,1.0,10.0,100.0,1000.0})
        for rec in out.values():
            self.assertEqual(rec['true_prediction'].shape,(672,)); self.assertEqual(len(rec['control_predictions']),1)
    def test_screen_can_select_nonreference_lambda(self):
        rng=np.random.default_rng(82); y=rng.normal(size=672)
        ref=0.35*y+1.0*rng.normal(size=672); strong=y+0.03*rng.normal(size=672)
        pred={}
        for lam in [0.1,1.0,10.0,100.0,1000.0]:
            true=strong if lam==10.0 else (ref if lam==1.0 else rng.normal(size=672))
            pred[lam]={'true_prediction':true,'control_predictions':tuple(rng.normal(size=672) for _ in range(8)),'active_feature_count':8,'input_feature_count':8}
        ds=r8.RidgeDataset(np.zeros((615,8)),np.zeros((672,8)),np.zeros(615),y,{})
        out=r8.screen_lambdas(ds,pred,SPEC)
        self.assertEqual(out['candidate_lambda'],10.0); self.assertTrue(out['records'][10.0]['screen_eligible'])
    def test_failure_result_is_invalid_for_claim(self):
        r=r8.failure_result(SPEC,'CONTRACT_MISMATCH','x'); validate_result_against_spec(r,SPEC)
        self.assertEqual(r['validity_status'],'INVALID'); self.assertEqual(r['claim_assessments'][0]['inference_status'],'INVALID_FOR_CLAIM')
    def test_allowlist_entry(self):
        e=json.loads((ROOT/'config/cb16_science_allowlist.json').read_text())['entrypoints'][r8.RESULT_COMMAND]
        self.assertIn('bounded_ridge_regularization_accessibility_screen_r8',' '.join(e['argv']))
        self.assertEqual(set(e['produces']),{'experiment_spec.json','RESULT.json','REPORT.md'})

if __name__=='__main__': unittest.main()
