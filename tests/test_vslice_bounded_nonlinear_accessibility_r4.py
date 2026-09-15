"""Implementation tests for VS-D R4 bounded nonlinear accessibility."""
from __future__ import annotations

import json, unittest
from pathlib import Path
import numpy as np
import torch

from cb16_science.evidence_scope import validate_result_against_spec
from cb16_science.vslice import bounded_nonlinear_accessibility_r4 as r4
from cb16_science.vslice import representation_accessibility_r3 as r3

ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'config/experiments/r12_vs_d_bounded_nonlinear_accessibility_r4.json').read_text())

class R4ImplementationTests(unittest.TestCase):
    def test_spec_sha_and_contract(self):
        self.assertEqual(r4.load_spec(), SPEC)
        r4.validate_spec(SPEC)

    def test_same_seed_initialization_is_bitwise_identical(self):
        a=r4._build_model(7,4501); b=r4._build_model(7,4501)
        for x,y in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.equal(x,y))

    def test_training_is_deterministic_for_same_seed_and_data(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(48,6)).astype(np.float64)
        y=(x[:,0]*x[:,1]+0.1*x[:,2]).astype(np.float64)
        ys,mean,std=r4._standardized_target(y)
        scaler=r3.fit_standardizer(x); xs=scaler.transform(x)
        a=r4._build_model(xs.shape[1],4502); b=r4._build_model(xs.shape[1],4502)
        r4._train_model(a,xs,ys,SPEC); r4._train_model(b,xs,ys,SPEC)
        for p,q in zip(a.parameters(),b.parameters()): self.assertTrue(torch.equal(p,q))
        self.assertTrue(np.array_equal(r4._predict(a,xs,mean,std),r4._predict(b,xs,mean,std)))

    def test_fit_seed_predictions_preserves_shapes(self):
        rng=np.random.default_rng(9)
        fit=rng.normal(size=(64,8)); val=rng.normal(size=(20,8)); y=rng.normal(size=64)
        out=r4.fit_seed_predictions(fit,val,y,SPEC,4501,14501)
        self.assertEqual(out['true_prediction'].shape,(20,))
        self.assertEqual(out['control_prediction'].shape,(20,))
        self.assertEqual(out['active_feature_count'],8)

    def test_evaluate_seed_paired_bootstrap_detects_strong_signal(self):
        rng=np.random.default_rng(12)
        y=rng.normal(size=696)
        true=y+0.05*rng.normal(size=696)
        control=rng.normal(size=696)
        linear=0.4*y+0.9*rng.normal(size=696)
        dataset=r3.ProbeDataset(np.zeros((679,64,5)),np.zeros(679),np.zeros((696,64,5)),y,np.repeat(np.arange(29),24),{})
        out=r4.evaluate_seed(dataset,true,control,linear,15401,SPEC)
        self.assertGreater(out['bootstrap']['mlp_true_corr_lcb'],0)
        self.assertGreater(out['bootstrap']['true_minus_control_corr_lcb'],0)
        self.assertGreater(out['bootstrap']['true_minus_linear_corr_lcb'],0)

    def test_failure_result_respects_v2_claim_authority(self):
        result=r4.failure_result(SPEC,'CONTRACT_MISMATCH','x')
        validate_result_against_spec(result,SPEC)
        self.assertEqual(result['validity_status'],'INVALID')
        self.assertTrue(all(x['inference_status']=='INVALID_FOR_CLAIM' for x in result['claim_assessments']))

    def test_allowlist_entry(self):
        allow=json.loads((ROOT/'config/cb16_science_allowlist.json').read_text())['entrypoints'][r4.RESULT_COMMAND]
        self.assertIn('bounded_nonlinear_accessibility_r4', ' '.join(allow['argv']))
        self.assertEqual(set(allow['produces']),{'experiment_spec.json','RESULT.json','REPORT.md'})

if __name__=='__main__': unittest.main()
