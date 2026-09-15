"""Implementation tests for VS-D R5 bounded horizon screening."""
from __future__ import annotations
import json, unittest
from pathlib import Path
import numpy as np
from cb16_science.evidence_scope import validate_result_against_spec
from cb16_science.vslice import bounded_horizon_accessibility_screen_r5 as r5

ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'config/experiments/r12_vs_d_bounded_horizon_accessibility_screen_r5.json').read_text())

class R5ImplementationTests(unittest.TestCase):
    def test_spec_sha_and_contract(self):
        self.assertEqual(r5.load_spec(),SPEC); r5.validate_spec(SPEC)

    def test_forward_target_exact_horizons(self):
        hourly=np.zeros((40,5),dtype=np.float64)
        hourly[:,0]=np.arange(100,140,dtype=np.float64)
        hourly[:,1]=hourly[:,0]+2; hourly[:,2]=hourly[:,0]-2
        hourly[:,3]=hourly[:,0]+1; hourly[:,4]=1
        idx=np.array([5,6],dtype=np.int64)
        h1=r5._forward_target(hourly,idx,1)
        self.assertTrue(np.allclose(h1,hourly[idx,3]/hourly[idx,0]-1))
        h4=r5._forward_target(hourly,idx,4)
        self.assertTrue(np.allclose(h4,hourly[idx+3,3]/hourly[idx,0]-1))
        h24=r5._forward_target(hourly,idx,24)
        self.assertTrue(np.allclose(h24,hourly[idx+23,3]/hourly[idx,0]-1))

    def test_circular_block_bootstrap_is_deterministic_and_bounded(self):
        a=r5.circular_block_samples(672,48,4,17001)
        b=r5.circular_block_samples(672,48,4,17001)
        self.assertEqual(len(a),4)
        for x,y in zip(a,b):
            self.assertTrue(np.array_equal(x,y)); self.assertEqual(x.shape,(672,))
            self.assertGreaterEqual(int(x.min()),0); self.assertLess(int(x.max()),672)

    def test_screen_selects_only_eligible_best_horizon(self):
        rng=np.random.default_rng(44); base=rng.normal(size=672)
        targets={1:base,4:base,12:base,24:base}
        # 1h is a mediocre reference; 4h is strongest; 12/24 fail control/gain.
        pred={
          1:{'true_prediction':0.45*base+0.9*rng.normal(size=672),'control_predictions':tuple(rng.normal(size=672) for _ in range(8))},
          4:{'true_prediction':base+0.03*rng.normal(size=672),'control_predictions':tuple(rng.normal(size=672) for _ in range(8))},
          12:{'true_prediction':0.35*base+1.0*rng.normal(size=672),'control_predictions':tuple(rng.normal(size=672) for _ in range(8))},
          24:{'true_prediction':rng.normal(size=672),'control_predictions':tuple(rng.normal(size=672) for _ in range(8))},
        }
        ds=r5.HorizonDataset(np.zeros((655,320)),np.zeros((672,320)),{h:np.zeros(655) for h in [1,4,12,24]},targets,{})
        out=r5.screen_horizons(ds,pred,SPEC)
        self.assertEqual(out['candidate_horizon_hours'],4)
        self.assertTrue(out['records'][4]['screen_eligible'])

    def test_failure_result_is_invalid_not_scientific_fail(self):
        r=r5.failure_result(SPEC,'CONTRACT_MISMATCH','x')
        validate_result_against_spec(r,SPEC)
        self.assertEqual(r['validity_status'],'INVALID')
        self.assertEqual(r['claim_assessments'][0]['inference_status'],'INVALID_FOR_CLAIM')
        self.assertEqual(r['claim_assessments'][0]['qualification_status'],'NOT_APPLICABLE')

    def test_allowlist_entry(self):
        e=json.loads((ROOT/'config/cb16_science_allowlist.json').read_text())['entrypoints'][r5.RESULT_COMMAND]
        self.assertIn('bounded_horizon_accessibility_screen_r5',' '.join(e['argv']))
        self.assertEqual(set(e['produces']),{'experiment_spec.json','RESULT.json','REPORT.md'})

if __name__=='__main__': unittest.main()
