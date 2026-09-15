"""Implementation tests for VS-D R6 context-length screening."""
from __future__ import annotations
import json, unittest
from pathlib import Path
import numpy as np
from cb16_science.evidence_scope import validate_result_against_spec
from cb16_science.normalization.transforms import n0_endpoint_log_ratios
from cb16_science.vslice import bounded_context_length_accessibility_screen_r6 as r6

ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'config/experiments/r12_vs_d_bounded_context_length_accessibility_screen_r6.json').read_text())

class R6ImplementationTests(unittest.TestCase):
    def test_spec_sha_and_contract(self):
        self.assertEqual(r6.load_spec(),SPEC); r6.validate_spec(SPEC)

    def test_n0_supports_all_frozen_context_lengths(self):
        for represented in [16,32,64,128]:
            raw=np.ones((3,represented+1,5),dtype=np.float64)
            base=np.linspace(100,101,represented+1)
            raw[:,:,0]=base; raw[:,:,1]=base*1.001; raw[:,:,2]=base*0.999; raw[:,:,3]=base; raw[:,:,4]=np.linspace(10,11,represented+1)
            self.assertEqual(n0_endpoint_log_ratios(raw).values.shape,(3,represented,5))

    def test_control_orders_are_deterministic_no_fixed_points(self):
        a=r6.control_orders(615,[18001,18002]); b=r6.control_orders(615,[18001,18002])
        for seed in [18001,18002]:
            self.assertTrue(np.array_equal(a[seed],b[seed]))
            self.assertFalse(np.any(a[seed]==np.arange(615)))
            self.assertTrue(np.array_equal(np.sort(a[seed]),np.arange(615)))

    def test_screen_selects_strongest_eligible_context(self):
        rng=np.random.default_rng(66); y=rng.normal(size=672)
        pred={
          16:{'true_prediction':0.5*y+0.8*rng.normal(size=672),'control_predictions':tuple(rng.normal(size=672) for _ in range(8)),'active_feature_count':80,'input_feature_count':80},
          32:{'true_prediction':y+0.03*rng.normal(size=672),'control_predictions':tuple(rng.normal(size=672) for _ in range(8)),'active_feature_count':160,'input_feature_count':160},
          64:{'true_prediction':0.42*y+0.9*rng.normal(size=672),'control_predictions':tuple(rng.normal(size=672) for _ in range(8)),'active_feature_count':320,'input_feature_count':320},
          128:{'true_prediction':0.3*y+1.0*rng.normal(size=672),'control_predictions':tuple(rng.normal(size=672) for _ in range(8)),'active_feature_count':640,'input_feature_count':640},
        }
        ds=r6.ContextDataset({c:np.zeros((615,c*5)) for c in [16,32,64,128]},{c:np.zeros((672,c*5)) for c in [16,32,64,128]},np.zeros(615),y,{})
        out=r6.screen_contexts(ds,pred,SPEC)
        self.assertEqual(out['candidate_context_hours'],32)
        self.assertTrue(out['records'][32]['screen_eligible'])

    def test_failure_result_is_invalid_not_scientific_result(self):
        r=r6.failure_result(SPEC,'CONTRACT_MISMATCH','x')
        validate_result_against_spec(r,SPEC)
        self.assertEqual(r['validity_status'],'INVALID')
        self.assertEqual(r['claim_assessments'][0]['qualification_status'],'NOT_APPLICABLE')
        self.assertEqual(r['claim_assessments'][0]['inference_status'],'INVALID_FOR_CLAIM')

    def test_allowlist_entry(self):
        e=json.loads((ROOT/'config/cb16_science_allowlist.json').read_text())['entrypoints'][r6.RESULT_COMMAND]
        self.assertIn('bounded_context_length_accessibility_screen_r6',' '.join(e['argv']))
        self.assertEqual(set(e['produces']),{'experiment_spec.json','RESULT.json','REPORT.md'})

if __name__=='__main__': unittest.main()
