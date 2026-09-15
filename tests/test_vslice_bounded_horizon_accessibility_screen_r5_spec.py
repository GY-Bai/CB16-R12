"""Preregistration invariants for VS-D R5 horizon screen."""
from __future__ import annotations
import hashlib, json, unittest
from pathlib import Path
from cb16_science.evidence_scope import validate_experiment_spec

ROOT=Path(__file__).resolve().parents[1]
SPEC_PATH=ROOT/'config/experiments/r12_vs_d_bounded_horizon_accessibility_screen_r5.json'
R4_DOC=ROOT/'docs/experiments/frozen/R12_VS_D_BOUNDED_NONLINEAR_ACCESSIBILITY_R4.md'
EXPECTED_SHA='f85afb41c66ec4b452185418626dabe51d0abadeb59dddc37cc89cb6c22e935e'

class R5SpecTests(unittest.TestCase):
    def setUp(self): self.spec=json.loads(SPEC_PATH.read_text())
    def test_spec_sha_and_v2_contract(self):
        self.assertEqual(hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),EXPECTED_SHA)
        self.assertEqual(set(validate_experiment_spec(self.spec)),{'C_HORIZON_SCREENING_PRIORITY'})
    def test_role_is_screening_and_not_qualification(self):
        self.assertEqual(self.spec['experiment_role'],'SCREENING')
        c=self.spec['claims'][0]
        self.assertEqual(c['vv_classification'],'NOT_APPLICABLE')
        self.assertTrue(self.spec['research_series_policy']['screening_is_not_qualification'])
    def test_only_jan_feb_and_common_windows(self):
        self.assertEqual([x['name'] for x in self.spec['data']['allowed_archives']],['BTCUSDT-1m-2020-01.zip','BTCUSDT-1m-2020-02.zip'])
        self.assertIn('BTCUSDT-1m-2020-03.zip',self.spec['data']['forbidden_reads'])
        self.assertEqual(self.spec['data']['common_fit']['expected_decisions'],655)
        self.assertEqual(self.spec['data']['common_validation']['expected_decisions'],672)
        self.assertEqual(self.spec['data']['common_validation']['calendar_days'],28)
    def test_horizon_set_target_and_probe_are_frozen(self):
        self.assertEqual(self.spec['screen']['horizons_hours'],[1,4,12,24])
        self.assertEqual(self.spec['screen']['candidate_horizons_hours'],[4,12,24])
        self.assertEqual(self.spec['screen']['ridge_lambda'],1.0)
        self.assertIn('h - 1',self.spec['screen']['target_formula'])
        self.assertEqual(self.spec['screen']['control_seeds'],list(range(16001,16009)))
    def test_bootstrap_is_dependence_aware_and_paired(self):
        u=self.spec['uncertainty_protocol']
        self.assertEqual(u['bootstrap_replicates'],4096)
        self.assertEqual(u['block_length_hours'],48)
        self.assertEqual(u['sample_length_hours'],672)
        self.assertTrue(u['same_resamples_across_all_horizons_and_controls'])
        self.assertEqual(u['lower_bound_order_index_zero_based'],204)
    def test_promotion_is_candidate_only(self):
        allowed=set(self.spec['allowed_promotion_decisions'])
        self.assertEqual(allowed,{'NO_PROMOTION','PROMOTE_H4_CANDIDATE_HYPOTHESIS','PROMOTE_H12_CANDIDATE_HYPOTHESIS','PROMOTE_H24_CANDIDATE_HYPOTHESIS'})
        invalid=set(self.spec['claims'][0]['invalid_inferences'])
        self.assertIn('does_not_qualify_any_horizon_for_historical_market_information',invalid)
        self.assertIn('does_not_make_screening_confirmatory',invalid)
    def test_r4_contract_is_frozen_not_active(self):
        self.assertTrue(R4_DOC.is_file())
        self.assertFalse((ROOT/'docs/tasks/active/R12_VS_D_BOUNDED_NONLINEAR_ACCESSIBILITY_R4.md').exists())

if __name__=='__main__': unittest.main()
