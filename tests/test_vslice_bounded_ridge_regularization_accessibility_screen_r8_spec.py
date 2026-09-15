"""Preregistration invariants for VS-D R8 ridge-scale screen."""
from __future__ import annotations
import hashlib, json, unittest
from pathlib import Path
from cb16_science.evidence_scope import validate_experiment_spec

ROOT=Path(__file__).resolve().parents[1]
SPEC_PATH=ROOT/'config/experiments/r12_vs_d_bounded_ridge_regularization_accessibility_screen_r8.json'
R7_DOC=ROOT/'docs/experiments/frozen/R12_VS_D_MARCH_TEMPORAL_ROBUSTNESS_R7.md'
EXPECTED_SHA='8e6b51a30b4f12ebeaf6ec8b6d4ad0092ada445e9cb1c1721d139e3fb12f6d50'

class R8SpecTests(unittest.TestCase):
    def setUp(self): self.spec=json.loads(SPEC_PATH.read_text())
    def test_spec_sha_and_v2_contract(self):
        self.assertEqual(hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),EXPECTED_SHA)
        self.assertEqual(set(validate_experiment_spec(self.spec)),{'C_RIDGE_REGULARIZATION_SCREENING_PRIORITY'})
    def test_role_is_screening_not_validation(self):
        self.assertEqual(self.spec['experiment_role'],'SCREENING')
        self.assertEqual(self.spec['claims'][0]['vv_classification'],'NOT_APPLICABLE')
        self.assertTrue(self.spec['research_series_policy']['screening_is_not_qualification'])
    def test_data_is_only_jan_feb_common_r6_window(self):
        self.assertEqual([x['name'] for x in self.spec['data']['allowed_archives']],['BTCUSDT-1m-2020-01.zip','BTCUSDT-1m-2020-02.zip'])
        self.assertIn('BTCUSDT-1m-2020-03.zip',self.spec['data']['forbidden_reads'])
        self.assertEqual(self.spec['data']['fit']['expected_decisions'],615)
        self.assertEqual(self.spec['data']['validation']['expected_decisions'],672)
        self.assertEqual(self.spec['data']['context']['represented_hours'],64)
    def test_lambda_grid_reference_and_controls_are_frozen(self):
        s=self.spec['screen']
        self.assertEqual(s['ridge_lambdas'],[0.1,1.0,10.0,100.0,1000.0])
        self.assertEqual(s['candidate_lambdas'],[0.1,10.0,100.0,1000.0])
        self.assertEqual(s['reference_lambda'],1.0)
        self.assertEqual(s['control_seeds'],list(range(21001,21009)))
        self.assertTrue(s['same_target_permutation_across_lambdas'])
    def test_bootstrap_is_paired_and_dependence_aware(self):
        u=self.spec['uncertainty_protocol']
        self.assertEqual((u['bootstrap_replicates'],u['bootstrap_seed'],u['block_length_hours'],u['sample_length_hours'],u['lower_bound_order_index_zero_based']),(4096,22001,48,672,204))
        self.assertTrue(u['same_resamples_across_all_lambdas_and_controls'])
    def test_promotions_are_candidate_only(self):
        allowed=set(self.spec['allowed_promotion_decisions'])
        self.assertIn('NO_PROMOTION',allowed)
        self.assertEqual(len(allowed),5)
        invalid=set(self.spec['claims'][0]['invalid_inferences'])
        self.assertIn('does_not_qualify_historical_market_information',invalid)
        self.assertIn('does_not_establish_optimal_regularization',invalid)
    def test_r7_is_frozen_not_active(self):
        self.assertTrue(R7_DOC.is_file())
        self.assertFalse((ROOT/'docs/tasks/active/R12_VS_D_MARCH_TEMPORAL_ROBUSTNESS_R7.md').exists())

if __name__=='__main__': unittest.main()
