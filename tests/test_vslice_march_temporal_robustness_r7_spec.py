"""Preregistration invariants for VS-D R7 March temporal robustness."""
from __future__ import annotations
import hashlib, json, unittest
from pathlib import Path
from cb16_science.evidence_scope import validate_experiment_spec

ROOT=Path(__file__).resolve().parents[1]
SPEC_PATH=ROOT/'config/experiments/r12_vs_d_march_temporal_robustness_r7.json'
R6_DOC=ROOT/'docs/experiments/frozen/R12_VS_D_BOUNDED_CONTEXT_LENGTH_ACCESSIBILITY_SCREEN_R6.md'
EXPECTED_SHA='31862bd6a50fcaa0ab9dc4379073d10e5ec55245a928fea7481d3e62bc4c8416'

class R7SpecTests(unittest.TestCase):
    def setUp(self): self.spec=json.loads(SPEC_PATH.read_text())
    def test_sha_and_v2_claim_contract(self):
        self.assertEqual(hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),EXPECTED_SHA)
        self.assertEqual(set(validate_experiment_spec(self.spec)),{'C_MARCH_TEMPORAL_ROBUSTNESS_OF_64H_LINEAR_HINT'})
    def test_role_is_robustness_not_confirmation(self):
        self.assertEqual(self.spec['experiment_role'],'ROBUSTNESS')
        self.assertEqual(self.spec['claims'][0]['vv_classification'],'NOT_APPLICABLE')
        self.assertTrue(self.spec['research_series_policy']['march_is_not_fresh_confirmation'])
        self.assertEqual(self.spec['data_exposure']['freshness_class'],'ADAPTIVELY_REUSED')
    def test_exact_allowed_archives_and_march_count(self):
        data=self.spec['data']
        self.assertEqual([x['name'] for x in data['allowed_archives']],['BTCUSDT-1m-2020-01.zip','BTCUSDT-1m-2020-02.zip','BTCUSDT-1m-2020-03.zip'])
        self.assertEqual(data['fit']['expected_decisions'],615)
        self.assertEqual(data['evaluation']['expected_decisions'],744)
        self.assertEqual(data['evaluation']['expected_utc_day_blocks'],31)
        self.assertTrue(data['evaluation']['target_fitting_forbidden'])
    def test_probe_matches_r6_reference_semantics(self):
        self.assertEqual(self.spec['data']['context']['represented_hours'],64)
        self.assertEqual(self.spec['data']['context']['raw_window_hours'],65)
        self.assertEqual(self.spec['probe']['ridge_lambda'],1.0)
        self.assertEqual(self.spec['probe']['control_seeds'],list(range(18001,18009)))
        self.assertIn('close[consequence_hour]',self.spec['probe']['target_formula'])
    def test_day_block_bootstrap_is_frozen(self):
        u=self.spec['uncertainty_protocol']
        self.assertEqual(u['bootstrap_replicates'],4096)
        self.assertEqual(u['bootstrap_seed'],20001)
        self.assertEqual(u['day_blocks'],31)
        self.assertEqual(u['hours_per_day'],24)
        self.assertEqual(u['lower_bound_order_index_zero_based'],204)
        self.assertTrue(u['same_resamples_across_true_and_controls'])
    def test_promotion_and_invalid_inference_are_local(self):
        self.assertEqual(set(self.spec['allowed_promotion_decisions']),{'NO_PROMOTION','PROMOTE_TEMPORAL_ROBUSTNESS_CANDIDATE_HYPOTHESIS'})
        invalid=set(self.spec['claims'][0]['invalid_inferences'])
        self.assertIn('does_not_make_March_fresh_confirmation',invalid)
        self.assertIn('does_not_qualify_historical_market_information',invalid)
        self.assertIn('does_not_prove_temporal_regime_shift_is_the_root_cause_if_gate_misses',invalid)
    def test_r6_is_frozen_not_active(self):
        self.assertTrue(R6_DOC.is_file())
        self.assertFalse((ROOT/'docs/tasks/active/R12_VS_D_BOUNDED_CONTEXT_LENGTH_ACCESSIBILITY_SCREEN_R6.md').exists())

if __name__=='__main__': unittest.main()
