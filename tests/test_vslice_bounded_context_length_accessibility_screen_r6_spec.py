"""Preregistration invariants for VS-D R6 context-length screen."""
from __future__ import annotations
import hashlib, json, unittest
from pathlib import Path
from cb16_science.evidence_scope import validate_experiment_spec

ROOT=Path(__file__).resolve().parents[1]
SPEC_PATH=ROOT/'config/experiments/r12_vs_d_bounded_context_length_accessibility_screen_r6.json'
R5_DOC=ROOT/'docs/experiments/frozen/R12_VS_D_BOUNDED_HORIZON_ACCESSIBILITY_SCREEN_R5.md'
EXPECTED_SHA='3e4706036bb7227aff885ec768d1c6a1db32e17e5d4f4aed997c42c091d3c8dd'

class R6SpecTests(unittest.TestCase):
    def setUp(self): self.spec=json.loads(SPEC_PATH.read_text())
    def test_spec_sha_and_v2_contract(self):
        self.assertEqual(hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),EXPECTED_SHA)
        self.assertEqual(set(validate_experiment_spec(self.spec)),{'C_CONTEXT_LENGTH_SCREENING_PRIORITY'})
    def test_role_and_vv_are_screening_local(self):
        self.assertEqual(self.spec['experiment_role'],'SCREENING')
        self.assertEqual(self.spec['claims'][0]['vv_classification'],'NOT_APPLICABLE')
        self.assertTrue(self.spec['research_series_policy']['screening_is_not_qualification'])
    def test_context_set_and_reference_are_frozen(self):
        self.assertEqual(self.spec['screen']['context_lengths_hours'],[16,32,64,128])
        self.assertEqual(self.spec['screen']['candidate_contexts_hours'],[16,32,128])
        self.assertEqual(self.spec['screen']['reference_context_hours'],64)
        self.assertEqual([(x['represented_hours'],x['raw_window_hours'],x['feature_width']) for x in self.spec['data']['contexts']],[(16,17,80),(32,33,160),(64,65,320),(128,129,640)])
    def test_common_jan_feb_windows(self):
        data=self.spec['data']
        self.assertEqual([x['name'] for x in data['allowed_archives']],['BTCUSDT-1m-2020-01.zip','BTCUSDT-1m-2020-02.zip'])
        self.assertIn('BTCUSDT-1m-2020-03.zip',data['forbidden_reads'])
        self.assertEqual(data['common_fit']['expected_decisions'],615)
        self.assertEqual(data['common_validation']['expected_decisions'],672)
    def test_target_probe_and_controls_are_frozen(self):
        s=self.spec['screen']
        self.assertIn('close[first_consequence_hour]',s['target_formula'])
        self.assertEqual(s['ridge_lambda'],1.0)
        self.assertEqual(s['control_seeds'],list(range(18001,18009)))
        self.assertTrue(s['same_target_permutation_across_contexts'])
    def test_bootstrap_is_shared_and_dependence_aware(self):
        u=self.spec['uncertainty_protocol']
        self.assertEqual(u['bootstrap_replicates'],4096)
        self.assertEqual(u['bootstrap_seed'],19001)
        self.assertEqual(u['block_length_hours'],48)
        self.assertEqual(u['sample_length_hours'],672)
        self.assertTrue(u['same_resamples_across_all_contexts_and_controls'])
        self.assertEqual(u['lower_bound_order_index_zero_based'],204)
    def test_dimension_regularization_is_explicitly_unresolved(self):
        alternatives=' '.join(self.spec['claims'][0]['credible_alternatives'])
        self.assertIn('dimension-dependent regularization',alternatives)
        self.assertIn('does_not_prove_context_length_is_the_unique_bottleneck',self.spec['claims'][0]['invalid_inferences'])
    def test_promotion_is_candidate_only(self):
        self.assertEqual(set(self.spec['allowed_promotion_decisions']),{'NO_PROMOTION','PROMOTE_CONTEXT16_CANDIDATE_HYPOTHESIS','PROMOTE_CONTEXT32_CANDIDATE_HYPOTHESIS','PROMOTE_CONTEXT128_CANDIDATE_HYPOTHESIS'})
    def test_r5_is_frozen_not_active(self):
        self.assertTrue(R5_DOC.is_file())
        self.assertFalse((ROOT/'docs/tasks/active/R12_VS_D_BOUNDED_HORIZON_ACCESSIBILITY_SCREEN_R5.md').exists())

if __name__=='__main__': unittest.main()
