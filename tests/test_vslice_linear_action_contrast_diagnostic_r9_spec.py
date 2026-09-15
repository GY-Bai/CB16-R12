"""Preregistration invariants for VS-D R9 linear action contrast."""
from __future__ import annotations
import hashlib,json,unittest
from pathlib import Path
from cb16_science.evidence_scope import validate_experiment_spec
ROOT=Path(__file__).resolve().parents[1]
SPEC_PATH=ROOT/'config/experiments/r12_vs_d_linear_action_contrast_diagnostic_r9.json'
R8_DOC=ROOT/'docs/experiments/frozen/R12_VS_D_BOUNDED_RIDGE_REGULARIZATION_ACCESSIBILITY_SCREEN_R8.md'
EXPECTED_SHA='0da7b886a53fcab1b4f0c3b80dfe66d3f9705d550b80d2bdbbb478ecc678a2ae'
class R9SpecTests(unittest.TestCase):
 def setUp(self): self.spec=json.loads(SPEC_PATH.read_text())
 def test_spec_sha_and_v2(self):
  self.assertEqual(hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),EXPECTED_SHA); self.assertEqual(set(validate_experiment_spec(self.spec)),{'C_LINEAR_ACTION_CONTRAST_ACCESSIBILITY'})
 def test_role_and_vv(self):
  self.assertEqual(self.spec['experiment_role'],'DIAGNOSTIC'); self.assertEqual(self.spec['claims'][0]['vv_classification'],'NOT_APPLICABLE')
 def test_reference_probe_is_unpromoted_lambda1(self):
  self.assertEqual(self.spec['probe']['ridge_lambda'],1.0); self.assertEqual(self.spec['data']['context']['represented_hours'],64)
  self.assertIn('no_lambda_change',self.spec['no_rescue'])
 def test_action_mapping_has_no_tuned_deadband(self):
  self.assertIn('exact zero maps FLAT',self.spec['probe']['action_mapping']); self.assertIn('no_action_threshold_or_FLAT_deadband',self.spec['no_rescue'])
 def test_data_is_jan_feb_only(self):
  self.assertEqual([x['name'] for x in self.spec['data']['allowed_archives']],['BTCUSDT-1m-2020-01.zip','BTCUSDT-1m-2020-02.zip']); self.assertIn('BTCUSDT-1m-2020-03.zip',self.spec['data']['forbidden_reads'])
  self.assertEqual((self.spec['data']['fit']['expected_decisions'],self.spec['data']['evaluation']['expected_decisions']),(615,672))
 def test_controls_and_bootstrap_frozen(self):
  self.assertEqual(self.spec['probe']['control_seeds'],list(range(23001,23009))); u=self.spec['uncertainty_protocol']; self.assertEqual((u['bootstrap_replicates'],u['bootstrap_seed'],u['block_length_hours'],u['sample_length_hours'],u['lower_bound_order_index_zero_based']),(4096,24001,48,672,204))
 def test_authority_is_not_profitability(self):
  inv=set(self.spec['claims'][0]['invalid_inferences']); self.assertIn('does_not_establish_profitability',inv); self.assertIn('does_not_survive_transaction_costs_or_funding',inv)
 def test_r8_frozen_not_active(self):
  self.assertTrue(R8_DOC.is_file()); self.assertFalse((ROOT/'docs/tasks/active/R12_VS_D_BOUNDED_RIDGE_REGULARIZATION_ACCESSIBILITY_SCREEN_R8.md').exists())
if __name__=='__main__': unittest.main()
