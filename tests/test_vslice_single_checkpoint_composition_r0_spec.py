"""Preregistration invariants for VS-E R0 single-checkpoint composition."""
from __future__ import annotations
import hashlib,json,unittest
from pathlib import Path
from cb16_science.evidence_scope import validate_experiment_spec
ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'config/experiments/r12_vs_e_single_checkpoint_composition_r0.json'
PARENT=ROOT/'config/experiments/r12_vs_c_joint_confirmation_r3.json'
EXPECTED_SHA='6ca58656d2c29c976db762f1eb3eb6259208017ac0cae5060e5d02012c446fc5'
PARENT_RAW_SHA='e08a8c10d417d34600c97599b8f149bf4092cbe683da7edfa5ad32b7740cb56d'
class VSER0SpecTests(unittest.TestCase):
 def setUp(self): self.s=json.loads(SPEC.read_text())
 def test_spec_sha_and_claim_schema(self):
  self.assertEqual(hashlib.sha256(SPEC.read_bytes()).hexdigest(),EXPECTED_SHA)
  self.assertEqual(set(validate_experiment_spec(self.s)),{'C_SINGLE_CHECKPOINT_DUAL_TASK_COMPOSITION'})
 def test_parent_authority_locked(self):
  self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(),PARENT_RAW_SHA)
  p=self.s['parent_authority']; self.assertEqual(p['raw_spec_sha256'],PARENT_RAW_SHA); self.assertEqual(p['formal_run_id'],34908126391)
  self.assertEqual(p['formal_commit'],'465c8c916d0f67472b1575e177e8e24b1e29a4cf')
 def test_fresh_seeds(self):
  seeds=self.s['paired_seeds']; self.assertEqual(seeds,list(range(2501,2509)))
  self.assertTrue(set(seeds).isdisjoint(set(range(1201,1209))|set(range(2201,2209))))
 def test_single_checkpoint_schedule(self):
  t=self.s['training_schedule']; self.assertEqual(t['cycles'],256); self.assertEqual(t['updates_per_cycle'],['TASK_A','TASK_B'])
  self.assertEqual((t['total_updates'],t['task_a_updates'],t['task_b_updates']),(512,256,256))
  self.assertFalse(t['parameter_reset_between_updates']); self.assertFalse(t['optimizer_reset_between_updates'])
  self.assertIn('global monotonic 0..511',t['learner_generation_id']); self.assertIn('cycle index 0..255',t['task_local_generation_index'])
 def test_exact_evaluation_checkpoints(self):
  t=self.s['training_schedule']; self.assertEqual(t['evaluation_checkpoints'],['PRE','AFTER_FINAL_TASK_A_UPDATE','AFTER_FINAL_TASK_B_UPDATE'])
  self.assertIn('AFTER_FINAL_TASK_B_UPDATE for both',self.s['reused_r3_final_gates']['evaluation_checkpoint'])
 def test_parent_task_shapes_and_optimizer(self):
  self.assertEqual((self.s['task_a']['generations'],self.s['task_a']['batch_trajectories']),(256,900))
  self.assertEqual((self.s['task_b']['generations'],self.s['task_b']['batch_trajectories']),(256,128))
  o=self.s['optimizer']; self.assertEqual((o['actor_lr'],o['critic_lr'],o['direction_entropy_coefficient']),(0.001,0.001,0.005)); self.assertFalse(o['replay'])
 def test_reused_r3_gates_exact(self):
  parent=json.loads(PARENT.read_text()); g=self.s['reused_r3_final_gates']
  self.assertEqual(g['task_a'],parent['joint_gate']['task_a']); self.assertEqual(g['task_b'],parent['joint_gate']['task_b'])
 def test_composition_gate(self):
  g=self.s['composition_gate']; self.assertEqual(g['positive_joint_seed_pass_count_after_final_A_min'],7); self.assertEqual(g['positive_joint_seed_pass_count_after_final_B_min'],7)
  self.assertEqual(g['control_joint_seed_pass_count_after_final_A_max'],1); self.assertEqual(g['control_joint_seed_pass_count_after_final_B_max'],1)
  self.assertTrue(g['require_final_task_a_r3_aggregate_gate']); self.assertTrue(g['require_final_task_b_r3_aggregate_gate']); self.assertIn('same learner checkpoint',g['joint_seed_definition'])
 def test_scope_is_synthetic_only(self):
  self.assertFalse(self.s['claim_scope']['historical_market_data_used']); self.assertFalse(self.s['claim_scope']['historical_profitability_claim'])
  inv=set(self.s['claims'][0]['invalid_inferences']); self.assertIn('does_not_establish_continuing_account_composition',inv); self.assertIn('does_not_override_any_VS_D_result',inv)
  self.assertIn('no_historical_data_read',self.s['no_rescue'])
 def test_only_scoped_promotion(self):
  self.assertEqual(self.s['allowed_promotion_decisions'],['NO_PROMOTION','PROMOTE_SINGLE_CHECKPOINT_DUAL_TASK_COMPOSITION'])
if __name__=='__main__': unittest.main()
