"""Implementation tests for VS-E R0 without executing the formal 256-cycle experiment."""
from __future__ import annotations
import json,unittest
from pathlib import Path
from unittest import mock
import torch
from cb16_science.vslice import controlled_tasks as tasks
from cb16_science.vslice import joint_confirmation_r3 as r3
from cb16_science.vslice import qualification as q
from cb16_science.vslice import task_a_direction_entropy_r2 as r2
from cb16_science.vslice import vectorized_tasks as vt
from cb16_science.vslice import single_checkpoint_composition_r0 as vse
ROOT=Path(__file__).resolve().parents[1]
SPEC_PATH=ROOT/'config/experiments/r12_vs_e_single_checkpoint_composition_r0.json'
ALLOWLIST=ROOT/'config/cb16_science_allowlist.json'
class VSER0ImplementationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.spec=json.loads(SPEC_PATH.read_text()); q.configure_deterministic_runtime()
  cls.authority=r2.parse_authority(cls.spec); cls.optimizer=r3.parse_optimizer(cls.spec)
  cls.env_a=tasks.task_a_environment(cls.spec); cls.env_b=tasks.task_b_environment(cls.spec)
 def learner(self,seed=2501):
  torch.manual_seed(seed); return q.build_learner(self.authority,self.optimizer,direction_entropy_coefficient=r3.DIRECTION_ENTROPY_COEFFICIENT)
 def test_spec_validates_and_parent_is_locked(self):
  vse.validate_spec(self.spec); self.assertEqual(vse.paired_seeds(self.spec),tuple(range(2501,2509)))
 def test_allowlist_wires_exact_runner(self):
  d=json.loads(ALLOWLIST.read_text())['entrypoints'][vse.RESULT_COMMAND]
  self.assertEqual(d['argv'][-1],'cb16_science.vslice.single_checkpoint_composition_r0'); self.assertNotIn('dry_run_only',d)
 def test_interleaved_task_b_builder_equals_parent_when_indices_match(self):
  left=self.learner(); right=self.learner(); q.require_identical_parameters(left,right)
  rng=q._collection_seed(2501,'task_b',tasks.ARM_CONTROL,0)
  torch.manual_seed(rng); parent=vt.build_task_b_generation(left,self.env_b,0,mode=tasks.ARM_CONTROL,seed=2501)
  torch.manual_seed(rng); new=vse.build_task_b_interleaved_generation(right,self.env_b,global_generation_id=0,local_cycle=0,mode=tasks.ARM_CONTROL,seed=2501)
  self.assertEqual(len(parent),len(new))
  for a,b in zip(parent,new):
   self.assertEqual(a.generation_id,b.generation_id); self.assertEqual(len(a.steps),len(b.steps))
   for x,y in zip(a.steps,b.steps):
    self.assertTrue(torch.equal(x.state,y.state)); self.assertEqual(x.direction,y.direction); self.assertEqual(x.requested_risk,y.requested_risk); self.assertEqual(x.reward,y.reward)
 def test_evaluation_is_read_only(self):
  learner=self.learner(); before=vse._evaluation_snapshot(learner); metrics=vse.evaluate_both(learner,self.env_a,self.env_b); after=vse._evaluation_snapshot(learner)
  vse._assert_same_snapshot(before,after); self.assertIn('target_exposure_mae',metrics['task_a']); self.assertIn('mean_true_delayed_log_growth',metrics['task_b'])
 def test_one_interleaved_cycle_keeps_same_learner_and_optimizers(self):
  learner=self.learner(); aid=id(learner.actor_optimizer); cid=id(learner.critic_optimizer)
  vse._task_a_update(learner,self.env_a,seed=2501,mode=tasks.ARM_POSITIVE,cycle=0); self.assertEqual(learner.generation_id,1)
  vse._task_b_update(learner,self.env_b,seed=2501,mode=tasks.ARM_POSITIVE,cycle=0); self.assertEqual(learner.generation_id,2)
  self.assertEqual((id(learner.actor_optimizer),id(learner.critic_optimizer)),(aid,cid)); self.assertTrue(all(x==2 for x in vse._optimizer_steps(learner)))
 def test_task_b_local_control_index_is_not_global_generation(self):
  learner=self.learner(); captured=[]
  original=tasks.no_fixed_point_permutation
  def spy(count,*,seed,generation): captured.append(generation); return original(count,seed=seed,generation=generation)
  with mock.patch.object(tasks,'no_fixed_point_permutation',side_effect=spy):
   torch.manual_seed(q._collection_seed(2501,'task_b',tasks.ARM_CONTROL,7))
   batch=vse.build_task_b_interleaved_generation(learner,self.env_b,global_generation_id=0,local_cycle=7,mode=tasks.ARM_CONTROL,seed=2501)
  self.assertEqual(captured,[7]); self.assertTrue(all(t.generation_id==0 for t in batch))
 def test_composition_gate_requires_same_checkpoint_joint_pass(self):
  records=[]
  for seed in range(2501,2509):
   records.append({'seed':seed,'task_a':{'positive':{'seed_gate':{'passed':True}},'control':{'seed_gate':{'passed':False}},'paired':{'positive_target_mae_improvement':0.20,'positive_post_target_mae_strictly_better_than_control':True}},'task_b':{'positive':{'seed_gate':{'passed':True}},'control':{'seed_gate':{'passed':False}},'paired':{'positive_true_delayed_log_growth_improvement':0.03,'positive_minus_control_true_delayed_log_growth':0.03}},'checkpoints':{'AFTER_FINAL_TASK_A_UPDATE':{'positive':{'joint_pass':True},'control':{'joint_pass':False}},'AFTER_FINAL_TASK_B_UPDATE':{'positive':{'joint_pass':True},'control':{'joint_pass':False}}}})
  gate=vse.composition_gate(self.spec,records); self.assertTrue(gate['passed']); self.assertEqual(gate['coexistence']['counts']['positive_joint_seed_pass_count_after_final_B'],8)
  records[0]['checkpoints']['AFTER_FINAL_TASK_B_UPDATE']['positive']['joint_pass']=False; records[1]['checkpoints']['AFTER_FINAL_TASK_B_UPDATE']['positive']['joint_pass']=False
  self.assertFalse(vse.composition_gate(self.spec,records)['passed'])
if __name__=='__main__': unittest.main()
