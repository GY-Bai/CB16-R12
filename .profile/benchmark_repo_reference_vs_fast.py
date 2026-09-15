import copy,sys,time,torch
sys.path.insert(0,'/home/bgy/CB16-R12-motherfold/.worktrees/sol-fast/science')
from cb16_science.vslice import controlled_tasks as tasks, qualification as q
from cb16_science.vslice import task_a_direction_entropy_r2 as r2, vectorized_tasks as vt
spec=copy.deepcopy(r2.load_spec()); env=tasks.task_a_environment(spec); q.configure_deterministic_runtime(); seed=1201

def build(coef):
    torch.manual_seed(seed)
    return q.build_learner(r2.parse_authority(spec),r2.parse_optimizer(spec),direction_entropy_coefficient=coef)
def params_equal(a,b): return all(torch.equal(x.detach(),y.detach()) for x,y in zip(a.parameters(),b.parameters()))
def make_sets():
    ref=[]; fast=[]
    for arm in r2.ARM_ORDER:
        coef=r2.arm_definition(spec,arm).direction_entropy_coefficient
        for mode in (tasks.ARM_POSITIVE,tasks.ARM_CONTROL):
            ref.append((arm,mode,build(coef))); fast.append((arm,mode,build(coef)))
    return ref,fast
ref,fast=make_sets(); gens=16
t=time.perf_counter()
for arm,mode,l in ref:
  for g in range(gens):
    torch.manual_seed(q._collection_seed(seed,'task_a',mode,g)); l.update(vt.build_task_a_generation(l,env,g,mode=mode))
ref_t=time.perf_counter()-t
t=time.perf_counter()
for arm,mode,l in fast:
  for g in range(gens):
    torch.manual_seed(q._collection_seed(seed,'task_a',mode,g)); l.update_one_step_batch(vt.build_task_a_one_step_batch(l,env,g,mode=mode))
fast_t=time.perf_counter()-t
print('reference_s',ref_t); print('fast_s',fast_t); print('speedup',ref_t/fast_t)
for (arm,mode,a),(_,_,b) in zip(ref,fast): print(arm,mode,'actor_exact',params_equal(a.actor,b.actor),'critic_exact',params_equal(a.critic,b.critic))
