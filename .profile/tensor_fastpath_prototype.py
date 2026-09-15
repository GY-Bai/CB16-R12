from __future__ import annotations
import copy, sys, time
from pathlib import Path
import torch
ROOT=Path('/home/bgy/CB16-R12-motherfold/.worktrees/sol-r2')
sys.path.insert(0,str(ROOT/'science'))
from cb16_science.vslice import controlled_tasks as tasks
from cb16_science.vslice import qualification as q
from cb16_science.vslice import qualification_vectorized as qv
from cb16_science.vslice import task_a_direction_entropy_r2 as r2
from cb16_science.vslice import vectorized_tasks as vt
from cb16_science.vslice.contracts import account_state_from_truth
from cb16_science.vslice.policy import DIRECTION_INDEX_FLAT, account_state_vector


def build(seed, coef):
    torch.manual_seed(seed)
    return q.build_learner(r2.parse_authority(SPEC), r2.parse_optimizer(SPEC),
                           direction_entropy_coefficient=coef)

def collect_tensor(learner, env, mode):
    schedule=env.positive_schedule if mode==tasks.ARM_POSITIVE else env.control_schedule
    actual=torch.tensor([a for a,_ in schedule],dtype=torch.long)
    observed=torch.tensor([o for _,o in schedule],dtype=torch.long)
    z=learner.encode(torch.from_numpy(env.market)).to(torch.float32)
    rows=torch.stack([account_state_vector(account_state_from_truth(c.truth,env.config)) for c in env.cases])
    states=torch.cat((z.expand(len(schedule),z.shape[-1]),rows.index_select(0,observed)),dim=-1)
    directions,risks=vt.sample_action_batch(learner,states)
    rewards=vt.task_a_reward_batch(env,actual,directions,risks)
    return states,directions,risks,rewards

def fast_update(learner,batch):
    states,directions,risks,returns=batch
    learner._require_generation_snapshot()
    if not (states.ndim==2 and directions.ndim==risks.ndim==returns.ndim==1): raise RuntimeError('shape')
    n=states.shape[0]
    if not (directions.numel()==risks.numel()==returns.numel()==n): raise RuntimeError('length')
    if not bool(torch.isfinite(states).all() & torch.isfinite(risks).all() & torch.isfinite(returns).all()): raise RuntimeError('finite')
    flat=directions==DIRECTION_INDEX_FLAT
    if bool((flat & (risks!=0)).any() | ((~flat)&(risks==0)).any()): raise RuntimeError('semantic')
    logp=learner.actor.log_prob_batch(states,directions,risks)
    values=learner.critic(states)
    advantages=returns-values.detach()
    entropy=learner.actor.direction_entropy_batch(states)
    actor_loss=-(logp*advantages).mean()-learner.direction_entropy_coefficient*entropy.mean()
    critic_loss=((values-returns)**2).mean()
    if not bool(torch.isfinite(actor_loss)&torch.isfinite(critic_loss)): raise RuntimeError('loss finite')
    learner.actor_optimizer.zero_grad(set_to_none=True); learner.critic_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward(); critic_loss.backward()
    learner.actor_optimizer.step(); learner.critic_optimizer.step()
    learner._generation_id += 1
    learner._actor_snapshot=learner._snapshot_actor_parameters()
    return float(actor_loss.detach()),float(critic_loss.detach())

def params_equal(a,b):
    return all(torch.equal(x.detach(),y.detach()) for x,y in zip(a.parameters(),b.parameters()))

def max_param_diff(a,b):
    return max(float((x.detach()-y.detach()).abs().max()) for x,y in zip(a.parameters(),b.parameters()))

SPEC=copy.deepcopy(r2.load_spec()); ENV=tasks.task_a_environment(SPEC)
q.configure_deterministic_runtime()
seed=1201; coef=0.005; mode=tasks.ARM_POSITIVE; stream=q._collection_seed(seed,'task_a',mode,0)
ref=build(seed,coef); fast=build(seed,coef); q.require_identical_parameters(ref,fast)
torch.manual_seed(stream); rb=vt.build_task_a_generation(ref,ENV,0,mode=mode); rr=ref.update(rb)
torch.manual_seed(stream); tb=collect_tensor(fast,ENV,mode); fa,fc=fast_update(fast,tb)
print('ONE_GEN actor_loss ref/fast',rr.actor_loss,fa,'diff',abs(rr.actor_loss-fa))
print('ONE_GEN critic_loss ref/fast',rr.critic_loss,fc,'diff',abs(rr.critic_loss-fc))
print('ONE_GEN actor exact',params_equal(ref.actor,fast.actor),'maxdiff',max_param_diff(ref.actor,fast.actor))
print('ONE_GEN critic exact',params_equal(ref.critic,fast.critic),'maxdiff',max_param_diff(ref.critic,fast.critic))

# benchmark 16 generations, four learners, same R2 streams

def run_reference(gens=16):
    learners=[]
    for arm in r2.ARM_ORDER:
        coef=r2.arm_definition(SPEC,arm).direction_entropy_coefficient
        learners += [(arm,tasks.ARM_POSITIVE,build(seed,coef)),(arm,tasks.ARM_CONTROL,build(seed,coef))]
    t=time.perf_counter()
    for arm,mode,l in learners:
        for g in range(gens):
            torch.manual_seed(q._collection_seed(seed,'task_a',mode,g))
            l.update(vt.build_task_a_generation(l,ENV,g,mode=mode))
    return time.perf_counter()-t

def run_fast(gens=16):
    learners=[]
    for arm in r2.ARM_ORDER:
        coef=r2.arm_definition(SPEC,arm).direction_entropy_coefficient
        learners += [(arm,tasks.ARM_POSITIVE,build(seed,coef)),(arm,tasks.ARM_CONTROL,build(seed,coef))]
    t=time.perf_counter()
    for arm,mode,l in learners:
        for g in range(gens):
            torch.manual_seed(q._collection_seed(seed,'task_a',mode,g))
            fast_update(l,collect_tensor(l,ENV,mode))
    return time.perf_counter()-t


if __name__ == '__main__':
    ref_t=run_reference(); fast_t=run_fast()
    print('BENCH reference_s',ref_t)
    print('BENCH fast_s',fast_t)
    print('BENCH speedup',ref_t/fast_t)
