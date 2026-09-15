import copy,cProfile,pstats,sys,time
sys.path.insert(0,'/home/bgy/CB16-R12-motherfold/.worktrees/sol-fast/science')
from cb16_science.vslice import qualification as q
from cb16_science.vslice import task_a_direction_entropy_r2 as r2
spec=copy.deepcopy(r2.load_spec()); spec['task_a']['generations']=16
q.configure_deterministic_runtime(); prof=cProfile.Profile(); t=time.perf_counter(); prof.enable(); r2.run_seed(spec,1201); prof.disable(); elapsed=time.perf_counter()-t
print('elapsed_s',elapsed); pstats.Stats(prof).strip_dirs().sort_stats('cumulative').print_stats(25)
