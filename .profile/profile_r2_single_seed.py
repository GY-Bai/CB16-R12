from __future__ import annotations
import copy, cProfile, pstats, sys, time
from pathlib import Path
ROOT=Path('/home/bgy/CB16-R12-motherfold/.worktrees/sol-r2')
sys.path.insert(0,str(ROOT/'science'))
from cb16_science.vslice import qualification as q
from cb16_science.vslice import task_a_direction_entropy_r2 as r2

spec=copy.deepcopy(r2.load_spec())
spec['task_a']['generations']=16
q.configure_deterministic_runtime()
prof=cProfile.Profile()
t0=time.perf_counter()
prof.enable()
record=r2.run_seed(spec,1201)
prof.disable()
elapsed=time.perf_counter()-t0
prof.dump_stats('/home/bgy/CB16-R12-motherfold/.profile/r2_16gen_seed1201.pstats')
print(f'elapsed_s={elapsed:.6f}')
for arm in r2.ARM_ORDER:
    b=record['arms'][arm]
    print(arm,'pos_last_actor=',b['positive']['diagnostics']['last_actor_loss'],
          'ctl_last_actor=',b['control']['diagnostics']['last_actor_loss'])
print('\n=== CUMULATIVE ===')
pstats.Stats(prof).strip_dirs().sort_stats('cumulative').print_stats(50)
print('\n=== TOTAL ===')
pstats.Stats(prof).strip_dirs().sort_stats('tottime').print_stats(50)
