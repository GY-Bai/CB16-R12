from __future__ import annotations
import cProfile,pstats,sys,time
sys.path.insert(0,'/home/bgy/CB16-R12-motherfold/.profile')
import tensor_fastpath_prototype as p
prof=cProfile.Profile(); t=time.perf_counter(); prof.enable(); elapsed=p.run_fast(16); prof.disable(); wall=time.perf_counter()-t
print(f'inner_elapsed_s={elapsed:.6f} wall_s={wall:.6f}')
pstats.Stats(prof).strip_dirs().sort_stats('cumulative').print_stats(45)
print('===TOTAL===')
pstats.Stats(prof).strip_dirs().sort_stats('tottime').print_stats(45)
