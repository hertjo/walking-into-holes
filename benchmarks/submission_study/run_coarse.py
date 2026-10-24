"""Additional coarse-multigrid and geometric-proximity baselines, run serially."""
import json,numpy as np
from run_study import *
dump(OUT/'coarse-protocol.json',{'baselines':['mg49','nearest'],'nearest_rule':'Euclidean distance to sensor center; same feasibility constraints and candidate pool','longest_axis':49,'reason':'Audit a cheaper deterministic design baseline and a simple objective-aware geometric control, added during the campaign.','seeds':SEEDS,'refinement_rounds':8,'evaluation':'same independent blocks as primary comparison','all_other_constraints':'unchanged'})
for name in SPECS:
 for ch,study in [(False,'pipeline'),(True,'channels')]:
  s,m=load(name)
  for method in ['mg49','nearest']:
   for seed in SEEDS:
    d=greedy(name,method,seed,ch)
    ref=refine(d,8) if not ch else None
    path=OUT/study/name/'evaluation'/str(seed)/'evaluation.json';ev=json.loads(path.read_text())
    if f'{method}-initial' in ev and (ch or f'{method}-refine8' in ev):continue
    sol=Solver(name,OUT/study/name/f'{method}-evaluation'/str(seed)/'raw')
    states=[('initial',d['centers'],d['radius'])]
    if not ch:states += [(f'refine{h["round"]}',h['centers'],R) for h in ref['history'] if h['round'] in [4,8]]
    for label,axes,r in states:
     holes=channel_balls(axes) if ch else np.array(axes)
     ev[f'{method}-{label}']=[objective(sol,f'{method}-{label}-block{b}',s,holes,r,880000+b,walks=512,nprobe=128) for b in range(4)]
    if ch:ev[f'{method}-volume']=[Volume(m,seed=66100+b,power=20).volume(channel_balls(d['centers']),d['radius']) for b in range(4)]
    if seed==SEEDS[0]:
     holes=channel_balls(d['centers']) if ch else d['centers']
     ev['grid_check'][method]=[objective(sol,f'grid{n}-{method}',s,holes,d['radius'],99123,mode='mg',n=n,nprobe=512) for n in [97,129]]
    sol.close();dump(path,ev);print('ADDITIONAL EVAL',name,method,seed,ch,flush=True)
dump(OUT/'coarse-complete.json',{'geometries':list(SPECS),'seeds':SEEDS,'completed':True})
