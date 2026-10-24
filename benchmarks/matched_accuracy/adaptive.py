"""Predeclared extension-only stopping after required accuracy is attained."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
IDX=np.random.default_rng(20260917).integers(0,5,(10000,5))
def values(p):return pd.read_csv(p).score.values

def unresolved(folder,reference,counts,prior):
 ids=json.loads((reference/'done.json').read_text())['spot_indices'];need={};status=[]
 for state in ['original','edited']:
  ref=values(reference/f'{state}-mg257/field.csv');coarse=values(reference/f'{state}-mg193/field.csv')
  mc=np.stack([values(reference/f'{state}-mc{i}/field.csv') for i in range(8)])
  den=np.linalg.norm(ref[ids]);eta=np.linalg.norm(mc.std(0,ddof=1)/np.sqrt(8))/den;delta=np.linalg.norm(mc.mean(0)-ref[ids])/den
  for q in counts:
   grid=np.linalg.norm(ref[:q]-coarse[:q])/np.linalg.norm(ref[:q])
   targets=[t for t in [.1,.2,.4] if grid<=t/4 and eta<=t/4 and delta<=t/4+2*eta]
   target=min(targets) if targets else None;met=False;eligible=[]
   for b in prior:
    paths=[folder/f'twos{b}-seed{s}'/f'{state}-prefix{q}.csv' for s in [1611,2711,3811,4911,6011]]
    if not all(p.exists() for p in paths):continue
    e=np.array([np.linalg.norm(values(p)-ref[:q])/np.linalg.norm(ref[:q]) for p in paths])
    hi=float(np.quantile(e[IDX].mean(1),.975))
    if target is not None and hi<=target:met=True;eligible.append(b)
   required=target is not None and not met
   status.append(dict(state=state,queries=q,strictest_resolved_target=target,met=met,eligible_budgets=eligible,needs_more=required))
   if required:need[state]=max(q,need.get(state,0))
 return need,status

def sweep_mc(r,name,p,hole,reference):
 folder=r.OUT/name;folder.mkdir(exist_ok=True);history=[];prior=[]
 for budget in [16,64,256,1024,4096]:
  if budget<=256:need={'original':512,'edited':512};status=[]
  else:need,status=unresolved(folder,reference,r.COUNTS,prior)
  history.append(dict(budget=budget,required_prefix=need,case_status=status))
  r.a.dump(folder/'adaptive-progress.json',history)
  if not need:break
  for seed in r.SEEDS:
   dest=folder/f'twos{budget}-seed{seed}'
   if (dest/'done.json').exists():continue
   sol=r.a.Solver(name,dest);rows=[]
   _,warm=r.call(sol,'mesh-setup',p[:1],walks=2,seed=123)
   for state,holes in [('original',[]),('edited',hole)]:
    if state not in need:continue
    elapsed=0;parts=[];kernels=[];last=0
    for q in r.COUNTS:
     if q>need[state]:continue
     f,k=r.call(sol,f'{state}-q{q}',p[last:q],holes=holes,walks=budget,seed=seed+last)
     elapsed+=k['request_wall_seconds'];parts.append(f);kernels.append(k);last=q
     pd.concat(parts,ignore_index=True).to_csv(dest/f'{state}-prefix{q}.csv',index=False)
     rows.append(dict(geometry=name,method='twos',budget=budget,seed=seed,state=state,queries=q,
       seconds=elapsed,increment_seconds=k['request_wall_seconds'],rss_mb=max(v['peak_rss_megabytes'] for v in kernels),
       distance_queries=sum(v['distance_queries'] for v in kernels),residual=0,mesh_setup_seconds=warm['request_wall_seconds']))
   sol.close();r.a.dump(dest/'done.json',rows)
   print('ADAPTIVE',name,budget,seed,round(sum(x['increment_seconds'] for x in rows),2),flush=True)
  prior.append(budget)
 need,status=unresolved(folder,reference,r.COUNTS,prior)
 r.a.dump(folder/'adaptive-done.json',dict(history=history,remaining_at_cap=status,unresolved_prefixes=need,budgets_measured=prior))
