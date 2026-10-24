from pathlib import Path
import json
import numpy as np,pandas as pd
import run as r
O=r.OUT
SEEDS=[71011+11000*i for i in range(20)]
IDX=np.random.default_rng(17092027).integers(0,20,(10000,20))
def field(p):return pd.read_csv(p).score.values

def main():
 selected=pd.read_csv(O/'selected.csv');claims=selected[(selected.target==.2)&(selected.twos_rb_speedup>1)]
 results=[]
 for _,c in claims.iterrows():
  name='bunny' if c.geometry=='bunny_compliance' else c.geometry
  base=O/name;ref=O/'compliance/bunny/reference' if c.geometry=='bunny_compliance' else base/'reference'
  p=np.loadtxt(base/'sites.txt');hole=np.loadtxt(base/'hole.txt').reshape(1,3);truth=field(ref/f'{c.state}-mg257/field.csv')[:int(c.queries)]
  folder=O/'holdout'/f'{c.geometry}-{c.state}-q{int(c.queries)}-T{int(c.twos_rb_budget)}-M{int(c.best_mg_budget)}-{c.best_mg_variant}';rows=[]
  for j,seed in enumerate(SEEDS):
   for method in (['twos_rb','mg'] if j%2==0 else ['mg','twos_rb']):
    dest=folder/f'seed{seed}'/method
    if (dest/'result.json').exists():rows.append(json.loads((dest/'result.json').read_text()));continue
    binary='wih_compliance_query' if c.geometry=='bunny_compliance' else ('wih_analytic_query' if method=='twos_rb' else ('wih_incremental_query' if c.best_mg_variant=='mg_update' else 'wih_reuse_query'))
    r.a.BIN=Path('/tmp/wih-reuse-build')/binary
    sol=r.a.Solver(name,dest);_,warm=r.call(sol,'mesh-setup',p[:1],walks=2,seed=123)
    budget=int(c.twos_rb_budget if method=='twos_rb' else c.best_mg_budget)
    if method=='mg' and c.state=='edited':r.call(sol,'original-field',p[:8],mode='mg',n=budget,mg_cycles=800 if c.geometry=='bunny_compliance' else 120)
    last=0;elapsed=0;parts=[]
    for q in r.COUNTS:
     if q>c.queries:break
     f,k=r.call(sol,f'q{q}',p[last:q],holes=hole if c.state=='edited' else [],mode='twos' if method=='twos_rb' else 'mg',walks=budget if method=='twos_rb' else 2,n=budget if method=='mg' else 65,seed=seed+last,mg_cycles=800 if c.geometry=='bunny_compliance' else 120)
     last=q;parts.append(f);elapsed+=k['request_wall_seconds']
    v=pd.concat(parts).score.values;sol.close()
    row=dict(geometry=c.geometry,state=c.state,queries=int(c.queries),method=method,budget=budget,seed=seed,seconds=elapsed,error=float(np.linalg.norm(v-truth)/np.linalg.norm(truth)))
    r.a.dump(dest/'result.json',row);rows.append(row)
  d=pd.DataFrame(rows);t=d[d.method=='twos_rb'].sort_values('seed');m=d[d.method=='mg'].sort_values('seed')
  ratio=m.seconds.values[IDX].mean(1)/t.seconds.values[IDX].mean(1);error=t.error.values[IDX].mean(1)
  row=dict(geometry=c.geometry,state=c.state,queries=int(c.queries),twos_budget=int(c.twos_rb_budget),mg_budget=int(c.best_mg_budget),mg_variant=c.best_mg_variant,
    twos_seconds=float(t.seconds.mean()),mg_seconds=float(m.seconds.mean()),speedup=float(m.seconds.mean()/t.seconds.mean()),speedup_lo=float(np.quantile(ratio,.025)),speedup_hi=float(np.quantile(ratio,.975)),
    mean_error=float(t.error.mean()),error_hi=float(np.quantile(error,.975)),verified=bool(np.quantile(error,.975)<=.2 and np.quantile(ratio,.025)>1))
  results.append(row);r.a.dump(folder/'summary.json',row);print('HOLDOUT',row,flush=True)
 pd.DataFrame(results).to_csv(O/'holdout-summary.csv',index=False)
 r.a.dump(O/'holdout-status.json',dict(candidate_cells=len(claims),completed_cells=len(results),seeds=SEEDS,verified_wins=sum(x['verified'] for x in results)))
if __name__=='__main__':main()
