from pathlib import Path
import json
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;O=H/'output'
RNG=np.random.default_rng(20260917);IDX=RNG.integers(0,5,(10000,5))
def ci(v):
 v=np.asarray(v);b=v[IDX].mean(1)
 return float(v.mean()),*np.quantile(b,[.025,.975]).tolist()
def field(p):return pd.read_csv(p).score.to_numpy()

def analyze():
 rows=[];checks=[]
 for name in ['housing','branch','ring','bunny','bunny_compliance']:
  f=O/'compliance'/'bunny' if name=='bunny_compliance' else O/name;ref=f/'reference'
  if not (ref/'done.json').exists():continue
  ids=json.loads((ref/'done.json').read_text())['spot_indices']
  for state in ['original','edited']:
   r=field(ref/f'{state}-mg257/field.csv');r0=field(ref/f'{state}-mg193/field.csv')
   mc=np.stack([field(ref/f'{state}-mc{i}/field.csv') for i in range(8)])
   denom=np.linalg.norm(r[ids]);se=np.linalg.norm(mc.std(0,ddof=1)/np.sqrt(8))/denom
   diff=np.linalg.norm(mc.mean(0)-r[ids])/denom
   for count in [8,32,128,512]:
    delta=np.linalg.norm(r0[:count]-r[:count])/np.linalg.norm(r[:count])
    checks.append(dict(geometry=name,state=state,queries=count,grid_delta=delta,spot_se=se,spot_delta=diff))
   for folder,variant in [(f,None),(O/'incremental'/name,'mg_update'),(O/'analytic'/name,'twos_rb')]:
    for p in folder.glob('*-seed*/done.json'):
     for row in json.loads(p.read_text()):
      if row['state']!=state:continue
      if variant:row['method']=variant
      row['geometry']=name
      if name=='bunny_compliance':row['method']={'twos':'twos_rb','mg':'mg_update'}[row['method']]
      q=row['queries'];v=field(p.parent/f'{state}-prefix{q}.csv')
      row['error']=np.linalg.norm(v-r[:q])/np.linalg.norm(r[:q]);row['reference_norm2']=float(np.dot(r[:q],r[:q]));row['field_path']=str((p.parent/f'{state}-prefix{q}.csv').relative_to(O));rows.append(row)
 d=pd.DataFrame(rows);c=pd.DataFrame(checks)
 if d.empty:return
 d.to_csv(O/'runs.csv',index=False);c.to_csv(O/'reference_checks.csv',index=False)
 summaries=[]
 for key,g in d.groupby(['geometry','state','queries','method','budget']):
  if len(g)!=5:continue
  row=dict(zip(['geometry','state','queries','method','budget'],key))
  g=g.sort_values('seed')
  for k in ['error','seconds','increment_seconds']:
   row[k],row[k+'_lo'],row[k+'_hi']=ci(g[k].values)
  row['rss_mb']=g.rss_mb.max();values=np.stack([field(O/p) for p in g.field_path]);row['normalized_variance']=float(values.var(0,ddof=1).sum()/g.reference_norm2.iloc[0]);summaries.append(row)
 s=pd.DataFrame(summaries);s.to_csv(O/'summary.csv',index=False)
 selected=[]
 for key,g in s.groupby(['geometry','state','queries']):
  check=c[(c.geometry==key[0])&(c.state==key[1])&(c.queries==key[2])].iloc[0]
  for target in [.1,.2,.4]:
   gate=(check.grid_delta<=target/4 and check.spot_se<=target/4 and check.spot_delta<=target/4+2*check.spot_se)
   row=dict(zip(['geometry','state','queries'],key));row.update(target=target,reference_resolved=bool(gate))
   for method in ['twos','mg','twos_rb','mg_update']:
    eligible=g[(g.method==method)&(g.error_hi<=target)] if gate else g.iloc[:0]
    if len(eligible):
     best=eligible.sort_values('seconds').iloc[0]
     for k in ['budget','seconds','seconds_lo','seconds_hi','error','error_hi','rss_mb']:row[method+'_'+k]=best[k]
   # Report the fastest eligible grid variant, including the stronger update.
   eligible=g[g.method.isin(['mg','mg_update']) & (g.error_hi<=target)] if gate else g.iloc[:0]
   if len(eligible):
    best=eligible.sort_values('seconds').iloc[0];row['best_mg_variant']=best['method']
    for k in ['budget','seconds','seconds_lo','seconds_hi','error','error_hi','rss_mb']:row['best_mg_'+k]=best[k]
   for method in ['twos','twos_rb']:
    if method+'_budget' in row and 'best_mg_budget' in row:
     x=[]
     for m,b in [(method,row[method+'_budget']),(row['best_mg_variant'],row['best_mg_budget'])]:
      v=d[(d.geometry==key[0])&(d.state==key[1])&(d.queries==key[2])&(d.method==m)&(d.budget==b)].sort_values('seed').seconds.values
      x.append(v)
     boots=x[1][IDX].mean(1)/x[0][IDX].mean(1)
     row[method+'_speedup']=x[1].mean()/x[0].mean()
     row[method+'_speedup_lo']=float(np.quantile(boots,.025));row[method+'_speedup_hi']=float(np.quantile(boots,.975))
   selected.append(row)
 pd.DataFrame(selected).to_csv(O/'selected.csv',index=False)
 print('Completed runs',len(d),'complete configurations',len(s))
 print(pd.DataFrame(selected).query('target==0.2').to_string(index=False))
if __name__=='__main__':analyze()
