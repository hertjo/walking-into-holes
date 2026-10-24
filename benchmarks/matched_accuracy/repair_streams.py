"""Remove structural legacy stream aliases by replaying only affected paths.

The discarded path identities are fixed before inspecting values. The remaining
4096-path sums are reconstructed from retained aggregates plus replay deltas.
"""
from pathlib import Path
import json,shutil,time
import numpy as np,pandas as pd
import run as r
O=r.OUT

def audit_keys():
 def mix(v):
  v=np.asarray(v,dtype=np.uint64)+np.uint64(0x9e3779b97f4a7c15)
  v=(v^(v>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
  v=(v^(v>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
  return v^(v>>np.uint64(31))
 aliases=sorted({s+offset-571 for s in r.SEEDS for offset in [0,8,32,128] if 0<=s+offset-571<4096})
 n=np.arange(4096,dtype=np.uint64);correct=n.copy();correct[aliases]+=1000000
 counts={}
 for label,walks in [('legacy',n),('repaired',correct)]:
  blocks=[]
  for seed in r.SEEDS:
   last=0
   for q in r.COUNTS:
    keys=mix(seed+last)^mix(np.arange(q-last,dtype=np.uint64)[:,None]+91347)^mix(walks[None,:]+571)
    blocks.append(keys.ravel());last=q
  keys=np.concatenate(blocks);keys.sort();counts[label]=int(np.sum(keys[1:]==keys[:-1]))
 assert counts['repaired']==0,counts
 return dict(paths_per_field=4096,sites_per_repetition=512,repetitions=5,replaced_indices=aliases,duplicate_keys=counts)

def family(root,kind):
 validations=[];affected=0
 for name in r.GEOMS:
  source=root/name
  folders=sorted(source.glob('twos4096-seed*'))
  if not folders:continue
  sol=None;first=True
  for folder in folders:
   if not (folder/'done.json').exists():continue
   if (folder/'stream-repaired.json').exists():
    affected+=1;continue
   backup=O/'legacy-streams'/folder.relative_to(O)
   if not backup.exists():shutil.copytree(folder,backup)
   r.a.BIN=Path('/tmp/wih-reuse-build')/f'wih_repair_{kind}'
   if sol is None:sol=r.a.Solver(name,O/'stream-replay'/kind/name)
   rows=json.loads((backup/'done.json').read_text());updated=[];patches=[]
   for state in ['original','edited']:
    parts=[];elapsed=0;steps=0;queries=0;rss=0
    for old in sorted([x for x in rows if x['state']==state],key=lambda x:x['queries']):
     q=old['queries'];batch=backup/f'{state}-q{q}';request=json.loads((batch/'request.json').read_text())
     seed=int(request[request.index('--seed')+1]);points=np.loadtxt(batch/'points.txt').reshape(-1,3)
     holesfile=batch/'holes.txt';holes=np.loadtxt(holesfile).reshape(-1,3) if holesfile.stat().st_size else []
     delta,k=r.call(sol,f'{folder.name}-{state}-q{q}',points,holes=holes,walks=4096,seed=seed)
     original=pd.read_csv(batch/'field.csv');new=original.copy();N=4096.
     su=original.u.to_numpy()*N+delta.u.to_numpy();sp=original.p.to_numpy()*N+delta.p.to_numpy();cross=original.same_index.to_numpy()*N+delta.same_index.to_numpy()
     new['u']=su/N;new['p']=sp/N;new['same_index']=cross/N;new['score']=(su*sp-cross)/(N*(N-1))
     assert np.isfinite(new[['u','p','same_index','score']].values).all()
     new['steps']=original.steps.to_numpy()+delta.steps.to_numpy()
     dest=folder/f'{state}-q{q}';new.to_csv(dest/'field.csv',index=False)
     meta=json.loads((batch/'kernel.json').read_text())
     for key in ['request_wall_seconds','total_seconds','distance_queries','steps','parity_rays']:
      meta[key]+=k[key]
     meta['peak_rss_megabytes']=max(meta['peak_rss_megabytes'],k['peak_rss_megabytes'])
     meta['stream_repair']='12 identities replaced; original plus full old/new replay cost charged'
     r.a.dump(dest/'kernel.json',meta)
     elapsed+=meta['request_wall_seconds'];parts.append(new);queries+=meta['distance_queries'];rss=max(rss,meta['peak_rss_megabytes'])
     pd.concat(parts,ignore_index=True).to_csv(folder/f'{state}-prefix{q}.csv',index=False)
     row=dict(old);row.update(seconds=elapsed,increment_seconds=meta['request_wall_seconds'],distance_queries=queries,rss_mb=rss)
     updated.append(row);patches.append(dict(state=state,queries=q,replay_seconds=k['request_wall_seconds']))
     if first:
      # A complete clean 4096-path run checks the algebraic reconstruction.
      r.a.BIN=Path('/tmp/wih-reuse-build')/f'wih_clean_{kind}'
      check=r.a.Solver(name,O/'stream-validation'/kind/name)
      clean,ck=r.call(check,'clean',points,holes=holes,walks=4096,seed=seed)
      check.close();r.a.BIN=Path('/tmp/wih-reuse-build')/f'wih_repair_{kind}'
      a=new[['u','p','same_index','score']].values;b=clean[['u','p','same_index','score']].values
      error=float(np.max(np.abs(a-b)));scale=float(np.max(np.abs(b)))
      assert error<=1e-10*max(scale,1e-3),(kind,name,error,scale)
      validations.append(dict(family=kind,geometry=name,max_absolute_discrepancy=error,scale=scale,sites=len(points)))
      first=False
   r.a.dump(folder/'done.json',updated);r.a.dump(folder/'stream-repaired.json',dict(patches=patches,source_backup=str(backup.relative_to(O))))
   affected+=1;print('STREAM REPAIRED',kind,name,folder.name,flush=True)
  if sol is not None:sol.close()
 return affected,validations

def main():
 report=audit_keys();total=0;checks=[]
 previous=O/"stream-repair-audit.json"
 if previous.exists():checks=json.loads(previous.read_text()).get("full_replay_checks",[])
 for root,kind in [(O,'raw'),(O/'analytic','analytic'),(O/'compliance','compliance')]:
  n,v=family(root,kind);total+=n;checks+=v
 report.update(corrected_method_seed_runs=total,full_replay_checks=checks,scope='Corrected calibration fields; numerical roundoff audit, not exact-arithmetic identity. Legacy outputs retained and all replay work charged.')
 r.a.dump(O/'stream-repair-audit.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
