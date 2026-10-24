"""Audit all retained requests, cache transitions, residuals and caps."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
H=Path(__file__).resolve().parent;O=H/'output'
def main():
 records=[];kernels=[p for p in O.glob('**/kernel.json') if 'legacy-streams' not in p.parts];cached=0;cache_delta=0.0
 for p in kernels:
  k=json.loads(p.read_text());assert k['caps']==0,p
  f=p.parent/'field.csv'
  if f.exists():assert np.isfinite(pd.read_csv(f)[['u','p','score','same_index']].values).all(),f
  assert k.get('relative_residual',0)<=1e-7,p
  if p.parent.name.endswith('cached-repeat'):
   assert k['distance_queries']==k['parity_rays']==0,p
   state=p.parent.name.split('-')[0]
   a=pd.read_csv(p.parent/'field.csv')[['u','p','score']].values
   b=pd.read_csv(p.parent.parent/f'{state}-prefix512.csv')[['u','p','score']].values
   delta=float(np.max(np.abs(a-b)));cache_delta=max(cache_delta,delta)
   assert np.allclose(a,b,rtol=8*np.finfo(float).eps,atol=0),p;cached+=1
  records.append(k)
 for p in O.glob('*/mg*-seed*/done.json'):
  for state in ['original','edited']:
   cold=json.loads((p.parent/f'{state}-q8/kernel.json').read_text())
   assert cold['parity_rays']>0,p
   for q in [32,128,512]:
    k=json.loads((p.parent/f'{state}-q{q}/kernel.json').read_text())
    assert k['parity_rays']==k['distance_queries']==0,p
 done=list(O.glob('*/*-seed*/done.json'))
 assert len(done)==4*5*7,len(done)
 assert cached==2*4*5*4*2+40,cached
 assert len(list((O/"incremental").glob("*/*-seed*/done.json")))==80
 assert len(list((O/"analytic").glob("*/adaptive-done.json")))==4
 assert len(list((O/"compliance").glob("*/adaptive-done.json")))==1
 na=len(list((O/"analytic").glob("*/*-seed*/done.json")));nc=len(list((O/"compliance").glob("*/*-seed*/done.json")))
 report=dict(completed_method_seed_runs=len(done)+80+na+nc,compliance_runs=nc,original_runs=len(done),incremental_grid_runs=80,analytic_source_runs=na,kernels=len(kernels),cached_repeats_verified=cached,
   maximum_residual=max(k.get('relative_residual',0) for k in records),walk_caps=sum(k['caps'] for k in records),
   cache_invalidation='Original baseline rebuilt after edits; incremental variant retained outer masks. All additional batches made zero geometry queries',
   identity='Cached values agree within eight machine epsilons after CSV serialization', maximum_cached_absolute_discrepancy=cache_delta,
   protocol_sha256=hashlib.sha256((O/'protocol.json').read_bytes()).hexdigest(),
   analysis_protocol_sha256=hashlib.sha256((H/'analysis_protocol.json').read_bytes()).hexdigest())
 (O/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
