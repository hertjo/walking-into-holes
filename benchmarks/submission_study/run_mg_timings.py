"""Repeat deterministic query timing; do not reinterpret runs as MC field variance."""
import json,numpy as np
from run_study import OUT,SPECS,Solver,dump

def main():
 for name in SPECS:
  folder=OUT/'estimators'/name;path=folder/'results.json';result=json.loads(path.read_text())
  if 'mg_timing_repeats' in result:continue
  sol=Solver(name,folder/'mg-timing-raw');p=np.loadtxt(folder/'sites.txt')
  sol.call('geometry-warmup',p[:1],walks=2,seed=28119)
  records=[]
  for rep in range(5):
   # Rotate resolution order, retaining geometry but rebuilding every hierarchy.
   for n in np.roll([49,65,97,129],rep):
    f,k=sol.call(f'grid{n}-rep{rep}',p,mode='mg',n=int(n))
    old=[r['score'] for r in result['rows'] if r['method']=='mg' and r['walks']==n]
    assert np.allclose(f.score,old,rtol=1e-10,atol=1e-15)
    records.append({'grid':int(n),'rep':rep,'seconds':k['request_wall_seconds']})
  sol.close();result['mg_timing_repeats']=records;dump(path,result);print('MG TIMING REPEATS',name,flush=True)
if __name__=='__main__':main()
