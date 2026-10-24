"""Frozen, resumable sparse-query workload sweep. Run on one machine serially."""
from pathlib import Path
import sys,os,json,hashlib,time,shutil,platform
import numpy as np,pandas as pd,trimesh
from scipy.stats import qmc
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'submission_study'))
import run_study as a
from geometry import SPECS,distance,admissible
OUT=HERE/'output'; OUT.mkdir(exist_ok=True)
a.BIN=Path('/tmp/wih-reuse-build/wih_reuse_query')
GEOMS=['housing','branch','ring','bunny']; COUNTS=[8,32,128,512]; SEEDS=[1611,2711,3811,4911,6011]
BUDGETS=[('twos',256),('twos',1024),('twos',4096),('mg',49),('mg',65),('mg',97),('mg',129)]
SPECS['bunny']=dict(sensor=[.15,-.15,0,.07],heat=[[-.15,-.15,0,.085,35],[.15,-.15,0,.06,4]],label='Stanford bunny')
p=a.HERE/'assets/bunny.ply'
if not p.exists():shutil.copy2(HERE.parent/'compiled_bunny/output/bunny-69674-faces.ply',p)
def setup(name):
 s,m=a.load(name);f=OUT/name;f.mkdir(exist_ok=True)
 if (f/'sites.txt').exists():return np.loadtxt(f/'sites.txt'),np.loadtxt(f/'hole.txt').reshape(1,3)
 v=qmc.Sobol(3,scramble=True,seed=17092026).random_base2(16)
 v=m.bounds[0]+v*np.ptp(m.bounds,axis=0)
 v=v[admissible(m,s,v,.04)]
 hole=v[-1:];v=v[np.linalg.norm(v-hole,axis=1)>.10][:512]
 assert len(v)==512,(name,len(v))
 np.savetxt(f/'sites.txt',v);np.savetxt(f/'hole.txt',hole)
 return v,hole

def call(sol,tag,p,**kw):
 # A session's cache transitions must execute even if a prior output exists.
 f,k=sol.call(tag,p,radius=.04,epsilon=.0001,**kw)
 assert k['caps']==0
 return f,k

def reference(name,p,hole):
 folder=OUT/name/'reference'
 if (folder/'done.json').exists():return
 sol=a.Solver(name,folder)
 for state,holes in [('original',[]),('edited',hole)]:
  for n in [193,257]:
   call(sol,f'{state}-mg{n}',p,holes=holes,mode='mg',n=n)
  # Independent MC checks at a fixed, spatially distributed subset; these do
  # not convert grid convergence or floating-point truncation into certificates.
  ids=np.linspace(0,511,32,dtype=int)
  for rep in range(8):
   call(sol,f'{state}-mc{rep}',p[ids],holes=holes,walks=8192,seed=900000+rep)
 sol.close();a.dump(folder/'done.json',{'complete':True,'spot_indices':ids.tolist()})
 print('REFERENCE',name,flush=True)

def sweep(name,p,hole):
 for rep,seed in enumerate(SEEDS):
  order=BUDGETS[rep:]+BUDGETS[:rep]
  for method,budget in order:
   folder=OUT/name/f'{method}{budget}-seed{seed}'
   if (folder/'done.json').exists():continue
   sol=a.Solver(name,folder);rows=[]
   # Common mesh construction measured separately, not silently omitted.
   _,warm=call(sol,'mesh-setup',p[:1],walks=2,seed=123)
   for state,holes in [('original',[]),('edited',hole)]:
    elapsed=0;parts=[];kernels=[];last=0
    for count in COUNTS:
     f,k=call(sol,f'{state}-q{count}',p[last:count],holes=holes,mode=method,
              walks=budget if method=='twos' else 2,n=budget if method=='mg' else 65,
              seed=seed+last)
     elapsed+=k['request_wall_seconds'];parts.append(f);kernels.append(k);last=count
     pd.concat(parts,ignore_index=True).to_csv(folder/f'{state}-prefix{count}.csv',index=False)
     rows.append(dict(geometry=name,method=method,budget=budget,seed=seed,state=state,
        queries=count,seconds=elapsed,increment_seconds=k['request_wall_seconds'],
        rss_mb=max(x['peak_rss_megabytes'] for x in kernels),
        distance_queries=sum(x['distance_queries'] for x in kernels),
        residual=max(x['relative_residual'] for x in kernels),mesh_setup_seconds=warm['request_wall_seconds']))
    if method=='mg':
     f,k=call(sol,f'{state}-cached-repeat',p,holes=holes,mode='mg',n=budget)
     assert k['distance_queries']==0 and k['parity_rays']==0
     assert np.array_equal(f[['u','p','score']].values,pd.concat(parts)[['u','p','score']].values)
   sol.close();a.dump(folder/'done.json',rows)
   print('RUN',name,method,budget,seed,round(sum(x['increment_seconds'] for x in rows),2),flush=True)

def freeze():
 protocol=dict(date='2026-09-17',geometries=GEOMS,counts=COUNTS,seeds=SEEDS,budgets=BUDGETS,
  geometry_edit='Insert fixed radius .04 cavity at final admissible Sobol point, never selected by score',
  candidates='512 fixed admissible nested Sobol sites, clearance .10 from future cavity',
  sampling='five independent paired seeds; fixed sites; nested query batches 8+24+96+384',
  threads=4,epsilon=.0001,max_steps=2048,mg_residual_tolerance=1e-7,
  reuse='Retain common mesh; MG retains solved primal/adjoint grids. A cavity edit invalidates those grids.',
  timing='Sequential wall time through each prefix; mesh setup recorded separately; serial rotated method order',
  targets=[.10,.20,.40],primary_target=.20,
  accuracy='Relative L2 score error to MG257. Reference resolution gate: MG193 versus MG257 <= target/4 for each prefix. Independent 8 x 8192-walk checks on 32 fixed sites; flag inconsistency or unresolved sampling. No continuum accuracy certificate.',
  selection='Fastest measured tested budget whose upper 95% bootstrap mean relative error is below target; pointwise bootstrap, five seeds. Apply reference gate before selecting.',
  exclusions='No optimization-quality claim, no formal bias bound, no universal solver ranking; all failed targets retained.',
  machine=platform.platform(),cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
  binary_sha256=hashlib.sha256(a.BIN.read_bytes()).hexdigest())
 path=OUT/'protocol.json'
 if path.exists():assert json.loads(path.read_text())==protocol,'protocol drift'
 else:a.dump(path,protocol)

if __name__=='__main__':
 freeze()
 for name in GEOMS:
  p,hole=setup(name)
  # Sweep precedes the costly independent reference, without any budget changes.
  sweep(name,p,hole);reference(name,p,hole)
