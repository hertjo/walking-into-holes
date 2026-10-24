#!/usr/bin/env python3
"""Matched, resumable experiment campaign. Raw solver requests remain inspectable."""
import argparse,json,time,subprocess,hashlib,os,resource
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import qmc
from scipy.spatial import cKDTree
from geometry import HERE,SPECS,load,candidates,probes,admissible,distance
BIN=Path(os.environ.get('WIH_STUDY_BINARY','/tmp/wih-study-build/wih_study'))
OUT=HERE/'output';R=.065;SEEDS=[1611,2711,3811,4911,6011]

def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)

def grid(m,n):
 extent=np.ptp(m.bounds,axis=0);a=np.maximum(17,2*np.ceil((n-1)*extent/extent.max()/2).astype(int)+1)
 return ','.join(map(str,a))

class Solver:
 def __init__(self,name,folder):self.name=name;self.folder=Path(folder);self.proc=None;self.calls=[];self.mesh=load(name)[1]
 def call(self,tag,points,holes=(),radius=R,mode='twos',walks=128,seed=7,hot=False,fd=False,diff=False,n=65,epsilon=.0002,offset=.004,mg_cycles=120):
  d=self.folder/tag;d.mkdir(parents=True,exist_ok=True)
  holes=np.asarray(holes).reshape(-1,3);points=np.asarray(points).reshape(-1,3)
  np.savetxt(d/'points.txt',points,fmt='%.17g');np.savetxt(d/'holes.txt',holes,fmt='%.17g')
  m=self.mesh
  args=[str(BIN),'--mesh',str(HERE/'assets'/f'{self.name}.ply'),'--config',str(HERE/'assets'/f'{self.name}.txt'),'--points',str(d/'points.txt'),'--cavities',str(d/'holes.txt') if len(holes) else '', '--cavity-radius',str(radius),'--output',str(d/'field.csv'),'--summary',str(d/'kernel.json'),'--mode',mode,'--source-mode','housing','--walks',str(walks),'--repeats','1','--seed',str(seed),'--threads','4','--max-steps','2048','--epsilon',str(epsilon),'--hot-only',str(int(hot)),'--fd',str(int(fd)),'--diff',str(int(diff)),'--grid',grid(m,n),'--mg-cycles',str(mg_cycles),'--mg-tolerance','1e-7','--diff-offset',str(offset)]
  dump(d/'request.json',args)
  if self.proc is None:
   self.folder.mkdir(parents=True,exist_ok=True);self.err=(self.folder/'server.log').open('w')
   self.proc=subprocess.Popen([str(BIN),'--serve'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,text=True,bufsize=1)
  start=time.perf_counter();self.proc.stdin.write(str(len(args))+'\n'+'\n'.join(args)+'\n');self.proc.stdin.flush()
  response=self.proc.stdout.readline().strip()
  if response!='DONE 0':raise RuntimeError(f'{d}: {response} '+(self.folder/'server.log').read_text())
  meta=json.loads((d/'kernel.json').read_text());meta['request_wall_seconds']=time.perf_counter()-start
  dump(d/'kernel.json',meta);self.calls.append(meta)
  return pd.read_csv(d/'field.csv'),meta
 def close(self):
  if self.proc is not None:self.proc.stdin.close();self.proc.wait();self.err.close();self.proc=None

def channel_balls(axes):
 axes=np.asarray(axes).reshape(-1,3);z=np.arange(-.48,.801,.04)
 a=np.repeat(axes[:,None,:],len(z),axis=1);a[:,:,2]=z
 return a.reshape(-1,3)

class Volume:
 cache={}
 def __init__(self,m,seed=8821,power=20):
  key=(len(m.faces),float(m.volume),seed,power)
  if key not in self.cache:
   path=HERE/'assets'/f'volume-{len(m.faces)}-{seed}-{power}.npz'
   if path.exists():
    a=np.load(path);self.cache[key]=(a['p'],float(a['factor']))
   else:
    p=qmc.Sobol(3,scramble=True,seed=seed).random_base2(power)
    p=m.bounds[0]+p*np.ptp(m.bounds,axis=0)
    inside=p[m.contains(p)];factor=np.prod(np.ptp(m.bounds,axis=0))/len(p)
    np.savez_compressed(path,p=inside,factor=factor);self.cache[key]=(inside,factor)
  self.p,self.factor=self.cache[key]
 def radius(self,axes,target):
  d=cKDTree(channel_balls(axes)).query(self.p,workers=1)[0]
  k=int(round(target/self.factor));r=float(np.partition(d,k)[k]);return r
 def volume(self,holes,r):return float(np.sum(cKDTree(holes).query(self.p,workers=1)[0]<r)*self.factor)

def valid(s,m,holes,channels=False,r=R):
 holes=np.asarray(holes).reshape(-1,3)
 if len(holes)==0:return True
 if not np.all(admissible(m,s,holes,r,channels)):return False
 a=holes[:,:2] if channels else holes
 d=np.linalg.norm(a[:,None]-a[None,:],axis=2)+np.eye(len(a))*100
 return d.min()>2*r+.025

def objective(sol,tag,s,holes,r,seed,walks=256,nprobe=128,mode='twos',n=65):
 p,w=probes(s,seed+837,nprobe)
 f,_=sol.call(tag,p,holes,r,mode=mode,walks=walks,seed=seed,hot=True,n=n)
 return float(np.mean(w*f.u.to_numpy()))

def greedy(name,method,seed,channels=False):
 folder=OUT/('channels' if channels else 'pipeline')/name/f'{method}-{seed}'
 if (folder/'design.json').exists():return json.loads((folder/'design.json').read_text())
 s,m=load(name);sol=Solver(name,folder/'raw');vol=Volume(m) if channels else None;t=time.perf_counter()
 pool=candidates(m,s,seed,384,channels);axes=[];r=R;history=[]
 for k in range(6):
  a=np.asarray(axes).reshape(-1,3);d=pool[:,:2] if channels else pool
  old=a[:,:2] if channels else a
  keep=np.ones(len(pool),bool) if not len(a) else np.min(np.linalg.norm(d[:,None]-old[None,:],axis=2),axis=1)>2*(.11 if channels else R)+.025
  p=pool[keep];assert len(p)>0
  holes=channel_balls(a) if channels else a
  if method=='random':idx=0
  elif method=='nearest':idx=int(np.argmin(np.linalg.norm(p-np.array(s['sensor'][:3]),axis=1)))
  else:
   mode='mg' if method in ['mg','mg49','hot'] else 'twos'
   f,_=sol.call(f'round{k}-scout',p,holes,r,mode=mode,walks=48,seed=seed+101*k,hot=method=='hot',n=49 if method=='mg49' else 65)
   if method=='twos':
    short=np.argsort(f.score.to_numpy())[-min(16,len(p)):]
    ff,_=sol.call(f'round{k}-rescore',p[short],holes,r,walks=1024,seed=seed+101*k+53)
    idx=int(short[np.argmax(ff.score.to_numpy())])
   else:idx=int(np.argmax(f.score.to_numpy()))
  axes.append(p[idx].tolist())
  if channels:
   r=vol.radius(axes,(k+1)*.0028)
   if not .02<r<.11:raise ValueError(('volume radius out of protected range',name,r))
  history.append({'centers':list(axes),'radius':r,'seconds':time.perf_counter()-t})
 sol.close();elapsed=time.perf_counter()-t
 result={'geometry':name,'method':method,'seed':seed,'channels':channels,'centers':axes,'radius':r,'history':history,'design_seconds':elapsed,'backend_peak_rss_mb':max([x['peak_rss_megabytes'] for x in sol.calls] or [0]),'walk_steps':sum(x['steps'] for x in sol.calls),'distance_queries':sum(x['distance_queries'] for x in sol.calls),'parity_rays':sum(x['parity_rays'] for x in sol.calls),'target_volume':.0168 if channels else 6*4*np.pi*R**3/3,'protocol':'compact-source-v2-sensor-load'}
 dump(folder/'design.json',result);print('DESIGN',name,method,seed,channels,round(elapsed,3),flush=True)
 return result

def refine(result,rounds=8):
 name=result['geometry'];method=result['method'];seed=result['seed'];folder=OUT/'pipeline'/name/f'{method}-{seed}'
 if (folder/'refinement.json').exists():return json.loads((folder/'refinement.json').read_text())
 s,m=load(name);sol=Solver(name,folder/'refine-raw');t=time.perf_counter();a=np.array(result['centers']);history=[{'round':0,'centers':a.tolist(),'seconds':result['design_seconds']}]
 for k in range(rounds):
  p,w=probes(s,seed+20001+k,64)
  f,_=sol.call(f'r{k}-gradient',p,a,R,walks=96 if k<4 else 192,seed=seed+6000+k,diff=True)
  g=np.array([[np.mean(w*f[f'dc{i}_{c}'].to_numpy()) for c in 'xyz'] for i in range(6)])
  scale=np.max(np.linalg.norm(g,axis=1));direction=-g/max(scale,1e-30)
  old=objective(sol,f'r{k}-old',s,a,R,seed+10000+k,walks=192,nprobe=64)
  accepted=False
  for j,step in enumerate([.035,.0175,.00875]):
   trial=a.copy()
   # Feasible block backtracking: one active constraint must not freeze all cavities.
   for i in np.roll(np.arange(6),k):
    for shrink in range(12):
     proposal=trial.copy();proposal[i]=a[i]+step*(.5**shrink)*direction[i]
     if valid(s,m,proposal):trial=proposal;break
   if np.linalg.norm(trial-a)<1e-10:continue
   value=objective(sol,f'r{k}-trial{j}',s,trial,R,seed+10000+k,walks=192,nprobe=64)
   if value<old:
    a=trial;accepted=True;break
  history.append({'round':k+1,'centers':a.tolist(),'seconds':result['design_seconds']+time.perf_counter()-t,'accepted':accepted})
 sol.close();out={'geometry':name,'method':method,'seed':seed,'radius':R,'history':history,'centers':a.tolist(),'total_seconds':result['design_seconds']+time.perf_counter()-t,'backend_peak_rss_mb':max(result['backend_peak_rss_mb'],max(x['peak_rss_megabytes'] for x in sol.calls)),'walk_steps':result['walk_steps']+sum(x['steps'] for x in sol.calls),'distance_queries':result['distance_queries']+sum(x['distance_queries'] for x in sol.calls),'parity_rays':result['parity_rays']+sum(x['parity_rays'] for x in sol.calls),'implementation':'Differential WoS Algorithm 2 / Eq. 13, local implementation; fixed-radius translation; feasible block backtracking, not certified'}
 dump(folder/'refinement.json',out);print('REFINE',name,method,seed,round(out['total_seconds'],3),flush=True);return out

def evaluate(name,seed,channels=False):
 s,m=load(name);folder=OUT/('channels' if channels else 'pipeline')/name/'evaluation'/str(seed)
 if (folder/'evaluation.json').exists():return
 sol=Solver(name,folder/'raw');vals={}
 for method in ['twos','mg','hot','random']:
  d=greedy(name,method,seed,channels)
  states=[('initial',d['centers'],d['radius'])]
  if not channels:
   ref=refine(d,16 if method in ['hot','random'] else 8)
   states += [(f'refine{h["round"]}',h['centers'],R) for h in ref['history'] if h['round'] in [4,8,16]]
  for label,axes,r in states:
   holes=channel_balls(axes) if channels else np.array(axes)
   vals[f'{method}-{label}']=[objective(sol,f'{method}-{label}-block{b}',s,holes,r,880000+b,walks=512,nprobe=128) for b in range(4)]
   if channels:
    checks=[Volume(m,seed=66100+b,power=20).volume(holes,r) for b in range(4)]
    vals[f'{method}-volume']=checks
    assert valid(s,m,np.array(axes),channels=True,r=r)
   else:assert valid(s,m,np.array(axes),r=r)
 vals['unperforated']=[objective(sol,f'base-block{b}',s,[],R,880000+b,walks=512,nprobe=128) for b in range(4)]
 # A distinct discretization checks the first paired seed, at two resolutions.
 if seed==SEEDS[0]:
  vals['grid_check']={}
  for method in ['twos','mg','hot','random']:
   d=greedy(name,method,seed,channels);holes=channel_balls(d['centers']) if channels else d['centers']
   vals['grid_check'][method]=[objective(sol,f'grid{n}-{method}',s,holes,d['radius'],99123,mode='mg',n=n,nprobe=512) for n in [97,129]]
  vals['grid_check']['unperforated']=[objective(sol,f'grid{n}-base',s,[],R,99123,mode='mg',n=n,nprobe=512) for n in [97,129]]
 sol.close();dump(folder/'evaluation.json',vals);print('EVAL',name,seed,channels,flush=True)

def estimators(name,pilot=False):
 s,m=load(name);folder=OUT/'estimators'/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'results.json').exists():return
 p=candidates(m,s,seed=851,n=128);p=p[admissible(m,s,p,.08)][:8];assert len(p)==8;np.savetxt(folder/'sites.txt',p)
 sol=Solver(name,folder/'raw');rows=[]
 for rep in range(8 if pilot else 16):
  f,meta=sol.call(f'ref-{rep}',p,walks=8192,seed=901200+rep,epsilon=.0001)
  for i,v in enumerate(f.score):rows.append(dict(method='reference',rep=rep,site=i,radius=0,walks=8192,score=v,seconds=meta['request_wall_seconds'],queries=meta['distance_queries'],parity_rays=meta['parity_rays'],rss=meta['peak_rss_megabytes']))
 for method in ['twos','coupled-fd']:
  for radius in ([.06] if method=='twos' else [.08,.04,.02,.01]):
   for walks in [256,2048]:
    for rep in range(4 if pilot else 16):
     f,meta=sol.call(f'{method}-{radius}-{walks}-{rep}',p,radius=radius,walks=walks,seed=731200+rep,fd=method=='coupled-fd',epsilon=.0001)
     for i,v in enumerate(f.score):rows.append(dict(method=method,rep=rep,site=i,radius=radius,walks=walks,score=v,seconds=meta['request_wall_seconds'],queries=meta['distance_queries'],parity_rays=meta['parity_rays'],rss=meta['peak_rss_megabytes']))
 for n in [49,65,97,129]:
  f,meta=sol.call(f'mg-{n}',p,mode='mg',n=n)
  for i,v in enumerate(f.score):rows.append(dict(method='mg',rep=0,site=i,radius=0,walks=n,score=v,seconds=meta['request_wall_seconds'],queries=meta['distance_queries'],parity_rays=meta['parity_rays'],rss=meta['peak_rss_megabytes']))
 # Finite-insertion grid references explicitly separate the radius remainder.
 base={n:objective(sol,f'base{n}',s,[],R,13131,mode='mg',n=n,nprobe=512) for n in [97,129]}
 finite=[]
 for r in [.08,.04,.02]:
  for n in [97,129]:
   for i,x in enumerate(p):
    val=objective(sol,f'finite{r}-{n}-{i}',s,[x],r,13131,mode='mg',n=n,nprobe=512)
    finite.append({'site':i,'radius':r,'grid':n,'score':(base[n]-val)/(4*np.pi*r)})
 sol.close();dump(folder/'results.json',{'rows':rows,'finite_grid_checks':finite});print('ESTIMATORS',name,flush=True)

def main():
 global OUT
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['estimators','pipeline','channels','all'],default='all');ap.add_argument('--geometry',default='all');ap.add_argument('--seeds',type=int,default=5);ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
 if args.pilot:OUT=HERE/'pilot'
 OUT.mkdir(exist_ok=True)
 protocol={'date':'2026-09-15','source_protocol':'compact-source-v2-sensor-load','pilot_disclosure':'Remote-source-only pilot left the branching sensor almost cold; add prescribed low-power sensor heat before measured study.','geometries':list(SPECS),'seeds':SEEDS,'radius':R,'holes':6,'protected_compact_heat_sources':True,'sensor_weight_integral':'32*pi*sensor_radius^3/105','candidate_count':384,'twos_walks':[48,1024],'twos_shortlist':16,'MG_primary_max_grid':65,'refinement_rounds':{'twos':8,'mg':8,'hot':16,'random':16},'refinement_walks':[96,192],'feasibility':'Per-cavity backtracking with up to 12 halvings; cyclic block order; three common global step sizes','gradient_offset':.004,'epsilon':.0002,'channel_target_volume':.0168,'calibration_QMC_power':20,'independent_volume_audit':'4 x 2^20','evaluation':'4 independent 128-point Sobol blocks, 512 walks each','threads':4,'method_order':'cyclic rotation by paired-seed index','author_code_commit':'f2b10ebd44904b9c81c063d80e244f0d26868f33','exclusions':['geometry construction','common input mesh preprocessing','independent final evaluation'],'claims':['bias and variance are reported separately where references resolve them','no fluid-flow model','channel score is an axis-point heuristic','local Differential WoS implementation, not the authors optimization application'],'binary_sha256':hashlib.sha256(BIN.read_bytes()).hexdigest(),'source_configs':{name: {'sensor':s['sensor'],'heat':s['heat']} for name,s in SPECS.items()}}
 if not (OUT/'protocol.json').exists():dump(OUT/'protocol.json',protocol)
 for name in SPECS if args.geometry=='all' else [args.geometry]:
  load(name)
  if args.stage in ['estimators','all']:estimators(name,args.pilot)
  for ch,stage in [(False,'pipeline'),(True,'channels')]:
   if args.stage in [stage,'all']:
    for seed in SEEDS[:args.seeds]:
     for method in np.roll(['twos','mg','hot','random'],SEEDS.index(seed)):
      method=str(method)
      d=greedy(name,method,seed,ch)
      if not ch:refine(d,16 if method in ['hot','random'] else 8)
     evaluate(name,seed,ch)
if __name__=='__main__':main()
