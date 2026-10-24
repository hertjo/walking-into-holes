"""Independent directional finite-difference check and offset sensitivity."""
import numpy as np,json
from pathlib import Path
from run_study import *
name='branch';s,m=load(name);folder=HERE/'validation-local-cavity'
a=np.array([np.array(s['sensor'][:3])-.22*np.array([.72,.48,0])/np.linalg.norm([.72,.48,0])])
assert valid(s,m,a)
sol=Solver(name,folder/'raw');p,w=probes(s,9113,128)
# Move the most sensor-proximal cavity along x, keeping all other parameters fixed.
i=int(np.argmin(np.linalg.norm(a-np.array(s['sensor'][:3]),axis=1)));h=.005
records=[]
for offset in [.002,.004,.008]:
 vals=[]
 for b in range(4):
  f,_=sol.call(f'gradient-{offset}-{b}',p,a,R,walks=1024,seed=813141+b,diff=True,offset=offset,epsilon=.0001)
  vals.append(float(np.mean(w*f[f'dc{i}_x'])))
 records.append({'method':'differential-wos','offset':offset,'values':vals})
vals=[]
for b in range(4):
 plus=a.copy();minus=a.copy();plus[i,0]+=h;minus[i,0]-=h
 assert valid(s,m,plus) and valid(s,m,minus)
 u=objective(sol,f'plus-{b}',s,plus,R,734111+b,walks=1024,nprobe=128)
 v=objective(sol,f'minus-{b}',s,minus,R,734111+b,walks=1024,nprobe=128)
 vals.append((u-v)/(2*h))
records.append({'method':'centered-finite-difference','step':h,'values':vals})
sol.close();dump(folder/'gradient-check.json',{'geometry':name,'cavity':i,'records':records})
print(json.dumps(records,indent=2))
