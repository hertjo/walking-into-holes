"""Replay and independently check the analytic capsule geometry and projection."""
from pathlib import Path
import json
import numpy as np
A=Path(__file__).parent/'assets';d=json.loads((A/'capsule-walks.json').read_text());L=d['half_length'];R=d['radius'];P=d['projection'];M=np.array([P['right'],-np.array(P['up'])]);scale=P['scale'];center=np.array(P['center'])
assert np.allclose(M@M.T,np.eye(2),atol=1e-14)
def boundary_distance(p):
 x,y,z=p;return R-np.sqrt(max(abs(x)-L,0.)**2+y*y+z*z)
def project(p):return center+scale*(M@p)
endpoints=np.array([project(np.array([-L,0,0])),project(np.array([L,0,0]))]);axis=endpoints[1]-endpoints[0]
def projected_axis_distance(p):
 f=np.clip(np.dot(p-endpoints[0],axis)/np.dot(axis,axis),0,1)
 return np.linalg.norm(p-endpoints[0]-f*axis)
count=0;max_replay=0.;max_projection=0.;min_clearance=float('inf');min_projected_clearance=float('inf')
for tr in d['traces']:
 rng=np.random.default_rng(tr['seed']);p=np.array(d['start'])
 for st in tr['steps']:
  x=np.array(st['x']);y=np.array(st['next']);r=st['radius'];v=rng.normal(size=3);v/=np.linalg.norm(v)
  assert np.allclose(x,p,atol=1e-14);assert boundary_distance(x)>=d['terminal_epsilon'];assert abs(r-.9999*boundary_distance(x))<1e-13
  max_replay=max(max_replay,np.linalg.norm(y-x-r*v));assert abs(np.linalg.norm(y-x)-r)<1e-13
  assert boundary_distance(y)>0;min_clearance=min(min_clearance,boundary_distance(x)-r)
  for key,q in [('screen_x',x),('screen_next',y)]:max_projection=max(max_projection,np.linalg.norm(project(q)-st[key]))
  assert abs(st['screen_radius']-scale*r)<1e-12
  gap=scale*R-projected_axis_distance(project(x))-scale*r;min_projected_clearance=min(min_projected_clearance,gap);assert gap>0
  # Orthographic projection contracts distances. An exit need not lie on the 2-D rim.
  assert np.linalg.norm(project(y)-project(x))<=scale*r+1e-12
  p=y;count+=1
 assert boundary_distance(p)<d['terminal_epsilon'];assert abs(boundary_distance(p)-tr['termination_distance'])<1e-12
assert max_replay<1e-12 and max_projection<1e-10
# Verify every displayed capsule interpolation uses the same affine projection.
frames=0;max_interpolation=0.
for f in range(19*24,31*24):
 q=(f/24-19)/12;progress=(q*3)%1;idx=min(int(q*3),2);steps=d['traces'][idx]['steps'];w=np.array([s['radius']**.35 for s in steps]);edges=np.r_[0,np.cumsum(w)/sum(w)];k=min(np.searchsorted(edges,progress,side='right')-1,len(steps)-1)
 assert 0<=k<len(steps)
 a=np.clip((progress-edges[k])/(edges[k+1]-edges[k]),0,1);a=a*a*(3-2*a);st=steps[k]
 p=(1-a)*np.array(st['x'])+a*np.array(st['next']);screen=(1-a)*np.array(st['screen_x'])+a*np.array(st['screen_next']);max_interpolation=max(max_interpolation,np.linalg.norm(project(p)-screen));frames+=1
assert max_interpolation<1e-10
report={'paths':len(d['traces']),'steps':count,'max_seed_replay_error':float(max_replay),'minimum_sphere_clearance':float(min_clearance),'minimum_projected_sphere_clearance_px':float(min_projected_clearance),'max_projection_error_px':float(max_projection),'all_paths_terminate_at_epsilon':True,'animation_frames_checked':frames,'max_animation_projection_error_px':float(max_interpolation),'method':'Independent capsule distance formula using axial excess; seed replay; orthonormal projection; exact projected-capsule containment; frame-by-frame interpolation check.','rendering_scope':'SVG sphere outlines represent orthographic sphere silhouettes. Raster placement is rounded to subpixel/pixel precision. Straight segments show jumps, not Brownian paths. Separate capsule and Bunny examples share relative time only.'}
(A/'capsule-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
