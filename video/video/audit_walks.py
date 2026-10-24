"""Independent numerical checks of the displayed WoS paths.

Triangle distances and solid-angle winding use a separate NumPy implementation,
not the VTK distance oracle that generated the Bunny walks.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from render_scene import ROOT,read_poly,vtk_to_numpy

A=Path(__file__).parent/'assets';tr=json.loads((A/'walk-traces.json').read_text())
mesh=read_poly(ROOT/'benchmarks/stanford_bunny/output/stanford-bunny-watertight.ply')
vertices=vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
faces=vtk_to_numpy(mesh.GetPolys().GetData()).reshape(-1,4)[:,1:];tri=vertices[faces]
a,b,c=tri[:,0],tri[:,1],tri[:,2];ab=b-a;ac=c-a
n=np.cross(ab,ac);n2=np.einsum('ij,ij->i',n,n);good=n2>1e-25

def distance(x):
 ap=x-a;t=np.einsum('ij,ij->i',ap,n)/np.maximum(n2,1e-30);p=x-t[:,None]*n
 v=p-a;d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac);d20=np.einsum('ij,ij->i',v,ab);d21=np.einsum('ij,ij->i',v,ac);den=d00*d11-d01*d01
 u=(d11*d20-d01*d21)/np.maximum(den,1e-30);w=(d00*d21-d01*d20)/np.maximum(den,1e-30);inside=good&(u>=0)&(w>=0)&(u+w<=1)
 best=np.where(inside,t*t*n2,np.inf)
 for q,r in [(a,b),(b,c),(c,a)]:
  edge=r-q;v=x-q;s=np.clip(np.einsum('ij,ij->i',v,edge)/np.maximum(np.einsum('ij,ij->i',edge,edge),1e-30),0,1);d=v-s[:,None]*edge;best=np.minimum(best,np.einsum('ij,ij->i',d,d))
 return float(np.sqrt(best.min()))

def winding(x):
 u=a-x;v=b-x;w=c-x;lu=np.linalg.norm(u,axis=1);lv=np.linalg.norm(v,axis=1);lw=np.linalg.norm(w,axis=1)
 numerator=np.einsum('ij,ij->i',u,np.cross(v,w));denominator=lu*lv*lw+np.einsum('ij,ij->i',u,v)*lw+np.einsum('ij,ij->i',v,w)*lu+np.einsum('ij,ij->i',w,u)*lv
 return float(np.arctan2(numerator,denominator).sum()/(2*np.pi))

records=[];seed_error=0.;jump_error=0.;radius_error=0.;min_gap=float('inf');winding_error=0.
for trace in tr['traces']:
 rng=np.random.default_rng(trace['seed']);x=np.array(tr['start']);value=0
 for k,st in enumerate(trace['steps']):
  p=np.array(st['x']);q=np.array(st['next']);r=st['radius'];assert np.linalg.norm(p-x)<1e-12
  direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
  seed_error=max(seed_error,float(np.linalg.norm(q-(p+r*direction))))
  jump_error=max(jump_error,abs(np.linalg.norm(q-p)-r));d=distance(p);gap=d-r;min_gap=min(min_gap,gap);radius_error=max(radius_error,abs(r-.9999*d));assert gap>0,(trace['seed'],k,gap)
  for y in [p,q]:
   we=abs(abs(winding(y))-1);winding_error=max(winding_error,we);assert we<1e-6,(trace['seed'],k,we)
  contribution=(1+.65*(p[1]+.5))*r*r/6;assert abs(contribution-st['contribution'])<1e-13;value+=contribution;x=q
  records.append({'seed':trace['seed'],'step':k,'independent_distance':d,'radius':r,'boundary_gap':gap})
 terminal=distance(x);assert terminal<tr['terminal_epsilon'];assert abs(value-trace['estimate_u'])<1e-12
 assert abs(terminal-trace['termination_distance'])<1e-8
assert seed_error<1e-12 and radius_error<1e-8 and jump_error<1e-12

# A diagnostic of the generator, not a proof from a few illustrated paths.
rng=np.random.default_rng(2026091900);directions=rng.normal(size=(200000,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
mean=directions.mean(0);second=directions.T@directions/len(directions)
assert np.max(abs(mean))<.008 and np.max(abs(second-np.eye(3)/3))<.008
report={'mesh_sha256':hashlib.sha256((ROOT/'benchmarks/stanford_bunny/output/stanford-bunny-watertight.ply').read_bytes()).hexdigest(),'bunny_paths':len(tr['traces']),'bunny_steps':len(records),'independent_geometry':'NumPy point-to-triangle distance (plane plus edge regions); solid-angle winding number for interior classification','max_seed_replay_error':seed_error,'max_jump_length_error':jump_error,'max_radius_rule_error':radius_error,'minimum_sphere_clearance':min_gap,'max_interior_winding_error':winding_error,'all_paths_terminate_at_declared_epsilon':True,'uniform_direction_generator':'Normalize three independent standard Gaussians; rotational invariance gives the uniform spherical law. Seeds are retained in source data.','direction_diagnostic':{'n':len(directions),'mean':mean.tolist(),'second_moment':second.tolist(),'scope':'Diagnostic only; not evidence that a finite set of sample paths is uniformly distributed.'},'animation_scope':'Straight moving segments interpolate WoS jumps for display; they do not depict Brownian trajectories. The capsule and Bunny are separate examples.','steps':records}
(A/'walk-independent-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='steps'},indent=2))
