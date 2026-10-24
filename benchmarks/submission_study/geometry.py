"""Reproducible watertight application domains; no geometry selected by results."""
from pathlib import Path
import sys,json
import numpy as np
import trimesh,igl
from scipy.stats import qmc
from skimage.measure import marching_cubes
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'engineering_housing'))
from run_housing_benchmark import housing_sdf,write_ascii_ply

def capsule(p,a,b,r):
 a=np.array(a);v=np.array(b)-a;t=np.clip((p-a)@v/(v@v),0,1)
 return np.linalg.norm(p-a-t[:,None]*v,axis=1)-r

def branch_sdf(p):
 d=capsule(p,[-.85,0,0],[0,0,0],.24)
 for y in [-.48,.48]:
  d=np.minimum(d,capsule(p,[0,0,0],[.72,y,0],.23))
 # Small parallel ribs and an elevated return loop create thin features and gaps.
 for x in np.linspace(-.65,-.10,6):
  d=np.minimum(d,capsule(p,[x,0,.10],[x,0,.48],.047))
 arc=np.stack([.52+.35*np.cos(np.linspace(-1.57,1.57,28)),.48*np.sin(np.linspace(-1.57,1.57,28)),np.full(28,.18)],1)
 for a,b in zip(arc[:-1],arc[1:]):d=np.minimum(d,capsule(p,a,b,.065))
 return d

def ring_sdf(p):
 d=np.sqrt((np.linalg.norm(p[:,:2],axis=1)-.67)**2+p[:,2]**2)-.215
 d=np.minimum(d,capsule(p,[-.68,0,0],[.68,0,0],.14))
 for a in np.linspace(0,2*np.pi,18,endpoint=False):
  c=np.array([.67*np.cos(a),.67*np.sin(a),0])
  d=np.minimum(d,capsule(p,c+[0,0,.08],c+[0,0,.40],.043))
 return d

SPECS={
 'housing':dict(sensor=[.55,.18,-.03,.10],heat=[[-.48,-.18,-.06,.15,35],[.36,-.18,-.04,.13,45]],bounds=[[-1.2,-.77,-.4],[1.2,.77,.77]],sdf=housing_sdf,label='Finned housing'),
 'branch':dict(sensor=[.57,.37,0,.10],heat=[[-.66,0,0,.15,35],[.57,-.38,0,.13,45]],bounds=[[-1.15,-.78,-.30],[1.10,.78,.57]],sdf=branch_sdf,label='Bifurcating manifold'),
 'ring':dict(sensor=[0,.67,0,.10],heat=[[-.64,-.18,0,.13,35],[.52,-.41,0,.13,45]],bounds=[[-.94,-.94,-.27],[.94,.94,.49]],sdf=ring_sdf,label='Ribbed annular spreader')}

# A protected sensor dissipates a small amount of heat in addition to the main components.
for _s in SPECS.values():
 _s['heat'].append([*_s['sensor'][:3],.085,4.0])

def load(name):
 spec=SPECS[name];path=HERE/'assets'/f'{name}.ply'
 if not path.exists():
  bounds=np.array(spec['bounds']);step=.016
  axes=[np.linspace(a,b,int(np.ceil((b-a)/step))+1) for a,b in zip(*bounds)]
  grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1);flat=grid.reshape(-1,3)
  f=np.concatenate([spec['sdf'](p) for p in np.array_split(flat,32)]).reshape(grid.shape[:3])
  v,faces,_,_=marching_cubes(f,0,spacing=[a[1]-a[0] for a in axes]);v+=bounds[0]
  m=trimesh.Trimesh(v,faces,process=True);m.fix_normals()
  assert m.is_watertight and m.is_volume
  write_ascii_ply(path,m.vertices,m.faces)
  cfg=HERE/'assets'/f'{name}.txt'
  cfg.write_text('\n'.join(' '.join(map(str,row)) for row in [spec['sensor']]+spec['heat'])+'\n')
 cfg=HERE/'assets'/f'{name}.txt'
 text='\n'.join(' '.join(map(str,row)) for row in [spec['sensor']]+spec['heat'])+'\n'
 if not cfg.exists() or cfg.read_text()!=text:cfg.write_text(text)
 m=trimesh.load(path,process=False)
 return spec,m

def distance(m,p):
 if not hasattr(m,'_wih_aabb'):
  m._wih_aabb=igl.AABB();m._wih_aabb.init(m.vertices,m.faces)
 return np.sqrt(m._wih_aabb.squared_distance(m.vertices,m.faces,np.asarray(p))[0])
def admissible(m,s,p,r=.065,channels=False):
 p=np.asarray(p);ok=m.contains(p)&(distance(m,p)>r+.012)
 for c in [s['sensor']]+s['heat']:
  dim=2 if channels else 3
  ok &= np.linalg.norm(p[:,:dim]-np.array(c)[:dim],axis=1)>c[3]+r+.018
 return ok

def candidates(m,s,seed,n=384,channels=False):
 p=qmc.Sobol(3,scramble=True,seed=seed).random_base2(15)
 p=m.bounds[0]+p*(m.bounds[1]-m.bounds[0])
 if channels:p[:,2]=s['sensor'][2]
 p=p[admissible(m,s,p,.11 if channels else .065,channels)]
 if len(p)<n:raise ValueError((len(p),n))
 return p[:n]

def probes(s,seed,n=128):
 # Uniform ball quadrature. q normalized by its exact volume average 8/35.
 a=qmc.Sobol(3,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n]
 z=1-2*a[:,0];phi=2*np.pi*a[:,1];r=a[:,2]**(1/3)
 p=np.stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z],1)*r[:,None]
 return p*s['sensor'][3]+s['sensor'][:3],(1-r*r)**2*35/8

if __name__=='__main__':
 report={}
 for name in SPECS:
  s,m=load(name)
  clearance=[]
  for c in [s['sensor']]+s['heat']:
   d=distance(m,np.array([c[:3]]))[0];assert m.contains([c[:3]])[0] and d>c[3]
   clearance.append(float(d-c[3]))
  report[name]={'vertices':len(m.vertices),'triangles':len(m.faces),'volume':m.volume,'source_sensor_clearance':clearance,'candidate_count':len(candidates(m,s,90)), 'heat_integral':sum(h[4]*32*np.pi*h[3]**3/105 for h in s['heat'])}
 (HERE/'assets'/'geometry-audit.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
