"""Validate the auxiliary heat animation, and solve cooling after saved insertions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,io,zipfile
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import cg
from scipy.interpolate import RegularGridInterpolator
from vtk.util.numpy_support import vtk_to_numpy
from prepare_scooped import OUT,operator
from render_scene import read_vtp

def write_npz(path,data):
 buffer=io.BytesIO();np.savez_compressed(buffer,**data);raw=buffer.getvalue()
 with zipfile.ZipFile(io.BytesIO(raw)) as z:assert z.testzip() is None
 temp=path.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(path)

# A separable Dirichlet eigenmode has a known semidiscrete evolution.
n=13;h=1/(n+1);a=np.arange(1,n+1)*h;x,y,z=np.meshgrid(a,a,a,indexing='ij');phi=(np.sin(np.pi*x)*np.sin(np.pi*y)*np.sin(np.pi*z)).ravel();A=operator(np.ones((n,n,n),bool),h);lam=12/h**2*np.sin(np.pi*h/2)**2;dt=.001;u=np.zeros_like(phi);B=eye(len(u),format='csr')+dt*A
for _ in range(20):u,info=cg(B,u+dt*lam*phi,rtol=1e-11);assert info==0
exact=(1-(1+dt*lam)**-20)*phi;err=float(np.linalg.norm(u-exact)/np.linalg.norm(exact));assert err<1e-8
report={'manufactured_dirichlet_eigenmode_relative_error':err,'grid_refinement':{},'time_refinement':{},'cooling':{}}
coarse=json.loads((OUT/'grid112-audit.json').read_text()) if (OUT/'grid112-audit.json').exists() else None
fine=json.loads((OUT/'thermal-audit.json').read_text())
if coarse:
 for tag in ['ours0','budget12']:
  v=fine['designs'][tag]['steady_mean'];old=coarse['designs'][tag]['steady_mean'];report['grid_refinement'][tag]={'112_to_144_mean_change_relative':abs(v-old)/v}

for previous,tag in [('ours0','ours1'),('ours1','ours3'),('ours3','ours6')]:
 path=OUT/f'{tag}-field.npz';d=dict(np.load(path));prev=dict(np.load(OUT/f'{previous}-field.npz'));axes=[d[k] for k in ['x','y','z']];mask=d['mask'];h=axes[0][1]-axes[0][0];A=operator(mask,h);identity=eye(A.shape[0],format='csr');f=np.broadcast_to(1+.65*(axes[1][None,:,None]+.5),mask.shape)[mask];old=prev['u'][mask].astype(float)
 q=vtk_to_numpy(read_vtp(OUT/f'{tag}-bowl.vtp').GetPoints().GetData());volume=np.zeros(mask.shape,np.float32);volume[mask]=old;interp=RegularGridInterpolator(axes,volume,bounds_error=False,fill_value=0)
 times=np.r_[0,np.geomspace(.00010,.065,48)];fields=[interp(q).astype('float32')];means=[float(old.mean())];worst=0
 for start,end in zip(times[:-1],times[1:]):
  dt=end-start;B=identity+dt*A;rhs=old+dt*f;new,info=cg(B,rhs,x0=old,rtol=5e-9,maxiter=500);assert info==0
  assert float((new-old).max())<2e-7
  worst=max(worst,float(np.linalg.norm(B@new-rhs)/np.linalg.norm(rhs)));old=new;volume[mask]=old;fields.append(interp(q).astype('float32'));means.append(float(old.mean()))
 assert np.max(np.diff(means))<1e-8
 d.update(cooling_times=times,cooling_surface=np.array(fields),cooling_mean=np.array(means));write_npz(path,d)
 report['cooling'][tag]={'previous_design':previous,'mean_decreases':True,'max_relative_residual':worst,'final_relative_steady_mean_error':float(abs(means[-1]-d['u'][mask].mean())/d['u'][mask].mean())}
 print('Cooling',tag,report['cooling'][tag],flush=True)

# Half time steps on the intact solid: compare at every stored time.
d=dict(np.load(OUT/'ours0-field.npz'));mask=d['mask'];h=d['x'][1]-d['x'][0];A=operator(mask,h);identity=eye(A.shape[0],format='csr');f=np.broadcast_to(1+.65*(d['y'][None,:,None]+.5),mask.shape)[mask];old=np.zeros_like(f);means=[0.]
for start,end in zip(d['times'][:-1],d['times'][1:]):
 dt=(end-start)/2;B=identity+dt*A
 for _ in range(2):old,info=cg(B,old+dt*f,x0=old,rtol=5e-9,maxiter=500);assert info==0
 means.append(float(old.mean()))
change=np.max(np.abs(np.asarray(means)-d['mean']))/d['u'][mask].mean();assert change<.025
report['time_refinement']['ours0']={'max_mean_change_divided_by_steady_mean':float(change),'comparison':'two half backward-Euler steps versus one full step on each stored interval'}
report['scope']='Checks of the auxiliary animation only. Residuals do not bound continuum error. Grid refinement is a sensitivity check, not a certified error bound; reported paper objectives are unchanged.'
(OUT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
