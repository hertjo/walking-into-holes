"""Curved display cuts and an auxiliary 3-D heat solve on saved designs.

The display cut never enters the PDE. This finite-difference visualization is
separate from the retained WoS optimization benchmarks and their objectives.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
from scipy.sparse import coo_matrix,eye
from scipy.sparse.linalg import cg
from scipy.interpolate import RegularGridInterpolator
from render_scene import ROOT,read_poly

OUT=ROOT/'video/assets/scooped';OUT.mkdir(exist_ok=True)
CENTER=np.array([-.025,-.15,.535]);AXES=np.array([.385,.335,.49])
TAGS=['ours0','ours1','ours3','ours6','random','voxel','budget6','budget12']

def designs():
 import csv
 rows=list(csv.DictReader(open(ROOT/'benchmarks/stanford_bunny/output/selected-centers.csv')))
 d={}
 for tag in TAGS:
  if tag.startswith('budget'):
   n=int(tag[6:]);units=6 if n==6 else 3
   a=json.loads((ROOT/f'benchmarks/bunny_budget/output/v{units}-n{n:02d}-s2026091501/twos/result.json').read_text())
   d[tag]=(np.array([v['center'] for v in a['selections']]),a['radius'])
  else:
   name='Random feasible' if tag=='random' else 'Voxel adjoint TD' if tag=='voxel' else 'TWoS + fresh validation'
   n=int(tag[4:]) if tag.startswith('ours') else 6
   d[tag]=(np.array([[float(v[k]) for k in ['x','y','z']] for v in rows if v['method']==name][:n]).reshape(-1,3),.05)
 return d

def clip(poly,implicit,inside=False):
 c=vtk.vtkClipPolyData();c.SetInputData(poly);c.SetClipFunction(implicit);c.SetInsideOut(inside);c.SetValue(0);c.Update()
 q=vtk.vtkPolyData();q.DeepCopy(c.GetOutput());return q

def save(poly,path):
 w=vtk.vtkXMLPolyDataWriter();w.SetFileName(str(path));w.SetInputData(poly);w.SetDataModeToBinary();w.Write()

def sphere(center,r,res=100):
 s=vtk.vtkSphereSource();s.SetRadius(r);s.SetCenter(*center);s.SetThetaResolution(res);s.SetPhiResolution(res//2);s.Update();return s.GetOutput()

def geometry():
 solid=read_poly(ROOT/'benchmarks/stanford_bunny/output/stanford-bunny-watertight.ply')
 dist=vtk.vtkImplicitPolyDataDistance();dist.SetInput(solid)
 quad=vtk.vtkQuadric();a=1/AXES**2;c=CENTER
 quad.SetCoefficients(a[0],a[1],a[2],0,0,0,*(-2*a*c),float(np.sum(a*c*c)-1))
 sh=clip(solid,quad);ghost=clip(solid,quad,True)
 save(sh,OUT/'shell.vtp');save(ghost,OUT/'ghost.vtp')
 bowl=vtk.vtkPolyData();bowl.DeepCopy(sphere([0,0,0],1,420))
 p=vtk_to_numpy(bowl.GetPoints().GetData());p[:]=p*AXES+CENTER;bowl.GetPoints().Modified()
 bowl=clip(bowl,dist,True)
 report={}
 for tag,(centers,r) in designs().items():
  b=vtk.vtkPolyData();b.DeepCopy(bowl);cav=vtk.vtkAppendPolyData()
  visible=0
  for center in centers:
   sph=vtk.vtkSphere();sph.SetCenter(*center);sph.SetRadius(r);b=clip(b,sph)
   wall=clip(sphere(center,r,112),quad)
   if wall.GetNumberOfPoints():cav.AddInputData(wall);visible+=1
  save(b,OUT/f'{tag}-bowl.vtp')
  if len(centers):cav.Update();save(cav.GetOutput(),OUT/f'{tag}-cavity.vtp')
  q=vtk_to_numpy(b.GetPoints().GetData())
  report[tag]={'bowl_points':len(q),'bowl_z_range':[float(q[:,2].min()),float(q[:,2].max())],'cavities':len(centers),'cavity_surfaces_retained':visible}
 (OUT/'geometry.json').write_text(json.dumps({'cut':'ellipsoidal scoop, display only','center':CENTER.tolist(),'axes':AXES.tolist(),'designs':report},indent=2))
 print('Scooped surfaces built',flush=True)

def grid(n):
 solid=read_poly(ROOT/'benchmarks/stanford_bunny/output/stanford-bunny-watertight.ply');bounds=solid.GetBounds();h=1.06/(n-1)
 axes=[np.arange(bounds[2*j]-.025,bounds[2*j+1]+.025+h*.1,h) for j in range(3)]
 # VTK stencil uses the actual watertight outer mesh, not a display cut.
 img=vtk.vtkImageData();img.SetDimensions(*(len(v) for v in axes));img.SetOrigin(*(v[0] for v in axes));img.SetSpacing(h,h,h)
 img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,1);vtk_to_numpy(img.GetPointData().GetScalars())[:]=1
 s=vtk.vtkPolyDataToImageStencil();s.SetInputData(solid);s.SetOutputOrigin(img.GetOrigin());s.SetOutputSpacing(img.GetSpacing());s.SetOutputWholeExtent(img.GetExtent());s.Update()
 f=vtk.vtkImageStencil();f.SetInputData(img);f.SetStencilConnection(s.GetOutputPort());f.SetBackgroundValue(0);f.Update()
 mask=vtk_to_numpy(f.GetOutput().GetPointData().GetScalars()).reshape(tuple(len(v) for v in axes)[::-1]).transpose(2,1,0).astype(bool)
 return axes,h,mask

def operator(mask,h):
 ids=np.full(mask.shape,-1,np.int32);ids[mask]=np.arange(mask.sum());rows=[];cols=[]
 for j in range(3):
  a=[slice(None)]*3;b=a.copy();a[j]=slice(None,-1);b[j]=slice(1,None)
  ia=ids[tuple(a)];ib=ids[tuple(b)];ok=(ia>=0)&(ib>=0)
  rows.extend([ia[ok],ib[ok]]);cols.extend([ib[ok],ia[ok]])
 count=int(mask.sum());row=np.concatenate(rows+[np.arange(count)]);col=np.concatenate(cols+[np.arange(count)])
 vals=np.r_[-np.ones(len(row)-count),np.full(count,6.)]/h**2
 return coo_matrix((vals,(row,col)),shape=(count,count)).tocsr()

def solve(n=112,only=None):
 axes,h,outer=grid(n);x,y,z=np.meshgrid(*axes,indexing='ij',sparse=True);source=np.broadcast_to(1+.65*(y+.5),outer.shape)
 times=np.r_[0,np.geomspace(.00012,.065,48)]
 report={'scheme':'7-point Cartesian Dirichlet mask; backward Euler in dimensionless time','grid_shape':list(outer.shape),'spacing':h,'PDE':'du/dtau - Delta u = 1 + .65*(y+.5); initial u=0; all physical boundaries u=0','scope':'Auxiliary thermal visualization, not a TWoS timing or accuracy experiment. The curved display cut is absent from the solver.','designs':{}}
 for tag,(centers,r) in designs().items():
  if only and tag not in only:continue
  st=time.monotonic();mask=outer.copy()
  for c in centers:mask&=((x-c[0])**2+(y-c[1])**2+(z-c[2])**2>r*r)
  A=operator(mask,h);f=source[mask];u,info=cg(A,f,rtol=2e-9,maxiter=1200);assert info==0,(tag,info)
  residual=float(np.linalg.norm(A@u-f)/np.linalg.norm(f));volume=np.zeros(mask.shape,np.float32);volume[mask]=u
  bowl=vtk.vtkXMLPolyDataReader();bowl.SetFileName(str(OUT/f'{tag}-bowl.vtp'));bowl.Update();q=vtk_to_numpy(bowl.GetOutput().GetPoints().GetData())
  interp=RegularGridInterpolator(axes,volume,bounds_error=False,fill_value=0);surface=interp(q).astype('float32')
  data=dict(x=axes[0],y=axes[1],z=axes[2],u=volume,mask=mask,surface=surface)
  stat={'active_nodes':int(mask.sum()),'steady_relative_residual':residual,'steady_mean':float(u.mean()),'steady_max':float(u.max())}
  if tag in ['ours0','budget12']:
   old=np.zeros_like(u);fields=[np.zeros(len(q),np.float32)];means=[0.];maxima=[0.];identity=eye(len(u),format='csr');worst=0
   for a,b in zip(times[:-1],times[1:]):
    dt=b-a;B=identity+dt*A;rhs=old+dt*f;old,info=cg(B,rhs,x0=old,rtol=5e-9,maxiter=500);assert info==0
    worst=max(worst,float(np.linalg.norm(B@old-rhs)/np.linalg.norm(rhs)));volume[mask]=old;fields.append(interp(q).astype('float32'));means.append(float(old.mean()));maxima.append(float(old.max()))
   volume[mask]=u;data.update(times=times,transient_surface=np.array(fields),mean=np.array(means),maximum=np.array(maxima))
   assert np.min(np.diff(means))>-1e-9 and old.max()<=u.max()+1e-7 and np.min(fields)>-1e-8
   stat.update(transient_max_relative_residual=worst,heating_monotone=True,final_fraction_of_steady_mean=float(means[-1]/u.mean()))
  import io,zipfile
  buffer=io.BytesIO();np.savez_compressed(buffer,**data);payload=buffer.getvalue()
  with zipfile.ZipFile(io.BytesIO(payload)) as archive:assert archive.testzip() is None
  target=OUT/f'{tag}-field.npz';temp=target.with_suffix('.tmp');temp.write_bytes(payload);temp.replace(target);report['designs'][tag]=stat
  print(tag,mask.sum(),'nodes;',round(time.monotonic()-st,1),'s',stat,flush=True)
 if only and (OUT/'thermal-audit.json').exists():
  previous=json.loads((OUT/'thermal-audit.json').read_text());previous['designs'].update(report['designs']);report=previous
 (OUT/'thermal-audit.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=144);p.add_argument('--geometry-only',action='store_true');a=p.parse_args();geometry()
 if not a.geometry_only:solve(a.grid)
