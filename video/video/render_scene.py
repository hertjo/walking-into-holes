"""Deterministic VTK rendering of saved scientific geometry and scalar fields."""
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from PIL import Image,ImageDraw,ImageFilter
from matplotlib import colormaps
ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'figures/hero-assets'
THERM=ASSET/'thermal'
SCOOP=ROOT/'video/assets/scooped'
PURPLE=(.64,.51,.72);TEAL=(.48,.65,.71)

def read_poly(p):
 r=vtk.vtkPLYReader();r.SetFileName(str(p));r.Update();d=vtk.vtkPolyData();d.DeepCopy(r.GetOutput());return d

def actor(poly,color,opacity=1,smooth=True):
 if smooth:
  poly.GetPointData().SetNormals(None);poly.GetCellData().SetNormals(None)
  n=vtk.vtkPolyDataNormals();n.SetInputData(poly);n.SplittingOff();n.ConsistencyOn();n.AutoOrientNormalsOff();n.Update();poly=n.GetOutput()
 m=vtk.vtkPolyDataMapper();m.SetInputData(poly);m.ScalarVisibilityOff()
 a=vtk.vtkActor();a.SetMapper(m);p=a.GetProperty();p.SetColor(*color);p.SetOpacity(opacity);p.SetAmbient(.28);p.SetDiffuse(.64);p.SetSpecular(.3);p.SetSpecularPower(38);p.SetInterpolationToPhong();return a

def read_vtp(p,reverse=False):
 r=vtk.vtkXMLPolyDataReader();r.SetFileName(str(p));r.Update();d=vtk.vtkPolyData();d.DeepCopy(r.GetOutput())
 if reverse:
  f=vtk.vtkReverseSense();f.SetInputData(d);f.ReverseCellsOn();f.ReverseNormalsOn();f.Update();d.DeepCopy(f.GetOutput())
 return d

class Scene:
 def __init__(self,w=1320,h=1080):
  self.w=w;self.h=h;self.ren=vtk.vtkRenderer();self.ren.SetBackground(1,1,1)
  self.win=vtk.vtkRenderWindow();self.win.SetOffScreenRendering(1);self.win.SetSize(w,h);self.win.SetMultiSamples(0);self.win.SetAlphaBitPlanes(1);self.win.AddRenderer(self.ren)
  self.ren.SetUseDepthPeeling(1);self.ren.SetMaximumNumberOfPeels(12);self.ren.SetOcclusionRatio(.03);self.ren.SetUseFXAA(True);self.ren.SetTwoSidedLighting(False)
  self.cam=self.ren.GetActiveCamera();self.cam.ParallelProjectionOn();self.groups={};self.extras=[];self.current=None;self.thermal={};self.datasets={}
  for pos,intensity in [((-2,3,4),.85),((3,1,2),.35),((0,2,-3),.35)]:
   l=vtk.vtkLight();l.SetLightTypeToSceneLight();l.SetPosition(*pos);l.SetFocalPoint(0,0,0);l.SetIntensity(intensity);self.ren.AddLight(l)
  self.capture=vtk.vtkWindowToImageFilter();self.capture.SetInput(self.win);self.capture.SetInputBufferTypeToRGB();self.capture.ReadFrontBufferOff()
 def load(self,tag,field=None,color=PURPLE):
  key=(tag,field,tuple(color))
  if key in self.groups:return self.groups[key]
  if tag!='input' and SCOOP.exists():return self.load_scooped(tag,field,color)
  acts=[]
  for name in ['shell','section','cavity']:
   p=ASSET/f'{tag}-{name}.ply'
   if not p.exists():continue
   poly=read_poly(p);a=actor(poly,color if name=='shell' else ((.35,.23,.43) if name=='cavity' else (.85,.81,.87)),smooth=name!='section')
   if name=='section' and field:
    is_score=field.endswith(':score');fname=field.split(':')[0];f=np.load(THERM/f'{fname}.npz');v=gaussian_filter(f['benefit' if is_score else 'u'],1);v[~f['active']]=0
    q=vtk_to_numpy(poly.GetPoints().GetData());uv=np.c_[(q[:,0]-f['x'][0])/(f['x'][-1]-f['x'][0]),(q[:,1]-f['y'][0])/(f['y'][-1]-f['y'][0])].astype('float32');poly.GetPointData().SetTCoords(numpy_to_vtk(uv,deep=True))
    rgba=(colormaps['viridis' if is_score else 'inferno'](np.clip(v/(.0005173628631419 if is_score else .025),0,1))*255).astype('uint8');img=vtk.vtkImageData();img.SetDimensions(v.shape[1],v.shape[0],1);img.GetPointData().SetScalars(numpy_to_vtk(rgba.reshape(-1,4),deep=True,array_type=vtk.VTK_UNSIGNED_CHAR));tex=vtk.vtkTexture();tex.SetInputData(img);tex.InterpolateOn();a.SetTexture(tex);a.GetProperty().SetColor(1,1,1);a.GetProperty().LightingOff()
   acts.append(a);self.ren.AddActor(a);a.VisibilityOff()
  self.groups[key]=acts;return acts
 def load_scooped(self,tag,field,color):
  key=(tag,field,tuple(color));acts=[]
  shell=actor(read_vtp(SCOOP/'shell.vtp'),color);shell.GetProperty().SetAmbient(.21);shell.GetProperty().SetDiffuse(.73);shell.GetProperty().SetSpecular(.22);acts.append(shell)
  bowl=read_vtp(SCOOP/f'{tag}-bowl.vtp',reverse=True);surface=actor(bowl,(.88,.82,.90));p=surface.GetProperty();p.SetAmbient(.54);p.SetDiffuse(.40);p.SetSpecular(.10);p.SetSpecularPower(55)
  if field:
   if tag not in self.datasets:self.datasets[tag]=dict(np.load(SCOOP/f'{tag}-field.npz'))
   data=self.datasets[tag];score=field.endswith(':score');v=data['surface']**2 if score else data['surface']
   # The physical field is evaluated in 3-D on the curved display surface.
   surface.GetMapper().SetInputData(bowl);surface.GetMapper().SetScalarModeToUsePointFieldData();surface.GetMapper().SelectColorArray('rgb');surface.GetMapper().SetColorModeToDirectScalars();surface.GetMapper().ScalarVisibilityOn()
   self.thermal[key]=(bowl,surface,data,score);self.set_surface(key,v)
  acts.append(surface)
  if field and not score:
   lines=vtk.vtkContourFilter();lines.SetInputData(bowl)
   for i,level in enumerate([.003,.006,.009,.012,.016,.020]):lines.SetValue(i,level)
   lm=vtk.vtkPolyDataMapper();lm.SetInputConnection(lines.GetOutputPort());lm.ScalarVisibilityOff();lm.SetResolveCoincidentTopologyToPolygonOffset();lm.SetRelativeCoincidentTopologyLineOffsetParameters(0,-3);la=vtk.vtkActor();la.SetMapper(lm);la.GetProperty().SetColor(.92,.88,.97);la.GetProperty().SetOpacity(.20);la.GetProperty().SetLineWidth(.9);la.GetProperty().LightingOff();acts.append(la)
  cavpath=SCOOP/f'{tag}-cavity.vtp'
  if cavpath.exists():
   cav=actor(read_vtp(cavpath,reverse=True),(.23,.35,.42));cav.GetProperty().SetAmbient(.22);cav.GetProperty().SetDiffuse(.72);cav.GetProperty().SetSpecular(.25);cav.GetProperty().SetSpecularPower(45);acts.append(cav)
  # A fine physical rim makes the bowl lip and the cavity mouths legible.
  for poly,col,width in [(read_vtp(SCOOP/'shell.vtp'),(.78,.70,.82),1.2),(bowl,(.36,.29,.40),.65)]:
   edges=vtk.vtkFeatureEdges();edges.SetInputData(poly);edges.BoundaryEdgesOn();edges.FeatureEdgesOff();edges.NonManifoldEdgesOff();edges.ManifoldEdgesOff();edges.Update()
   e=actor(edges.GetOutput(),col,smooth=False);e.GetProperty().LightingOff();e.GetProperty().SetLineWidth(width);acts.append(e)
  for a in acts:self.ren.AddActor(a);a.VisibilityOff()
  self.groups[key]=acts;return acts
 def set_surface(self,key,v):
  bowl,surface,data,score=self.thermal[key];v=np.asarray(v)
  rgb=np.uint8(colormaps['viridis' if score else 'inferno'](np.clip(v/(.000625 if score else .025),0,1))[:,:3]*255)
  a=numpy_to_vtk(rgb,deep=True,array_type=vtk.VTK_UNSIGNED_CHAR);a.SetName('rgb');bowl.GetPointData().AddArray(a);bowl.Modified()
  scalar=numpy_to_vtk(v.astype('float32'),deep=True);scalar.SetName('temperature');bowl.GetPointData().SetScalars(scalar)
 def heat_time(self,tau):
  bowl,surface,data,score=self.thermal[self.current];times=data['times'];k=min(int(np.searchsorted(times,tau,side='right')),len(times)-1);j=max(k-1,0);a=0 if k==j else np.clip((tau-times[j])/(times[k]-times[j]),0,1)
  self.set_surface(self.current,(1-a)*data['transient_surface'][j]+a*data['transient_surface'][k])
  return float(np.interp(tau,times,data['mean']))
 def cooling_time(self,tau):
  bowl,surface,data,score=self.thermal[self.current];times=data['cooling_times'];k=min(int(np.searchsorted(times,tau,side='right')),len(times)-1);j=max(k-1,0);a=0 if k==j else np.clip((tau-times[j])/(times[k]-times[j]),0,1)
  self.set_surface(self.current,(1-a)*data['cooling_surface'][j]+a*data['cooling_surface'][k])
 def show(self,tag,field=None,color=PURPLE):
  key=(tag,field,tuple(color))
  if key!=self.current:
   for group in self.groups.values():
    for a in group:a.VisibilityOff()
   for a in self.load(tag,field,color):a.VisibilityOn()
   self.current=key
 def camera(self,angle=0,zoom=1,target=None):
  a=np.radians(angle);offset=np.array([1.05*np.cos(a)+3.5*np.sin(a),.66,3.5*np.cos(a)-1.05*np.sin(a)])
  target=np.array([-.01,-.015,0]) if target is None else np.array(target)
  self.cam.SetPosition(*(target+offset));self.cam.SetFocalPoint(*target);self.cam.SetViewUp(0,1,0);self.cam.SetParallelScale(.57/zoom);self.cam.SetClippingRange(.01,20)
 def clear_extras(self):
  for a in self.extras:self.ren.RemoveActor(a)
  self.extras=[]
 def add_poly(self,poly,color,opacity=1):
  a=actor(poly,color,opacity);self.ren.AddActor(a);self.extras.append(a);return a
 def sphere(self,c,r,color=PURPLE,opacity=1,wire=False):
  s=vtk.vtkSphereSource();s.SetCenter(*c);s.SetRadius(r);s.SetThetaResolution(36);s.SetPhiResolution(24);s.Update();a=self.add_poly(s.GetOutput(),color,opacity)
  if wire:a.GetProperty().SetRepresentationToWireframe();a.GetProperty().SetLineWidth(1)
  return a
 def line(self,points,color,width=.0018):
  if len(points)<2:return
  p=vtk.vtkPoints();p.SetData(numpy_to_vtk(np.asarray(points),deep=True));cells=vtk.vtkCellArray();cells.InsertNextCell(len(points))
  for i in range(len(points)):cells.InsertCellPoint(i)
  poly=vtk.vtkPolyData();poly.SetPoints(p);poly.SetLines(cells);tube=vtk.vtkTubeFilter();tube.SetInputData(poly);tube.SetRadius(width);tube.SetNumberOfSides(8);tube.Update();return self.add_poly(tube.GetOutput(),color)
 def sphere_guides(self,c,r,color=PURPLE):
  # Three true great circles replace dense triangle wireframe (which aliases).
  theta=np.linspace(0,2*np.pi,97);c=np.asarray(c)
  for axes in [(0,1),(0,2),(1,2)]:
   pts=np.tile(c,(len(theta),1));pts[:,axes[0]]+=r*np.cos(theta);pts[:,axes[1]]+=r*np.sin(theta)
   a=self.line(pts,color,max(r*.006,.00012));a.GetProperty().SetOpacity(.32)
 def ghost(self,opacity=.08):
  # Only the removed visualization window, not the entire surface duplicated.
  if not hasattr(self,'ghost_poly') and (SCOOP/'ghost.vtp').exists():self.ghost_poly=read_vtp(SCOOP/'ghost.vtp')
  if not hasattr(self,'ghost_poly'):
   p=read_poly(ASSET/'input-shell.ply');pts=vtk_to_numpy(p.GetPoints().GetData());ids=np.where((pts[:,2]>.09)&((pts[:,0]+.025)**2+(pts[:,1]+.18)**2<.355**2))[0];inside=np.zeros(len(pts),bool);inside[ids]=True
   con=vtk_to_numpy(p.GetPolys().GetData()).reshape(-1,4)[:,1:];keep=np.all(inside[con],axis=1);arr=np.c_[np.full(keep.sum(),3),con[keep]].astype('int64');ca=vtk.vtkCellArray();ca.SetCells(int(keep.sum()),numpy_to_vtk(arr.ravel(),deep=True,array_type=vtk.VTK_ID_TYPE));g=vtk.vtkPolyData();g.SetPoints(p.GetPoints());g.SetPolys(ca);self.ghost_poly=g
  return self.add_poly(self.ghost_poly,(.57,.47,.65),opacity)
 def render(self):
  self.win.Render();self.capture.Modified();self.capture.Update();a=vtk_to_numpy(self.capture.GetOutput().GetPointData().GetScalars()).reshape(self.h,self.w,3);
  rgb=a[::-1].copy();mask=rgb.min(2)<247
  if mask.any():
   ys,xs=np.where(mask);cx=(xs.min()+xs.max())/2;y=ys.max()-3;ww=(xs.max()-xs.min())*.36
   shadow=Image.new('L',(self.w,self.h),0);ImageDraw.Draw(shadow).ellipse((cx-ww,y-15,cx+ww,y+12),fill=28);shadow=shadow.filter(ImageFilter.GaussianBlur(13));alpha=np.asarray(shadow)/255
   rgb=np.where(mask[:,:,None],rgb,np.uint8(255*(1-alpha[:,:,None])))
  return Image.fromarray(rgb)
 def project(self,p):
  self.ren.SetWorldPoint(*p,1);self.ren.WorldToDisplay();q=self.ren.GetDisplayPoint();return q[0],self.h-q[1]

if __name__=='__main__':
 s=Scene();s.show('budget12','budget12');s.camera(-4);s.ghost(.06);s.render().save(Path(__file__).parent/'preview/test.png')
