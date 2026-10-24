"""Penrose geometry for shaded 3-D capsule examples; genuine random WoS paths."""
from pathlib import Path
import importlib.util,json
import numpy as np
from scipy.spatial import ConvexHull
R=Path(__file__).resolve().parents[1];A=R/'video/assets'
spec=importlib.util.spec_from_file_location('diagram',R/'scripts/redesign-method-diagrams.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
L=.65;RAD=.70;SCALE=225.;CX=400.;CY=206.
az=np.radians(22);el=np.radians(16)
RIGHT=np.array([np.cos(az),0,-np.sin(az)]);UP=np.array([-np.sin(az)*np.sin(el),np.cos(el),-np.cos(az)*np.sin(el)])
def project(p,scale=SCALE,center=(CX,CY)):
 p=np.asarray(p);return np.stack([center[0]+scale*(p@RIGHT),center[1]-scale*(p@UP)],axis=-1)
def clearance(p):
 p=np.asarray(p);axis=np.array([np.clip(p[0],-L,L),0.,0.]);return RAD-float(np.linalg.norm(p-axis))
def outline(scale=SCALE,center=(CX,CY)):
 t=np.linspace(0,2*np.pi,240,endpoint=False);centers=project(np.array([[-L,0,0],[L,0,0]]),scale,center);points=np.concatenate([q+scale*RAD*np.c_[np.cos(t),np.sin(t)] for q in centers]);return points[ConvexHull(points).vertices]
def capsule(f,scale=SCALE,center=(CX,CY)):
 p=outline(scale,center);shadow=f.path(p+[3,9],width=0,fill='grey',closed=True);body=f.path(p,width=1.4,fill='ground',closed=True)
 for x in [-L,L]:
  for lo,hi,dash in [(-np.pi/2,np.pi/2,None),(np.pi/2,3*np.pi/2,'3,5')]:
   t=np.linspace(lo,hi,90);p3=np.c_[np.full(len(t),x),RAD*np.cos(t),RAD*np.sin(t)];f.path(project(p3,scale,center),'line',.9,dash=dash)
 return {shadow:'shadow',body:'domain'}
paint={};f=m.Figure(800,450);paint['video-capsule']=capsule(f);f.eq(705,342,r'\Omega',28,'muted');f.eq(400,410,r'x_{k+1}=x_k+r_k\omega_k',34);f.write('video-capsule')
f=m.Figure(240,240);shape=f.circle(120,120,105,'tealpale','teal',1.1);paint['video-sphere']={shape:'sphere'}
# Two meridians give depth without altering the projected ball boundary.
for angle in [0,np.pi/2]:
 t=np.linspace(0,2*np.pi,130);x=105*np.cos(t);y=32*np.sin(t);p=np.c_[120+x*np.cos(angle)-y*np.sin(angle),120+x*np.sin(angle)+y*np.cos(angle)];f.path(p,'teal',.6,dash='2,5')
f.write('video-sphere')
f=m.Figure(800,280);roles={};roles.update(capsule(f,90,(190,112)));roles.update(capsule(f,90,(610,112)));f.line(359,112,441,112,'muted',1.5,arrow=True)
hole=f.circle(618,115,23,'pale','purple',1.3);roles[hole]='cavity';f.line(618,115,641,115,'purple',1);f.eq(658,94,r'\varepsilon',22,'purple')
f.eq(190,237,r'\Omega',31);f.eq(610,237,r'\Omega\setminus B_\varepsilon(z)',31)
f.write('video-topology');paint['video-topology']=roles
f=m.Figure(800,210);f.eq(400,53,r'-\Delta u=f',45,'orange');f.eq(400,147,r'u|_{\partial\Omega}=0',37,'teal');f.write('video-model')

traces=[]
for seed in [2026091901,2026091902,2026091903]:
 rng=np.random.default_rng(seed);x=np.array([.0,-.10,.03]);steps=[]
 for k in range(1024):
  d=clearance(x)
  if d<.0005:break
  r=.9999*d;v=rng.normal(size=3);v/=np.linalg.norm(v);y=x+r*v
  assert clearance(y)>0 and abs(np.linalg.norm(y-x)-r)<1e-12
  steps.append({'x':x.tolist(),'next':y.tolist(),'radius':r,'distance':d,'screen_x':project(x).tolist(),'screen_next':project(y).tolist(),'screen_radius':r*SCALE});x=y
 else:raise RuntimeError('capsule path did not terminate')
 traces.append({'seed':seed,'steps':steps,'termination_distance':d})
(A/'capsule-walks.json').write_text(json.dumps({'domain':'3D capsule: distance to segment [-.65,.65] on x-axis < .70','half_length':L,'radius':RAD,'start':[0.,-.10,.03],'terminal_epsilon':.0005,'step_rule':'.9999 * exact distance to capsule boundary','sampling':'Independent normalized 3D Gaussian directions; no path selection or hand-chosen angles','projection':{'right':RIGHT.tolist(),'up':UP.tolist(),'scale':SCALE,'center':[CX,CY]},'traces':traces},indent=2)+'\n')
(A/'penrose-depth-paint.json').write_text(json.dumps(paint,indent=2)+'\n')
print('Capsule steps:',[len(t['steps']) for t in traces])
