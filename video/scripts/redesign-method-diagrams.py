"""Generate deterministic Penrose sources for the two method diagrams.
Geometry is sampled explicitly; paths and walk radii are reproducible.
All content is schematic, not a numerical experimental result.
"""
from pathlib import Path
import json, math
import numpy as np
R=Path(__file__).resolve().parents[1]
WALK_AUDIT=[]
C={'ink':'#302c35','muted':'#756e7a','line':'#d8d2dc','ground':'#faf8fb','purple':'#9365a8','pale':'#eee5f3','teal':'#6997aa','tealpale':'#eaf1f4','orange':'#bd824b','red':'#ae7778','grey':'#e8e5e9','white':'#ffffff'}
class Figure:
 def __init__(self,w,h):self.w=w;self.h=h;self.s=[];self.n=0;self.audit=[]
 def xy(self,x,y):return f'({x-self.w/2:.4f},{self.h/2-y:.4f})'
 def shape(self,kind,props):
  self.n+=1;name=f's{self.n}';self.s.append(f'  shape {name} = {kind} {{\n'+''.join(f'    {k}: {v}\n' for k,v in props.items())+'  }');return name
 def text(self,x,y,t,size=17,color='ink',bold=False,anchor='middle'):
  self.audit.append(dict(text=t,x=x,y=y,size=size,anchor=anchor))
  return self.shape('Text',dict(center=self.xy(x,y),string=json.dumps(t),fontFamily='"Nimbus Roman"',fontSize=f'"{size}px"',fontWeight='"700"' if bold else '"400"',fillColor='Colors.'+color,textAnchor=json.dumps(anchor)))
 def eq(self,x,y,t,size=18,color='ink'):
  return self.shape('Equation',dict(center=self.xy(x,y),string='"'+t+'"',fontSize=f'"{size}px"',fillColor='Colors.'+color))
 def line(self,x1,y1,x2,y2,color='line',width=1,dash=None,arrow=False):
  p=dict(start=self.xy(x1,y1),end=self.xy(x2,y2),strokeColor='Colors.'+color,strokeWidth=str(width))
  if dash:p['strokeDasharray']=json.dumps(dash)
  if arrow:p.update(endArrowhead='"line"',arrowheadSize='0.65')
  return self.shape('Line',p)
 def rect(self,x,y,w,h,fill='white',stroke='line',width=1,corner=0):
  return self.shape('Rectangle',dict(center=self.xy(x,y),width=str(w),height=str(h),fillColor='none()' if fill is None else 'Colors.'+fill,strokeColor='none()' if stroke is None else 'Colors.'+stroke,strokeWidth=str(width),cornerRadius=str(corner)))
 def circle(self,x,y,r,fill=None,stroke='purple',width=1,dash=None):
  p=dict(center=self.xy(x,y),r=str(r),fillColor='none()' if fill is None else 'Colors.'+fill,strokeColor='none()' if stroke is None else 'Colors.'+stroke,strokeWidth=str(width))
  if dash:p['strokeDasharray']=json.dumps(dash)
  return self.shape('Circle',p)
 def path(self,points,color='ink',width=1,fill=None,closed=False,dash=None):
  q=[[round(float(x)-self.w/2,4),round(self.h/2-float(y),4)] for x,y in points]
  p=dict(d='pathFromPoints('+json.dumps('closed' if closed else 'open')+', '+json.dumps(q)+')',fillColor='none()' if fill is None else 'Colors.'+fill,strokeColor='Colors.'+color,strokeWidth=str(width),strokeLinecap='"round"',strokeLinejoin='"round"')
  if dash:p['strokeDasharray']=json.dumps(dash)
  return self.shape('Path',p)
 def write(self,name):
  s=f'canvas {{ width = {self.w}\n height = {self.h} }}\nColors {{\n'+''.join(f' color {k} = {v}\n' for k,v in C.items())+'}\nGlobal {\n'+'\n'.join(self.s)+'\n'+''.join(f'  layer s{i} below s{i+1}\n' for i in range(1,self.n))+'}\n'
  (R/'penrose'/f'{name}.style').write_text(s)
  (R/'penrose'/f'{name}.domain').write_text('type Diagram\n')
  (R/'penrose'/f'{name}.substance').write_text('Diagram method\n')
  (R/'penrose'/f'{name}-layout.json').write_text(json.dumps(dict(canvas=[self.w,self.h],labels=self.audit),indent=2)+'\n')

def boundary(cx,cy,sx,sy):
 t=np.linspace(0,2*np.pi,240,endpoint=False)
 # Smooth nonconvex solid; polygonal outline is also used for containment.
 r=1+.18*np.cos(2*t)+.09*np.sin(3*t)
 return np.c_[cx+sx*r*np.cos(t),cy+sy*r*np.sin(t)]
def distance(p,poly):
 a=poly;b=np.roll(a,-1,axis=0);v=b-a;t=np.clip(((p-a)*v).sum(1)/(v*v).sum(1),0,1)
 return float(np.linalg.norm(p-(a+t[:,None]*v),axis=1).min())
def walk(poly,start,seed,cap=16,n=12,directions=None):
 rng=np.random.default_rng(seed);p=np.array(start,dtype=float);pts=[p.copy()];balls=[]
 for k in range(n):
  d=distance(p,poly);r=min(.97*d,cap)
  if r<.45:break
  balls.append((p.copy(),r));t=rng.uniform(0,2*np.pi) if directions is None else directions[k % len(directions)]
  WALK_AUDIT.append(dict(center=p.tolist(),boundary_distance=d,radius=r,contained=bool(r<d),schematic_directions=directions is not None))
  p=p+r*np.array([np.cos(t),np.sin(t)]);pts.append(p.copy())
 return pts,balls

def crossfit():
 # The exact all-cross-pairs estimator, organized by its leave-one-out identity.
 f=Figure(1120,384)
 for x,title,subtitle in [(183,'(a) Reuse geometric work','Independent randomizations'),(550,'(b) Hold out the matching adjoint','Example: use the second primal'),(929,'(c) Multiply, then average','Two independent factors per term')]:
  f.text(x,20,title,21,bold=True)
  f.text(x,51,subtitle,17,color='muted')
 for x in [371,748]:f.line(x,75,x,305,'line',.85)
 # Three complete schematic randomizations from the same point. The purple
 # realization corresponds to the retained primal in the worked example.
 poly=boundary(182,178,129,75);f.path(poly,width=1.65,fill='ground',closed=True)
 start=np.array([180.,187.]);endpoints=[]
 specs=[(1,'muted',[-2.55,-1.85,-2.75,-2.1,-1.45,-2.15],1.35),(3,'muted',[.35,.95,.15,.8,.4,.9],1.35),(2,'purple',[-.8,-1.8,-.6,-1.2,-.5,-1.35],2.2)]
 for idx,color,angles,width in specs:
  pts,balls=walk(poly,start,20+idx,cap=24,n=32,directions=angles)
  if idx==2:
   for p,r in balls[:6]:f.circle(*p,r,stroke='line',width=.8)
  f.path(pts,color,width)
  for p in pts[1:]:f.circle(*p,1.7,color,None)
  endpoints.append((idx,pts[-1]))
 f.circle(*start,4,'ink',None);f.eq(167,209,'z',22)
 # Labels live outside the boundary; all coordinates are pinned and audited.
 for idx,p in endpoints:
  offsets={1:(-11,-15),2:(13,-14),3:(15,10)};dx,dy=offsets[idx]
  f.eq(p[0]+dx,p[1]+dy,f'X_{idx}',21,'purple' if idx==2 else 'muted')
 f.eq(184,279,r'X_i=(U_i,P_i)',23)
 f.text(184,309,'Primal and adjoint may share the walk',16,color='muted')
 # A worked example: retain U2, omit only its own potentially correlated P2.
 f.text(417,88,'pair',16,color='muted')
 f.text(519,88,'primal',16,color='orange')
 f.text(653,88,'adjoint',16,color='teal')
 for i,y in enumerate([124,184,244],1):
  if i==2:f.rect(552,y,321,49,'ground',None)
  f.eq(417,y,f'X_{i}',20,'purple' if i==2 else 'muted')
  f.eq(518,y,f'U_{i}-h_z',23,'orange' if i==2 else 'muted')
  if i!=2:
   f.rect(653,y,100,41,'tealpale',None,corner=3);f.eq(653,y,f'P_{i}',24,'teal')
  else:
   f.eq(653,y,'P_2',24,'muted')
   f.line(638,y,669,y,'muted',1.25)
   f.text(653,y+25,'held out',14,color='muted')
 f.rect(518,184,113,42,None,'orange',1.1,corner=3)
 # The bracket collects the kept blue entries; the matching entry is omitted.
 f.line(714,123,726,123,'teal',1.1);f.line(726,123,726,245,'teal',1.1);f.line(714,245,726,245,'teal',1.1)
 f.eq(549,293,r'\bar P_{-2}=\frac{P_1+P_3}{2}',23,'teal')
 # Explicit one-term construction, followed by the exact N-pair identity.
 f.eq(936,128,r'C_2=-(U_2-h_z)\bar P_{-2}',25)
 f.text(934,172,'Repeat this holdout for every pair',17,color='muted')
 f.eq(934,224,r'\widehat D_T=\frac{1}{N}\sum\nolimits_{i=1}^N C_i',25,'purple')
 f.eq(934,288,r'\bar P_{-i}=\frac{S_P-P_i}{N-1}',23,'teal')
 f.line(16,329,1104,329,'line',.85)
 f.text(23,351,'ONE ADJOINT SUM',14,bold=True,color='muted',anchor='start')
 f.eq(233,351,r'\textstyle S_P=\sum_j P_j',21,'teal')
 f.text(463,351,'O(N) total work',18,bold=True)
 f.text(862,351,'N independent pairs; products share samples',17,color='muted')
 f.write('crossfit')

def pipeline():
 f=Figure(1120,375)
 centers=[137,417,697,977]
 titles=['1  Score the current domain','2  Freeze a finite proposal','3  Check with fresh samples','4  Insert, then refine']
 for x,t in zip(centers,titles):f.text(x,20,t,20,bold=True)
 for x in [277,557,837]:f.line(x,49,x,320,'line',.8)
 for x in centers:f.text(x,46,['No trial cavity','Fix center and radius','Account for all error terms','Only after the test passes'][centers.index(x)],16,color='muted')
 z=(24,8);polys=[]
 for x in centers:
  poly=boundary(x,140,92,66);polys.append(poly)
 # 1: same finite candidate locations throughout the geometric progression.
 f.path(polys[0],width=1.6,fill='ground',closed=True)
 sites=[(-65,-8),(-42,-31),(-37,20),(-13,-4),(24,8),(45,-26),(61,18),(10,40)]
 scores=[.2,.43,.62,.39,1,.46,.71,.25]
 for (dx,dy),s in zip(sites,scores):
  f.circle(centers[0]+dx,140+dy,3+3*s,'purple' if s>.65 else 'line',None)
 f.circle(centers[0]+z[0],140+z[1],10,None,'purple',1.1)
 f.eq(centers[0],239,r'\widehat D_T(z)= -\operatorname{avg}_{i\ne j}(U_i-h_z)P_j',16)
 f.text(centers[0],267,'Rank feasible, separated candidates',16,color='muted')
 # 2: outline is a proposal, not a hole; interior fill stays intact.
 f.path(polys[1],width=1.6,fill='ground',closed=True)
 cx,cy=centers[1]+z[0],140+z[1];rad=18
 f.circle(cx,cy,rad,None,'purple',1.5,'4,3');f.circle(cx,cy,2.7,'purple',None)
 f.line(cx,cy,cx+rad,cy,'purple',1);f.eq(cx+rad/2,cy-10,r'\varepsilon',17,'purple')
 f.eq(cx-12,cy+13,'z',17,'purple')
 
 f.text(centers[1],239,'Choose the proposal before validation',17)
 f.text(centers[1],267,'The dashed circle is not yet a hole',16,color='muted')
 # 3: fresh walks on the same unperforated domain.
 f.path(polys[2],width=1.6,fill='ground',closed=True)
 sx,sy=centers[2]+z[0],140+z[1]
 for seed,color,angles in [(11,'teal',[.2,-.8,.4,-.3,.6,.1]),(27,'orange',[-2.6,-1.5,-2.8,-2.0,-1.3,-2.4])]:
  pts,balls=walk(polys[2],[sx,sy],seed,cap=17,n=32,directions=angles)
  for p,r in balls[:4]:f.circle(*p,r,stroke='line',width=.7)
  f.path(pts,color,1.35)
  for p in pts[1:]:f.circle(*p,1.35,color,None)
 f.circle(sx,sy,3,'purple',None);f.eq(sx-11,sy+13,'z',17,'purple')
 # Actual guard, with deliberately schematic interval values.
 f.line(621,245,784,245,'muted',1)
 f.line(757,228,757,260,'ink',1);f.eq(757,273,'0',16)
 f.line(639,245,728,245,'purple',3)
 f.circle(667,245,3.3,'purple',None)
 f.line(639,239,639,251,'purple',1.2);f.line(728,237,728,253,'purple',1.2)
 f.text(674,219,'covered upper bound',15,color='purple')
 f.text(697,291,'Pass only when the upper bound is < 0',16,color='muted')
 # 4: a genuine white hole; refinement is optional, displayed after acceptance.
 f.path(polys[3],width=1.6,fill='ground',closed=True)
 hx,hy=centers[3]+z[0],140+z[1]
 f.circle(hx,hy,rad,'white','purple',1.6)
 f.circle(hx,hy,2,None,None)
 # Nearby translated boundary is the optional shape step, not uniform expansion.
 f.circle(hx+6,hy-5,rad,None,'teal',1,'3,3')
 f.line(hx+rad+2,hy-3,hx+rad+14,hy-13,'teal',1.3,arrow=True)
 f.eq(centers[3],239,r'\Omega\leftarrow\Omega\setminus K_{\varepsilon,z}',20)
 f.text(centers[3],267,'Optional Differential WoS shape step',16,color='muted')
 f.text(697,316,'Fail: resample under a valid schedule, or skip',15,color='muted')
 f.line(16,330,1104,330,'line',.9)
 f.text(25,351,'General-objective guard',17,bold=True,anchor='start')
 f.eq(624,352,r'\widehat D_T+r_N(\delta)+b_{\rm term}+b_{\rm cache}+C_z\rho_z(\varepsilon)^\alpha<0',21)
 f.write('pipeline')

if __name__=='__main__':
 crossfit();pipeline()
 assert all(x['contained'] for x in WALK_AUDIT)
 (R/'penrose/method-geometry-audit.json').write_text(json.dumps(dict(walk_balls=WALK_AUDIT,all_strictly_contained=True,note='Fixed schematic directions chosen for legibility; the numerical algorithm samples uniform exit directions. All displayed radii are submaximal and strictly contained.'),indent=2)+'\n')
