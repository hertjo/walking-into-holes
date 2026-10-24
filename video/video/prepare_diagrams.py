from pathlib import Path
import importlib.util,json
import numpy as np
R=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('diagram',R/'scripts/redesign-method-diagrams.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
f=m.Figure(800,650)
f.text(400,27,'Cross-fitting',27,bold=True)
f.text(90,85,'pair',25,color='muted')
f.text(292,85,'primal',25,color='orange')
f.text(566,85,'adjoint',25,color='teal')
for i,y in enumerate([144,233,322],1):
 if i==2:f.rect(386,y,704,69,'ground',None,corner=4)
 f.eq(90,y,f'X_{i}',31,'purple' if i==2 else 'muted')
 f.eq(292,y,f'U_{i}',35,'orange' if i==2 else 'muted')
 if i!=2:
  f.rect(566,y,150,61,'tealpale',None,corner=4)
  f.eq(566,y,f'P_{i}',35,'teal')
 else:
  f.eq(566,y,'P_2',35,'muted')
  f.line(544,y,588,y,'muted',1.8)
  f.text(684,y,'omit',24,color='muted')
f.rect(292,233,150,61,None,'orange',1.7,corner=4)
f.line(41,364,755,364,'line',1.1)
f.eq(400,426,r'\bar P_{-2}=\frac{P_1+P_3}{2}',34,'teal')
f.eq(400,509,r'C_2=-U_2\bar P_{-2}',37)
f.eq(400,618,r'\widehat D_T=\frac{1}{N}\sum\nolimits_{i=1}^N C_i',35,'purple')
f.write('video-estimator')
f=m.Figure(800,650)
f.eq(207,332,r'D_T(z)=',51,'purple');f.eq(367,332,'-',51)
f.eq(461,332,r'u(z)',55,'orange');f.eq(614,332,r'p(z)',55,'teal')
f.line(414,377,504,377,'orange',1.5);f.line(568,377,660,377,'teal',1.5)
f.text(461,411,'temperature',25,color='orange');f.text(614,411,'influence',25,color='teal')
f.line(55,468,747,468,'line',1)
f.eq(400,536,r'\Delta J=4\pi\varepsilon D_T(z)+o(\varepsilon)',35)
f.eq(400,612,r'\varepsilon\to0,\qquad p=u',29,'muted')
f.write('video-sensitivity')

f=m.Figure(800,210)
f.eq(400,55,r'r_k<d(x_k,\partial\Omega)',35,'teal')
f.eq(400,151,r'U\leftarrow U+\frac{r_k^2}{6}\,f(x_k)',38,'orange')
f.write('video-increment')

f=m.Figure(1040,110)
f.eq(208,54,r'\frac{\partial u}{\partial\tau}',45,'orange')
f.eq(426,54,r'=\Delta u',45,'purple')
f.eq(668,54,r'+\,f',45,'orange')
f.text(912,54,'u = 0 on walls',26,color='teal')
f.write('video-heat-equation')
