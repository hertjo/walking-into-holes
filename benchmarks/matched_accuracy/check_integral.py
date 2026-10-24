"""Independent direct Green-volume quadrature versus the compiled closed form."""
from pathlib import Path
import subprocess,json
import numpy as np
from scipy.integrate import quad
H=Path(__file__).resolve().parent
rng=np.random.default_rng(17092026)
cases=[]
for i in range(300):
 R=10**rng.uniform(-2,0);A=10**rng.uniform(-1,2)
 cases.append((rng.uniform(0,3)*R,10**rng.uniform(-3,.7)*R,R,A))
for d,r in [(0,.001),(0,.3),(0,1),(0,2),(1,1e-4),(2,1),(2,1+1e-8),(1,2),(1e-14,.5),(.5,.5)]:cases.append((d,r,1.,1.))
def integral(d,r,R,A):
 def angular(t):
  if d*t==0:return A*max(0,1-(d*d+t*t)/(R*R))**2
  a=1-(d*d+t*t)/(R*R);b=2*d*t/(R*R)
  if a+b<=0:return 0.
  hi=min(1,a/b)
  def P(c):return a*a*c-a*b*c*c+b*b*c*c*c/3
  return A*(P(hi)-P(-1))/2
 cuts=sorted(set([0.,r]+[t for t in [abs(d-R),d+R] if 0<t<r]))
 return sum(quad(lambda t:(t-t*t/r)*angular(t),a,b,epsabs=1e-13*A*R*R,epsrel=1e-10)[0] for a,b in zip(cuts[:-1],cuts[1:]))
x=''.join(' '.join(map(str,c))+'\n' for c in cases)
y=subprocess.run(['/tmp/wih-reuse-build/wih_analytic_integral_audit'],input=x,text=True,capture_output=True,check=True)
values=np.array([float(v) for v in y.stdout.split()]);ref=np.array([integral(*c) for c in cases]);scale=np.array([A*R*R for d,r,R,A in cases])
error=np.abs(values-ref)/scale
assert np.all(error<1e-9),(error.max(),cases[error.argmax()])
report=dict(cases=len(cases),max_absolute_error_normalized_by_AR2=float(error.max()),minimum_integral=float(values.min()),scope='Compiled ball integral versus independently integrated Green volume source; numerical diagnostic, not rounding certification')
(H/'output/integral-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
