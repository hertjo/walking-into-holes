"""Exact rational check of the concentric 3D Poisson cavity benchmark.

No Monte Carlo estimates, fitted constants, or PDE reference solver are used.
Numbers below omit the common 4*pi factor until forming dimensionless ratios.
"""
from fractions import Fraction as F
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[2]

def evaluate(a):
 c=a*(1+a)/6
 def primitive(s):return (F(1,6)+c)*s**3/3-s**5/30-c*s**2/2
 direct=F(1,45)-(primitive(F(1))-primitive(a))
 energy=a**5/45+c*c*(1/a-1)
 closed=a*(1+a)**2*(1-a)/36+a**5/45
 lower=a*(1-a*a)**2/36
 gap=a*a*(5+5*a-5*a*a-a**3)/180
 assert direct==energy==closed
 assert closed-lower==gap and gap>0
 assert F(1,6)*(1-a*a)-c*(1/a-1)==0
 return {'radius':float(a),'normalized_benefit':float(closed/a),
         'capacity_normalized_benefit':float(closed*(1-a)/a),
         'trace_bound_fraction_of_benefit':float(lower/closed),
         'leading_TD_relative_error':float(abs(closed/a-F(1,36))/F(1,36))}
rows=[evaluate(a) for a in [F(1,1000),F(1,100),F(1,20),F(1,10),F(1,4),F(1,2)]]
for i in range(1,1000):evaluate(F(i,1000))
out=Path(__file__).resolve().parent
with (out/'results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(out/'audit.json').write_text(json.dumps({'rational_radii_checked':999,'identity_residual':'exactly zero in rational arithmetic','checks':['Dirichlet cavity trace','direct objective integration = difference energy = closed form','explicit nonnegative certificate gap'],'scope':'Analytic model and bound only; not the mesh solver or floating-point sampler.','rows':rows},indent=2)+'\n')
print(json.dumps(rows,indent=2))
