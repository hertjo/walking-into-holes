from pathlib import Path
import numpy as np
import run as r
from adaptive import sweep_mc
r.a.BIN=Path('/tmp/wih-reuse-build/wih_analytic_query')
r.BUDGETS=[('twos',16),('twos',64),('twos',256),('twos',1024),('twos',4096)]
original=r.OUT;r.OUT=original/'analytic';r.OUT.mkdir(exist_ok=True)
for name in r.GEOMS:
 p=np.loadtxt(original/name/'sites.txt');hole=np.loadtxt(original/name/'hole.txt').reshape(1,3)
 sweep_mc(r,name,p,hole,original/name/'reference')
