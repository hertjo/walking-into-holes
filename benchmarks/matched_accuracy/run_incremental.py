from pathlib import Path
import json,numpy as np,pandas as pd
import run as r
r.a.BIN=Path('/tmp/wih-reuse-build/wih_incremental_query')
r.BUDGETS=[('mg',49),('mg',65),('mg',97),('mg',129)]
# Separate output tree: reuse input geometry, not previous solver timings.
original=r.OUT;r.OUT=original/'incremental';r.OUT.mkdir(exist_ok=True)
for name in r.GEOMS:
 p=np.loadtxt(original/name/'sites.txt');hole=np.loadtxt(original/name/'hole.txt').reshape(1,3)
 r.sweep(name,p,hole)
 for f in (r.OUT/name).glob('mg*-seed*/done.json'):
  for state in ['original','edited']:
   x=pd.read_csv(f.parent/f'{state}-prefix512.csv').score.values
   y=pd.read_csv(original/name/f.parent.name/f'{state}-prefix512.csv').score.values
   delta=float(np.linalg.norm(x-y)/np.linalg.norm(y))
   assert delta<2e-5,(name,f,delta)
   k=json.loads((f.parent/f'{state}-q8/kernel.json').read_text())
   if state=='edited':assert k['parity_rays']==0
 print('INCREMENTAL VERIFIED',name,flush=True)
