from pathlib import Path
import numpy as np,json
import run as r
from adaptive import sweep_mc
r.a.BIN=Path('/tmp/wih-reuse-build/wih_compliance_query')
r.BUDGETS=[('twos',16),('twos',64),('twos',256),('twos',1024),('twos',4096),('mg',49),('mg',65),('mg',97),('mg',129)]
original=r.OUT;r.OUT=original/'compliance';r.OUT.mkdir(exist_ok=True)
base_call=r.call
def compliance_call(*args,**kwargs):
 if kwargs.get('mode')=='mg':kwargs['mg_cycles']=800
 return base_call(*args,**kwargs)
r.call=compliance_call
p=np.loadtxt(original/'bunny/sites.txt');hole=np.loadtxt(original/'bunny/hole.txt').reshape(1,3)
r.reference('bunny',p,hole)
r.BUDGETS=[('mg',49),('mg',65),('mg',97),('mg',129)]
r.sweep('bunny',p,hole)
sweep_mc(r,'bunny',p,hole,r.OUT/'bunny/reference')
for f in (r.OUT/'bunny').glob('mg*-seed*/done.json'):
 k=json.loads((f.parent/'edited-q8/kernel.json').read_text());assert k['parity_rays']==0
print('COMPLIANCE COMPLETE',flush=True)
