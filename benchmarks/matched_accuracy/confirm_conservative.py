"""Disclosed fixed-budget follow-up; never retune after this confirmation."""
import json
from pathlib import Path
import numpy as np,pandas as pd
import holdout as h
O=h.O
protocol=dict(reason='Both calibration-selected 64-path cells failed the fresh-seed 20% mean-error upper-bound rule. Test one fixed conservative 256-path budget in BOTH states; retain failed confirmations.',twos_budget=256,mg_budget=49,seeds=[510011+11000*i for i in range(20)],states=['original','edited'],target=.2,acceptance='Same reference gates, upper 97.5 percentile bootstrap mean error <=20%, lower 2.5 percentile runtime ratio >1; no further budget search')
p=Path(__file__).with_name('conservative_confirmation_protocol.json')
if p.exists():assert json.loads(p.read_text())==protocol
else:p.write_text(json.dumps(protocol,indent=2)+'\n')
# Reuse the exact holdout measurement routine with a temporary, fully specified
# selection input; restore calibration selection even if interrupted by errors.
original=h.pd.read_csv
selection=original(O/'selected.csv');selection=selection[(selection.geometry=='bunny_compliance')&(selection.target==.2)&(selection.queries==8)].copy()
selection['twos_rb_budget']=256;selection['best_mg_budget']=49;selection['twos_rb_speedup']=2.
h.SEEDS=protocol['seeds'];h.IDX=np.random.default_rng(17092028).integers(0,20,(10000,20))
def reader(p,*args,**kwargs):
 if Path(p)==O/'selected.csv':return selection
 return original(p,*args,**kwargs)
h.pd.read_csv=reader
# main writes its standard summary; preserve stage-one results and write stage two separately.
backup={n:(O/n).read_bytes() for n in ['holdout-summary.csv','holdout-status.json']}
try:
 h.main()
 for n in backup:(O/('conservative-'+n)).write_bytes((O/n).read_bytes())
finally:
 for n,b in backup.items():(O/n).write_bytes(b)
