"""Summarize independent numerical checks without calling them ground truth."""
import json
import numpy as np
import pandas as pd
from run_study import OUT,SPECS,SEEDS

def reference_audit():
 records=[]
 for study in ['pipeline','channels']:
  for name in SPECS:
   ev=json.loads((OUT/study/name/'evaluation'/str(SEEDS[0])/'evaluation.json').read_text())
   base=np.asarray(ev['grid_check']['unperforated'])
   fine_path=OUT/'fine-reference'/study/name/'results.json';fine=json.loads(fine_path.read_text()) if fine_path.exists() else None
   for method in ['twos','mg','mg49','nearest','hot','random']:
    grid=np.asarray(ev['grid_check'][method]);reduction=100*(1-grid/base)
    mc=100*(1-np.mean(ev[method+'-initial'])/np.mean(ev['unperforated']))
    record=dict(study=study,geometry=name,method=method,MC_reduction_pct=mc,grid97_reduction_pct=reduction[0],grid129_reduction_pct=reduction[1],grid_change_pp=reduction[1]-reduction[0],MC_minus_grid129_pp=mc-reduction[1])
    if fine and fine['triggered']:
     rr=100*(1-np.asarray(fine['values'][method])/np.asarray(fine['values']['unperforated']))
     record.update(grid193_reduction_pct=rr[0],grid257_reduction_pct=rr[1],fine_grid_change_pp=rr[1]-rr[0],MC_minus_grid257_pp=mc-rr[1])
    records.append(record)
 pd.DataFrame(records).to_csv(OUT/'reference-audit.csv',index=False)
 rows=[]
 for name in SPECS:
  data=json.loads((OUT/'estimators'/name/'results.json').read_text())
  raw=pd.DataFrame(data['rows']);ref=raw[raw.method=='reference'].groupby('site').score.mean()
  checks=pd.DataFrame(data['finite_grid_checks'])
  for r,g in checks.groupby('radius'):
   a=g.pivot(index='site',columns='grid',values='score')
   norm=np.linalg.norm(ref)
   for n in [97,129]:
    rows.append(dict(geometry=name,radius=r,grid=n,relative_error_to_TD=np.linalg.norm(a[n]-ref)/norm,relative_grid_change=np.linalg.norm(a[129]-a[97])/norm,scope='Finite radius, discretization and reference error are combined; no certified separation'))
 pd.DataFrame(rows).to_csv(OUT/'finite-grid-audit.csv',index=False)
 print('Independent-reference audit saved.')
if __name__=='__main__':reference_audit()
