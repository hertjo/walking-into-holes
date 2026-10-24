"""Refine cases where independent references disagree materially.

The rule was added after observing housing-channel reference disagreement
and is applied uniformly to all cases: >1 pp grid change or >2 pp
MC-versus-grid discrepancy. Every method in a triggered case is checked.
This does not turn either numerical reference into exact ground truth.
"""
import json,numpy as np
from run_study import OUT,SPECS,SEEDS,Solver,load,objective,channel_balls,dump
METHODS=['twos','mg','mg49','nearest','hot','random']

def main():
 for study in ['pipeline','channels']:
  for name in SPECS:
   folder=OUT/'fine-reference'/study/name
   if (folder/'results.json').exists():continue
   path=OUT/study/name/'evaluation'/str(SEEDS[0])/'evaluation.json';ev=json.loads(path.read_text())
   gb=np.array(ev['grid_check']['unperforated']);mb=np.mean(ev['unperforated']);changes=[];disagreements=[]
   for method in METHODS:
    g=100*(1-np.array(ev['grid_check'][method])/gb);mc=100*(1-np.mean(ev[method+'-initial'])/mb)
    changes.append(abs(g[1]-g[0]));disagreements.append(abs(mc-g[1]))
   trigger=max(changes)>1 or max(disagreements)>2
   result={'triggered':bool(trigger),'rule':'max over all methods: grid97-to129 change >1 pp OR MC-to-grid129 difference >2 pp','original_max_grid_change_pp':max(changes),'original_max_reference_difference_pp':max(disagreements),'states':'first seed, initial sphere designs or complete channels','values':{}}
   if trigger:
    s,m=load(name);sol=Solver(name,folder/'raw')
    result['values']['unperforated']=[objective(sol,f'base-{n}',s,[],.065,99123,mode='mg',n=n,nprobe=512) for n in [193,257]]
    for method in METHODS:
     d=json.loads((OUT/study/name/f'{method}-{SEEDS[0]}'/'design.json').read_text());holes=channel_balls(d['centers']) if study=='channels' else d['centers']
     result['values'][method]=[objective(sol,f'{method}-{n}',s,holes,d['radius'],99123,mode='mg',n=n,nprobe=512) for n in [193,257]]
    sol.close()
   dump(folder/'results.json',result);print('FINE REFERENCE',study,name,trigger,flush=True)
if __name__=='__main__':main()
