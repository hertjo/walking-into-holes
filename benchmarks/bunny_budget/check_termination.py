#!/usr/bin/env python3
"""Fourfold thinner WoS shell on pre-specified full-budget endpoint designs.

Same saved geometries, evaluation points, and base seeds. This checks
sensitivity; it does not bound the residual bias or provide exact truth.
"""
import json
import numpy as np
from run_study import OUT,evaluate,save_json,SEEDS

def main():
    epsilon=.000125
    initial=evaluate(OUT/'termination-check/initial',np.empty((0,3)),0,epsilon)
    original_initial=np.array(json.loads((OUT/'initial.json').read_text())['block_objectives'])
    rows=[]
    for n in [2,12]:
        for method in ['twos','voxel']:
            folder=OUT/f'v6-n{n:02d}-s{SEEDS[0]}'/method
            result=json.loads((folder/'result.json').read_text())
            centers=np.loadtxt(folder/'selected-centers.csv',delimiter=',').reshape(-1,3)
            values=evaluate(OUT/f'termination-check/n{n:02d}-{method}',centers,result['radius'],epsilon)
            coarse=100*(1-np.array(result['block_objectives'])/original_initial)
            fine=100*(1-np.array(values)/np.array(initial))
            rows.append(dict(count=n,method=method,seed=SEEDS[0],fine_block_objectives=values,
                coarse_cooling_percent=float(coarse.mean()),fine_cooling_percent=float(fine.mean()),
                change_pp=float(np.mean(fine-coarse))))
    allocation=[]
    for units,n in [(3,12),(6,6)]:
        folder=OUT/f'v{units}-n{n:02d}-s{SEEDS[0]}'/'twos'
        result=json.loads((folder/'result.json').read_text())
        centers=np.loadtxt(folder/'selected-centers.csv',delimiter=',').reshape(-1,3)
        values=evaluate(OUT/f'termination-check/allocation-v{units}-n{n:02d}',centers,result['radius'],epsilon)
        fine=100*(1-np.array(values)/np.array(initial))
        allocation.append(dict(volume_units=units,count=n,seed=SEEDS[0],fine_block_objectives=values,
            fine_cooling_percent=float(fine.mean()),coarse_cooling_percent=result['cooling_percent'],
            change_pp=float(fine.mean()-result['cooling_percent'])))
    save_json(OUT/'termination-check.json',dict(epsilon=epsilon,initial_blocks=initial,
      results=rows,maximum_absolute_change_pp=max(abs(r['change_pp']) for r in rows),
      scope='First design seed; two pre-specified full-budget endpoint counts, both TD selectors; sensitivity only, not a bias certificate.',
      allocation_check=allocation,allocation_scope='Additional first-seed audit of the half-volume comparison selected after inspecting the study, not a pre-specified check.'))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
