#!/usr/bin/env python3
"""Independent saved-center and solver-summary audit (no benchmark reruns)."""
import json
import numpy as np
import igl
from run_study import OUT,MESH

def main():
    rows=json.loads((OUT/'results.json').read_text())['results']
    vertices,faces=igl.read_triangle_mesh(str(MESH))
    tree=igl.AABB();tree.init(vertices,faces)
    outer=[];pair_gaps=[]
    for r in rows:
        folder=OUT/f'v{r["volume_units"]}-n{r["count"]:02d}-s{r["seed"]}'/r['method']
        centers=np.loadtxt(folder/'selected-centers.csv',delimiter=',').reshape(-1,3)
        assert len(centers)==r['count']
        signed=igl.signed_distance(centers,vertices,faces,igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]
        gap=np.sqrt(tree.squared_distance(vertices,faces,centers)[0])-r['radius']
        assert np.all(signed<0) and np.all(gap>=.012-1e-10)
        pair=np.linalg.norm(centers[:,None]-centers[None,:],axis=2)+np.eye(len(centers))*1e3
        pair_gap=float(pair.min()-2*r['radius'])
        assert pair_gap>=.014-1e-10
        outer.append(float(gap.min()));pair_gaps.append(pair_gap)
    solvers=[json.loads(p.read_text()) for p in OUT.rglob('*.json')
        if p.name in ['field.json','scout.json','rescore.json','voxel.json']]
    caps=sum(s.get('max_step_limit_hits',0) for s in solvers)
    residual=max(s.get('cg_reported_error',0) for s in solvers)
    assert caps==0 and residual<=1.1e-9
    result=dict(independently_checked_designs=len(rows),minimum_outer_wall=min(outer),
        minimum_pair_gap=min(pair_gaps),solver_calls_checked=len(solvers),
        path_cap_hits=caps,largest_cg_reported_residual=residual)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()
