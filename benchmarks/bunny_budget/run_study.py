#!/usr/bin/env python3
"""Equal-volume Bunny cavity allocation. All results are measured, cached, auditable.

Independent design seeds; common candidates within each seed/budget/count;
four independently scrambled evaluation blocks, shared by all designs.
Evaluation streams are separate from selection. This is not a timing study.
"""
from pathlib import Path
import argparse, concurrent.futures, dataclasses, hashlib, json, math, os
import subprocess, sys, time
import igl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'
sys.path.insert(0, str(ROOT/'benchmarks/compiled_bunny'))
from run_compiled_benchmark import build_cpp, load_reference_module
ref = load_reference_module()
MESH = ROOT/'benchmarks/stanford_bunny/output/stanford-bunny-watertight.ply'
EXE = Path('/tmp/walking-into-holes-embree-build/wih_twos')
COUNTS = [2, 4, 6, 12]
BUDGETS = [3, 6]  # Volume of this many radius-0.05 balls.
SEEDS = [2026091501, 2026091502, 2026091503]
CONFIG = dict(counts=COUNTS, volume_units=BUDGETS, design_seeds=SEEDS,
    radius_reference=.05, candidates=1024, scout_walks=12, shortlist=20,
    rescore_walks=128, voxel_grid=[96,112,84], outer_clearance=.012,
    pair_clearance=.014, evaluation_blocks=4, points_per_block=512,
    evaluation_walks=128, epsilon=.0005, max_steps=512, threads_per_process=4,
    evaluation_seed=2026091520, evaluation_walk_seed=2026091540,
    source_y_slope=.65, cooled_boundary_value=0,
    protocol='v1: fixed-volume allocations; no radius refinement; three independent design seeds')

def save_json(path, data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def xyz(path, points):
    np.savetxt(path,points,delimiter=',',fmt='%.17g')

def oracle(folder, name, points, centers, radius, walks, seed, mode='twos', epsilon=.0005):
    folder.mkdir(parents=True,exist_ok=True)
    field=folder/f'{name}.csv'; summary=folder/f'{name}.json'
    if field.exists() and summary.exists():
        return pd.read_csv(field)
    pts=folder/f'{name}-points.csv';cav=folder/'centers.csv'
    xyz(pts,points);xyz(cav,centers)
    command=[str(EXE),'--mode',mode,'--mesh',str(MESH),'--points',str(pts),
       '--output',str(field),'--summary',str(summary),'--walks',str(walks),
       '--seed',str(seed),'--threads','4','--repeats','1','--epsilon',str(epsilon),
       '--max-steps','512','--source-y-slope','.65']
    if len(centers):command+=['--cavities',str(cav),'--cavity-radius',str(radius)]
    if mode=='voxel':command+=['--grid','96,112,84']
    result=subprocess.run(command,text=True,capture_output=True,
       env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
    (folder/f'{name}.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError(result.stderr)
    df=pd.read_csv(field)
    if 'limit_hits' in df and df.limit_hits.sum():raise RuntimeError('path cap hit')
    columns=['score'] if mode=='voxel' else ['u','score']
    if not np.isfinite(df[columns].values).all():raise RuntimeError('nonfinite estimate')
    return df

def mesh_object():
    v,f=igl.read_triangle_mesh(str(MESH));tree=igl.AABB();tree.init(v,f)
    return ref.BunnyMesh(v,f,v,f,tree,v.min(0),v.max(0),{})

def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    config=dict(CONFIG,mesh_sha256=hashlib.sha256(MESH.read_bytes()).hexdigest(),
       solver_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest())
    target=OUT/'configuration.json'
    if target.exists() and json.loads(target.read_text())!=config:
        raise RuntimeError('Configuration differs from retained run; use a new output directory')
    save_json(target,config)
    mesh=mesh_object()
    for b in range(4):
        path=OUT/f'evaluation-{b}.csv'
        if not path.exists():xyz(path,ref.sobol_interior(mesh,512,0,CONFIG['evaluation_seed']+b))
    for seed in SEEDS:
        for units in BUDGETS:
            for n in COUNTS:
                folder=OUT/f'v{units}-n{n:02d}-s{seed}';folder.mkdir(exist_ok=True)
                radius=.05*(units/n)**(1/3)
                if not (folder/'candidates.csv').exists():
                    points=ref.sobol_interior(mesh,1024,radius+.012,seed)
                    xyz(folder/'candidates.csv',points)
                    np.save(folder/'outer-distance.npy',mesh.unsigned_distance(points))
    if not (OUT/'initial.json').exists():
        values=evaluate(OUT/'initial',np.empty((0,3)),0)
        save_json(OUT/'initial.json',dict(block_objectives=values,objective=float(np.mean(values))))

def evaluate(folder, centers, radius, epsilon=.0005):
    values=[]
    for b in range(4):
        points=np.loadtxt(OUT/f'evaluation-{b}.csv',delimiter=',')
        active=~ref.inside_cavities(points,centers,radius)
        df=oracle(folder/f'block{b}','field',points[active],centers,radius,128,
            CONFIG['evaluation_walk_seed']+b,epsilon=epsilon)
        # Fixed original-domain normalization, including exact zeros in voids.
        values.append(float(np.sum((1+.65*(points[active,1]+.5))*df.u.values)/len(points)))
    return values

def design(job):
    units,n,seed,method=job
    parent=OUT/f'v{units}-n{n:02d}-s{seed}';folder=parent/method
    folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    radius=.05*(units/n)**(1/3)
    candidates=np.loadtxt(parent/'candidates.csv',delimiter=',')
    distances=np.load(parent/'outer-distance.npy')
    centers=np.empty((0,3));rng=np.random.default_rng(seed+900)
    selections=[];start=time.perf_counter()
    for k in range(n):
        feasible=distances>=radius+.012-1e-12
        if len(centers):feasible &= np.min(np.linalg.norm(candidates[:,None]-centers[None,:],axis=2),axis=1)>=2*radius+.014
        indices=np.flatnonzero(feasible)
        if len(indices)==0:raise RuntimeError(f'Infeasible allocation {job}, inserted {k}/{n}')
        points=candidates[indices]
        if method=='random':chosen=int(rng.integers(len(points)));score=None
        elif method=='voxel':
            field=oracle(folder/f'step{k:02d}','voxel',points,centers,radius,12,seed,mode='voxel')
            chosen=int(np.argmax(field.score.values));score=float(field.score.iloc[chosen])
        else:
            field=oracle(folder/f'step{k:02d}','scout',points,centers,radius,12,seed+1000+2*k)
            m=min(20,len(points));short=np.argsort(field.score.values,kind='stable')[-m:]
            check=oracle(folder/f'step{k:02d}','rescore',points[short],centers,radius,128,seed+1001+2*k)
            # Independent shortlist rescoring heuristic. Selecting its max is
            # not a post-selection confidence certificate for finite insertion.
            lower=check.score.values-1.96*check.score_se.values
            j=int(np.argmax(lower));chosen=int(short[j]);score=float(check.score.iloc[j])
        centers=np.vstack([centers,points[chosen]])
        selections.append(dict(step=k,feasible=len(points),center=points[chosen].tolist(),score=score))
    xyz(folder/'selected-centers.csv',centers)
    blocks=evaluate(folder/'evaluation',centers,radius)
    initial=np.array(json.loads((OUT/'initial.json').read_text())['block_objectives'])
    cooling=100*(1-np.array(blocks)/initial)
    pair=np.linalg.norm(centers[:,None]-centers[None,:],axis=2)+np.eye(n)*1e3
    result=dict(method=method,volume_units=units,count=n,seed=seed,radius=radius,
       removed_volume=n*4*math.pi*radius**3/3,surface_area=n*4*math.pi*radius**2,
       isolated_capacity_sum=n*4*math.pi*radius,
       objective=float(np.mean(blocks)),block_objectives=blocks,
       cooling_percent=float(np.mean(cooling)),block_cooling_percent=cooling.tolist(),
       block_standard_error_percent=float(np.std(cooling,ddof=1)/2),
       minimum_pair_gap=float(pair.min()-2*radius),
       minimum_outer_gap=float(min(distances[np.argmin(np.linalg.norm(candidates-c,axis=1))] for c in centers)-radius),
       elapsed_seconds_not_benchmark=time.perf_counter()-start,selections=selections)
    if result['minimum_pair_gap']<.014-1e-10 or result['minimum_outer_gap']<.012-1e-10:
        raise RuntimeError('clearance violation')
    save_json(folder/'result.json',result)
    print(f'{method:6s} V={units} n={n:2d} seed={seed}: {result["cooling_percent"]:.2f}% cooling',flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    global EXE
    if not EXE.exists():EXE=build_cpp()
    initialize()
    jobs=[(v,n,s,m) for v in BUDGETS for n in COUNTS for s in SEEDS for m in ['twos','voxel','random']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results=list(executor.map(design,jobs))
    save_json(OUT/'results.json',dict(configuration=CONFIG,results=results))
    scalar=[{k:v for k,v in r.items() if not isinstance(v,list)} for r in results]
    pd.DataFrame(scalar).to_csv(OUT/'results.csv',index=False)
    print('Completed',len(results),'designs',flush=True)

if __name__=='__main__':main()
