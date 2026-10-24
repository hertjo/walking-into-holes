"""Render and evaluate the first-seed full/half-volume design pair for the hero.

Uses saved designs without repositioning any cavity. The physical oracle
never sees the display cut. Rendered labels report these designs' individual
objective evaluations, not the three-seed mean shown in the tradeoff plot.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import numpy as np
import pandas as pd
import igl
import drjit as dr
from render_mesh_hero import ROOT,OUT,BUNNY,PURPLE,parts,render
from hero_wire_details import run as wire
from hero_thermal import project

THERM=OUT/'thermal'
SPECS=[('budget6',6,6),('budget12',3,12)]
SEED=2026091501

def specifications():
    specs=[]
    for tag,units,n in SPECS:
        folder=ROOT/f'benchmarks/bunny_budget/output/v{units}-n{n:02d}-s{SEED}/twos'
        c=np.loadtxt(folder/'selected-centers.csv',delimiter=',').reshape(-1,3)
        result=json.loads((folder/'result.json').read_text())
        assert len(c)==n
        specs.append((tag,c,result,folder))
    return specs

def fields():
    executable=Path('/tmp/walking-into-holes-embree-build/wih_twos')
    if not executable.exists():
        sys.path.insert(0,str(ROOT/'benchmarks/compiled_bunny'))
        from run_compiled_benchmark import build_cpp
        executable=build_cpp()
    v,f=igl.read_triangle_mesh(str(BUNNY/'stanford-bunny-watertight.ply'))
    x=np.linspace(v[:,0].min(),v[:,0].max(),160)
    y=np.linspace(v[:,1].min(),v[:,1].max(),200)
    xx,yy=np.meshgrid(x,y)
    points=np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,.09)])
    inside=igl.signed_distance(points,v,f,igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]<0
    for tag,c,result,folder in specifications():
        if (THERM/f'{tag}.npz').exists():continue
        radius=result['radius']
        active=inside & (np.min(np.linalg.norm(points[:,None]-c[None,:],axis=2),axis=1)>radius)
        pd.DataFrame(points[active],columns=['x','y','z']).to_csv(THERM/f'{tag}-points.csv',index=False)
        pd.DataFrame(c,columns=['x','y','z']).to_csv(THERM/f'{tag}-centers.csv',index=False)
        cmd=[str(executable),'--mesh',str(BUNNY/'stanford-bunny-watertight.ply'),
            '--points',str(THERM/f'{tag}-points.csv'),'--output',str(THERM/f'{tag}-field.csv'),
            '--summary',str(THERM/f'{tag}-solver.json'),'--walks','256','--repeats','1',
            '--threads','4','--epsilon','.0005','--max-steps','512','--seed','2026091407',
            '--source-y-slope','.65','--cavities',str(THERM/f'{tag}-centers.csv'),'--cavity-radius',str(radius)]
        print('Evaluating',tag,int(active.sum()),'section sites',flush=True)
        subprocess.run(cmd,check=True)
        df=pd.read_csv(THERM/f'{tag}-field.csv');assert df.limit_hits.sum()==0
        assert np.isfinite(df[['u','score']].values).all() and df.u.min()>=0
        u=np.zeros(len(points));benefit=u.copy();u[active]=df.u.values;benefit[active]=df.score.values
        np.savez_compressed(THERM/f'{tag}.npz',x=x,y=y,u=u.reshape(xx.shape),benefit=benefit.reshape(xx.shape),
            active=active.reshape(xx.shape),centers=c,radius=radius)

def surfaces():
    dr.set_thread_count(4)
    for tag,c,result,folder in specifications():
        meshes={k:OUT/f'{tag}-{k}.ply' for k in ['shell','section','cavity'] if (OUT/f'{tag}-{k}.ply').exists()}
        if len(meshes)!=3:meshes=parts(tag,cut=True,centers_override=c,radius_override=result['radius'])
        for zoom in [False,True]:
            target=OUT/(tag+('-detail' if zoom else '')+'.png')
            if not target.exists():render(tag,meshes,PURPLE,1200,64,zoom)
        wire(tag,None,len(c),centers_override=c,radius=result['radius'])

def finish():
    dr.set_thread_count(4)
    records=[]
    for tag,c,result,folder in specifications():
        cached=np.load(THERM/f'{tag}.npz')
        assert np.array_equal(cached['centers'],c) and float(cached['radius'])==result['radius']
        solver=json.loads((THERM/f'{tag}-solver.json').read_text())
        assert solver['walks_per_site']==256 and solver['max_step_limit_hits']==0
        for zoom in [False,True]:project(tag,tag,tag,False,zoom)
        records.append(dict(tag=tag,source=str(folder.relative_to(ROOT)),seed=SEED,
          count=len(c),section_intersections=int(np.sum(np.abs(c[:,2]-.09)<result['radius'])),radius=result['radius'],removed_volume=result['removed_volume'],
          cooling_percent=result['cooling_percent'],
          field_sites=solver['sites'],path_cap_hits=solver['max_step_limit_hits'],
          centers_sha256=hashlib.sha256((folder/'selected-centers.csv').read_bytes()).hexdigest()))
    (OUT/'budget-render-manifest.json').write_text(json.dumps(dict(designs=records,
      selection='First pre-specified design seed, not selected by appearance or objective.',
      rendering='Mitsuba, 1200px, 64 spp, OIDN display radiance; identical existing camera and display cut.',
      fields='256 fresh WoS paths/site, 160x200 section at z=.09; same [0,.025] temperature scale and one-cell display-only smoothing.',
      labels='Individual saved objective estimates; tradeoff plot separately averages three design seeds.',
      visibility='A planar cut intersects only some of the 3D cavities; hidden cavities remain in the geometry and PDE.'),indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['fields','surfaces','finish','all'],default='all')
    args=ap.parse_args()
    if args.stage in ['fields','all']:fields()
    if args.stage in ['surfaces','all']:surfaces()
    if args.stage in ['finish','all']:finish()
