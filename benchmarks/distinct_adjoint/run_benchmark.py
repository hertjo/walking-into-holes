"""Predeclared distinct-adjoint comparison; all designs and raw fields retained.

The cold cavities are ideal sinks. This experiment does not assert coolant
transport or transfer the compliance-only finite-hole certificate to J_q.
"""
from pathlib import Path
import os, sys, json, time, subprocess, hashlib, argparse
import numpy as np
import pandas as pd
import igl
from scipy.stats import qmc

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/engineering_housing'))
import run_housing_benchmark as housing
OUT=Path(os.environ.get('WIH_ADJOINT_OUTPUT',str(HERE/'output')))
EXE=Path(os.environ.get('WIH_ADJOINT_EXE','/tmp/wih-adjoint-build/wih_adjoint'))
MESH=ROOT/'benchmarks/engineering_housing/assets/stylized-finned-housing.ply'
R=.095
METHODS=['twos_sparse','twos_dense','wos_hot','mg_td','mg_hot','random']
CONFIG={'protocol':'distinct-adjoint-v1','sides':[-1,1],
 'seeds':[2026091500+i for i in range(5)],'candidates':768,'rounds':6,
 'radius':R,'sensor_radius':.13,'sensor_clearance':.02,'outer_clearance':.023,
 'intercavity_gap':.025,'sparse':[12,192],'dense':[48,768],'shortlist':24,
 'grid':[113,73,57],'reference_grid':[169,109,85],
 'threads':8,'epsilon':.0005,'walk_cap':1024,
 'evaluation_blocks':4,'evaluation_points_per_block':128,'evaluation_walks':256,
 'methods':METHODS,'acceptance':'fixed-count fresh ranking; no finite-benefit certificate',
 'objective':'integral q*u divided by exact integral q; q=(1-|x-c|^2/.13^2)_+^2'}

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n')

def center(side):return np.array([.55*side,.18*side,-.03])

def solver(tag,points,holes,side,mode='twos',walks=12,seed=0,hot=False,grid=None,epsilon=.0005):
    folder=OUT/tag;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():
        return pd.read_csv(folder/'field.csv',float_precision='round_trip'),json.loads((folder/'summary.json').read_text())
    start=time.perf_counter()
    np.savetxt(folder/'points.csv',points,delimiter=',',fmt='%.17g')
    np.savetxt(folder/'holes.csv',np.asarray(holes).reshape(-1,3),delimiter=',',fmt='%.17g')
    cmd=[str(EXE),'--mesh',str(MESH),'--points',str(folder/'points.csv'),
        '--cavity-radius',str(R),
        '--output',str(folder/'field.csv'),'--summary',str(folder/'kernel.json'),
        '--mode',mode,'--source-mode','housing','--walks',str(walks),'--seed',str(seed),
        '--threads','8','--repeats','1','--epsilon',str(epsilon),'--max-steps','1024',
        '--sensor-side',str(side),'--hot-only',str(int(hot)),
        '--grid',','.join(map(str,grid or CONFIG['grid'])),'--mg-cycles','150','--mg-tolerance','1e-8']
    if len(holes):cmd+=['--cavities',str(folder/'holes.csv')]
    env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_PROC_BIND='true')
    run=subprocess.run(cmd,capture_output=True,text=True,env=env)
    (folder/'solver.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError(f'{tag}: {run.stderr}')
    frame=pd.read_csv(folder/'field.csv',float_precision='round_trip')
    info=json.loads((folder/'kernel.json').read_text())
    info['wall_seconds']=time.perf_counter()-start
    info['command']=cmd
    save(folder/'summary.json',info)
    return frame,info

def candidates(seed,side):
    points=qmc.Sobol(3,scramble=True,seed=seed).random_base2(13)
    points=np.array([-.84,-.41,-.135])+points*np.array([1.68,.82,.27])
    points=points[housing.housing_sdf(points)<-(R+.023)]
    points=points[np.linalg.norm(points-center(side),axis=1)>=R+.13+.02]
    v,f=igl.read_triangle_mesh(str(MESH))
    ds,_,_=igl.point_mesh_squared_distance(points,v,f)
    points=points[np.sqrt(ds)>R+.023]
    assert len(points)>=CONFIG['candidates']
    return points[:CONFIG['candidates']]

def probes(side):
    blocks=[]
    for b in range(CONFIG['evaluation_blocks']):
        a=qmc.Sobol(3,scramble=True,seed=990015+b).random_base2(7)
        z=2*a[:,0]-1;phi=2*np.pi*a[:,1];rr=.13*a[:,2]**(1/3)
        d=np.column_stack((np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z))
        blocks.append(center(side)+rr[:,None]*d)
    x=np.concatenate(blocks)
    q=np.maximum(0,1-np.sum((x-center(side))**2,axis=1)/.13**2)**2
    return x,q

def objective(u,q):
    # Exact spherical bump normalization: mean over the sensor ball is 8/35.
    return (np.asarray(u)*q).reshape(4,128).mean(axis=1)/(8/35)

def design(side,seed,method,x):
    tag=f'side{side}/seed{seed}/{method}';path=OUT/tag/'design.json'
    if path.exists():return json.loads(path.read_text())
    holes=[];rounds=[];elapsed=0.;kernel=0.;rss=0
    rng=np.random.default_rng(seed+981)
    for t in range(CONFIG['rounds']):
        tick=time.perf_counter()
        ok=np.ones(len(x),bool)
        if holes:ok=np.all(np.linalg.norm(x[:,None,:]-np.array(holes)[None,:,:],axis=2)>=2*R+.025,axis=1)
        eligible=x[ok];assert len(eligible)>=24
        infos=[]
        if method=='random': chosen=eligible[rng.integers(len(eligible))]
        elif method.startswith('mg'):
            field,info=solver(f'{tag}/round{t}/solve',eligible,holes,side,'mg',hot=method=='mg_hot')
            infos.append(info);chosen=eligible[int(field.score.argmax())]
        else:
            hot=method=='wos_hot';n,m=CONFIG['dense' if method=='twos_dense' else 'sparse']
            field,info=solver(f'{tag}/round{t}/scout',eligible,holes,side,walks=n,seed=seed+100*t,hot=hot)
            infos.append(info);short=eligible[np.argsort(field.score.to_numpy())[-24:]]
            field,info=solver(f'{tag}/round{t}/rescore',short,holes,side,walks=m,seed=seed+100*t+51,hot=hot)
            infos.append(info);chosen=short[int(field.score.argmax())]
        elapsed+=time.perf_counter()-tick
        # Saved per-call times preserve measured timing on resumed executions.
        wall=sum(a['wall_seconds'] for a in infos)
        kernel+=sum(a['total_seconds'] for a in infos)
        rss=max([rss]+[a['peak_rss_megabytes'] for a in infos])
        holes.append(chosen.tolist());rounds.append({'wall_seconds':wall,'kernel_seconds':sum(a['total_seconds'] for a in infos)})
    result={'side':side,'seed':seed,'method':method,'holes':holes,'rounds':rounds,
        'solver_wall_seconds':sum(a['wall_seconds'] for a in rounds),'kernel_seconds':kernel,
        'driver_seconds':elapsed,'peak_rss_megabytes':rss}
    save(path,result);print('design',side,seed,method,round(kernel,3),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'configuration.json').exists():
        save(OUT/'configuration.json',CONFIG)
        save(OUT/'provenance.json',{'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),HERE/'adjoint.cpp',ROOT/'benchmarks/compiled_bunny/src/wih_twos.cpp',MESH]},
            'executable_sha256':hashlib.sha256(EXE.read_bytes()).hexdigest()})
    else:assert json.loads((OUT/'configuration.json').read_text())==CONFIG
    rows=[]
    for side in CONFIG['sides']:
        xp,q=probes(side)
        u0,_=solver(f'side{side}/reference/initial_wos',xp,[],side,walks=256,seed=990010,hot=True)
        b0=objective(u0.u,q)
        g0,_=solver(f'side{side}/reference/initial_grid',xp,[],side,'mg',hot=True,grid=CONFIG['reference_grid'])
        j0g=objective(g0.u,q).mean()
        for seed in CONFIG['seeds'][:1 if args.pilot else 5]:
            setup=OUT/f'side{side}/seed{seed}/candidate_setup.json'
            if setup.exists():x=np.array(json.loads(setup.read_text())['points'])
            else:
                tick=time.perf_counter();x=candidates(seed,side);common=time.perf_counter()-tick
                save(setup,{'seconds':common,'points':x.tolist()})
            order=np.random.default_rng(seed).permutation(METHODS)
            for method in order:
                d=design(side,seed,str(method),x);holes=d['holes']
                tag=f'side{side}/seed{seed}/{method}'
                u,ui=solver(tag+'/evaluate_wos',xp,holes,side,walks=256,seed=880000+seed,hot=True)
                j=objective(u.u,q);reduction=1-j.mean()/b0.mean()
                g,gi=solver(tag+'/evaluate_grid',xp,holes,side,'mg',hot=True,grid=CONFIG['reference_grid'])
                gr=1-objective(g.u,q).mean()/j0g
                row={k:v for k,v in d.items() if k not in ['holes','rounds']}
                row.update(reduction=reduction,grid_reduction=gr,
                    conditional_block_se=np.std(1-j/b0,ddof=1)/2,
                    reference_wall_seconds=ui['wall_seconds']+gi['wall_seconds'])
                rows.append(row);pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False)
                print('evaluated',side,seed,method,round(100*reduction,3),round(100*gr,3),flush=True)
    print(pd.DataFrame(rows).groupby(['side','method'])[['reduction','grid_reduction','kernel_seconds']].mean(),flush=True)

if __name__=='__main__':main()
