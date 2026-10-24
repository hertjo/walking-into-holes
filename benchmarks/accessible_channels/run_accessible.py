"""Accessible channel placement: a point-score heuristic, not a channel derivative.

Run sequentially; complete design timing retains the fixed outer mesh for every
solver-based method. No finite-benefit certificate or flow model is enabled.
"""
from pathlib import Path
import sys, os, time, json, hashlib, subprocess
import numpy as np
import pandas as pd
from scipy.stats import qmc
import igl

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/distinct_adjoint'))
import run_benchmark as b
import run_persistent as p

OUT=Path(os.environ.get('WIH_ACCESSIBLE_OUTPUT',str(HERE/'output')))
R=.075
Z=np.linspace(-.45,.825,18)
DEVICES=np.array([[-.48,-.18,-.08],[.43,.19,-.07],[.02,-.31,-.02]])
METHODS=['twos_sparse','wos_hot','mg57','mg113','random']
CONFIG={'protocol':'accessible-through-channels-v1','sides':[-1,1],
 'seeds':[2026091600+i for i in range(5)],'methods':METHODS,'candidates':512,
 'rounds':6,'radius':R,'chain_z':Z.tolist(),'score_plane_z':-.03,
 'sensor_radius':.13,'component_radius':.07,'protected_gap':.02,'channel_gap':.025,
 'sparse':[12,192],'shortlist':24,'threads':8,'epsilon':.0005,
 'reference_grid':[113,73,57],'sensitivity_grid':[169,109,85],
 'evaluation':'four shared independent scrambled blocks of 128 sensor probes; 256 fresh walks',
 'claim':'point TD used as a heuristic for whole-channel placement; no channel asymptotic or finite-benefit certificate',
 'physics':'zero-temperature Dirichlet walls, prescribed volumetric source, no coolant flow',
 'ports':'optimized horizontal positions; both chain endpoints outside fixed outer mesh z bounds',
 'budget':'same six channels, radius, center spacing and axial extent; removed volume reported, not constrained equal'}

def chain(x):
    x=np.asarray(x).reshape(-1,3)
    return np.concatenate([np.column_stack((np.repeat(a[0],len(Z)),np.repeat(a[1],len(Z)),Z)) for a in x]) if len(x) else np.empty((0,3))

def candidates(seed,side):
    xy=qmc.Sobol(2,scramble=True,seed=seed).random_base2(13)*[1.68,.82]+[-.84,-.41]
    ok=np.linalg.norm(xy-b.center(side)[:2],axis=1)>=R+.13+.02
    ok &= np.all(np.linalg.norm(xy[:,None,:]-DEVICES[None,:,:2],axis=2)>=R+.07+.02,axis=1)
    x=np.column_stack((xy[ok],np.repeat(-.03,ok.sum())))
    v,f=igl.read_triangle_mesh(str(b.MESH));d,_,_=igl.point_mesh_squared_distance(x,v,f)
    x=x[np.sqrt(d)>R+.023]
    assert len(x)>=CONFIG['candidates']
    return x[:CONFIG['candidates']]

def geometry_audit(x,side,bounds):
    x=np.asarray(x);balls=chain(x)
    neighbor=np.max(np.diff(Z));throat=float(np.sqrt(R*R-(neighbor/2)**2))
    sensor=float(np.min(np.linalg.norm(x[:,:2]-b.center(side)[:2],axis=1))-R-.13)
    component=float(np.min(np.linalg.norm(x[:,None,:2]-DEVICES[None,:,:2],axis=2))-R-.07)
    ds=np.linalg.norm(x[:,None,:2]-x[None,:,:2],axis=2);np.fill_diagonal(ds,np.inf)
    gap=float(ds.min()-2*R)
    assert neighbor<2*R and sensor>=.02-1e-12 and component>=.02-1e-12 and gap>=.025-1e-12
    assert Z[0]<bounds[0,2] and Z[-1]>bounds[1,2]
    return {'connected_chains':len(x),'balls':len(balls),'throat_radius_lower_bound':throat,
      'sensor_gap':sensor,'component_gap':component,'between_channel_gap':gap,
      'lower_endpoint_margin':float(bounds[0,2]-Z[0]),'upper_endpoint_margin':float(Z[-1]-bounds[1,2])}

def design(side,seed,method,x):
    folder=OUT/f'side{side}/seed{seed}/{method}';path=folder/'design.json'
    if path.exists():return json.loads(path.read_text())
    if folder.exists() and list(folder.rglob('summary.json')):
        raise RuntimeError(f'Incomplete measured design: preserve {folder} before restarting')
    folder.mkdir(parents=True,exist_ok=True);p.context_folder=folder
    oldrun=subprocess.run;subprocess.run=p.request
    tick=time.perf_counter();selected=[];rounds=[]
    try:
      for t in range(6):
        ok=np.ones(len(x),bool)
        if selected:ok=np.all(np.linalg.norm(x[:,None,:2]-np.array(selected)[None,:,:2],axis=2)>=2*R+.025,axis=1)
        eligible=x[ok];assert len(eligible)>=24
        tag=f'side{side}/seed{seed}/{method}/round{t}';holes=chain(selected)
        if method=='random':
            index=np.random.default_rng(seed+113*t).integers(len(eligible));chosen=eligible[index];info=[]
        elif method.startswith('mg'):
            grid=[57,37,29] if method=='mg57' else [113,73,57]
            field,a=b.solver(tag+'/solve',eligible,holes,side,'mg',grid=grid)
            index=int(field.score.argmax());chosen=eligible[index];info=[a]
        else:
            hot=method=='wos_hot'
            field,a=b.solver(tag+'/scout',eligible,holes,side,walks=12,seed=seed+100*t,hot=hot)
            short=eligible[np.argsort(field.score.to_numpy())[-24:]]
            field,c=b.solver(tag+'/rescore',short,holes,side,walks=192,seed=seed+100*t+51,hot=hot)
            index=int(field.score.argmax());chosen=short[index];info=[a,c]
        selected.append(chosen.tolist())
        rounds.append({'chosen':chosen.tolist(),'kernel_seconds':sum(a['total_seconds'] for a in info),
                       'steps':sum(a['steps'] for a in info),'eligible':len(eligible)})
    finally:p.close();subprocess.run=oldrun
    seconds=time.perf_counter()-tick
    d={'side':side,'seed':seed,'method':method,'centers':selected,'rounds':rounds,
       'driver_seconds':seconds,'kernel_seconds':sum(a['kernel_seconds'] for a in rounds)}
    b.save(path,d);print('design',side,seed,method,round(seconds,3),flush=True)
    return d

def volume_probes(v,f):
    path=OUT/'volume_probes.npz'
    if path.exists():return dict(np.load(path))
    low=v.min(0);extent=v.max(0)-low
    x=np.concatenate([low+extent*qmc.Sobol(3,scramble=True,seed=664901+i).random_base2(15) for i in range(4)])
    inside=np.abs(igl.fast_winding_number(v,f,x))>.5
    data={'points':x,'inside':inside,'box_volume':np.prod(extent)}
    np.savez_compressed(path,**data);return data

def volume_stats(centers,data):
    x=data['points'];xy=np.asarray(centers)[:,:2]
    radial=np.min(np.sum((x[:,None,:2]-xy[None,:,:])**2,axis=2),axis=1)
    axial=np.min((x[:,None,2]-Z[None,:])**2,axis=1)
    removed=(radial+axial<=R*R)&data['inside']
    v=removed.reshape(4,-1).mean(1)*data['box_volume']
    fraction=removed.mean()/data['inside'].mean()
    source=b.housing.source(x)
    heat=float(np.sum(source*removed)/np.sum(source*data['inside']))
    return {'removed_volume':float(v.mean()),'removed_volume_block_se':float(v.std(ddof=1)/2),
            'removed_material_fraction':float(fraction),'removed_source_fraction':heat}

def main():
    OUT.mkdir(parents=True,exist_ok=True);b.OUT=OUT;b.R=R;b.EXE=p.EXE
    path=OUT/'configuration.json'
    if path.exists():assert json.loads(path.read_text())==CONFIG
    else:
        b.save(path,CONFIG)
        files=[Path(__file__),b.HERE/'adjoint.cpp',b.HERE/'run_benchmark.py',b.HERE/'run_persistent.py',b.MESH]
        b.save(OUT/'provenance.json',{'sources':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in files},
           'executable_sha256':hashlib.sha256(p.EXE.read_bytes()).hexdigest()})
    v,f=igl.read_triangle_mesh(str(b.MESH));bounds=np.array([v.min(0),v.max(0)])
    data=volume_probes(v,f);rows=[];audits=[]
    for side in [-1,1]:
      xp,q=b.probes(side)
      initial,_=b.solver(f'side{side}/reference/initial_wos',xp,[],side,walks=256,seed=441091,hot=True)
      j0=b.objective(initial.u,q).mean()
      grid0,_=b.solver(f'side{side}/reference/initial_grid',xp,[],side,'mg',hot=True,grid=CONFIG['reference_grid'])
      g0=b.objective(grid0.u,q).mean()
      for seed in CONFIG['seeds']:
        setup=OUT/f'side{side}/seed{seed}/candidate_setup.json'
        if setup.exists():cs=json.loads(setup.read_text());x=np.array(cs['points'])
        else:
            tick=time.perf_counter();x=candidates(seed,side)
            cs={'seconds':time.perf_counter()-tick,'points':x.tolist()};b.save(setup,cs)
        for method in np.random.default_rng(seed).permutation(METHODS):
          d=design(side,seed,str(method),x);holes=chain(d['centers']);tag=f'side{side}/seed{seed}/{method}'
          audit=geometry_audit(d['centers'],side,bounds);audits.append(dict(side=side,seed=seed,method=method,**audit))
          u,ui=b.solver(tag+'/evaluate_wos',xp,holes,side,walks=256,seed=559000+seed,hot=True)
          g,gi=b.solver(tag+'/evaluate_grid',xp,holes,side,'mg',hot=True,grid=CONFIG['reference_grid'])
          row={'side':side,'seed':seed,'method':method,'reduction':1-b.objective(u.u,q).mean()/j0,
            'grid_reduction':1-b.objective(g.u,q).mean()/g0,'complete_design_seconds':d['driver_seconds']+cs['seconds'],
            'kernel_seconds':d['kernel_seconds'],'reference_seconds':ui['wall_seconds']+gi['wall_seconds'],
            **volume_stats(d['centers'],data)}
          rows.append(row);pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False);b.save(OUT/'geometry_audit.json',audits)
          print('evaluated',side,seed,method,round(100*row['reduction'],3),round(100*row['grid_reduction'],3),flush=True)
    assert len(rows)==50

if __name__=='__main__':main()
