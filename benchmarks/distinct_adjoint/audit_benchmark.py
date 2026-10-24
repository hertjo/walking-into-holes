"""Independent discretization/shell checks and supplementary MG budget sweep.

Run after the primary benchmark to avoid concurrent timing interference.
The extra grids are declared after the first-seed pilot and before their runs.
They are retained regardless of whether they favor TWoS.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import igl
import run_benchmark as b

def main():
    b.save(b.OUT/'audit_protocol.json',{'extra_design_grids':[[57,37,29],[85,55,43],[169,109,85]],
        'seeds':b.CONFIG['seeds'],'sides':b.CONFIG['sides'],
        'reference_checks':'first seed all methods, 225x145x113 and epsilon=0.000125',
        'status':'supplementary grid sweep specified after primary first-seed pilot'})
    extra=[];checks=[]
    for side in b.CONFIG['sides']:
        xp,q=b.probes(side)
        u0,_=b.solver(f'side{side}/reference/initial_wos',xp,[],side,walks=256,seed=990010,hot=True)
        j0=b.objective(u0.u,q).mean()
        g0,_=b.solver(f'side{side}/reference/initial_grid',xp,[],side,'mg',hot=True,grid=b.CONFIG['reference_grid'])
        j0g=b.objective(g0.u,q).mean()
        for seed in b.CONFIG['seeds']:
            setup=json.loads((b.OUT/f'side{side}/seed{seed}/candidate_setup.json').read_text())
            x=np.array(setup['points'])
            for grid in [[57,37,29],[85,55,43],[169,109,85]]:
                tag=f'side{side}/seed{seed}/mg{grid[0]}'
                path=b.OUT/tag/'design.json'
                if path.exists():d=json.loads(path.read_text())
                else:
                    start=time.perf_counter();holes=[];infos=[]
                    for t in range(6):
                        feasible=x if not holes else x[np.all(np.linalg.norm(x[:,None,:]-np.array(holes)[None,:,:],axis=2)>=2*b.R+.025,axis=1)]
                        field,info=b.solver(f'{tag}/round{t}',feasible,holes,side,'mg',grid=grid)
                        holes.append(feasible[int(field.score.argmax())].tolist());infos.append(info)
                    d={'side':side,'seed':seed,'method':f'mg{grid[0]}','holes':holes,
                       'driver_seconds':time.perf_counter()-start,
                       'kernel_seconds':sum(i['total_seconds'] for i in infos),
                       'solver_wall_seconds':sum(i['wall_seconds'] for i in infos),
                       'peak_rss_megabytes':max(i['peak_rss_megabytes'] for i in infos)}
                    b.save(path,d)
                u,ui=b.solver(tag+'/evaluate_wos',xp,d['holes'],side,walks=256,seed=880000+seed,hot=True)
                g,gi=b.solver(tag+'/evaluate_grid',xp,d['holes'],side,'mg',hot=True,grid=b.CONFIG['reference_grid'])
                row={k:v for k,v in d.items() if k!='holes'}
                row.update(reduction=1-b.objective(u.u,q).mean()/j0,
                    grid_reduction=1-b.objective(g.u,q).mean()/j0g,
                    reference_wall_seconds=ui['wall_seconds']+gi['wall_seconds'])
                extra.append(row);pd.DataFrame(extra).to_csv(b.OUT/'grid_sweep.csv',index=False)
                print('sweep',side,seed,grid[0],round(100*row['reduction'],2),flush=True)
        # Fourfold shell refinement and separate finer grid on preselected seed.
        tight,_=b.solver(f'side{side}/reference/tight_wos',xp,[],side,walks=256,seed=990010,hot=True,epsilon=.000125)
        fine,_=b.solver(f'side{side}/reference/fine_grid',xp,[],side,'mg',hot=True,grid=[225,145,113])
        for method in b.METHODS:
            tag=f'side{side}/seed{b.CONFIG["seeds"][0]}/{method}'
            d=json.loads((b.OUT/tag/'design.json').read_text())
            t,_=b.solver(tag+'/tight_wos',xp,d['holes'],side,walks=256,seed=880000+b.CONFIG['seeds'][0],hot=True,epsilon=.000125)
            g,_=b.solver(tag+'/fine_grid',xp,d['holes'],side,'mg',hot=True,grid=[225,145,113])
            checks.append({'side':side,'method':method,
                'tight_reduction':1-b.objective(t.u,q).mean()/b.objective(tight.u,q).mean(),
                'fine_grid_reduction':1-b.objective(g.u,q).mean()/b.objective(fine.u,q).mean()})
            pd.DataFrame(checks).to_csv(b.OUT/'reference_sensitivity.csv',index=False)
        # Field convergence: same predetermined first 64 candidates.
        pts=np.array(json.loads((b.OUT/f'side{side}/seed{b.CONFIG["seeds"][0]}/candidate_setup.json').read_text())['points'])[:64]
        for mode,n,grid in [('twos',4096,None),('mg',2,[113,73,57]),('mg',2,[169,109,85]),('mg',2,[225,145,113])]:
            b.solver(f'side{side}/field_audit/{mode}{n if grid is None else grid[0]}',pts,[],side,mode,walks=n,seed=777000,grid=grid)
    audit_geometry_and_scores()

def audit_geometry_and_scores():
    v,f=igl.read_triangle_mesh(str(b.MESH));count=0;minouter=1e9;minsensor=1e9;mingap=1e9;decisions=0
    for side in [-1,1]:
        c=b.center(side)[None,:];ds,_,_=igl.point_mesh_squared_distance(c,v,f)
        assert abs(igl.fast_winding_number(v,f,c)[0])>.99 and np.sqrt(ds[0])>.13
    for path in b.OUT.glob('side*/seed*/*/design.json'):
        d=json.loads(path.read_text());x=np.array(d['holes']);assert len(x)==6
        ds,_,_=igl.point_mesh_squared_distance(x,v,f)
        assert np.all(np.abs(igl.fast_winding_number(v,f,x))>.99)
        outer=np.sqrt(ds)-b.R;sensor=np.linalg.norm(x-b.center(d['side']),axis=1)-b.R-.13
        dd=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)-2*b.R;dd[np.eye(6,dtype=bool)]=np.inf
        assert min(outer)>.023-1e-10 and min(sensor)>.02-1e-10 and dd.min()>.025-1e-10
        minouter=min(minouter,float(min(outer)));minsensor=min(minsensor,float(min(sensor)));mingap=min(mingap,float(dd.min()));count+=1
        if d['method']!='random':
            for step,chosen in enumerate(x):
                folder=path.parent/f'round{step}'
                if (folder/'rescore').exists():folder=folder/'rescore'
                elif (folder/'solve').exists():folder=folder/'solve'
                points=np.loadtxt(folder/'points.csv',delimiter=',').reshape(-1,3)
                field=pd.read_csv(folder/'field.csv',float_precision='round_trip')
                assert np.array_equal(chosen,points[int(field.score.argmax())])
                decisions+=1
    scores=0;maxres=0;steps=0
    for path in b.OUT.rglob('summary.json'):
        info=json.loads(path.read_text());cmd=info['command'];maxres=max(maxres,info['relative_residual']);steps+=info['steps']
        frame=pd.read_csv(path.parent/'field.csv',float_precision='round_trip')
        assert np.isfinite(frame.to_numpy()).all() and (frame.caps==0).all()
        if cmd[cmd.index('--mode')+1]=='twos' and cmd[cmd.index('--hot-only')+1]=='0':
            n=int(cmd[cmd.index('--walks')+1]);reconstructed=(n*frame.u*frame.p-frame.same_index)/(n-1)
            assert np.allclose(reconstructed,frame.score,rtol=1e-12,atol=1e-18);scores+=len(frame)
    # Exact elementary manufactured identities, independently differentiated.
    import sympy as s
    x,y,z,bb=s.symbols('x y z bb');u=(1-x*x-y*y-z*z)/6;p=(1-x*x-y*y-z*z)*(s.Rational(1,6)+bb*x/10)
    lap=lambda a:sum(s.diff(a,t,2) for t in [x,y,z])
    assert s.expand(-lap(u)-1)==0 and s.expand(-lap(p)-(1+bb*x))==0
    assert s.diff(u*p,x).subs({x:0,y:0,z:0})==bb/60
    b.save(b.OUT/'audit.json',{'designs':count,'selected_argmax_decisions':decisions,'cross_products_reconstructed':scores,
        'steps_no_cap':steps,'maximum_mg_residual':maxres,'min_outer_gap':minouter,
        'min_sensor_gap':minsensor,'min_intercavity_gap':mingap,'manufactured_symbolic_check':True,
        'qualification':'numerical audit, not a proof of ideal sampling laws or statistical coverage'})

if __name__=='__main__':main()
