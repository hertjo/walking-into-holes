"""Independent saved-output checks and reference refinements, after timed runs."""
import json
import numpy as np
import pandas as pd
import run_accessible as a

def main():
    b=a.b;b.OUT=a.OUT;b.R=a.R;b.EXE=a.p.EXE
    data=pd.read_csv(a.OUT/'results.csv');assert len(data)==50
    assert data.groupby(['side','method']).size().eq(5).all()
    sensitivity=[]
    for side in [-1,1]:
        xp,q=b.probes(side)
        t0,_=b.solver(f'side{side}/reference/tight_initial',xp,[],side,walks=256,seed=441091,hot=True,epsilon=.000125)
        g0,_=b.solver(f'side{side}/reference/fine_initial',xp,[],side,'mg',hot=True,grid=a.CONFIG['sensitivity_grid'])
        for method in a.METHODS:
            seed=a.CONFIG['seeds'][0];tag=f'side{side}/seed{seed}/{method}'
            d=json.loads((a.OUT/tag/'design.json').read_text());holes=a.chain(d['centers'])
            t,_=b.solver(tag+'/tight_evaluation',xp,holes,side,walks=256,seed=559000+seed,hot=True,epsilon=.000125)
            g,_=b.solver(tag+'/fine_evaluation',xp,holes,side,'mg',hot=True,grid=a.CONFIG['sensitivity_grid'])
            sensitivity.append({'side':side,'seed':seed,'method':method,
                'tight_reduction':1-b.objective(t.u,q).mean()/b.objective(t0.u,q).mean(),
                'fine_grid_reduction':1-b.objective(g.u,q).mean()/b.objective(g0.u,q).mean()})
            pd.DataFrame(sensitivity).to_csv(a.OUT/'reference_sensitivity.csv',index=False)
    decisions=0;products=0;steps=0;maxres=0
    for row in data.itertuples():
        folder=a.OUT/f'side{row.side}/seed{row.seed}/{row.method}'
        d=json.loads((folder/'design.json').read_text())
        for i,c in enumerate(d['centers']):
            for stage in ['scout','rescore','solve']:
                path=folder/f'round{i}'/stage
                if not path.exists():continue
                v=pd.read_csv(path/'field.csv',float_precision='round_trip')
                info=json.loads((path/'summary.json').read_text());cmd=info['command']
                if row.method=='twos_sparse':
                    n=int(cmd[cmd.index('--walks')+1])
                    expected=(n*v.u*v.p-v.same_index)/(n-1)
                    assert np.allclose(v.score,expected,rtol=5e-12,atol=1e-18);products+=len(v)
                if stage in ['solve','rescore']:
                    x=np.loadtxt(path/'points.csv',delimiter=',').reshape(-1,3)
                    assert np.array_equal(np.array(c),x[int(v.score.argmax())]);decisions+=1
        for path in folder.rglob('kernel.json'):
            v=json.loads(path.read_text());assert v['caps']==0
            steps+=v['steps'];maxres=max(maxres,v['relative_residual'])
    assert decisions==240 and maxres<=1e-8
    merged=data.merge(pd.DataFrame(sensitivity),on=['side','seed','method'])
    report={'designs':50,'selected_argmax_checks':decisions,'cross_product_rows':products,
      'walk_steps_in_design_folders':steps,'max_mg_relative_residual':maxres,
      'max_shell_sensitivity_pp':float(100*abs(merged.reduction-merged.tight_reduction).max()),
      'max_grid_sensitivity_pp':float(100*abs(merged.grid_reduction-merged.fine_grid_reduction).max()),
      'max_reference_disagreement_pp':float(100*abs(data.reduction-data.grid_reduction).max()),
      'scope':'saved-data and geometry predicate audit; numerical references approximate; no physical flow or independent expert proof review'}
    b.save(a.OUT/'audit.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
