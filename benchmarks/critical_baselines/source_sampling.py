"""True Poisson-compliance TD: -Delta u=1, u=0, p=u on unit disk.

Constant-source Green integral on each 2D WoS ball is exactly r²/4.
The source kernel is shared by all methods. This is a direct estimator
comparison, with no claimed finite-insertion certification experiment.
"""
import time,json
import numpy as np
import pandas as pd
from sampling import walks,crossfit,OUT

def run():
    starts=np.array([0j,.25+.15j,.5+0j,.65+.2j,-.55-.3j])
    truth=-((1-np.abs(starts)**2)/4)**2
    rows=[]
    for n in [64,256,1024]:
      for rep in range(32):
       for method in np.random.default_rng(462+rep).permutation(['mc','rqmc','array']):
        t=time.perf_counter();a=[];queries=moves=0
        for batch in range(4):
            seed=int(np.random.SeedSequence([20263914,n,rep,batch]).generate_state(1)[0])
            _,payoff,q,m=walks(starts,n,seed,'disk',method,source_payoff=True)
            a.append(payoff.mean(axis=1));queries+=q;moves+=m
        a=np.array(a);td=crossfit(a,a)
        rows.append(dict(method=method,n=n,rep=rep,mse=float(np.mean((td-truth)**2)),
           seconds=time.perf_counter()-t,queries=queries,moves=moves,paths=n*4*len(starts)))
       if rep%8==0:print('Poisson',n,rep,flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'source_sampling_runs.csv',index=False)
    (OUT/'source_manifest.json').write_text(json.dumps(dict(
        pde='-Delta u=1, homogeneous Dirichlet, unit disk',objective='J=integral u',
        adjoint='p=u',exact_td='-(1-|z|^2)^2/16',batches=4,repetitions=32,tolerance=1e-7,
        seeds='SeedSequence([20263914,n,rep,batch])',sites=[[z.real,z.imag] for z in starts],
        source_integral='r^2/4 on each ball, exact constant-source integration',
        shell_bias_bound='marginal <= eps/2; product <= eps/4+eps^2/4',
        neighbor_status='not applied: harmonic terminal-only reweighting is insufficient for a source PDE'),indent=2))
    print(f.groupby(['method','n'])[['mse','seconds']].mean().to_string(),flush=True)

if __name__=='__main__':run()
