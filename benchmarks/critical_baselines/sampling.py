"""Matched MC/RQMC/Array-RQMC and neighbor-reuse TD experiments.

Independent implementation of Ho--Owen Algorithm 1 (shifted Korobov,
13-bit Hilbert ordering, converged paths last) and Czekanski et al.
equations (17)--(18). The replication, not a dependent path, is the
cross-fitting unit. Every method solves the same harmonic Dirichlet BVP.
"""
from pathlib import Path
import argparse, json, time, platform
import numpy as np
import pandas as pd
from scipy.stats import qmc

OUT = Path(__file__).parent / 'output'
GENERATORS = {16:7,32:7,64:19,128:49,256:75,512:189,1024:283,2048:857}

def fields(z):
    x,y=z.real,z.imag
    return (0.28+0.58*x-0.30*y+0.22*(x*x-y*y),
            0.46-0.38*x+0.34*y+0.40*x*y)

def distance(z, domain):
    if domain=='disk': return 1-np.abs(z)
    return 1-np.maximum(np.abs(z.real),np.abs(z.imag))

def project(z, domain):
    if domain=='disk': return z/np.abs(z)
    x,y=z.real.copy(),z.imag.copy()
    side=np.abs(x)>=np.abs(y)
    x[side]=np.sign(x[side]); y[~side]=np.sign(y[~side])
    return x+1j*y

def hilbert(z, bits=13):
    """Vectorized 2D xy-to-Hilbert distance; verified against hilbertcurve."""
    side=1<<bits
    x=np.clip(np.floor((z.real+1)*side/2),0,side-1).astype(np.int64)
    y=np.clip(np.floor((z.imag+1)*side/2),0,side-1).astype(np.int64)
    key=np.zeros(z.shape,dtype=np.int64)
    s=side//2
    while s:
        rx=(x&s)>0; ry=(y&s)>0
        key += s*s*((3*rx.astype(np.int64))^ry.astype(np.int64))
        turn=~ry; flip=turn&rx
        x=np.where(flip,s-1-x,x); y=np.where(flip,s-1-y,y)
        x,y=np.where(turn,y,x),np.where(turn,x,y)
        s//=2
    return key

def walks(starts,n,seed,domain,method,eps=1e-7,cap=256,source_payoff=False):
    rng=np.random.default_rng(seed)
    z=np.broadcast_to(starts[:,None],(len(starts),n)).copy()
    first=None; queries=0; moves=0
    source=np.zeros(z.shape)
    if method=='rqmc':
        uniforms=np.stack([qmc.Sobol(cap,scramble=True,seed=int(s)).random_base2(int(np.log2(n)))
                           for s in rng.integers(0,2**31-1,len(starts))])
    for step in range(cap):
        r=distance(z,domain); queries+=z.size
        active=r>eps
        if not active.any(): break
        if source_payoff: source[active]+=r[active]**2/4
        if method=='array':
            keys=np.where(active,hilbert(z),np.iinfo(np.int64).max)
            order=np.argsort(keys,axis=1,kind='stable')
            u=(((GENERATORS[n]*np.arange(n))%n)[None,:]/n+rng.random((len(starts),1)))%1
            assigned=np.empty_like(u); np.put_along_axis(assigned,order,u,axis=1)
        elif method=='rqmc': assigned=uniforms[:,:,step]
        else: assigned=rng.random(z.shape)
        z[active]+=r[active]*np.exp(2j*np.pi*assigned[active])
        moves+=int(active.sum())
        if first is None: first=z.copy()
    if np.any(distance(z,domain)>eps):
        raise RuntimeError('Unconverged walks: do not silently project capped paths')
    return first, (source if source_payoff else project(z,domain)), queries, moves

def reused_means(starts,first,last,domain):
    """Equation (18) weights from geometry, never data-adaptive weights."""
    u,p=fields(last); r=distance(starts,domain)
    out=np.zeros((len(starts),2)); kernel_evals=0
    for k,x in enumerate(starts):
        a=np.abs(x-starts)/r; valid=a<1-1e-12
        a=a[valid]; rv=r[valid]
        w=(1-a)**4/(1-a*a)**2
        kernel=(rv[:,None]**2-np.abs(x-starts[valid,None])**2)/np.abs(x-first[valid])**2
        out[k,0]=np.sum(w*np.mean(kernel*u[valid],axis=1))/w.sum()
        out[k,1]=np.sum(w*np.mean(kernel*p[valid],axis=1))/w.sum()
        kernel_evals+=kernel.size
    return out[:,0],out[:,1],kernel_evals

def crossfit(a,b):
    r=a.shape[0]
    return -(a.sum(axis=0)*b.sum(axis=0)-(a*b).sum(axis=0))/(r*(r-1))

def verify():
    from hilbertcurve.hilbertcurve import HilbertCurve
    rng=np.random.default_rng(841)
    z=rng.uniform(-1,1,1000)+1j*rng.uniform(-1,1,1000)
    ij=np.floor(np.stack([z.real+1,z.imag+1],1)*4096).astype(int)
    assert np.array_equal(hilbert(z),HilbertCurve(13,2).distances_from_points(ij))
    for domain in ['disk','square']:
        starts=np.array([0j,.3+.2j,-.4+.1j])
        for method in ['mc','rqmc','array']:
            _,ex,_,_=walks(starts,64,14,domain,method)
            assert np.max(np.abs(distance(ex,domain)))<1e-12
    # Exact antithetic counterexample: E[cos(theta)]=0, n=2.
    theta=2*np.pi*np.arange(10000)/10000
    a=np.stack([np.cos(theta),-np.cos(theta)])
    assert abs(crossfit(a,a).mean()-.5)<1e-12
    return {'hilbert_1000_points':'pass','exit_geometry':'pass',
            'antithetic_diagonal_deletion_bias':.5}

def run(reps=32,batches=4,sizes=(64,256,1024)):
    OUT.mkdir(exist_ok=True,parents=True)
    checks=verify(); rows=[]; pointrows=[]
    grid=np.linspace(-.55,.55,5)
    starts=(grid[:,None]+1j*grid[None,:]).ravel()
    exact=-np.prod(np.stack(fields(starts)),axis=0)
    methods=['mc','rqmc','array','neighbors']
    for domain_id,domain in enumerate(['disk','square']):
      for n in sizes:
       for rep in range(reps):
        # Paired MC and neighbor runs reuse the exact same timed walk data.
        for method in np.random.default_rng(991+rep).permutation(methods):
         a=[]; b=[]; queries=moves=kernel_evals=0; walk_a=[];walk_b=[]
         t=time.perf_counter()
         for batch in range(batches):
            seed=int(np.random.SeedSequence([20260914,domain_id,n,rep,batch]).generate_state(1)[0])
            first,last,q,m=walks(starts,n,seed,domain,'mc' if method=='neighbors' else method)
            u,p=fields(last); walk_a.append(u);walk_b.append(p)
            if method=='neighbors': u,p,k=reused_means(starts,first,last,domain);kernel_evals+=k
            else: u,p=u.mean(axis=1),p.mean(axis=1)
            a.append(u);b.append(p);queries+=q;moves+=m
         a,b=np.array(a),np.array(b)
         td=crossfit(a,b); elapsed=time.perf_counter()-t
         mse=float(np.mean((td-exact)**2))
         # Diagnostic only: applying the iid formula to dependent individual walks.
         wa=np.concatenate(walk_a,axis=1).T;wb=np.concatenate(walk_b,axis=1).T
         wrong=crossfit(wa,wb)
         rows.append(dict(domain=domain,method=method,n=n,batches=batches,rep=rep,
                          mse=mse,seconds=elapsed,distance_queries=queries,moves=moves,
                          paths=len(starts)*n*batches,kernel_evaluations=kernel_evals,
                          top5_overlap=len(set(np.argsort(td)[:5])&set(np.argsort(exact)[:5]))/5,
                          naive_walk_mse=float(np.mean((wrong-exact)**2))))
         for j,z in enumerate(starts):pointrows.append(dict(domain=domain,method=method,n=n,rep=rep,
                     site=j,x=z.real,y=z.imag,exact=exact[j],estimate=td[j],naive=wrong[j]))
        if rep%8==0: print(domain,n,rep,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'sampling_runs.csv',index=False)
    points=pd.DataFrame(pointrows);points.to_csv(OUT/'sampling_fields.csv',index=False)
    summary=frame.groupby(['domain','method','n']).agg(mse=('mse','mean'),seconds=('seconds','median'),
           distance_queries=('distance_queries','mean'),top5_overlap=('top5_overlap','mean')).reset_index()
    summary.to_csv(OUT/'sampling_summary.csv',index=False)
    (OUT/'sampling_manifest.json').write_text(json.dumps(dict(seed=20260914,repetitions=reps,
       batches=batches,sizes=list(sizes),sites=len(starts),tolerance=1e-7,cap=256,
       python=platform.python_version(),checks=checks,
       scope='harmonic 2D disk and square; no Poisson source, GPU or optimization timing',
       estimator='leave-diagonal-out across independent full batches, shared within-batch primal/adjoint',
       array_reference='https://github.com/hoval58/RQMC-WoS/tree/5b4909207386438ecadd891b5fd4f3122e9dd75d',
       neighbor_reference='https://arxiv.org/html/2404.17692v3#S4.SS2'),indent=2))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=32)
    ap.add_argument('--sizes',type=int,nargs='+',default=[64,256,1024]);args=ap.parse_args()
    run(args.reps,sizes=args.sizes)
