"""Controlled oracle-state specialization of probabilistic shape calculus.

This is NOT a reproduction of the Schmidt--Wuerschmidt PINN/FEniCS optimizer.
Both optimizers use the same two-parameter ellipse family and exact Poisson
state. Uniform-start importance weighting replaces signed rejection starts
in equation (3.1); pushforward signs are converted per their Remark 3.1.
"""
from pathlib import Path
import time, json
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss

OUT=Path(__file__).parent/'output'
TARGET=np.array([1.,.65])

def state(x,ab):
    c=1/(2*np.sum(ab**-2))
    return c*(1-np.sum((x/ab)**2,axis=-1))

def target(x):return np.maximum(state(x,TARGET),0)

def quadrature(ab,nr=80,nt=256):
    nodes,w=leggauss(nr);s=(nodes+1)/2;w=w/2
    angle=2*np.pi*(np.arange(nt)+.5)/nt
    unit=np.sqrt(s)[:,None,None]*np.stack([np.cos(angle),np.sin(angle)],axis=-1)[None,:,:]
    x=unit*ab; weights=np.broadcast_to(w[:,None],(nr,nt))*np.pi*np.prod(ab)/nt
    return x,weights

def objective(ab):
    x,w=quadrature(ab)
    return float(np.sum(.5*(state(x,ab)-target(x))**2*w))

def deterministic_gradient(ab):
    """Differentiate fixed unit-disk pullback; independent of exit formula."""
    x,w=quadrature(ab); u=state(x,ab);c=1/(2*np.sum(ab**-2))
    residual=u-target(x);out=[]
    ct=1/(2*np.sum(TARGET**-2))
    active=state(x,TARGET)>0
    for j in range(2):
        du=(4*c/ab[j]**3)*u
        dz=np.where(active,-2*ct*x[...,j]**2/(TARGET[j]**2*ab[j]),0)
        out.append(np.sum((residual*(du-dz)+.5*residual**2/ab[j])*w))
    return np.array(out)

def probabilistic_gradient(ab,n,seed):
    rng=np.random.default_rng(seed)
    ang=rng.uniform(0,2*np.pi,n);rad=np.sqrt(rng.random(n))
    starts=np.stack([rad*np.cos(ang),rad*np.sin(ang)],axis=-1)*ab
    residual=state(starts,ab)-target(starts)
    x=starts.copy();queries=0
    for step in range(1024):
        # min singular value times distance in unit disk is an inscribed-ball bound.
        r=np.min(ab)*(1-np.linalg.norm(x/ab,axis=1));active=r>1e-8
        queries+=len(x)
        if not np.any(active):break
        a=rng.uniform(0,2*np.pi,int(active.sum()))
        x[active]+=r[active,None]*np.stack([np.cos(a),np.sin(a)],axis=1)
    else:raise RuntimeError('Ellipse exit cap exhausted')
    x/=np.linalg.norm(x/ab,axis=1)[:,None]
    c=1/(2*np.sum(ab**-2));grad=-2*c*x/(ab*ab)
    volume=-np.pi*np.prod(ab)*np.mean(residual[:,None]*x/ab*grad,axis=0)
    # Boundary term: Va dot n ds=b cos(theta)^2 dtheta; Vb gives a sin².
    theta=2*np.pi*(np.arange(4096)+.5)/4096
    boundary=np.stack([ab[0]*np.cos(theta),ab[1]*np.sin(theta)],axis=1)
    cost=.5*target(boundary)**2
    surface=2*np.pi*np.array([np.mean(cost*ab[1]*np.cos(theta)**2),
                              np.mean(cost*ab[0]*np.sin(theta)**2)])
    return volume+surface,queries

def run():
    OUT.mkdir(parents=True,exist_ok=True);audit=[];traces=[]
    ab0=np.array([.85,.85]);exact=deterministic_gradient(ab0)
    fd=np.array([(objective(ab0+np.eye(2)[j]*1e-5)-objective(ab0-np.eye(2)[j]*1e-5))/2e-5 for j in range(2)])
    assert np.max(np.abs(fd-exact))<1e-8
    for n in [256,1024,4096]:
      for rep in range(48):
        t=time.perf_counter();g,q=probabilistic_gradient(ab0,n,20261000+rep)
        audit.append(dict(n=n,rep=rep,ga=g[0],gb=g[1],exact_a=exact[0],exact_b=exact[1],
                          error=float(np.linalg.norm(g-exact)),seconds=time.perf_counter()-t,queries=q))
      print('shape audit',n,flush=True)
    for method in ['deterministic','probabilistic']:
      for seed in (range(5) if method=='probabilistic' else range(1)):
        ab=ab0.copy();cumulative=0.;qtotal=0
        for step in range(31):
          value=objective(ab)
          traces.append(dict(method=method,seed=seed,step=step,a=ab[0],b=ab[1],objective=value,
                             seconds=cumulative,queries=qtotal))
          if step==30:break
          t=time.perf_counter()
          if method=='probabilistic':g,q=probabilistic_gradient(ab,4096,20262000+100*seed+step)
          else:g,q=deterministic_gradient(ab),0
          # Identical predeclared step schedule and exact-objective Armijo test.
          direction=-g;alpha=20.
          for trial in range(12):
            cand=np.clip(ab+alpha*direction,.45,1.25)
            if objective(cand)<=value-1e-4*alpha*np.dot(g,g):ab=cand;break
            alpha*=.5
          cumulative+=time.perf_counter()-t;qtotal+=q
        print('shape optimization',method,seed,objective(ab),flush=True)
    pd.DataFrame(audit).to_csv(OUT/'shape_gradient_audit.csv',index=False)
    pd.DataFrame(traces).to_csv(OUT/'shape_optimization.csv',index=False)
    (OUT/'shape_manifest.json').write_text(json.dumps(dict(target=TARGET.tolist(),initial=ab0.tolist(),
        exact_gradient=exact.tolist(),central_difference=fd.tolist(),repetitions=48,
        scope='oracle-state, two-parameter, fixed-topology diagnostic; not full PINN/FEniCS baseline',
        upstream='https://github.com/max-wuer/ProbabilisticShape/tree/52a894588ba0235150b401f9cae97b0d2747347a',
        equation='arXiv:2603.01141 equation (3.1), pushforward sign per Remark 3.1; uniform weighted starts',
        tolerance=1e-8,iterations=30,paths_per_step=4096),indent=2))

if __name__=='__main__':run()
