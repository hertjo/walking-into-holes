"""Fresh sequential minimum-benefit validation on application meshes.

Fixed protocol: six attempts per object; one scout-selected finalist each;
four nested validation stages; total ideal-sampling failure budget 0.01 across
both objects and all stages. References and baseline scale never drive a
statistical ratio claim. New data live separately from earlier benchmarks.
"""
from pathlib import Path
from fractions import Fraction as Q
import argparse, hashlib, json, math, os, subprocess, sys, time
import igl
import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/'output'
EXE=Path('/tmp/wih-certification-build/wih_twos')
FAST=Path('/tmp/walking-into-holes-embree-build/wih_twos')
STAGES=[1024,4096,16384,65536]
DELTA=.01/(2*6*len(STAGES))
CONFIG=dict(protocol='application-minimum-benefit-v1',attempts=6,stages=STAGES,
    total_delta=.01,delta_per_stage=DELTA,delta_per_stage_exact='1/4800',
    confidence_log_upper=8.5,threads=8,seed=2026091601,
    max_steps=128,epsilon=1e-6,radius_cap=.1,clearance=.012,
    scout_walks=12,rescore_walks=192,shortlist=24,
    primary_target_fraction_of_fixed_scale=.001,
    diagnostic_target_fractions=[.001,.005,.01,.02],
    value_caps={'bunny':.06,'housing':.18},fixed_scales={'bunny':.002,'housing':.015},
    reference_blocks=4,reference_points_per_block=512,reference_walks=256,
    scope='Conditional on ideal sampling laws. Geometry/source safeguards do not constitute a full finite-precision probability proof.')

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n');tmp.replace(path)

def xyz(path,points):
    np.savetxt(path,np.asarray(points).reshape(-1,3),delimiter=',',fmt='%.17g')

def exact_inside(point,V,F):
    """Exact rational positive-axis parity for binary64 coordinates.

    Bounding-box culling uses exact comparisons; projected predicates and
    ray-intersection signs use fractions. Retry an axis on edge/coplanar hits.
    Assumes a closed embedded triangle boundary (audited separately).
    """
    for axis in range(3):
        yz=[a for a in range(3) if a!=axis]
        tri=V[F]
        mask=np.all((tri[:,:,yz].min(1)<=point[yz])&(tri[:,:,yz].max(1)>=point[yz]),axis=1)
        hits=0;degenerate=False
        p=[Q(float(x)) for x in point]
        def orient(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        for t in tri[mask]:
            verts=[[Q(float(x)) for x in row] for row in t]
            a,b,c=[[row[k] for k in yz] for row in verts];q=[p[k] for k in yz]
            area=orient(a,b,c)
            if area==0:
                if orient(a,b,q)==0 and orient(a,c,q)==0:degenerate=True;break
                continue
            weights=[orient(q,b,c),orient(a,q,c),orient(a,b,q)]
            sign=1 if area>0 else -1
            if any(w*sign<0 for w in weights):continue
            if any(w==0 for w in weights):degenerate=True;break
            hit=sum(w*row[axis] for w,row in zip(weights,verts))/area
            if hit==p[axis]:return False,{'axis':axis,'boundary':True}
            hits+=hit>p[axis]
        if not degenerate:return bool(hits%2),{'axis':axis,'intersections':int(hits),'rational_predicates':True}
    raise RuntimeError('All axis rays degenerate; no certified initial classification')

def exp_lower(t):
    a=np.maximum(0.,np.nextafter(1-np.nextafter(t/1024,np.inf),-np.inf))
    for _ in range(10):a=np.maximum(0.,np.nextafter(a*a,-np.inf))
    return a

def exp_upper(t):
    a=np.nextafter(1+np.maximum(0.,np.nextafter(t/1024,-np.inf)),-np.inf)
    for _ in range(10):a=np.nextafter(a*a,-np.inf)
    return np.nextafter(1/a,np.inf)

CENTERS=np.array([[-.48,-.18,-.08],[.43,.19,-.07],[.02,-.31,-.02]])
WIDTHS=np.array([[.24,.22,.16],[.21,.20,.15],[.30,.16,.19]])
AMPL=np.array([5.2,4.6,2.])

def source_bounds(z,R,kind):
    """Outward arithmetic enclosures over a containing coordinate box."""
    dn=lambda x:np.nextafter(x,-np.inf)
    up=lambda x:np.nextafter(x,np.inf)
    if kind=='bunny':
        return max(0.,float(dn(1+dn(.65*dn(dn(z[1]-R)+.5))))),float(up(1+up(.65*up(up(z[1]+R)+.5))))
    d=np.abs(z-CENTERS)
    far=up(up(up(d)+R)/WIDTHS)
    near=np.maximum(0.,dn(dn(d)-R));near=np.maximum(0.,dn(near/WIDTHS))
    upper=np.zeros(3);lower=np.zeros(3)
    for axis in range(3):
        upper=up(upper+up(far[:,axis]**2))
        lower=np.maximum(0.,dn(lower+np.maximum(0.,dn(near[:,axis]**2))))
    lo=.08;hi=.08
    for a,l,h in zip(AMPL,exp_lower(upper),exp_upper(lower)):
        lo=float(dn(lo+dn(a*l)));hi=float(up(hi+up(a*h)))
    return max(0.,lo),hi

def benefit(L,z,r,clearance,kind):
    # R grid affects tightness only: one state event covers every admissible R.
    if clearance<=r:return 0.,None
    radii=np.linspace(np.nextafter(r,np.inf),.99*clearance,128)
    best=(-1.,None)
    for R in radii:
        if not r<R<clearance:continue
        flo,fhi=source_bounds(z,R,kind)
        dn=lambda x:float(np.nextafter(x,-np.inf))
        up=lambda x:float(np.nextafter(x,np.inf))
        diff=max(0.,dn(dn(R*R)-up(r*r)))
        qlo=max(0.,dn(dn(flo*diff)/6))
        qhi=up(up(fhi*up(R*R))/6)
        t=up(r/R);den=up(up(1+t)**2)
        c=max(0.,dn(dn(1-t)/den))
        h=max(0.,dn(L-qhi))
        m=max(0.,dn(qlo+max(0.,dn(c*h))))
        # Machin's formula and alternating rational sums supply pi from below.
        aa=sum((Q((-1)**k,(2*k+1)*5**(2*k+1)) for k in range(12)),Q(0))
        bb=sum((Q((-1)**k,(2*k+1)*239**(2*k+1)) for k in range(5)),Q(0))
        pi_lo=dn(float(16*aa-4*bb))
        b=max(0.,dn(dn(dn(4*pi_lo)*r)*max(0.,dn(m*m))))
        if b>best[0]:best=(b,dict(R=float(R),f_lower=flo,f_upper=fhi,temperature_floor=m))
    return best

def run_oracle(folder,name,mesh,points,holes,radius,kind,mode,walks,seed):
    folder.mkdir(parents=True,exist_ok=True)
    out=folder/f'{name}.csv';summary=folder/f'{name}.json'
    if out.exists() and summary.exists():return pd.read_csv(out,float_precision='round_trip'),json.loads(summary.read_text())
    pts=folder/f'{name}-points.csv';cav=folder/f'{name}-holes.csv'
    xyz(pts,points);xyz(cav,holes)
    command=[str(EXE if mode=='validate' else FAST),'--mode',mode,'--mesh',str(mesh),
      '--points',str(pts),'--output',str(out),'--summary',str(summary),
      '--walks',str(walks),'--seed',str(seed),'--threads','8','--repeats','1',
      '--source-mode','affine' if kind=='bunny' else 'housing','--source-y-slope','.65',
      '--epsilon',str(CONFIG['epsilon'] if mode=='validate' else .0005),
      '--max-steps',str(CONFIG['max_steps'] if mode=='validate' else 512)]
    if len(holes):command+=['--cavities',str(cav),'--cavity-radius',str(radius)]
    if mode=='validate':command+=['--validation-radius-cap',str(CONFIG['radius_cap']),
                                  '--validation-value-cap',str(CONFIG['value_caps'][kind])]
    start=time.perf_counter()
    proc=subprocess.run(command,capture_output=True,text=True,
        env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1'))
    (folder/f'{name}.log').write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(f'{name} failed: {proc.stderr}')
    meta=json.loads(summary.read_text());meta['process_seconds']=time.perf_counter()-start
    save(summary,meta)
    return pd.read_csv(out,float_precision='round_trip'),meta

def reference(kind,mesh,V,F,holes,r,folder):
    result=folder/'reference.json'
    if result.exists():return json.loads(result.read_text())
    low,high=V.min(0),V.max(0)
    volume=abs(float(np.einsum('ij,ij->i',V[F[:,0]],np.cross(V[F[:,1]],V[F[:,2]])).sum()/6))
    vals=[]
    for block in range(4):
        path=OUT/kind/f'probes-{block}.csv'
        if not path.exists():
            samples=low+qmc.Sobol(3,scramble=True,seed=2026091690+block).random_base2(16)*(high-low)
            wn=igl.fast_winding_number(V,F,samples)
            pts=samples[np.abs(wn)>.5][:512]
            if len(pts)!=512:raise RuntimeError('Insufficient probes')
            xyz(path,pts)
        pts=np.loadtxt(path,delimiter=',');active=np.ones(len(pts),bool)
        if len(holes):active=np.min(np.linalg.norm(pts[:,None]-holes[None],axis=2),axis=1)>r
        field,_=run_oracle(folder,f'reference-{block}',mesh,pts[active],holes,r,kind,'twos',256,2026091680+block)
        if kind=='bunny':f=1+.65*(pts[active,1]+.5)
        else:f=.08+(AMPL*np.exp(-np.sum(((pts[active,None]-CENTERS)/WIDTHS)**2,axis=2))).sum(axis=1)
        vals.append(float(volume*np.sum(f*field.u.values)/len(pts)))
    out=dict(blocks=vals,mean=float(np.mean(vals)),se=float(np.std(vals,ddof=1)/2),mesh_volume=volume,
        scope='Independent approximate WoS reference with four randomized spatial blocks; shell and geometry bias not certified.')
    save(result,out);return out

def design(kind):
    folder=OUT/kind;folder.mkdir(parents=True,exist_ok=True)
    if kind=='bunny':
        mesh=ROOT/'benchmarks/stanford_bunny/output/stanford-bunny-watertight.ply';r=.05
        candidates=np.loadtxt(ROOT/'benchmarks/bunny_budget/output/v6-n06-s2026091501/candidates.csv',delimiter=',')
    else:
        mesh=ROOT/'benchmarks/engineering_housing/assets/stylized-finned-housing.ply';r=.095
        sys.path.insert(0,str(ROOT/'benchmarks/engineering_housing'))
        from run_housing_benchmark import candidate_points
        candidates=candidate_points(768)
    V,F=igl.read_triangle_mesh(str(mesh));tree=igl.AABB();tree.init(V,F)
    dist=np.sqrt(tree.squared_distance(V,F,candidates)[0])
    holes=np.empty((0,3));attempted=[];records=[]
    target=CONFIG['primary_target_fraction_of_fixed_scale']*CONFIG['fixed_scales'][kind]
    initial=reference(kind,mesh,V,F,holes,r,folder/'initial')
    previous=initial
    for step in range(6):
        sf=folder/f'attempt-{step:02d}';sf.mkdir(exist_ok=True)
        final=sf/'decision.json'
        if final.exists():
            record=json.loads(final.read_text());records.append(record)
            attempted.append(record['candidate_index'])
            if record['accepted']:holes=np.vstack([holes,record['center']]);previous=record['reference_after']
            continue
        feasible=dist>r+.012
        if len(holes):feasible &= np.min(np.linalg.norm(candidates[:,None]-holes[None],axis=2),axis=1)>2*r+.014
        feasible[attempted]=False;ids=np.flatnonzero(feasible);pts=candidates[ids]
        scout,scmeta=run_oracle(sf,'scout',mesh,pts,holes,r,kind,'twos',12,CONFIG['seed']+100*step)
        short=np.argsort(scout.score.values,kind='stable')[-24:]
        rescore,re_meta=run_oracle(sf,'rescore',mesh,pts[short],holes,r,kind,'twos',192,CONFIG['seed']+100*step+1)
        j=int(np.argmax(rescore.score.values-1.96*rescore.score_se.values));idx=int(ids[short[j]]);z=candidates[idx]
        inside,parity=exact_inside(z,V,F)
        if not inside:raise RuntimeError('Finalist failed exact parity')
        # Exact squared-distance comparison outside each retained cavity.
        for c in holes:
            assert sum((Q(float(a))-Q(float(b)))**2 for a,b in zip(z,c))>Q(float(r))**2
        samples,meta=run_oracle(sf,'validation',mesh,z[None],holes,r,kind,'validate',65536,CONFIG['seed']+100*step+2)
        assert meta['containment_failures']==0
        C=CONFIG['value_caps'][kind]
        assert np.all((samples.value>=0)&(samples.value<=C))
        stages=[]
        for st in meta['stages']:
            up=lambda x:float(np.nextafter(x,np.inf))
            dn=lambda x:float(np.nextafter(x,-np.inf))
            margin=up(C*up(math.sqrt(up(CONFIG['confidence_log_upper']/(2*st['n'])))))
            # Exact dyadic summation of stored binary64 samples, then round down.
            exact_sum=sum((Q(float(w)) for w in samples.value.values[:st['n']]),Q(0))
            mean_lower=dn(float(exact_sum/st['n']))
            L=max(0.,dn(mean_lower-margin))
            bound,geometry=benefit(L,z,r,st['center_clearance_lower'],kind)
            source_only,_=benefit(0.,z,r,st['center_clearance_lower'],kind)
            stages.append(dict(**st,sampling_margin=margin,state_lower=L,benefit_lower=bound,
                source_only_lower=source_only,geometry=geometry,passes=bound>=target))
        best=max(stages,key=lambda x:x['benefit_lower'])
        accepted=best['benefit_lower']>=target
        old_holes=holes.copy()
        if accepted:holes=np.vstack([holes,z])
        # Measure the proposed finite insertion even when rejected. The reference
        # does not feed back into selection or acceptance.
        trial=np.vstack([old_holes,z])
        trial_ref=reference(kind,mesh,V,F,trial,r,sf/'trial')
        delta=np.asarray(previous['blocks'])-np.asarray(trial_ref['blocks'])
        record=dict(step=step,candidate_index=idx,center=z.tolist(),radius=r,
            accepted=bool(accepted),target_absolute=target,fixed_scale=CONFIG['fixed_scales'][kind],
            benefit_lower=best['benefit_lower'],source_only_lower=best['source_only_lower'],
            first_passing_n=next((s['n'] for s in stages if s['passes']),None),stages=stages,
            validation_seconds=meta['stages'][-1]['elapsed_seconds'],
            validation_build_seconds=meta['distance_tree_build_seconds'],
            validation_process_seconds=meta['process_seconds'],
            selection_seconds=scmeta['runtime_seconds_median']+re_meta['runtime_seconds_median'],
            exact_initial_parity=parity,containment_failures=meta['containment_failures'],
            measured_trial_benefit=float(delta.mean()),measured_trial_benefit_se=float(delta.std(ddof=1)/2),
            reference_before=previous,reference_trial=trial_ref,
            reference_after=trial_ref if accepted else previous)
        save(final,record);records.append(record);attempted.append(idx)
        if accepted:previous=trial_ref
        print(kind,step,'accept',accepted,'bound',best['benefit_lower'],'measured',delta.mean(),
              'validation_s',record['validation_seconds'],flush=True)
    result=dict(kind=kind,mesh=str(mesh.relative_to(ROOT)),mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
        radius=r,initial_reference=initial,final_reference=previous,accepted_count=len(holes),centers=holes.tolist(),
        sum_accepted_benefit_lower=sum(x['benefit_lower'] for x in records if x['accepted']),
        validation_seconds=sum(x['validation_seconds'] for x in records),
        validation_process_seconds=sum(x['validation_process_seconds'] for x in records),
        selection_seconds=sum(x['selection_seconds'] for x in records),records=records)
    save(folder/'result.json',result);xyz(folder/'accepted-centers.csv',holes)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--geometry',choices=['both','bunny','housing'],default='both');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'configuration.json'
    if path.exists():assert json.loads(path.read_text())==CONFIG,'Retained protocol differs'
    else:save(path,CONFIG)
    save(OUT/'provenance.json',dict(validator_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(),
        fast_solver_sha256=hashlib.sha256(FAST.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for kind in ['bunny','housing'] if args.geometry=='both' else [args.geometry]:design(kind)

if __name__=='__main__':main()
