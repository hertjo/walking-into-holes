"""Raw-data audit and paired summaries; never assumes a favorable result."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr,t
from reference_audit import reference_audit
from tables_study import tables
from report_results import report
from uncertainty import design_intervals,estimator_intervals,NBOOT,BOOT_SEED
from run_study import HERE,OUT,SPECS,SEEDS,R,load,valid,channel_balls,dump

def summarize():
 rows=[];checks=[]
 for study in ['pipeline','channels']:
  for name in SPECS:
   s,m=load(name)
   for seed in SEEDS:
    ev=json.loads((OUT/study/name/'evaluation'/str(seed)/'evaluation.json').read_text());base=np.array(ev['unperforated']);assert np.all(base>0)
    for method in ['twos','mg','mg49','nearest','hot','random']:
     d=json.loads((OUT/study/name/f'{method}-{seed}'/'design.json').read_text());assert valid(s,m,np.array(d['centers']),study=='channels',d['radius']),(study,name,method,seed,'initial')
     states=[('initial',d['design_seconds'],d['centers'],d['backend_peak_rss_mb'],d['radius'])]
     if study=='pipeline':
      ref=json.loads((OUT/study/name/f'{method}-{seed}'/'refinement.json').read_text())
      for h in ref['history']:
       assert valid(s,m,np.array(h['centers']),r=R),(study,name,method,seed,h['round'])
       if h['round'] in [4,8,16]:states.append((f'refine{h["round"]}',h['seconds'],h['centers'],ref['backend_peak_rss_mb'],R))
     for label,seconds,centers,rss,r in states:
      vals=np.array(ev[f'{method}-{label}']);ratio=1-vals/base
      rows.append({'study':study,'geometry':name,'seed':seed,'method':method,'state':label,'reduction_pct':100*(1-vals.mean()/base.mean()),'evaluation_block_se_pp':100*ratio.std(ddof=1)/2,'seconds':seconds,'backend_peak_rss_mb':rss,'radius':r,'objective':vals.mean(),'unperforated_objective':base.mean()})
     if study=='channels':
      v=np.array(ev[f'{method}-volume']);err=100*(v.mean()/.0168-1)
      checks.append({'geometry':name,'method':method,'seed':seed,'volume':v.mean(),'volume_relative_error_pct':err,'scramble_se_volume':v.std(ddof=1)/2,'radius':d['radius'],'tube_core_radius':np.sqrt(d['radius']**2-.02**2),'heat_input_exact':sum(h[4]*32*np.pi*h[3]**3/105 for h in s['heat'])})
      assert d['radius']>.02
      for c in [s['sensor']]+s['heat']:
       assert np.min(np.linalg.norm(np.array(d['centers'])[:,:2]-np.array(c)[:2],axis=1))>c[3]+d['radius']
      balls=channel_balls(d['centers']);assert balls[:,2].min()<m.bounds[0,2] and balls[:,2].max()>m.bounds[1,2]
 df=pd.DataFrame(rows);df.to_csv(OUT/'design-results.csv',index=False)
 pd.DataFrame(checks).to_csv(OUT/'volume-audit.csv',index=False)
 group=df.groupby(['study','geometry','method','state']).agg(radius=('radius','mean'),reduction_pct=('reduction_pct','mean'),seed_sd_pp=('reduction_pct','std'),seconds=('seconds','mean'),backend_peak_rss_mb=('backend_peak_rss_mb','mean'),evaluation_se_pp=('evaluation_block_se_pp','mean')).reset_index();group=group.merge(pd.DataFrame(design_intervals(df)),on=['study','geometry','method','state'],validate='one_to_one');group.to_csv(OUT/'group-results.csv',index=False)
 paired=[]
 for study in ['pipeline','channels']:
  for name in SPECS:
   states=['initial','refine8'] if study=='pipeline' else ['initial']
   comparisons=[(state,state,b) for state in states for b in ['mg','mg49','nearest','hot','random']]
   if study=='pipeline':comparisons += [('refine8','refine16',b) for b in ['hot','random']]
   for state,bstate,b in comparisons:
    aa=df.query('study==@study and geometry==@name and state==@state and method=="twos"').set_index('seed').loc[SEEDS]
    bb=df.query('study==@study and geometry==@name and state==@bstate and method==@b').set_index('seed').loc[SEEDS]
    x=(aa.reduction_pct-bb.reduction_pct).to_numpy();mean=x.mean();hw=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    blocks=[]
    for seed in SEEDS:
     ev=json.loads((OUT/study/name/'evaluation'/str(seed)/'evaluation.json').read_text())
     blocks.append([ev[f'twos-{state}'],ev[f'{b}-{bstate}'],ev['unperforated']])
    blocks=np.array(blocks);rng=np.random.default_rng(923111);boot=[]
    for _ in range(4000):
     si=rng.integers(0,5,5);bi=rng.integers(0,4,4);v=blocks[si][:,:,bi].mean(axis=2)
     boot.append(float(np.mean(100*(v[:,1]-v[:,0])/v[:,2])))
    blo,bhi=np.quantile(boot,[.025,.975])
    paired.append({'study':study,'geometry':name,'state':state,'baseline_state':bstate,'contrast':f'twos - {b}','paired_difference_pp':mean,'ci95_low':mean-hw,'ci95_high':mean+hw,'n':len(x),'joint_bootstrap_low':float(blo),'joint_bootstrap_high':float(bhi)})
 pd.DataFrame(paired).to_csv(OUT/'paired-results.csv',index=False)
 est=[]
 for name in SPECS:
  raw=json.loads((OUT/'estimators'/name/'results.json').read_text());f=pd.DataFrame(raw['rows']);reference=f[f.method=='reference'].groupby('site').score.mean();den=float(np.sum(reference**2));refse=f[f.method=='reference'].groupby('site').score.sem();ref_samples=f[f.method=='reference'].pivot(index='rep',columns='site',values='score').to_numpy();
  for (method,radius,walks),g in f[f.method!='reference'].groupby(['method','radius','walks']):
   a=g.pivot(index='rep',columns='site',values='score').to_numpy();bias=np.mean(a,axis=0)-reference.to_numpy();mse=np.mean(np.sum((a-reference.to_numpy())**2,axis=1))/den
   var=float(np.sum(np.var(a,axis=0,ddof=1))/den) if len(a)>1 else None
   ranks=[spearmanr(x,reference.to_numpy()).statistic for x in a]
   times=np.array([x['seconds'] for x in raw.get('mg_timing_repeats',[]) if x['grid']==walks]) if method=='mg' else g.groupby('rep').seconds.first().to_numpy()
   if len(times)==0:times=g.groupby('rep').seconds.first().to_numpy()
   if method=='twos' and 'twos_sparse_timing_repeats' in raw:times=np.array([r['seconds'] for r in raw['twos_sparse_timing_repeats'] if r['walks']==walks])
   est.append({'geometry':name,'method':method,'radius':radius,'budget':walks,'relative_rmse':float(np.sqrt(mse)),'relative_bias_norm':float(np.linalg.norm(bias)/np.sqrt(den)),'normalized_variance':var,'nonzero_repeats':int(np.count_nonzero(np.any(a!=0,axis=1))),'variance_status':'unresolved_all_zero' if np.all(a==0) and len(a)>1 else ('empirical' if len(a)>1 else 'deterministic'),'spearman':float(np.nanmean(ranks)),'top1_match':float(np.mean(np.argmax(a,axis=1)==np.argmax(reference.to_numpy()))),'seconds':times.mean(),'distance_queries':g.groupby('rep').queries.first().mean(),'parity_rays':g.groupby('rep').parity_rays.first().mean(),'backend_peak_rss_mb':g.rss.max(),'reference_relative_se_norm':float(np.linalg.norm(refse)/np.sqrt(den)),'reference_zero_sites':int(np.sum(reference.to_numpy()==0)),**estimator_intervals(a,ref_samples,times)})
 pd.DataFrame(est).to_csv(OUT/'estimator-results.csv',index=False)
 # Every completed kernel is required to satisfy its own solver guard.
 kernels=list(OUT.rglob('kernel.json'));caps=0;residual=0;products=0;product_error=0
 for p in kernels:
  k=json.loads(p.read_text());caps+=k.get('caps',0);residual=max(residual,k.get('relative_residual',0));assert k.get('distance_queries',0)>=0
  request=json.loads((p.parent/'request.json').read_text());args=dict(zip(request[1::2],request[2::2]))
  if 'hit-raw' not in p.parts and args.get('--mode')=='twos' and all(args.get(x,'0')=='0' for x in ['--hot-only','--fd','--diff']):
   field=pd.read_csv(p.parent/'field.csv');n=int(args['--walks']);reconstructed=(n*field.u*field.p-field.same_index)/(n-1)
   scale=max(float(np.max(np.abs(n*field.u*field.p))),float(np.max(np.abs(field.same_index))),float(np.max(np.abs(field.score))),1e-30)
   err=float(np.max(np.abs(reconstructed-field.score))/scale);assert err<1e-10,(str(p),err)
   products+=len(field);product_error=max(product_error,err)
 assert caps==0 and residual<=1.001e-7
 manifest={'complete_designs':180,'pipeline_initializations':90,'pipeline_refinement_runs':90,'channel_designs':90,'seed_count':5,'geometries':3,'evaluated_states':len(df),'kernel_count':len(kernels),'capped_paths':caps,'maximum_MG_relative_residual':residual,'cross_product_rows_checked':products,'maximum_scaled_product_reconstruction_error':product_error,'maximum_channel_volume_deviation_pct':max(abs(x['volume_relative_error_pct']) for x in checks),'geometry_constraints':'all saved states checked against triangle meshes and protected supports','volume_scope':'independent randomized-QMC estimate; not a certified interval','timing_scope':'per-design measured costs, reference and visualization excluded','inference_scope':'five paired candidate/optimization seeds; evaluation uncertainty also reported','source_hashes':{str(p.relative_to(HERE.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*HERE.glob('*.py'),*HERE.glob('*.cpp'),HERE/'CMakeLists.txt',HERE.parent/'compiled_bunny/src/wih_twos.cpp']}}
 manifest['complete_designs']=180
 manifest['uncertainty']={'type':'pointwise percentile bootstrap, not simultaneous or certified','replicates':NBOOT,'rng_seed':BOOT_SEED,'optimization_units':'five paired design seeds and four shared independent evaluation blocks','estimator_units':'sixteen independent eight-site repeat vectors and sixteen independent reference batches','runtime_units':'design seeds or estimator repeats; includes algorithm variation and machine timing variation','deterministic_MG':'RMSE interval from reference sampling only; timing interval from five rotated-order repeats when available','limitations':'small samples; numerical bias and termination error are not covered'}
 reference_audit();tables();report();dump(OUT/'audit.json',manifest);print(group.to_string(index=False));print(json.dumps(manifest,indent=2))
if __name__=='__main__':summarize()
