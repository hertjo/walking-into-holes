"""Pointwise bootstrap intervals from saved independent units.

Optimization: resample paired design seeds and shared evaluation blocks.
Estimator: resample repetition vectors and independent reference repetitions.
These are descriptive small-sample intervals, not certified or simultaneous bounds.
"""
import json
import numpy as np
from run_study import OUT, SEEDS
NBOOT = 4000
BOOT_SEED = 824301

def limits(x):
 return tuple(float(v) for v in np.quantile(x,[.025,.975]))

def design_intervals(df):
 records=[]
 rng=np.random.default_rng(BOOT_SEED)
 si=rng.integers(0,len(SEEDS),(NBOOT,len(SEEDS)))
 bi=rng.integers(0,4,(NBOOT,4))
 for (study,name),part in df.groupby(['study','geometry']):
  ev=[json.loads((OUT/study/name/'evaluation'/str(seed)/'evaluation.json').read_text()) for seed in SEEDS]
  base=np.array([x['unperforated'] for x in ev])
  base_boot=base[si[:,:,None],bi[:,None,:]].mean(axis=2)
  for (method,state),g in part.groupby(['method','state']):
   vals=np.array([x[f'{method}-{state}'] for x in ev])
   val_boot=vals[si[:,:,None],bi[:,None,:]].mean(axis=2)
   quality=(100*(1-val_boot/base_boot)).mean(axis=1)
   times=g.set_index('seed').loc[SEEDS,'seconds'].to_numpy()
   qlo,qhi=limits(quality);tlo,thi=limits(times[si].mean(axis=1))
   records.append(dict(study=study,geometry=name,method=method,state=state,
    cooling_ci95_low=qlo,cooling_ci95_high=qhi,seconds_ci95_low=tlo,seconds_ci95_high=thi))
 return records

def estimator_intervals(samples,reference,times):
 """Keep the eight sites together; bootstrap only independent repetitions.

 For deterministic multigrid, uncertainty comes only from the MC reference;
 no Monte Carlo field variance is inferred. Runtime repeats are independent
 of this deterministic field output.
 """
 samples=np.asarray(samples);reference=np.asarray(reference);times=np.asarray(times)
 rng=np.random.default_rng(BOOT_SEED)
 ri=rng.integers(0,len(reference),(NBOOT,len(reference)))
 ref=reference[ri].mean(axis=1);den=np.sum(ref**2,axis=1)
 ix=rng.integers(0,len(samples),(NBOOT,len(samples)))
 boot=samples[ix]
 rmse=np.sqrt(np.mean(np.sum((boot-ref[:,None,:])**2,axis=2),axis=1)/den)
 lo,hi=limits(rmse)
 out={'rmse_ci95_low':lo,'rmse_ci95_high':hi}
 if len(samples)>1:
  var=np.sum(boot.var(axis=1,ddof=1),axis=1)/den
  lo,hi=limits(var);out.update(variance_ci95_low=lo,variance_ci95_high=hi)
  lo,hi=limits(times[ix].mean(axis=1));out.update(seconds_ci95_low=lo,seconds_ci95_high=hi)
 else:
  out.update(variance_ci95_low=np.nan,variance_ci95_high=np.nan,seconds_ci95_low=np.nan,seconds_ci95_high=np.nan)
 if len(samples)==1 and len(times)>1:
  ti=rng.integers(0,len(times),(NBOOT,len(times)));lo,hi=limits(times[ti].mean(axis=1))
  out.update(seconds_ci95_low=lo,seconds_ci95_high=hi)
 if len(samples)>1 and np.all(samples==0):
  # A percentile bootstrap cannot discover unsampled rare corrections.
  # Do not present collapsed zero-output intervals as resolved uncertainty.
  out.update(rmse_ci95_low=np.nan,rmse_ci95_high=np.nan,
             variance_ci95_low=np.nan,variance_ci95_high=np.nan)
 return out
