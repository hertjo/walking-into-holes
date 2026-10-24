"""Stronger rare-event baseline, separate binary/provenance, serial measured run."""
import json,hashlib
import numpy as np
import run_study as a
from pathlib import Path
a.BIN=Path('/tmp/wih-hit-build/wih_hit_difference')
for name in a.SPECS:
 folder=a.OUT/'estimators'/name;result=json.loads((folder/'results.json').read_text())
 if any(r['method']=='hit-correction' for r in result['rows']):continue
 sol=a.Solver(name,folder/'hit-raw');p=np.loadtxt(folder/'sites.txt');new=[]
 sol.call('warmup',p[:1],radius=.04,walks=2,seed=119,epsilon=.0001)
 for radius in [.08,.04,.02,.01]:
  for walks in [256,2048]:
   for rep in range(16):
    f,meta=sol.call(f'hit-{radius}-{walks}-{rep}',p,radius=radius,walks=walks,seed=731200+rep,epsilon=.0001)
    for i,v in enumerate(f.score):new.append(dict(method='hit-correction',rep=rep,site=i,radius=radius,walks=walks,score=v,seconds=meta['request_wall_seconds'],queries=meta['distance_queries'],parity_rays=meta['parity_rays'],rss=meta['peak_rss_megabytes'],hits=int(f.hits.iloc[i])))
 sol.close();result['rows']+=new;a.dump(folder/'results.json',result);print('HIT CORRECTION',name,flush=True)
a.dump(a.OUT/'hit-correction-protocol.json',{'representation':'Exact harmonic correction for finite removal, estimated by a perforated-domain exit walk and an independent original-domain state walk only on a new-cavity hit.','numerical_limits':'finite epsilon, ideal sampling, geometry approximation','binary_sha256':hashlib.sha256(a.BIN.read_bytes()).hexdigest(),'radii':[.08,.04,.02,.01],'walks':[256,2048],'repeats':16,'geometries':list(a.SPECS)})
# A single deterministic field is sufficient, but one wall-clock sample is not.
a.BIN=Path('/tmp/wih-study-build/wih_study')
from run_mg_timings import main as repeat_mg_timings
repeat_mg_timings()
from run_reference_refinement import main as refine_references
refine_references()
from run_sparse_timings import main as repeat_sparse_timings
repeat_sparse_timings()
