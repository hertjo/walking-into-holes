"""Correct sparse-query worker occupancy, preserving all original numerical data."""
from pathlib import Path
import json,hashlib,subprocess,sys,numpy as np,pandas as pd
import run_study as a

def main():
 subprocess.run([sys.executable,str(a.HERE/'build_backends.py'),'--sparse-only'],check=True)
 original=a.BIN;a.BIN=Path('/tmp/wih-sparse-build/wih_sparse_query')
 for name in a.SPECS:
  folder=a.OUT/'estimators'/name;path=folder/'results.json';result=json.loads(path.read_text())
  if 'twos_sparse_timing_repeats' in result:continue
  p=np.loadtxt(folder/'sites.txt');sol=a.Solver(name,folder/'sparse-timing-raw');records=[]
  sol.call('geometry-warmup',p[:1],walks=2,seed=38101,epsilon=.0001)
  for walks in [256,2048]:
   for rep in range(16):
    f,k=sol.call(f'twos-{walks}-{rep}',p,radius=.06,walks=walks,seed=731200+rep,epsilon=.0001)
    old=pd.read_csv(folder/'raw'/f'twos-0.06-{walks}-{rep}'/'field.csv')
    fields=['u','p','score','same_index'];assert np.array_equal(f[fields].to_numpy(),old[fields].to_numpy()),(name,walks,rep)
    records.append({'walks':walks,'rep':rep,'seconds':k['request_wall_seconds'],'bitwise_field_match':True})
  sol.close();result['twos_sparse_timing_repeats']=records;a.dump(path,result);print('SPARSE TIMING REPLAY',name,flush=True)
 a.dump(a.OUT/'sparse-timing-protocol.json',{'scope':'Eight-site estimator timing only; full-design measurements unchanged','original_chunk_size':4,'query_chunk_size':1,'configured_OpenMP_threads':4,'reason':'Eight sites with chunk size four schedule only two chunks. Chunk size one permits all four workers.','all_replayed_fields':'bitwise identical to retained original outputs','original_timings':'retained in estimator rows; corrected timing repeats stored separately','binary_sha256':hashlib.sha256(a.BIN.read_bytes()).hexdigest()})
 a.BIN=original
if __name__=='__main__':main()
