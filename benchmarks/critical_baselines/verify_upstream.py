"""Run with --upstream pointing at pinned hoval58/RQMC-WoS checkout.

Only inject common shifts and the benchmark's boundary function. The authors'
walk/update/sort routine is executed unchanged. No upstream source is bundled.
"""
import argparse,sys,json
import numpy as np
from sampling import fields,walks,OUT

def run(path):
    sys.path.insert(0,path)
    import unit_disk_array_rqmc as ref
    rows=[]
    for n in [64,256]:
      for seed in [14,87]:
        rng=np.random.default_rng(seed)
        def lattice(n,a,seed=None):return (((a*np.arange(n))%n)/n+rng.random())%1
        ref.lattice_korobov=lattice
        ref.boundary_value=lambda xy:float(fields(xy[0]+1j*xy[1])[0])
        z=.3+.2j
        expected,_=ref.wos_unit_disk_array_rqmc([z.real,z.imag],n,qmc_type='lattice',
                                              eps=1e-7,max_steps=256,seed=seed)
        _,last,_,_=walks(np.array([z]),n,seed,'disk','array')
        actual=fields(last)[0].mean()
        rows.append(dict(n=n,seed=seed,upstream=float(expected),ours=float(actual),
                         difference=float(abs(expected-actual))))
        assert abs(expected-actual)<1e-12
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'array_upstream_parity.json').write_text(json.dumps(rows,indent=2))
    print(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--upstream',required=True);run(p.parse_args().upstream)
