"""Archive implementation provenance and check retained numerical artifacts."""
from pathlib import Path
import hashlib,json,platform,os,importlib.metadata
import numpy as np
import pandas as pd
import run_benchmark as b

def main():
    legacy=json.loads((b.OUT/'provenance.json').read_text())
    for name in ['adjoint.cpp','run_benchmark.py']:
        p=b.OUT/'implementation'/name
        expected=legacy['sources'][str((b.HERE/name).relative_to(b.ROOT))]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    root=b.HERE/'persistent_output'
    provenance=json.loads((root/'provenance.json').read_text())
    for name in ['adjoint.cpp','run_persistent.py']:
        p=root/'implementation'/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==provenance['source_sha256'][name]
    assert len(json.loads((root/'geometry_equivalence.json').read_text()))==70
    frame=pd.read_csv(root/'results.csv');assert len(frame)==70
    assert np.isfinite(frame.select_dtypes('number').to_numpy()).all()
    assert (frame.complete_design_seconds>=frame.driver_seconds).all()
    for folder in root.glob('side*/seed*/*'):
        if not (folder/'design.json').exists():continue
        d=json.loads((folder/'design.json').read_text())
        original=json.loads((b.OUT/f'side{d["side"]}/seed{d["seed"]}/{d["method"]}/design.json').read_text())
        assert np.array_equal(d['holes'],original['holes'])
        for summary in folder.rglob('summary.json'):
            info=json.loads(summary.read_text())
            assert info['caps']==0 and info['relative_residual']<=1e-8
    hardware={'platform':platform.platform(),'logical_cpus':os.cpu_count(),
       'allowed_cpus':sorted(os.sched_getaffinity(0)),
       'python':platform.python_version(),'packages':{name:importlib.metadata.version(name)
          for name in ['numpy','scipy','pandas','matplotlib','libigl','sympy']}}
    b.save(b.OUT/'final_audit.json',{'legacy_snapshots_match':True,'persistent_snapshots_match':True,
       'primary_and_sweep_designs':90,'persistent_designs':70,
       'all_persistent_geometries_identical':True,'all_persistent_mg_residuals_pass':True,
       'hardware':hardware,'scope':'numerical/provenance audit; no independent expert proof review'})

if __name__=='__main__':main()
