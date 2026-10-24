"""Full sequential designs with a retained outer-mesh acceleration structure.

Same optimization decisions and sampling budgets as the primary experiment.
Every algorithm gets a fresh process for each complete design. No requests
are replayed ahead of the actual adaptive choice. Run after other timings.
"""
import os,time,json,hashlib,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import run_benchmark as b

REFERENCE=b.OUT
OUT=Path(os.environ.get('WIH_PERSISTENT_OUTPUT',str(b.HERE/'persistent_output')))
EXE=Path('/tmp/wih-adjoint-server-build/wih_adjoint')
REAL_RUN=subprocess.run
session=None
session_log=None
context_folder=None

def request(cmd,**kwargs):
    global session,session_log
    if str(cmd[0])!=str(EXE):return REAL_RUN(cmd,**kwargs)
    if session is None:
        session_log=(context_folder/'session.log').open('w')
        session=subprocess.Popen([str(EXE),'--serve'],stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=session_log,text=True,bufsize=1,env=kwargs.get('env'))
    assert all('\n' not in a for a in cmd)
    session.stdin.write(str(len(cmd))+'\n'+'\n'.join(cmd)+'\n');session.stdin.flush()
    reply=session.stdout.readline().strip()
    if reply!='DONE 0':raise RuntimeError(f'persistent backend returned {reply!r}; inspect {context_folder}')
    return SimpleNamespace(returncode=0,stdout='',stderr='')

def close():
    global session,session_log
    if session is not None:
        session.stdin.close();code=session.wait(timeout=60);session.stdout.close();session_log.close()
        session=None;session_log=None
        if code:raise RuntimeError(f'server exit status {code}')

ORIGINAL_SOLVER=b.solver
def solve(tag,*args,**kwargs):
    if '/mg57/' in tag:kwargs['grid']=[57,37,29]
    return ORIGINAL_SOLVER(tag,*args,**kwargs)

def main():
    global context_folder
    OUT.mkdir(parents=True,exist_ok=True)
    b.OUT=OUT;b.EXE=EXE;b.solver=solve;subprocess.run=request
    methods=b.METHODS+['mg57']
    protocol={'basis':b.CONFIG,'methods':methods,'mode':'one fresh retained-mesh server per full design',
       'reference_reuse':'only after exact final-geometry and per-round decision agreement',
       'timing':'complete adaptive driver including server startup/shutdown, plus measured common candidate setup'}
    if not (OUT/'configuration.json').exists():
        b.save(OUT/'configuration.json',protocol)
        b.save(OUT/'provenance.json',{'executable_sha256':hashlib.sha256(EXE.read_bytes()).hexdigest(),
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [Path(__file__),b.HERE/'adjoint.cpp',b.HERE/'run_benchmark.py']}})
    else:assert json.loads((OUT/'configuration.json').read_text())==protocol
    refs=pd.concat([pd.read_csv(REFERENCE/'results.csv'),pd.read_csv(REFERENCE/'grid_sweep.csv')])
    rows=[];checks=[]
    try:
      for side in b.CONFIG['sides']:
        for seed in b.CONFIG['seeds']:
            setup=OUT/f'side{side}/seed{seed}/candidate_setup.json'
            if setup.exists():info=json.loads(setup.read_text());points=np.array(info['points'])
            else:
                tick=time.perf_counter();points=b.candidates(seed,side)
                info={'seconds':time.perf_counter()-tick,'points':points.tolist()};b.save(setup,info)
            old=np.array(json.loads((REFERENCE/f'side{side}/seed{seed}/candidate_setup.json').read_text())['points'])
            assert np.array_equal(points,old)
            for method in np.random.default_rng(seed+741).permutation(methods):
                context_folder=OUT/f'side{side}/seed{seed}/{method}';path=context_folder/'design.json'
                if not path.exists() and context_folder.exists() and list(context_folder.rglob('summary.json')):
                    raise RuntimeError(f'incomplete timed design; preserve and restart this exact directory: {context_folder}')
                if path.exists():d=json.loads(path.read_text())
                else:
                    context_folder.mkdir(parents=True,exist_ok=True)
                    tick=time.perf_counter();d=b.design(side,seed,str(method),points);close()
                    d['driver_seconds']=time.perf_counter()-tick;b.save(path,d)
                original=json.loads((REFERENCE/f'side{side}/seed{seed}/{method}/design.json').read_text())
                assert np.array_equal(np.asarray(d['holes']),np.asarray(original['holes'])),(side,seed,method)
                checks.append({'side':side,'seed':seed,'method':method,'all_six_sequential_centers_identical':True})
                ref=refs[(refs.side==side)&(refs.seed==seed)&(refs.method==method)].iloc[0]
                row={k:v for k,v in d.items() if k not in ['holes','rounds']}
                row.update(candidate_setup_seconds=info['seconds'],complete_design_seconds=d['driver_seconds']+info['seconds'],
                    reduction=float(ref.reduction),grid_reduction=float(ref.grid_reduction),
                    one_shot_driver_seconds=float(ref.driver_seconds),
                    reference_wall_seconds=float(ref.reference_wall_seconds))
                rows.append(row);pd.DataFrame(rows).to_csv(OUT/'results.csv',index=False)
                b.save(OUT/'geometry_equivalence.json',checks)
                print('persistent',side,seed,method,round(row['complete_design_seconds'],3),flush=True)
    finally:close();subprocess.run=REAL_RUN
    assert len(rows)==70

if __name__=='__main__':main()
