"""Independent deterministic objective reference; never used for acceptance."""
from pathlib import Path
import json,os,subprocess
import numpy as np
import pandas as pd
import run_application_validation as protocol

def main():
    for kind in ['bunny','housing']:
        folder=protocol.OUT/kind
        result=json.loads((folder/'result.json').read_text())
        mesh=protocol.ROOT/result['mesh'];r=result['radius']
        points=np.vstack([np.loadtxt(folder/f'probes-{b}.csv',delimiter=',') for b in range(4)])
        volume=result['initial_reference']['mesh_volume']
        targets=[('initial',np.empty((0,3)))];retained=[]
        for rec in result['records']:
            trial=np.asarray(retained+[rec['center']]).reshape(-1,3)
            targets.append((f"attempt-{rec['step']:02d}/trial",trial))
            if rec['accepted']:retained.append(rec['center'])
        for tag,holes in targets:
            where=folder/tag;summary=where/'multigrid-reference.json'
            if summary.exists():continue
            active=np.ones(len(points),dtype=bool)
            if len(holes):active=np.min(np.linalg.norm(points[:,None]-holes[None],axis=2),axis=1)>r
            pts=where/'mg-probes.csv';cav=where/'mg-cavities.csv';out=where/'mg-field.csv';meta=where/'mg-solver.json'
            protocol.xyz(pts,points[active]);protocol.xyz(cav,holes)
            grid='97,113,85' if kind=='bunny' else '113,73,57'
            command=[str(protocol.FAST),'--mode','mg','--mesh',str(mesh),'--points',str(pts),
                '--output',str(out),'--summary',str(meta),'--grid',grid,'--threads','8','--repeats','1',
                '--source-mode','affine' if kind=='bunny' else 'housing','--source-y-slope','.65',
                '--mg-tolerance','1e-8','--mg-cycles','100']
            if len(holes):command+=['--cavities',str(cav),'--cavity-radius',str(r)]
            proc=subprocess.run(command,capture_output=True,text=True,
                env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1'))
            (where/'mg-reference.log').write_text(proc.stdout+proc.stderr)
            if proc.returncode:raise RuntimeError(proc.stderr)
            field=pd.read_csv(out,float_precision='round_trip')
            p=points[active]
            if kind=='bunny':source=1+.65*(p[:,1]+.5)
            else:source=.08+(protocol.AMPL*np.exp(-np.sum(((p[:,None]-protocol.CENTERS)/protocol.WIDTHS)**2,axis=2))).sum(axis=1)
            values=np.zeros(len(points));values[active]=source*np.sqrt(np.maximum(0.,field.score.values))
            blocks=volume*values.reshape(4,512).mean(axis=1)
            protocol.save(summary,dict(grid=grid,blocks=blocks.tolist(),mean=float(blocks.mean()),
                scope='Voxelized multigrid approximation on the same independent probes; no continuum error certificate.'))
        print(kind,'multigrid references complete',flush=True)

if __name__=='__main__':main()
