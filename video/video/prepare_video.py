"""Prepare reproducible walk traces and an evidence manifest; never alter experiments."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,vtk
from render_scene import ROOT,read_poly
OUT=Path(__file__).parent/'assets';OUT.mkdir(exist_ok=True)
mesh=ROOT/'benchmarks/stanford_bunny/output/stanford-bunny-watertight.ply'
poly=read_poly(mesh);dist=vtk.vtkImplicitPolyDataDistance();dist.SetInput(poly)
design=pd.read_csv(ROOT/'benchmarks/stanford_bunny/output/selected-centers.csv');z=design[design.method=='TWoS + fresh validation'].iloc[0][['x','y','z']].to_numpy(float)
traces=[]
for seed in [2026091801,2026091802,2026091803]:
 rng=np.random.default_rng(seed);x=z.copy();steps=[];value=0
 for k in range(512):
  d=abs(dist.EvaluateFunction(x));r=.9999*d
  if d<.0005:break
  direction=rng.normal(size=3);direction/=np.linalg.norm(direction);y=x+r*direction
  contribution=(1+.65*(x[1]+.5))*r*r/6;value+=contribution
  steps.append(dict(x=x.tolist(),radius=r,next=y.tolist(),contribution=contribution));x=y
 else:raise RuntimeError('walk cap reached')
 traces.append(dict(seed=seed,steps=steps,estimate_u=value,termination_distance=d))
assert max(t['termination_distance'] for t in traces)<.0005
(OUT/'walk-traces.json').write_text(json.dumps({'start':z.tolist(),'p_equals_u':True,'step_radius':'0.9999 * distance to original triangle boundary','exit_sampling':'Independent normalized Gaussian directions, uniform on the 3D sphere','terminal_epsilon':.0005,'scope':'New illustrative sample paths in the actual unperforated Bunny; not the optimization seeds or performance trials.','traces':traces},indent=2))
df=pd.read_csv(ROOT/'benchmarks/stanford_bunny/output/design-trajectories.csv');metrics={}
for n,g in df.groupby('method'):
 metrics[n]={'objective':g.objective.tolist(),'cooling_percent':(100*(1-g.objective/g.iloc[0].objective)).tolist()}
budget=json.loads((ROOT/'figures/hero-assets/budget-render-manifest.json').read_text());ranges={}
for units,n in [(6,6),(3,12)]:
 vals=[]
 for seed in [2026091501,2026091502,2026091503]:
  p=ROOT/f'benchmarks/bunny_budget/output/v{units}-n{n:02d}-s{seed}/twos/result.json';vals.append(json.loads(p.read_text())['cooling_percent'])
 ranges[f'{units}-{n}']={'mean':float(np.mean(vals)),'min':min(vals),'max':max(vals),'n':3,'individuals':vals}
manifest={'title':'Walking into Holes','video':'66 seconds, 2560 x 1440, 24 fps; no audio required','mesh_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest(),'PDE':'-Delta u = 1 + 0.65*(y+0.5), zero Dirichlet temperature on the exterior and every cavity wall','objective':'Source-weighted temperature over fixed original-domain probes, normalized to the intact design','adjoint':'For this compliance example p=u. General objectives require a distinct adjoint.','temperature':'Saved 256-path WoS section estimates, same [0,0.025] color scale; one-cell Gaussian display smoothing only','animation':'Camera motion, exact illustrative WoS traces, scientific diagram emphasis, and transitions between saved steady-state designs. No time-dependent heat or coolant-flow simulation is asserted.','cutaway':'Display-only cylindrical window; removed exterior shell is shown faintly. The window never enters a PDE solve.','cavity_growth':'Insertion transitions are visual edits, not physical growth or certified intermediate-radius designs.','comparison':metrics,'budget_designs':budget,'budget_three_seed_ranges':ranges,'scope':'The six-hole random/TWoS/voxel panels show one retained design realization. Their percentages are not replicated means. The budget pair uses the first prespecified seed, with three-seed ranges explicitly labeled. Grid-based adjoint TD remains competitive; this video makes no general runtime superiority claim.','certificate':'Independent rescoring in this practical Bunny design is not asserted to be the separate finite-insertion certification experiment.'}
(OUT/'evidence.json').write_text(json.dumps(manifest,indent=2))
print('walk steps',[len(t['steps']) for t in traces]);print('metrics',[(n,v['cooling_percent'][-1]) for n,v in metrics.items()]);print('ranges',ranges)
