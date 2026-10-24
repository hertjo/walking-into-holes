from pathlib import Path
import json,hashlib
import numpy as np
from render_scene import ROOT,read_poly,vtk
V=Path(__file__).parent;E=json.loads((V/'assets/evidence.json').read_text());T=json.loads((V/'assets/walk-traces.json').read_text())
p=ROOT/'benchmarks/stanford_bunny/output/stanford-bunny-watertight.ply';assert hashlib.sha256(p.read_bytes()).hexdigest()==E['mesh_sha256']
d=vtk.vtkImplicitPolyDataDistance();d.SetInput(read_poly(p));issues=[];n=0
for trace in T['traces']:
 for step in trace['steps']:
  x=np.array(step['x']);y=np.array(step['next']);r=step['radius'];sd=d.EvaluateFunction(x)
  assert sd<0 and r<abs(sd) and np.isclose(np.linalg.norm(y-x),r)
  assert d.EvaluateFunction(y)<0
  n+=1
 assert trace['termination_distance']<T['terminal_epsilon']
for p in (V/'preview').glob('layout-*.json'):
 a=json.loads(p.read_text())
 for x in a:
  b=x['bbox']
  if b[0]<80 or b[2]>2480 or b[1]<0 or b[3]>1430:issues.append((p.name,'bounds',x))
 for i,x in enumerate(a):
  for y in a[i+1:]:
   b=x['bbox'];c=y['bbox']
   if min(b[2],c[2])-max(b[0],c[0])>2 and min(b[3],c[3])-max(b[1],c[1])>2:issues.append((p.name,'overlap',x['text'],y['text']))
a,b=E['budget_designs']['designs'];assert np.isclose(b['removed_volume']/a['removed_volume'],.5)
assert round(E['comparison']['TWoS + fresh validation']['cooling_percent'][-1],1)==47.8
assert not issues,issues
result={'walk_steps_checked':n,'all_centers_and_jumps_inside_original_mesh':True,'all_spheres_strictly_contained':True,'all_walks_terminated_below_epsilon':True,'preview_layouts_checked':len(list((V/'preview').glob('layout-*.json'))),'text_clashes':issues,'half_volume_verified':True,'scientific_scope':'This is an asset/geometry/layout audit, not an independent revalidation of the paper or a new statistical experiment.'}
(V/'assets/audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
