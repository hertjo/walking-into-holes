from pathlib import Path
import json,zipfile,sys
H=Path(__file__).resolve().parent;R=H.parents[1];O=H/'output'
out=Path(sys.argv[1]);files=set(p for p in H.iterdir() if p.is_file())
files.update(p for p in (H/'binaries').rglob('*') if p.is_file())
for p in O.glob('*.json'):files.add(p)
for p in O.glob('*.csv'):files.add(p)
for p in O.glob('**/adaptive-*.json'):files.add(p)
for p in (O/'holdout').glob('*/summary.json'):files.update(x for x in p.parent.rglob('*') if x.is_file())
for folder in ['stream-replay','stream-validation']:
 files.update(p for p in (O/folder).rglob('*') if p.is_file())
for name in ['housing','branch','ring','bunny']:
 for n in ['sites.txt','hole.txt']:
  if (O/name/n).exists():files.add(O/name/n)
for p in O.glob('**/done.json'):
 files.update(x for x in p.parent.rglob('*') if x.is_file())
S=R/'benchmarks/submission_study'
for p in S.glob('*.cpp'):files.add(p)
for n in ['CMakeLists.txt','geometry.py','run_study.py','build_backends.py']:files.add(S/n)
for name in ['housing','branch','ring','bunny']:
 for suffix in ['ply','txt']:
  p=S/'assets'/f'{name}.{suffix}'
  if p.exists():files.add(p)
files.update(p for p in (R/'benchmarks/compiled_bunny/src').rglob('*') if p.is_file())
files.add(R/'benchmarks/engineering_housing/run_housing_benchmark.py')
for p in (R/'figures').glob('matched-accuracy-*'):files.add(p)
for p in (R/'sections').glob('2[23]_*.tex'):files.add(p)
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(files):z.write(p,Path('walking-into-holes-matched-accuracy')/p.relative_to(R))
 z.writestr('walking-into-holes-matched-accuracy/STATUS.json',json.dumps({'completed_solver_runs':sum('legacy-streams' not in p.parts for p in O.glob('**/*-seed*/done.json')),'original_complete_references':len(list(O.glob('*/reference/done.json'))),'full_audit_complete':(O/'audit.json').exists()},indent=2))
print(out.stat().st_size,len(files))
