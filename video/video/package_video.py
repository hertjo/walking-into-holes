"""Package the exact rendering sources and minimal numerical/geometry inputs."""
from pathlib import Path
import json,zipfile,os
ROOT=Path(__file__).resolve().parents[1];OUT=Path(os.environ.get('TWOS_VIDEO_OUTPUT',str(ROOT/'video/output')));OUT.mkdir(parents=True,exist_ok=True)
segments=[(0,7,'Where should a new cooling cavity go?\nPredict the benefit before making the hole.'),(7,12,'Switch on the same heat source in both designs.\nThe exterior and cavity walls stay cold.'),(12,19,'An auxiliary heat-equation simulation shows the difference.\nTWoS optimizes the steady-state design.'),(19,23,'Separate 3D examples: a capsule and the Bunny.\nEvery walk sphere stays inside its domain.'),(23,27,'Sample uniform sphere exits.\nThe lines show jumps, not Brownian paths.'),(27,31,'Repeat with independent random seeds.\nHere the compliance adjoint is p = u.'),(31,35,'Temperature and objective influence predict\nthe effect of a small new cavity.'),(35,39,'Keep one temperature estimate.\nHold out its matching adjoint and average the others.'),(39,43,'Multiply the independent factors, then average.\nThe products themselves can share samples.'),(43,47,'Score feasible sites in the current geometry.\nRescore a shortlist with fresh samples.'),(47,51,'Insert the selected cavity and update the domain.\nThe cooling animation shows thermal relaxation.'),(51,55,'The percentages remain the paper’s\nsteady-state benchmark estimates.'),(55,60,'With the same six-hole budget, TWoS gives\n47.8% lower heat objective in this retained run.'),(60,65,'Random placement gives 39.5%;\ngrid-based adjoint placement gives 48.1%.'),(65,70,'Six larger cavities: 47.4% lower heat objective.\nTwelve smaller cavities: 47.1%.'),(70,76,'The twelve smaller cavities remove\nhalf as much material as the six larger ones.'),(76,82,'Walking into Holes.\nMonte Carlo topological derivatives for PDE design.')]
def stamp(t):return f'{int(t)//3600:02d}:{int(t)//60%60:02d}:{int(t)%60:02d},000'
srt='\n\n'.join(f'{i}\n{stamp(a)} --> {stamp(b)}\n{s}' for i,(a,b,s) in enumerate(segments,1))+'\n'
(OUT/'Walking_into_Holes_Hero_Captions.srt').write_text(srt);(ROOT/'video/assets/captions.srt').write_text(srt)
files=set()
for pat in ['video/*.py','video/*.md','video/*.txt','video/assets/**/*','video/preview/layout-*.json','video/preview/contact.png','penrose/video-*','scripts/render-penrose.mjs','scripts/redesign-method-diagrams.py','figures/video-*.svg','package.json','package-lock.json']:
 files.update(p for p in ROOT.glob(pat) if p.is_file())
for tag in ['input','ours0','ours1','ours3','ours6','random','voxel','budget6','budget12']:
 files.update((ROOT/'figures/hero-assets').glob(f'{tag}-*.ply'))
for tag in ['initial','ours1','ours3','ours6','random','voxel','budget6','budget12']:
 files.add(ROOT/f'figures/hero-assets/thermal/{tag}.npz')
for name in ['budget-render-manifest.json','render-manifest.json','thermal/manifest.json','thermal/validation.json']:
 files.add(ROOT/'figures/hero-assets'/name)
for name in ['stanford-bunny-watertight.ply','design-trajectories.csv','selected-centers.csv','summary.json']:
 files.add(ROOT/'benchmarks/stanford_bunny/output'/name)
for units,n in [(6,6),(3,12)]:
 for seed in [2026091501,2026091502,2026091503]:
  files.add(ROOT/f'benchmarks/bunny_budget/output/v{units}-n{n:02d}-s{seed}/twos/result.json')
# Human- and machine-readable integrity inventory, relative to extracted root.
import hashlib
manifest={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}
(ROOT/'video/source-manifest.json').write_text(json.dumps(manifest,indent=2));files.add(ROOT/'video/source-manifest.json')
out=OUT/'Walking_into_Holes_Hero_Video_Source.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(files):z.write(p,Path('walking-into-holes-video')/p.relative_to(ROOT))
print(len(files),'files;',out.stat().st_size,'bytes;',out)
