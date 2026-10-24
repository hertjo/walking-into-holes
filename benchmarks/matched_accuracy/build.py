"""Build portable source targets; locate the Embree SDK supplied by embreex."""
from pathlib import Path
import sys,os,shutil,subprocess
H=Path(__file__).resolve().parent
cmake=shutil.which('cmake') or str(Path.home()/'.local/bin/cmake')
embree=os.environ.get('embree_DIR') or os.environ.get('EMBREE_DIR')
if not embree:
 for root in [Path(sys.prefix),Path.home()/'.local/lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages']:
  for pattern in ['lib64/cmake/embree-*','lib/cmake/embree-*']:
   matches=sorted(root.glob(pattern))
   if matches:embree=str(matches[-1])
args=[cmake,'-S',str(H.parent/'submission_study'),'-B','/tmp/wih-reuse-build','-DCMAKE_BUILD_TYPE=Release']
if embree:args+=['-Dembree_DIR='+embree]
subprocess.run(args,check=True)
targets=['wih_incremental_query','wih_analytic_query','wih_analytic_integral_audit','wih_compliance_query']
if '--repair-only' in sys.argv:targets=[f'wih_{kind}_{family}' for family in ['raw','analytic','compliance'] for kind in ['repair','clean']]
elif '--extensions-only' not in sys.argv:targets=['wih_reuse_query']+targets
subprocess.run([cmake,'--build','/tmp/wih-reuse-build','--target',*targets,'-j','4'],check=True)
