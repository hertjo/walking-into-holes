"""Build the study backends; keep machine-specific dependency paths out of code."""
import os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 cmake=shutil.which('cmake') or str(Path.home()/'.local/bin/cmake')
 embree=os.environ.get('embree_DIR') or os.environ.get('EMBREE_DIR')
 if not embree:
  roots=[Path(sys.prefix),Path.home()/'.local/lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages']
  choices=[p for root in roots for pattern in ['lib64/cmake/embree-*','lib/cmake/embree-*'] for p in root.glob(pattern)]
  if choices:embree=str(sorted(choices)[-1])
 targets=[('/tmp/wih-study-build','wih_study'),('/tmp/wih-hit-build','wih_hit_difference'),('/tmp/wih-sparse-build','wih_sparse_query')]
 if '--sparse-only' in sys.argv:targets=targets[-1:]
 for directory,target in targets:
  args=[cmake,'-S',str(HERE),'-B',directory,'-DCMAKE_BUILD_TYPE=Release']
  if embree:args+=['-Dembree_DIR='+embree]
  subprocess.run(args,check=True)
  subprocess.run([cmake,'--build',directory,'--target',target,'-j','4'],check=True)
if __name__=='__main__':main()
