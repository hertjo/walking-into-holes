"""Build missing backends without replacing retained benchmark executables."""
from pathlib import Path
import subprocess,sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'benchmarks/compiled_bunny'
sys.path.insert(0,str(SOURCE))
from run_compiled_benchmark import cmake_binary,embree_cmake_dir

def main():
    for directory in [Path('/tmp/wih-certification-build'),Path('/tmp/walking-into-holes-embree-build')]:
        if (directory/'wih_twos').is_file():
            print('Using existing backend:',directory/'wih_twos');continue
        subprocess.run([cmake_binary(),'-S',str(SOURCE),'-B',str(directory),
            '-DCMAKE_BUILD_TYPE=Release',f'-Dembree_DIR={embree_cmake_dir()}'],check=True)
        subprocess.run([cmake_binary(),'--build',str(directory),'--parallel','2'],check=True)

if __name__=='__main__':main()
