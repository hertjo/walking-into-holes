"""Resume the full campaign serially, preserving every completed measured design."""
from pathlib import Path
import os,subprocess,sys,datetime
from run_study import HERE,OUT,dump

def main():
 logs=OUT/'logs';logs.mkdir(exist_ok=True,parents=True)
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
 stages=[('build',['build_backends.py']),('primary',['run_study.py','--stage','all']),('additional',['run_coarse.py']),('strong-baseline-and-references',['run_hit_difference.py']),('audit',['analyze_study.py']),('render',['render_study.py']),('compose',['compose_study.py'])]
 for name,args in stages:
  record={'stage':name,'status':'running','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  dump(OUT/'campaign-status.json',record);print('STAGE',name,flush=True)
  with (logs/(name+'.log')).open('a') as log:
   code=subprocess.call([sys.executable,str(HERE/args[0]),*args[1:]],cwd=HERE.parents[1],env=env,stdout=log,stderr=subprocess.STDOUT)
  if code:
   record.update(status='failed',returncode=code);dump(OUT/'campaign-status.json',record)
   raise SystemExit(f'{name} failed; see {logs/(name+".log")}')
 dump(OUT/'campaign-status.json',{'status':'complete','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print('CAMPAIGN COMPLETE; figures still require visual inspection.',flush=True)
if __name__=='__main__':main()
