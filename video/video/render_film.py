"""Render independent chapters concurrently, concatenate, and verify both exports."""
from pathlib import Path
import os,sys,subprocess,time,json,hashlib,shutil
V=Path(__file__).parent;R=V.parent
OUT=Path(os.environ.get('TWOS_VIDEO_OUTPUT',str(V/'output')));OUT.mkdir(parents=True,exist_ok=True)
WORK=OUT/'render-parts';WORK.mkdir(exist_ok=True)
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LP_NUM_THREADS='4')
parts=[];processes=[]
for i,(start,end) in enumerate([(0,19),(19,43),(43,65),(65,82)]):
 path=WORK/f'part-{i}.mp4';log=open(WORK/f'part-{i}.log','w')
 p=subprocess.Popen([sys.executable,str(V/'make_video.py'),'--start',str(start),'--end',str(end),'--out',str(path)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
 parts.append(path);processes.append((p,log,start,end))
while any(p.poll() is None for p,_,_,_ in processes):
 print('Chapters complete:',sum(p.poll()==0 for p,_,_,_ in processes),'/',len(processes),flush=True)
 time.sleep(15)
for p,log,start,end in processes:
 log.close()
 if p.returncode:raise RuntimeError(f'Render failed for {start}-{end}; inspect chapter log')
lst=WORK/'concat.txt';lst.write_text(''.join("file '"+p.name+"'\n" for p in parts))
master=OUT/'Walking_into_Holes_Hero_1440p.mp4';share=OUT/'Walking_into_Holes_Hero_LinkedIn_1080p.mp4';cmd=['ffmpeg','-hide_banner','-loglevel','error','-y']
subprocess.run(cmd+['-f','concat','-safe','0','-i',str(lst),'-c','copy','-movflags','+faststart',str(master)],check=True)
subprocess.run(cmd+['-i',str(master),'-vf','scale=1920:1080:flags=lanczos','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',str(share)],check=True)
audit={}
for p in [master,share]:
 data=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)]));s=data['streams'][0];num,den=map(float,s['r_frame_rate'].split('/'));fps=num/den
 assert int(s['nb_frames'])==1968 and fps==24 and abs(float(data['format']['duration'])-82)<.001
 audit[p.name]={'width':s['width'],'height':s['height'],'frames':int(s['nb_frames']),'fps':fps,'seconds':float(data['format']['duration']),'codec':s['codec_name'],'pixel_format':s['pix_fmt'],'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(V/'assets/media-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
for t in [12,17,22,32.5,39,47,50,53,60,71,79]:
 subprocess.run(cmd+['-ss',str(t),'-i',str(master),'-frames:v','1','-update','1',str(V/'preview'/f'encoded-{t}.png')],check=True)
shutil.copyfile(V/'preview/encoded-17.png',OUT/'Walking_into_Holes_Hero_Cover.png')
print(json.dumps(audit,indent=2),flush=True)
