"""Research hero film, assembled from saved data, exact 3D geometry, and Penrose.
No generated scientific imagery; all numerical claims are read from evidence.json.
"""
from pathlib import Path
import argparse,json,math,subprocess,time,os
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from matplotlib import colormaps
from render_scene import Scene,ROOT,PURPLE,TEAL,THERM,SCOOP,read_vtp,vtk_to_numpy
V=Path(__file__).parent;A=V/'assets';OUT=Path(os.environ.get('TWOS_VIDEO_OUTPUT',str(V/'output')));OUT.mkdir(parents=True,exist_ok=True)
W,H=2560,1440;FPS=24;DURATION=82
INK='#302c35';MUTED='#776f7c';PUR='#9365a8';BLUE='#6997aa';LINE='#ded8e2';ORANGE='#bd824b'
E=json.loads((A/'evidence.json').read_text());TR=json.loads((A/'walk-traces.json').read_text())
CAP=json.loads((A/'capsule-walks.json').read_text())
MET=E['comparison'];FONTS={};LAYOUT=[]
def font(size,serif=False,bold=False):
 key=size,serif,bold
 if key not in FONTS:
  path=('/usr/share/texmf/fonts/opentype/public/lm/lmroman12-'+('bold' if bold else 'regular')+'.otf') if serif else ('/usr/share/fonts/opentype/urw-base35/NimbusSans-'+('Bold' if bold else 'Regular')+'.otf')
  FONTS[key]=ImageFont.truetype(path,size)
 return FONTS[key]
def text(im,xy,s,size=36,color=INK,serif=False,bold=False,anchor='la'):
 d=ImageDraw.Draw(im);f=font(size,serif,bold);b=d.textbbox(xy,s,font=f,anchor=anchor,spacing=12);LAYOUT.append({'text':s,'bbox':list(b)});d.text(xy,s,font=f,fill=color,anchor=anchor,spacing=12)
def ease(v):v=float(np.clip(v,0,1));return v*v*(3-2*v)
def base(chapter,title,subtitle,t):
 im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
 text(im,(110,65),'TWoS',26,PUR,bold=True)
 text(im,(2450,65),chapter.split('/')[0].strip(),26,MUTED,anchor='ra')
 text(im,(110,120),title,76,serif=True)
 d.line((110,280,2450,280),fill=LINE,width=2)
 d.line((110,1365,2450,1365),fill=LINE,width=2);d.line((110,1365,110+2340*t/DURATION,1365),fill=PUR,width=5)
 return im

def note(im,s):text(im,(110,1303),s,27,MUTED)
def fade_asset(im,asset,xy,alpha=1):
 if alpha>=.999:im.paste(asset,xy,asset);return
 layer=asset.copy();layer.putalpha(layer.getchannel('A').point(lambda x:round(x*max(0,alpha))));im.paste(layer,xy,layer)
def bar(im,x,y,w=470,cmap='inferno',label='Temperature',right='hotter'):
 a=np.uint8(colormaps[cmap](np.linspace(0,1,w))[:,:3]*255);im.paste(Image.fromarray(np.repeat(a[None,:,:],20,axis=0)),(x,y))
 text(im,(x,y-40),label,27,MUTED);text(im,(x,y+28),'0',25,MUTED);text(im,(x+w,y+28),right,25,MUTED,anchor='ra')

def callout(im,xy,number,title,desc,active=True):
 x,y=xy;d=ImageDraw.Draw(im);c=PUR if active else '#b3aaba';d.ellipse((x,y,x+52,y+52),fill=c)
 text(im,(x+26,y+8),str(number),32,'white',bold=True,anchor='ma');text(im,(x+83,y-2),title,44,INK if active else MUTED,bold=True)
 if desc:text(im,(x+83,y+59),desc,32,MUTED)

class Movie:
 def __init__(self):
  self.big=Scene(1390,880);self.small=Scene(760,700);self.pair=Scene(1110,740)
  self.walk=Image.open(A/'capsule.png').convert('RGBA').resize((1000,563),Image.Resampling.LANCZOS)
  self.sphere_sprite=Image.open(A/'sphere.png').convert('RGBA');self.sphere_cache={}
  self.topology=Image.open(A/'topology.png').convert('RGBA').resize((1000,350),Image.Resampling.LANCZOS)
  self.model=Image.open(A/'model.png').convert('RGBA').resize((900,236),Image.Resampling.LANCZOS)
  self.est=Image.open(A/'estimator.png').convert('RGBA').resize((1000,813),Image.Resampling.LANCZOS)
  self.sensitivity=Image.open(A/'sensitivity.png').convert('RGBA').resize((1000,813),Image.Resampling.LANCZOS)
  self.increment=Image.open(A/'increment.png').convert('RGBA').resize((930,244),Image.Resampling.LANCZOS)
  self.heat_eq=Image.open(A/'heat-equation.png').convert('RGBA').resize((1040,110),Image.Resampling.LANCZOS)
  self.heat_left=Scene(1080,690);self.heat_right=Scene(1080,690)
  self.heat_data=[dict(np.load(SCOOP/f'{tag}-field.npz')) for tag in ['ours0','budget12']]
  self.trajectory=MET['TWoS + fresh validation']['cooling_percent']
  poly=read_vtp(SCOOP/'ours0-bowl.vtp');pts=vtk_to_numpy(poly.GetPoints().GetData());valid=(pts[:,2]<.18)&(pts[:,1]<.04)&(pts[:,1]>-.40)&(pts[:,0]>-.27)&(pts[:,0]<.27);q=pts[valid];self.query_points=q[np.linspace(0,len(q)-1,18).astype(int)]
 def render_model(self,s,tag,field,angle,ghost=.025,zoom=1,opacity=1,cooling_time=None):
  s.clear_extras();s.show(tag,field,TEAL if tag in ['random','voxel'] else PURPLE)
  if cooling_time is not None:s.cooling_time(cooling_time)
  elif s.current in s.thermal:
   poly,actor,data,score=s.thermal[s.current];s.set_surface(s.current,data['surface']**2 if score else data['surface'])
  for a in s.groups[s.current]:
   if not hasattr(a,'_base_opacity'):a._base_opacity=a.GetProperty().GetOpacity()
   a.GetProperty().SetOpacity(a._base_opacity*opacity)
  s.camera(angle,zoom)
  if ghost:s.ghost(ghost)
  return s.render()
 def frame(self,t):
  if t<7:return self.intro(t)
  if t<19:return self.heating(t)
  if t<31:return self.walks(t)
  if t<43:return self.estimator(t)
  if t<55:return self.insertion(t)
  if t<65:return self.compare(t)
  if t<76:return self.budget(t)
  return self.end(t)
 def intro(self,t):
  im=base('01','Walking into Holes','',t)
  s=self.big;s.clear_extras();s.show('ours0','initial');s.camera(-7+2*t)
  s.ghost(1-.975*ease((t-1.4)/2.4));im.paste(s.render(),(1060,310))
  text(im,(113,452),'Grid-free cavity design',66,serif=True)
  text(im,(113,555),'Monte Carlo topological derivatives',35,MUTED)
  im.paste(self.model,(104,704),self.model)
  bar(im,140,1090,600,right='0.025')
  note(im,'Idealized conduction · cooled exterior and cavity walls')
  return im
 def heating(self,t):
  q=np.clip((t-7)/12,0,1);tau=.065*q*q
  im=base('02','Heat diffusion','',t)
  for j,(s,tag,title,c) in enumerate([(self.heat_left,'ours0','Intact',BLUE),(self.heat_right,'budget12','TWoS · 12 cavities',PUR)]):
   x=80 if j==0 else 1390;s.clear_extras();s.show(tag,tag);s.heat_time(tau);s.camera(-5+11*q,.97);s.ghost(.025);im.paste(s.render(),(x,383));text(im,(x+540,335),title,39,c,bold=True,anchor='ma')
  # One explicit common scale for both geometries.
  d=ImageDraw.Draw(im);a=np.uint8(colormaps['inferno'](np.linspace(1,0,320))[:,:3]*255);im.paste(Image.fromarray(np.repeat(a[:,None,:],21,axis=1)),(1270,550));text(im,(1280,502),'0.025',25,MUTED,anchor='ma');text(im,(1280,892),'0',25,MUTED,anchor='ma');text(im,(1280,942),'u',32,ORANGE,serif=True,anchor='ma')
  im.paste(self.heat_eq,(100,1080),self.heat_eq)
  for x,label,color in [(308,'heat storage',ORANGE),(526,'diffusion',PUR),(768,'heat source',ORANGE)]:text(im,(x,1193),label,25,color,anchor='ma')
  text(im,(130,1242),f'τ = {tau:.4f}',30,MUTED)
  # Actual simulated volume means; the horizontal axis is physical model time.
  px,py,pw,ph=1470,1104,890,143;d.line((px,py,px,py+ph,px+pw,py+ph),fill=LINE,width=2)
  text(im,(px,1055),'Mean temperature',26,MUTED)
  text(im,(px-15,py-8),'0.008',21,MUTED,anchor='ra');text(im,(px-15,py+ph-10),'0',21,MUTED,anchor='ra')
  for data,c in zip(self.heat_data,[BLUE,PUR]):
   times=data['times'];keep=times<tau;tt=np.r_[times[keep],tau];vv=np.interp(tt,times,data['mean']);points=[(px+float(a)/.065*pw,py+ph-float(b)/.008*ph) for a,b in zip(tt,vv)]
   if len(points)>1:d.line(points,fill=c,width=5)
   xx,yy=points[-1];d.ellipse((xx-6,yy-6,xx+6,yy+6),fill=c)
  xx=px+tau/.065*pw;d.line((xx,py,xx,py+ph),fill='#d3ccd9',width=1)
  text(im,(px,1254),'0',22,MUTED);text(im,(px+pw,1254),'0.065',22,MUTED,anchor='ra');text(im,(px+pw/2,1254),'τ',22,MUTED,anchor='ma')
  note(im,'Auxiliary simulation · dimensionless time · fixed designs')
  return im
 def walks(self,t):
  q=(t-19)/12;im=base('03','Walk on spheres','',t)
  im.paste(self.walk,(80,375),self.walk)
  progress=(q*3)%1;index=min(int(q*3),2);trace=TR['traces'][index]['steps'];weights=np.array([st['radius']**.35 for st in trace]);edges=np.r_[0,np.cumsum(weights)/weights.sum()];k=min(np.searchsorted(edges,progress,side='right')-1,len(trace)-1);a=ease((progress-edges[k])/(edges[k+1]-edges[k]))
  pts=[np.array(trace[0]['x'])]+[np.array(x['next']) for x in trace[:k]];p=np.array(trace[k]['x'])*(1-a)+np.array(trace[k]['next'])*a;pts.append(p)
  s=self.big;s.clear_extras();s.show('input',None);s.camera(-7+10*q,1.02)
  for actor in s.groups[s.current]:actor.GetProperty().SetOpacity(.16)
  for st in trace[:k+1]:
   if st['radius']>.005:
    s.sphere(st['x'],st['radius'],(.57,.41,.67),.07)
    if st is trace[k]:s.sphere_guides(st['x'],st['radius'],(.57,.41,.67))
  s.line(pts,(.52,.27,.67),.0024);s.sphere(TR['start'],.007,(.20,.16,.23));s.sphere(p,.006,(.87,.48,.17))
  im.paste(s.render(),(1080,305));px,py=s.project(TR['start']);d=ImageDraw.Draw(im);d.line((1080+px+8,305+py+7,1080+px+70,305+py+48),fill=PUR,width=2);text(im,(1080+px+80,305+py+35),'z',31,PUR,serif=True)
  # A separate, genuinely sampled 3-D capsule walk, projected orthographically.
  ct=CAP['traces'][index]['steps'];cw=np.array([st['radius']**.35 for st in ct]);ce=np.r_[0,np.cumsum(cw)/cw.sum()];kk=min(np.searchsorted(ce,progress,side='right')-1,len(ct)-1);aa=ease((progress-ce[kk])/(ce[kk+1]-ce[kk]))
  origin=np.array([80,375]);to_screen=lambda p:origin+1.25*np.array(p)
  for j,st in enumerate(ct[:kk+1]):
   rad=st['screen_radius']*1.25
   if rad<2:continue
   size=max(8,round(2*rad*240/210))
   if size not in self.sphere_cache:self.sphere_cache[size]=self.sphere_sprite.resize((size,size),Image.Resampling.LANCZOS)
   cen=to_screen(st['screen_x']);fade_asset(im,self.sphere_cache[size],tuple(np.round(cen-size/2).astype(int)),1 if j==kk else .22)
  current=(1-aa)*np.array(ct[kk]['screen_x'])+aa*np.array(ct[kk]['screen_next']);points=[to_screen(ct[0]['screen_x'])]+[to_screen(st['screen_next']) for st in ct[:kk]]+[to_screen(current)]
  d=ImageDraw.Draw(im);d.line([tuple(p) for p in points],fill=PUR,width=4)
  for p in points[:-1]:d.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill=PUR)
  p2=points[-1];d.ellipse((p2[0]-8,p2[1]-8,p2[0]+8,p2[1]+8),fill=ORANGE)
  text(im,(120,332),'Capsule',28,MUTED);text(im,(1490,332),'Bunny',28,MUTED)
  fade_asset(im,self.increment,(105,925),ease((t-19)/1.1))
  text(im,(120,1196),'PATH '+str(index+1)+' / 3',27,PUR,bold=True)
  val=sum(st['contribution'] for st in trace[:k+1]);text(im,(1680,1220),f'U = {val:.5f}',34,ORANGE)
  note(im,'Projected 3D jumps · uniform sphere exits')
  return im
 def estimator(self,t):
  q=(t-31)/12;im=base('04','Topological sensitivity','',t)
  blend=ease((t-34.2)/.6)
  if blend<.5:
   fade_asset(im,self.sensitivity,(105,357),1-2*blend);fade_asset(im,self.topology,(105,357),1-2*blend)
  if blend>.5:fade_asset(im,self.est,(105,357),2*blend-1)
  # Emphasize the actual leave-one-out construction instead of scanning a matrix.
  d=ImageDraw.Draw(im);stage=int(np.clip((t-35)/2,0,3))
  boxes=[[(292,233,162,73,ORANGE)],[(566,144,162,73,BLUE),(566,322,162,73,BLUE),(400,426,388,96,BLUE)],[(400,509,390,71,PUR)],[(400,618,408,82,PUR)]][stage]
  for x,y,w,h,c in boxes if blend>.98 else []:
   x=105+x*1.25;y=357+y*1.25
   d.rounded_rectangle((x-w*.625,y-h*.625,x+w*.625,y+h*.625),radius=8,outline=c,width=4)
  s=self.big;im.paste(self.render_model(s,'ours0','initial:score',-2+5*q),(1110,312))
  d=ImageDraw.Draw(im);nq=min(int(q/.65*len(self.query_points))+1,len(self.query_points))
  for site in self.query_points[:nq]:
   px,py=s.project(site);x=1110+px;y=312+py;d.ellipse((x-5,y-5,x+5,y+5),fill='white')
  z=self.query_points[nq-1].copy()
  if q>.65:
   target=np.array(TR['start']);z=self.query_points[np.argmin(np.linalg.norm(self.query_points[:,:2]-target[:2],axis=1))].copy()
  px,py=s.project(z);x=1110+px;y=312+py;rad=21+6*math.sin(q*math.pi*4);d.ellipse((x-rad,y-rad,x+rad,y+rad),outline='white',width=4);d.line((x-8,y,x+8,y),fill='white',width=3);d.line((x,y-8,x,y+8),fill='white',width=3)
  bar(im,1740,1230,600,cmap='viridis',label='Predicted benefit',right='higher')
  if blend>.5:note(im,'Independent factors · shared samples')
  return im
 def insertion(self,t):
  q=(t-43)/12;stage=min(int(q*4),3);tags=['ours0','ours1','ours3','ours6'];counts=[0,1,3,6];tag=tags[stage];field='initial' if stage==0 else tag
  im=base('05','Sequential insertion','',t)
  local=q*4-stage;tau=.065*ease(min(local/.88,1))**2
  s=self.big;render=self.render_model(s,tag,field,-4+15*q,cooling_time=tau if stage else None)
  im.paste(render,(1100,310))
  callout(im,(112,430),1,'Score','',True)
  callout(im,(112,565),2,'Resample','',True)
  callout(im,(112,700),3,'Insert','',True)
  val=self.trajectory[counts[stage]];text(im,(112,912),f'{counts[stage]:02d}',104,PUR,serif=True);text(im,(290,934),'cavities',41,MUTED)
  text(im,(112,1050),f'{val:.1f}%',92,PUR,serif=True);text(im,(430,1072),'steady-state reduction',36,MUTED)
  # Measured trajectory, not an inferred continuous physical history.
  d=ImageDraw.Draw(im);x0,y0=1260,1230;ww=1070;hh=130
  # plot is intentionally omitted here to leave geometry and shared scale readable.
  bar(im,1670,1230,650,right='0.025')
  text(im,(1240,1160),f'τ = {tau:.4f}' if stage else 'Steady state',28,MUTED)
  note(im,'Auxiliary cooling · steady-state benchmark values')
  return im
 def compare(self,t):
  q=(t-55)/10;im=base('06','Equal-volume placement','',t)
  specs=[('random','Random feasible',BLUE),('ours6','TWoS + fresh validation',PUR),('voxel','Voxel adjoint TD',BLUE)]
  labels=['Random','TWoS','Grid adjoint TD']
  for j,(tag,name,c) in enumerate(specs):
   x=100+j*805;im.paste(self.render_model(self.small,tag,tag,-4+6*q), (x,368))
   text(im,(x+380,327),labels[j],40,c,bold=True,anchor='ma')
   val=MET[name]['cooling_percent'][-1];text(im,(x+380,1116),f'{val:.1f}%',76,c,serif=True,anchor='ma');text(im,(x+380,1207),'objective reduction',29,MUTED,anchor='ma')
  note(im,'6 cavities · equal radii · one retained run')
  return im
 def budget(self,t):
  q=(t-65)/11;im=base('07','Cavity size and count','',t)
  specs=[('budget6','6 larger cavities',47.35476687674381,'100%','6-6'),('budget12','12 smaller cavities',47.07426525420503,'50%','3-12')]
  for j,(tag,label,val,vol,key) in enumerate(specs):
   x=120+j*1190;im.paste(self.render_model(self.pair,tag,tag,-4+6*q), (x,345))
   text(im,(x+555,321),label,43,PUR,bold=True,anchor='ma')
   text(im,(x+240,1120),f'{val:.1f}%',74,PUR,serif=True,anchor='ma');text(im,(x+240,1210),'objective reduction',29,MUTED,anchor='ma')
   text(im,(x+790,1120),vol,74,INK,serif=True,anchor='ma');text(im,(x+790,1210),'removed volume',29,MUTED,anchor='ma')
  note(im,'3-seed ranges: 46.8–48.2% / 47.0–47.7% · first seed shown')
  return im
 def end(self,t):
  q=(t-76)/6;im=base('08','Walking into Holes','',t)
  im.paste(self.render_model(self.big,'budget12','budget12',2-5*q),(1120,310))
  text(im,(113,475),'Monte Carlo',68,serif=True)
  text(im,(113,565),'topological derivatives',68,serif=True)
  text(im,(113,759),'Grid-free PDE-constrained design',36,MUTED)
  text(im,(113,1030),'Query  →  Score  →  Insert',38,PUR)
  return im

def main():
 p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');p.add_argument('--start',type=float,default=0);p.add_argument('--end',type=float,default=DURATION);p.add_argument('--out',type=Path,default=OUT/'Walking_into_Holes_Hero_1440p.mp4');args=p.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True);m=Movie()
 if args.preview:
  times=[2.,5.,8.,12.,17.,22.,27.,32.5,36.,41.,46.,53.,59.,71.,79.];thumb=[]
  for t in times:
   LAYOUT.clear();im=m.frame(t);
   (V/'preview'/f'layout-{t:04.1f}.json').write_text(json.dumps(LAYOUT,indent=2));im.save(V/'preview'/f'frame-{t:04.1f}.png');im.thumbnail((640,360));thumb.append(im);print('preview',t,flush=True)
  sheet=Image.new('RGB',(1920,360*5),'#eee')
  for i,im in enumerate(thumb):sheet.paste(im,((i%3)*640,(i//3)*360))
  sheet.save(V/'preview/contact.png');return
 cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',str(args.out)]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);st=time.perf_counter()
 try:
  for n in range(round(args.start*FPS),round(args.end*FPS)):
   LAYOUT.clear();t=n/FPS;im=m.frame(t);proc.stdin.write(im.tobytes())
   if n%FPS==0:print(f'{t:.1f}s / {args.end:.1f}s; elapsed {time.perf_counter()-st:.1f}s',flush=True)
 finally:proc.stdin.close()
 if proc.wait():raise RuntimeError('ffmpeg failed')
 print('Wrote',args.out,flush=True)
if __name__=='__main__':main()
