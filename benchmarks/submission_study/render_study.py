"""Data-grounded Mitsuba cutaways. Run only after all timed experiments finish."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np,pandas as pd,trimesh
import matplotlib as mpl
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from run_study import HERE,OUT,SEEDS,R,Solver,load,channel_balls,dump,SPECS
sys.path.insert(0,str(HERE.parents[1]/'figures'))
from render_mesh_hero import mi,T,material,denoise
ASSETS=HERE/'render_assets';ASSETS.mkdir(exist_ok=True)
CAMERA=np.array([-2.8,-4.5,8.]);SIZE=1600;SPP=96
BLANK_CACHE={}
def save_png(im,path):
 tmp=path.with_suffix('.tmp.png');im.save(tmp);tmp.replace(path)

COLORS={'nearest':[.55,.62,.49],'twos':[.55,.42,.64],'mg':[.40,.59,.66],'hot':[.70,.65,.57],'initial':[.66,.68,.72]}

def state(name,study,method):
 if method=='initial':return np.empty((0,3)),R
 d=json.loads((OUT/study/name/f'{method}-{SEEDS[0]}'/('refinement.json' if study=='pipeline' else 'design.json')).read_text())
 return (channel_balls(d['centers']) if study=='channels' else np.array(d['centers'])),d['radius']

def field(name,study,method):
 if method=='initial':study='shared'
 path=ASSETS/f'{name}-{study}-{method}-field.npz'
 if path.exists():return np.load(path)
 s,m=load(name);cut=s['sensor'][2];holes,r=state(name,study,method)
 x=np.linspace(m.bounds[0,0],m.bounds[1,0],320);y=np.linspace(m.bounds[0,1],m.bounds[1,1],256)
 xx,yy=np.meshgrid(x,y);p=np.stack([xx.ravel(),yy.ravel(),np.full(xx.size,cut)],1)
 sol=Solver(name,ASSETS/'field-raw'/f'{study}-{name}-{method}')
 f,_=sol.call('field',p,holes,r,mode='mg',n=193,hot=True);sol.close()
 mask=m.contains(p)
 if len(holes):mask &= cKDTree(holes).query(p)[0]>r
 tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,x=x,y=y,u=f.u.to_numpy().reshape(xx.shape),mask=mask.reshape(xx.shape));tmp.replace(path)
 return np.load(path)

def parts(name,study,method):
 folder=ASSETS/f'{name}-{study}-{method}';folder.mkdir(exist_ok=True)
 paths={k:folder/f'{k}.ply' for k in ['shell','section','walls','ghost']}
 if (folder/'geometry.json').exists():return {k:p for k,p in paths.items() if p.exists()}
 s,base=load(name);cut=s['sensor'][2];holes,r=state(name,study,method);cutters=[]
 if study=='channels' and len(holes):
  axes=np.unique(holes[:,:2],axis=0);zz=np.unique(holes[:,2])
  z=np.unique(np.concatenate([np.linspace(c-r,c+r,25) for c in zz]));rad=np.sqrt(np.maximum(0,r*r-np.min((z[:,None]-zz[None,:])**2,axis=1)));rad[[0,-1]]=0
  for a in axes:
   c=trimesh.creation.revolve(np.stack([rad,z],1),sections=80);c.apply_translation([a[0],a[1],0]);cutters.append(c)
 else:
  for h in holes:
   c=trimesh.creation.icosphere(subdivisions=3,radius=r);c.apply_translation(h);cutters.append(c)
 physical=trimesh.boolean.difference([base,*cutters],engine='manifold') if cutters else base
 assert physical.is_volume
 # Broad common display window retains the far rim and original external silhouette.
 radius=.89 if name=='housing' else 1.03
 window=trimesh.creation.cylinder(radius=radius,height=2.,sections=192);window.apply_translation([0,0,cut+1.])
 visible=trimesh.boolean.difference([physical,window],engine='manifold');assert visible.is_volume
 tri=visible.triangles_center;section=np.abs(tri[:,2]-cut)<2e-5
 near=cKDTree(holes).query(tri)[0] if len(holes) else np.full(len(tri),99.)
 walls=(np.abs(near-r)<.002)&~section
 for k,mask in [('shell',~(section|walls)),('section',section),('walls',walls)]:
  if mask.any():visible.submesh([np.flatnonzero(mask)],append=True,repair=False).export(paths[k])
 ghost=trimesh.boolean.intersection([physical,window],engine='manifold');tri=ghost.triangles_center
 artificial=(np.abs(tri[:,2]-cut)<2e-5)|(np.abs(np.linalg.norm(tri[:,:2],axis=1)-radius)<.001)
 if (~artificial).any():ghost.submesh([np.flatnonzero(~artificial)],append=True,repair=False).export(paths['ghost'])
 dump(folder/'geometry.json',{'source_centers':holes.tolist(),'radius':r,'physical_volume_render_mesh':physical.volume,'display_cut':{'z':cut,'radius':radius},'all_PDE_solves_use_uncut_geometry':True,'tessellated_voids_for_display_only':True})
 return {k:p for k,p in paths.items() if p.exists()}

def project(q,target,scale,size):
 direction=-CAMERA/np.linalg.norm(CAMERA);left=np.cross([0,0,1],direction);left/=np.linalg.norm(left);up=np.cross(direction,left)
 return np.stack([.5-(q-target)@left/(2*scale),.5-(q-target)@up/(2*scale)],1)*size

def render(name,study,method,zoom=False):
 dest=ASSETS/f'{name}-{study}-{method}{"-detail" if zoom else ""}.png'
 if dest.exists() and dest.with_suffix('.json').exists():return
 s,m=load(name);paths=parts(name,study,method);cut=s['sensor'][2]
 target=np.array(s['sensor'][:3]) if zoom else np.array([0,0,.06])
 scale=.29 if zoom else (1.39 if name=='housing' else 1.15);size=1000 if zoom else SIZE
 cfg={'type':'scene','integrator':{'type':'aov','aovs':'albedo:albedo,normal:sh_normal','beauty':{'type':'path','max_depth':8}},
 'sensor':{'type':'orthographic','to_world':T.look_at(origin=target+CAMERA,target=target,up=[0,0,1])@T.scale([scale,scale,1]),'sampler':{'type':'independent','sample_count':SPP},'film':{'type':'hdrfilm','width':size,'height':size,'pixel_format':'rgba','rfilter':{'type':'tent'}}},
 'environment':{'type':'constant','radiance':{'type':'rgb','value':.3}},
 'key':{'type':'rectangle','to_world':T.look_at(origin=[-3,3.5,7],target=[0,0,0],up=[0,1,0])@T.scale([3,3,1]),'emitter':{'type':'area','radiance':{'type':'rgb','value':5.}}},
 'floor':{'type':'rectangle','to_world':T.translate([0,0,m.bounds[0,2]-.02])@T.scale(30),'bsdf':material([1,1,1])}}
 for k,p in paths.items():
  if k=='ghost':continue
  color=COLORS[method] if k=='shell' else ([.73,.73,.73] if k=='section' else [.3,.31,.35])
  cfg[k]={'type':'ply','filename':str(p),'face_normals':k=='section','bsdf':material(color)}
 scene=mi.load_dict(cfg);arr=np.asarray(mi.render(scene,seed=6501,spp=SPP));names=scene.integrator().aov_names()
 albedo=arr[:,:,names.index('albedo.R'):names.index('albedo.B')+1];normals=arr[:,:,names.index('normal.X'):names.index('normal.Z')+1];rgb=arr[:,:,:3]
 blank={k:v for k,v in cfg.items() if k not in paths};blank_key=(name,zoom,size,SPP)
 if blank_key not in BLANK_CACHE:BLANK_CACHE[blank_key]=np.asarray(mi.render(mi.load_dict(blank),seed=6501,spp=SPP))[:,:,:3].copy()
 clean=BLANK_CACHE[blank_key]
 foreground=(albedo.min(2)<.999)&(albedo.max(2)>0);rgb=np.where(foreground[:,:,None],rgb,np.clip(rgb/np.maximum(clean,1e-5),0,1))
 rgb=np.clip(denoise(np.clip(rgb,0,1),albedo,normals),0,1);rgb=np.where(foreground[:,:,None],rgb,gaussian_filter(rgb,sigma=(2,2,0)));rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055)
 yy,xx=np.mgrid[:size,:size];sample=mi.Point2f(np.array([(xx.ravel()+.5)/size,(yy.ravel()+.5)/size],dtype=np.float32));ray,_=scene.sensors()[0].sample_ray(0,0,sample,mi.Point2f(.5));hit=scene.ray_intersect(ray)
 p=np.asarray(hit.p).T;valid=np.asarray(hit.is_valid());cap=valid&(np.abs(p[:,2]-cut)<2e-5)&(np.abs(np.asarray(hit.n).T[:,2])>.999)
 d=field(name,study,method);base=field(name,study,'initial')
 get=lambda a,k:RegularGridInterpolator((a['y'],a['x']),a[k],bounds_error=False,fill_value=0)(p[cap][:,[1,0]])
 if method=='initial':
  values=get(d,'u');maxval=float(d['u'][d['mask']].max());colors=mpl.colormaps['inferno'](np.clip(values/maxval,0,1))[:,:3]
 else:
  values=get(d,'u');before=get(base,'u');cool=1-values/np.maximum(before,1e-12)
  colors=mpl.colormaps['viridis'](np.clip(cool,0,1))[:,:3]
  resolved=before>1e-4*float(base['u'][base['mask']].max());colors[~resolved]=[.86,.86,.88]
 rgb=rgb.reshape(-1,3);rgb[cap]=colors
 # A faint depth-tested shell shows the removed top structure without inventing material.
 if 'ghost' in paths and not zoom:
  ghost_cfg={'type':'scene','ghost':{'type':'ply','filename':str(paths['ghost']),'bsdf':material([.7,.72,.77])}}
  gh=mi.load_dict(ghost_cfg).ray_intersect(ray);front=np.asarray(gh.is_valid())&(np.asarray(gh.t)<np.asarray(hit.t)-1e-5)
  gn=np.asarray(gh.n).T;view=CAMERA/np.linalg.norm(CAMERA);edge=np.abs(gn@view)<.17
  alpha=np.where(edge,.32,.09)*front;rgb=rgb*(1-alpha[:,None])+np.array([.46,.49,.56])*alpha[:,None]
 im=Image.fromarray(np.uint8(np.clip(rgb.reshape(size,size,3)*255,0,255)));overlay=Image.new('RGBA',(size,size));draw=ImageDraw.Draw(overlay);tt=np.linspace(0,2*np.pi,240)
 for c,col in [(s['sensor'],'#00ccda')]+[(h,'#ec9844') for h in s['heat']]:
  rad2=c[3]**2-(cut-c[2])**2
  if rad2<=0:continue
  rad=np.sqrt(rad2);q=np.stack([c[0]+rad*np.cos(tt),c[1]+rad*np.sin(tt),np.full(len(tt),cut)],1);pix=project(q,target,scale,size)
  draw.line([tuple(v) for v in pix],fill=col,width=max(2,size//350))
 # Mark only physically visible section segments.
 a=np.asarray(overlay).copy();a[:,:,3]*=cap.reshape(size,size).astype(np.uint8)
 save_png(Image.alpha_composite(im.convert('RGBA'),Image.fromarray(a)).convert('RGB'),dest)
 dump(dest.with_suffix('.json'),{'camera':CAMERA.tolist(),'target':target.tolist(),'scale':scale,'size':size,'samples':SPP,'cut':cut,'sensor_pixel':(project(np.array([s['sensor'][:3]]),target,scale,size)[0]/size).tolist(),'field':'MG max-axis 193; bilinear section interpolation','map':'absolute temperature / unperforated maximum' if method=='initial' else '1 - u_design / u_unperforated, clipped to [0,1] for display','low_temperature_mask':'Relative cooling is gray where baseline temperature is below 1e-4 of its section maximum.','ghost':'physical upper surface, depth tested, 9% opacity with 32% silhouettes','seed':SEEDS[0]})
 print('RENDER',dest.name,flush=True)

def render_opaque(name):
 dest=ASSETS/f'{name}-opaque.png'
 if dest.exists():return
 s,m=load(name);scale=1.4 if name=='housing' else 1.18;target=np.array([0,0,.10])
 cfg={'type':'scene','integrator':{'type':'aov','aovs':'albedo:albedo,normal:sh_normal','beauty':{'type':'path','max_depth':8}},
 'sensor':{'type':'orthographic','to_world':T.look_at(origin=target+CAMERA,target=target,up=[0,0,1])@T.scale([scale,scale,1]),'sampler':{'type':'independent','sample_count':SPP},'film':{'type':'hdrfilm','width':SIZE,'height':SIZE,'pixel_format':'rgba','rfilter':{'type':'tent'}}},
 'environment':{'type':'constant','radiance':{'type':'rgb','value':.3}},
 'key':{'type':'rectangle','to_world':T.look_at(origin=[-3,3.5,7],target=[0,0,0],up=[0,1,0])@T.scale([3,3,1]),'emitter':{'type':'area','radiance':{'type':'rgb','value':5.}}},
 'floor':{'type':'rectangle','to_world':T.translate([0,0,m.bounds[0,2]-.02])@T.scale(30),'bsdf':material([1,1,1])},
 'body':{'type':'ply','filename':str(HERE/'assets'/f'{name}.ply'),'bsdf':material([.60,.53,.68])}}
 scene=mi.load_dict(cfg);a=np.asarray(mi.render(scene,seed=8401,spp=SPP));names=scene.integrator().aov_names();albedo=a[:,:,names.index('albedo.R'):names.index('albedo.B')+1];normal=a[:,:,names.index('normal.X'):names.index('normal.Z')+1]
 bg={k:v for k,v in cfg.items() if k!='body'};clean=np.asarray(mi.render(mi.load_dict(bg),seed=8401,spp=SPP))[:,:,:3];fg=(albedo.min(2)<.999)&(albedo.max(2)>0)
 rgb=np.where(fg[:,:,None],a[:,:,:3],np.clip(a[:,:,:3]/np.maximum(clean,1e-5),0,1));rgb=np.clip(denoise(rgb,albedo,normal),0,1);rgb=np.where(fg[:,:,None],rgb,gaussian_filter(rgb,sigma=(2,2,0)));rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055)
 save_png(Image.fromarray(np.uint8(np.clip(rgb*255,0,255))),dest);print('RENDER',dest.name,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--geometry',default='all');ap.add_argument('--study',default='all');ap.add_argument('--preview',action='store_true');args=ap.parse_args()
 if args.preview:SIZE=720;SPP=24
 for name in SPECS if args.geometry=='all' else [args.geometry]:
  render_opaque(name)
  for study in ['pipeline','channels'] if args.study=='all' else [args.study]:
   for method in ['nearest','twos','mg']:
    render(name,study,method)
    if method!='initial':render(name,study,method,True)
