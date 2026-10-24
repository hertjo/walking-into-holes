"""Path-traced cutaways and exact-location temperature overlays for saved designs.

Display booleans never enter the PDE. Data uses the uncut physical domains.
All methods share camera, lighting, cut window and absolute temperature scale.
"""
from pathlib import Path
import sys,json,time,argparse,hashlib
import numpy as np
import pandas as pd
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
from scipy.interpolate import RegularGridInterpolator
from PIL import Image,ImageDraw
import run_accessible as a
sys.path.insert(0,str(a.ROOT/'figures'))
from render_mesh_hero import mi,T,material,denoise

ASSETS=a.HERE/'render_assets';ASSETS.mkdir(exist_ok=True)
CUT=-.03
SEED=a.CONFIG['seeds'][0]
METHODS=['wos_hot','twos_sparse','mg57']
CAMERA=np.array([-3.,-4.5,8.])

def centers(method):
    if method=='initial':return np.empty((0,3))
    return np.array(json.loads((a.OUT/f'side1/seed{SEED}/{method}/design.json').read_text())['centers'])

def fields():
    b=a.b;b.OUT=ASSETS/'fields';b.R=a.R;b.EXE=a.p.EXE
    x=np.linspace(-1.16,1.16,256);y=np.linspace(-.76,.76,176)
    xx,yy=np.meshgrid(x,y);points=np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,CUT)))
    for method in ['initial']+METHODS:
        path=ASSETS/f'{method}-field.npz'
        if path.exists():continue
        c=centers(method)
        values,_=b.solver(method,points,a.chain(c),1,mode='mg',grid=[169,109,85])
        mask=a.b.housing.housing_sdf(points)<0
        if len(c):mask &= np.min(np.linalg.norm(points[:,None,:]-a.chain(c)[None,:,:],axis=2),axis=1)>a.R
        u=values.u.to_numpy().reshape(xx.shape);p=values.p.to_numpy().reshape(xx.shape)
        np.savez_compressed(path,x=x,y=y,u=u,p=p,score=u*p,mask=mask.reshape(xx.shape))
    initial=np.load(ASSETS/'initial-field.npz')
    return float(np.ceil(initial['u'][initial['mask']].max()/.01)*.01)

def channel(c):
    # Include centers and seam planes; tessellation approximates the analytic
    # ball union for display only, not a cylindrical surrogate PDE domain.
    zz=np.unique(np.concatenate([np.linspace(z-a.R,z+a.R,25) for z in a.Z]))
    rr=np.sqrt(np.maximum(0,a.R*a.R-np.min((zz[:,None]-a.Z[None,:])**2,axis=1)))
    rr[0]=rr[-1]=0
    mesh=trimesh.creation.revolve(np.column_stack((rr,zz)),sections=96)
    mesh.apply_translation([c[0],c[1],0]);assert mesh.is_volume
    return mesh

def geometry(method):
    paths={k:ASSETS/f'{method}-{k}.ply' for k in ['shell','section','walls']}
    if all(p.exists() for p in paths.values()):return paths
    base=trimesh.load(a.b.MESH,process=False)
    c=centers(method);cutters=[channel(x) for x in c]
    window=trimesh.creation.cylinder(radius=.86,height=2.,sections=192)
    window.apply_translation([0,0,CUT+1.]);cutters.append(window)
    mesh=trimesh.boolean.difference([base,*cutters],engine='manifold')
    assert mesh.is_volume
    xyz=mesh.triangles_center;section=np.abs(xyz[:,2]-CUT)<1e-5
    near=np.full(len(xyz),np.inf)
    for ball in a.chain(c):near=np.minimum(near,np.abs(np.linalg.norm(xyz-ball,axis=1)-a.R))
    walls=(near<.001)&~section
    for key,mask in [('shell',~(section|walls)),('section',section),('walls',walls)]:
        part=mesh.submesh([np.flatnonzero(mask)],append=True,repair=False);part.export(paths[key])
    (ASSETS/f'{method}-geometry.json').write_text(json.dumps({'source_centers':c.tolist(),
        'triangles':len(mesh.faces),'volume_before_display_cut_not_recomputed':True,
        'display_cut':{'radius':.86,'z_min':CUT},'watertight_display_mesh':True},indent=2)+'\n')
    return paths

def scene_config(paths,size,spp,zoom):
    target=np.array([.55,.18,CUT]) if zoom else np.array([0,0,.08])
    scale=.31 if zoom else 1.43
    cfg={'type':'scene','integrator':{'type':'aov','aovs':'albedo:albedo,normal:sh_normal',
       'beauty':{'type':'path','max_depth':8}},
      'sensor':{'type':'orthographic','to_world':T.look_at(origin=target+CAMERA,target=target,up=[0,0,1])@T.scale([scale,scale,1]),
       'sampler':{'type':'independent','sample_count':spp},
       'film':{'type':'hdrfilm','width':size,'height':size,'pixel_format':'rgba','rfilter':{'type':'tent'}}},
      'environment':{'type':'constant','radiance':{'type':'rgb','value':.30}},
      'key':{'type':'rectangle','to_world':T.look_at(origin=[-3,3.5,7],target=[0,0,0],up=[0,1,0])@T.scale([3,3,1]),
       'emitter':{'type':'area','radiance':{'type':'rgb','value':5.}}},
      'floor':{'type':'rectangle','to_world':T.translate([0,0,-.303])@T.scale(30),'bsdf':material([1,1,1])}}
    for key,path in paths.items():
        color={'shell':[.62,.65,.69],'section':[.68,.68,.68],'walls':[.35,.37,.40]}[key]
        cfg[key]={'type':'ply','filename':str(path),'face_normals':key=='section','bsdf':material(color)}
    return cfg,target,scale

def project_points(q,target,scale,size):
    direction=-CAMERA/np.linalg.norm(CAMERA)
    left=np.cross([0,0,1],direction);left/=np.linalg.norm(left);up=np.cross(direction,left)
    return np.column_stack((.5-(q-target)@left/(2*scale),.5-(q-target)@up/(2*scale)))*size

def render(method,paths,size,spp,vmax,zoom=False):
    name=method+('-detail' if zoom else '')
    cfg,target,scale=scene_config(paths,size,spp,zoom)
    scene=mi.load_dict(cfg);arr=np.asarray(mi.render(scene,seed=19051,spp=spp));names=scene.integrator().aov_names()
    albedo=arr[:,:,names.index('albedo.R'):names.index('albedo.B')+1]
    normals=arr[:,:,names.index('normal.X'):names.index('normal.Z')+1]
    rgb=arr[:,:,:3]
    blank={k:v for k,v in cfg.items() if k not in paths}
    clean=np.asarray(mi.render(mi.load_dict(blank),seed=19051,spp=spp))[:,:,:3]
    foreground=(albedo.min(2)<.999)&(albedo.max(2)>0)
    rgb=np.where(foreground[:,:,None],rgb,np.clip(rgb/np.maximum(clean,1e-5),0,1))
    rgb=np.clip(denoise(np.clip(rgb,0,1),albedo,normals),0,1)
    rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055)
    yy,xx=np.mgrid[:size,:size]
    sample=mi.Point2f(np.array([(xx.ravel()+.5)/size,(yy.ravel()+.5)/size],dtype=np.float32))
    ray,_=scene.sensors()[0].sample_ray(0,0,sample,mi.Point2f(.5));hit=scene.ray_intersect(ray)
    points=np.asarray(hit.p).T;valid=np.asarray(hit.is_valid())
    cap=valid&(np.abs(points[:,2]-CUT)<2e-5)&(np.abs(np.asarray(hit.n).T[:,2])>.999)
    data=np.load(ASSETS/f'{method}-field.npz')
    field=RegularGridInterpolator((data['y'],data['x']),data['u'],bounds_error=False,fill_value=0.)
    colors=mpl.colormaps['inferno'](np.clip(field(points[cap][:,[1,0]])/vmax,0,1))[:,:3]
    result=rgb.reshape(-1,3).copy();result[cap]=colors
    image=Image.fromarray(np.uint8(np.clip(result.reshape(size,size,3)*255,0,255)))
    overlay=Image.new('RGBA',(size,size),(0,0,0,0));draw=ImageDraw.Draw(overlay)
    tt=np.linspace(0,2*np.pi,240)
    # The sensor center lies on the cut plane, so this is its exact section.
    for center,radius,color in [(a.b.center(1),.13,'#20c9db')]:
        q=center+np.column_stack((radius*np.cos(tt),radius*np.sin(tt),np.zeros_like(tt)))
        pix=project_points(q,target,scale,size)
        draw.line([tuple(x) for x in pix],fill=color,width=max(2,round(size/450)))
    alpha=np.asarray(overlay).copy();alpha[:,:,3]*=cap.reshape(size,size).astype(np.uint8)
    image=Image.alpha_composite(image.convert('RGBA'),Image.fromarray(alpha)).convert('RGB')
    image.save(ASSETS/f'{name}.png')
    np.savez_compressed(ASSETS/f'{name}-projection.npz',sensor_pixel=project_points(a.b.center(1)[None,:],target,scale,size)[0]/size)
    print('rendered',name,size,spp,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    vmax=fields();size=720 if args.preview else 1500;spp=24 if args.preview else 128
    methods=['twos_sparse'] if args.preview else METHODS
    for method in methods:
        paths=geometry(method)
        for zoom in [False,True]:render(method,paths,size,spp,vmax,zoom)
    meta={'display_only':True,'preview':args.preview,'source_seed':SEED,'sensor_side':1,
      'physical_domains':'Original housing minus saved analytic ball unions; display window excluded from all PDE solves.',
      'renderer':'Mitsuba '+mi.__version__,'size':size,'samples_per_pixel':spp,
      'temperature_scale_max':vmax,'temperature_reference':'169x109x85 multigrid; approximate numerical field',
      'field_display':'Bilinear interpolation on z=-0.03; no per-panel rescaling or denoising of scientific values.',
      'radiance_denoising':'OIDN on display illumination only; temperature pixels replaced afterward.',
      'geometry':'Watertight manifold booleans of tessellated ball-union channels and a common cylindrical display cut.',
      'render_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ASSETS/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()
