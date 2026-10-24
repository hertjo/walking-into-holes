"""Render actual saved Bunny cavity designs for the SIGGRAPH teaser.

Mitsuba path tracing; Manifold booleans are visualization only. Neither the
simulation mesh nor stored designs is changed. All cameras/cutaway boxes are
identical across the Bunny design comparison.
"""
from pathlib import Path
import argparse,json,hashlib,ctypes,importlib.util,time
import numpy as np
import pandas as pd
import trimesh
import mitsuba as mi
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures/hero-assets'
BUNNY=ROOT/'benchmarks/stanford_bunny/output'
mi.set_variant('llvm_ad_rgb')
T=mi.ScalarTransform4f
PURPLE=[.49,.36,.59]
BLUE=[.36,.56,.65]

def denoise(color,albedo,normal):
    """OIDN is used only for display radiance, never numerical results."""
    package=Path(importlib.util.find_spec('oidn').origin).parent
    tbb=package/'lib.linux.x64/libtbb.so.12.5'
    if tbb.exists():ctypes.CDLL(str(tbb),mode=ctypes.RTLD_GLOBAL)
    import oidn
    device=oidn.NewDevice();oidn.CommitDevice(device)
    filt=oidn.NewFilter(device,'RT')
    buffers={k:np.ascontiguousarray(v,dtype=np.float32) for k,v in
             [('color',color),('albedo',albedo),('normal',normal)]}
    buffers['output']=np.empty_like(buffers['color'])
    for name,data in buffers.items():
        oidn.SetSharedFilterImage(filt,name,data,oidn.FORMAT_FLOAT3,color.shape[1],color.shape[0])
    native=ctypes.CDLL(str(package/'lib.linux.x64/libOpenImageDenoise.so.1.4.3'))
    set_bool=native.oidnSetFilter1b
    set_bool.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_bool]
    set_bool(filt,b'hdr',True)
    oidn.CommitFilter(filt);oidn.ExecuteFilter(filt)
    error=oidn.GetDeviceError(device)
    oidn.ReleaseFilter(filt);oidn.ReleaseDevice(device)
    if error:raise RuntimeError(f'OIDN error {error}')
    return buffers['output']

def material(color):
    return {'type':'diffuse','reflectance':{'type':'rgb','value':color}}

def parts(tag,method=None,count=0,cut=True,centers_override=None,radius_override=None):
    base=trimesh.load(BUNNY/'stanford-bunny-watertight.ply',process=False)
    centers=pd.read_csv(BUNNY/'selected-centers.csv')
    centers=centers[centers.method==method].sort_values('order')[['x','y','z']].values[:count]
    radius=json.loads((BUNNY/'summary.json').read_text())['configuration']['cavity_radius']
    if centers_override is not None:centers=np.asarray(centers_override,dtype=float)
    if radius_override is not None:radius=float(radius_override)
    cutters=[]
    for c in centers:
        sphere=trimesh.creation.icosphere(subdivisions=4,radius=radius)
        sphere.apply_translation(c);cutters.append(sphere)
    if cut:
        window=trimesh.creation.cylinder(radius=.355,height=2,sections=192)
        window.apply_translation([-.025,-.18,1.09]);cutters.append(window)
    mesh=trimesh.boolean.difference([base,*cutters],engine='manifold') if cutters else base
    tri=mesh.triangles_center
    cap=np.abs(tri[:,2]-.09)<1e-5 if cut else np.zeros(len(tri),bool)
    inner=np.zeros(len(tri),bool)
    for c in centers:
        inner|=np.abs(np.linalg.norm(tri-c,axis=1)-radius)<radius*.004
    result={}
    for name,mask in [('shell',~(cap|inner)),('section',cap&~inner),('cavity',inner)]:
        if mask.any():
            part=mesh.submesh([np.flatnonzero(mask)],append=True,repair=False)
            path=OUT/f'{tag}-{name}.ply';part.export(path)
            result[name]=path
    return result

def scene_dict(meshes,color,size,spp,zoom=False):
    target=np.array([-.01,-.015,0.])
    offset=np.array([1.3,.52,3.5])
    scale=.62
    if zoom:target=np.array([.065,-.16,.065]);scale=.225
    scene={'type':'scene',
      'integrator':{'type':'aov','aovs':'albedo:albedo,normal:sh_normal',
                    'beauty':{'type':'path','max_depth':8}},
      'sensor':{'type':'orthographic','to_world':T.look_at(origin=target+offset,target=target,up=[0,1,0])@T.scale([scale,scale,1]),
        'sampler':{'type':'independent','sample_count':spp},
        'film':{'type':'hdrfilm','width':size,'height':size,'pixel_format':'rgba',
                'rfilter':{'type':'tent'}}},
      'environment':{'type':'constant','radiance':{'type':'rgb','value':.35}},
      'key':{'type':'rectangle','to_world':T.look_at(origin=[-2,4,4],target=[0,0,0],up=[0,1,0])@T.scale([2,2,1]),
             'emitter':{'type':'area','radiance':{'type':'rgb','value':5.0}}},
      'floor':{'type':'rectangle','to_world':T.translate([0,-.501,0])@T.rotate([1,0,0],-90)@T.scale(30),
               'bsdf':material([1,1,1])}}
    for name,p in meshes.items():
        c=color if name=='shell' else ([.82,.78,.84] if name=='section' else [v*.67 for v in color])
        scene[name]={'type':'ply','filename':str(p),'face_normals':name=='section','bsdf':material(c)}
    return scene

def render(tag,meshes,color,size,spp,zoom=False):
    started=time.perf_counter()
    path=OUT/(tag+('-detail' if zoom else '')+'.png')
    cfg=scene_dict(meshes,color,size,spp,zoom)
    scene=mi.load_dict(cfg)
    arr=np.asarray(mi.render(scene,seed=1731,spp=spp))
    names=scene.integrator().aov_names()
    print(tag,arr.shape,names,flush=True)
    # AOV integrator exposes beauty, albedo, and shading-normal channels.
    channels=names
    rgb=arr[:,:,:3]
    albedo=arr[:,:,channels.index('albedo.R'):channels.index('albedo.B')+1]
    normal=arr[:,:,channels.index('normal.X'):channels.index('normal.Z')+1]
    # A floor-only render supplies a physically lit shadow-catcher reference.
    blank={k:v for k,v in cfg.items() if k not in meshes}
    clean=np.asarray(mi.render(mi.load_dict(blank),seed=1731,spp=spp))[:,:,:3]
    foreground=(albedo.min(axis=2)<.999)&(albedo.max(axis=2)>0)
    shadow=np.clip(rgb/np.maximum(clean,1e-5),0,1)
    rgb=np.where(foreground[:,:,None],rgb,shadow)
    rgb=np.clip(denoise(np.clip(rgb,0,1),albedo,normal),0,1)
    srgb=np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055)
    Image.fromarray(np.uint8(np.clip(srgb*255,0,255))).save(path)
    print('finished',path.name,round(time.perf_counter()-started,2),'seconds',flush=True)
    return path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true')
    ap.add_argument('--size',type=int,default=1500);ap.add_argument('--spp',type=int,default=128)
    args=ap.parse_args();OUT.mkdir(exist_ok=True)
    spec=[('input',None,0,False,PURPLE),('voxel','Voxel adjoint TD',6,True,BLUE),
          ('random','Random feasible',6,True,BLUE),('ours0','TWoS + fresh validation',0,True,PURPLE),
          ('ours1','TWoS + fresh validation',1,True,PURPLE),('ours3','TWoS + fresh validation',3,True,PURPLE),
          ('ours6','TWoS + fresh validation',6,True,PURPLE)]
    if args.preview:spec=[spec[0],spec[-1]]
    for tag,method,count,cut,color in spec:
        p=parts(tag,method,count,cut)
        render(tag,p,color,args.size,args.spp)
        if not args.preview:render(tag,p,color,args.size,args.spp,zoom=True)
    data={'source_mesh':str((BUNNY/'stanford-bunny-watertight.ply').relative_to(ROOT)),
      'source_sha256':hashlib.sha256((BUNNY/'stanford-bunny-watertight.ply').read_bytes()).hexdigest(),
      'designs':'benchmarks/stanford_bunny/output/selected-centers.csv',
      'renderer':'Mitsuba '+mi.__version__,'size':args.size,'samples_per_pixel':args.spp,
      'geometry':'Actual saved spherical cavities, radius 0.05; icosphere subdivision 4 for display only.',
      'cutaway':'Identical cylindrical display window: radius .355 about (x,y)=(-.025,-.18), z>.09; never part of physical design.',
      'lighting':'Area light plus environment; physical floor shadow divided by floor-only reference.',
      'denoising':'OIDN RT with albedo and normal guides; display radiance only, no experimental data filtered.',
      'preview':args.preview}
    (OUT/'render-manifest.json').write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
