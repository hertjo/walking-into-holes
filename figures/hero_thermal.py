"""Measured WoS temperature fields on the hero's physical interior section.

The large display cut never enters the PDE domain. Fields use the original
Bunny plus the saved analytic cavities. Ray intersections project those
fields onto exactly the existing render's section. A documented display-only
Gaussian filter suppresses pixel-scale Monte Carlo grain. Raw fields and all
reported metrics are retained without filtering or per-panel rescaling.
"""
from pathlib import Path
import argparse,json,subprocess,sys,time
import numpy as np
import pandas as pd
import igl
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from PIL import Image,ImageDraw
from render_mesh_hero import mi,scene_dict,PURPLE

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'figures/hero-assets'
OUT=ASSETS/'thermal';OUT.mkdir(exist_ok=True)
BUNNY=ROOT/'benchmarks/stanford_bunny/output'
SPECS=[('initial',None,0),('random','Random feasible',6),
       ('voxel','Voxel adjoint TD',6),('ours1','TWoS + fresh validation',1),
       ('ours3','TWoS + fresh validation',3),('ours6','TWoS + fresh validation',6)]
CMAP='inferno'
BENEFIT_CMAP='viridis'

def fields(walks=256):
    executable=Path('/tmp/walking-into-holes-embree-build/wih_twos')
    if not executable.exists():
        sys.path.insert(0,str(ROOT/'benchmarks/compiled_bunny'))
        from run_compiled_benchmark import build_cpp
        executable=build_cpp()
    accuracy_check(executable)
    v,f=igl.read_triangle_mesh(str(BUNNY/'stanford-bunny-watertight.ply'))
    x=np.linspace(v[:,0].min(),v[:,0].max(),160)
    y=np.linspace(v[:,1].min(),v[:,1].max(),200)
    xx,yy=np.meshgrid(x,y)
    points=np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,.09)])
    inside=igl.signed_distance(points,v,f,igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]<0
    centers=pd.read_csv(BUNNY/'selected-centers.csv')
    for tag,method,n in SPECS:
        if (OUT/f'{tag}.npz').exists():
            cached=json.loads((OUT/f'{tag}-solver.json').read_text())
            if cached['walks_per_site']!=walks:
                raise ValueError(f'{tag}: cached field uses a different path budget; preserve or move it before recomputing')
            continue
        c=centers[centers.method==method].sort_values('order')[['x','y','z']].values[:n]
        active=inside.copy()
        if len(c):active &= np.min(np.linalg.norm(points[:,None]-c[None,:],axis=2),axis=1)>.05
        pd.DataFrame(points[active],columns=['x','y','z']).to_csv(OUT/f'{tag}-points.csv',index=False)
        pd.DataFrame(c,columns=['x','y','z']).to_csv(OUT/f'{tag}-centers.csv',index=False)
        cmd=[str(executable),'--mesh',str(BUNNY/'stanford-bunny-watertight.ply'),
          '--points',str(OUT/f'{tag}-points.csv'),'--output',str(OUT/f'{tag}-field.csv'),
          '--summary',str(OUT/f'{tag}-solver.json'),'--walks',str(walks),'--repeats','1',
          '--threads','4','--epsilon','0.0005','--max-steps','512','--seed','2026091407',
          '--source-y-slope','0.65','--cavity-radius','0.05']
        if len(c):cmd += ['--cavities',str(OUT/f'{tag}-centers.csv')]
        print('Evaluating',tag,int(active.sum()),'section points;',walks,'paths each',flush=True)
        subprocess.run(cmd,check=True)
        df=pd.read_csv(OUT/f'{tag}-field.csv')
        if df.limit_hits.sum():raise RuntimeError('WoS path cap hit')
        u=np.zeros(len(points));benefit=u.copy()
        u[active]=df.u.values;benefit[active]=df.score.values
        np.savez_compressed(OUT/f'{tag}.npz',x=x,y=y,u=u.reshape(xx.shape),
           benefit=benefit.reshape(xx.shape),active=active.reshape(xx.shape),centers=c)
    write_manifest(walks)

def write_manifest(walks):
    initial=np.load(OUT/'initial.npz')
    umax=float(initial['u'].max())
    temperature_max=float(np.ceil(umax/.005)*.005)
    metadata=dict(pde='-Delta u = 1 + 0.65*(y+0.5)',boundary='u=0 on exterior and analytic cavity walls',
      section_z=.09,grid_shape=[200,160],paths_per_point=walks,epsilon=.0005,
      temperature_max=temperature_max,initial_section_max=umax,
      benefit_max=float(initial['benefit'].max()),temperature_colormap=CMAP,
      benefit_colormap=BENEFIT_CMAP,scalar_smoothing='display-only Gaussian, sigma=1.0 grid cells; zero restored outside material',
      interpolation='bilinear on display field; raw point estimates retained separately',
      geometry_cut='visualization only; excluded from all PDE solves',
      objective='source-weighted temperature averaged over fixed original-domain probes',
      benefit='cross-index product (sum(U)^2-sum(U^2))/(N*(N-1)); normalized by initial section maximum',
      notes='Fresh display evaluations separate from optimization, with a common base seed across panels; saved objective trajectories are unchanged. Ideal cooled cavities, not a fluid-flow simulation.')
    (OUT/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Shared temperature scale',temperature_max,flush=True)

def accuracy_check(executable):
    retained=ROOT/'benchmarks/compiled_bunny/output'
    reference=OUT/'accuracy-reference-64.csv'
    if not reference.exists():
        subprocess.run([str(executable),'--mesh',str(BUNNY/'stanford-bunny-watertight.ply'),
          '--points',str(retained/'evaluation-points-0064.csv'),'--output',str(reference),
          '--summary',str(OUT/'accuracy-reference-64.json'),'--walks','16384','--repeats','1',
          '--threads','4','--epsilon','0.0005','--max-steps','512','--seed','2026091419',
          '--source-y-slope','0.65'],check=True)
    df=pd.read_csv(reference);ref=df.score.values
    result={}
    for method in ['twos','voxel']:
        values=pd.read_csv(retained/f'field-output-{method}-0064.csv').score.values
        result[method+'_rrmse_16384']=float(np.linalg.norm(values-ref)/np.linalg.norm(ref))
    result.update(reference_relative_score_se=float(np.linalg.norm(df.score_se)/np.linalg.norm(ref)),
      reference_walks_per_site=16384,reference_is_exact=False,
      timings_source='Retained matched C++ output-scaling.csv, 8 threads, median of three repeats; fresh reference timing is not a performance comparison.')
    (OUT/'accuracy-check.json').write_text(json.dumps(result,indent=2)+'\n')

def project(tag,base,field_name,benefit=False,zoom=False):
    """Replace only the scientific section colors; retain shaded surface renders.

    The colormap is applied without illumination modulation, so the shared
    colorbar maps exactly to the field. Cavity-wall shading remains visible.
    """
    meta=json.loads((OUT/'manifest.json').read_text())
    data=np.load(OUT/f'{field_name}.npz')
    suffix='-detail' if zoom else ''
    original=np.asarray(Image.open(ASSETS/f'{base}{suffix}.png').convert('RGB'))/255.
    size=original.shape[0]
    meshes={k:ASSETS/f'{base}-{k}.ply' for k in ['shell','section','cavity'] if (ASSETS/f'{base}-{k}.ply').exists()}
    scene=mi.load_dict(scene_dict(meshes,PURPLE,size,1,zoom))
    yy,xx=np.mgrid[:size,:size]
    sample=mi.Point2f(np.array([(xx.ravel()+.5)/size,(yy.ravel()+.5)/size],dtype=np.float32))
    ray,_=scene.sensors()[0].sample_ray(0,0,sample,mi.Point2f(.5))
    hit=scene.ray_intersect(ray)
    p=np.asarray(hit.p).T
    valid=np.asarray(hit.is_valid())
    cap=valid & (np.abs(p[:,2]-.09)<2e-5) & (np.abs(np.asarray(hit.n).T[:,2])>.999)
    raw=data['benefit'] if benefit else data['u']
    displayed=gaussian_filter(raw,sigma=1.0,mode='constant')
    displayed[~data['active']]=0.
    scale=meta['benefit_max'] if benefit else meta['temperature_max']
    field=RegularGridInterpolator((data['y'],data['x']),displayed,bounds_error=False,fill_value=0.)
    values=field(np.column_stack([p[cap,1],p[cap,0]]))/scale
    colors=mpl.colormaps[BENEFIT_CMAP if benefit else CMAP](np.clip(values,0,1))[:,:3]
    rgb=original.reshape(-1,3).copy();rgb[cap]=colors
    # Exterior and cavity-wall material shading are geometric context. Only
    # the planar section carries the scalar colormap, preserving concavity.
    image=Image.fromarray(np.uint8(np.clip(rgb.reshape(size,size,3)*255,0,255)))
    target=np.array([.065,-.16,.065]) if zoom else np.array([-.01,-.015,0.])
    scale_view=.225 if zoom else .62
    direction=-np.array([1.3,.52,3.5]);direction/=np.linalg.norm(direction)
    left=np.cross([0,1,0],direction);left/=np.linalg.norm(left);up=np.cross(direction,left)
    def pixels(q):
        return np.column_stack([.5-(q-target)@left/(2*scale_view),.5-(q-target)@up/(2*scale_view)])*size
    # Constant-width isotherms, clipped to visible section pixels. Drawing
    # level bands would incorrectly make contours thick in low-gradient areas.
    fig,ax=plt.subplots()
    cs=ax.contour(data['x'],data['y'],displayed/scale,levels=[.15,.3,.45,.6,.75,.9])
    overlay=Image.new('RGBA',(size,size),(0,0,0,0));draw=ImageDraw.Draw(overlay)
    for segs in cs.allsegs:
        for seg in segs:
            if len(seg)<2:continue
            xy=pixels(np.column_stack([seg,np.full(len(seg),.09)]))
            draw.line([tuple(v) for v in xy],fill=(255,255,255,55),width=2)
    plt.close(fig)
    rgba=np.array(overlay);rgba[:,:,3]*=cap.reshape(size,size)
    image=Image.alpha_composite(image.convert('RGBA'),Image.fromarray(rgba)).convert('RGB')
    if benefit:
        c=pd.read_csv(BUNNY/'selected-centers.csv')
        first=c[c.method=='TWoS + fresh validation'].sort_values('order')[['x','y','z']].values[0].copy()
        first[2]=.09
        px,py=pixels(first[None,:])[0];r=12 if not zoom else 16
        draw=ImageDraw.Draw(image)
        for a,b in [((px-r,py),(px+r,py)),((px,py-r),(px,py+r))]:
            draw.line([a,b],fill='black',width=5)
            draw.line([a,b],fill='white',width=3)
    image.save(OUT/f'{tag}{suffix}.png')
    print('Projected',tag,suffix,'onto',int(cap.sum()),'section pixels',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-only',action='store_true');ap.add_argument('--walks',type=int,default=256)
    args=ap.parse_args()
    if not args.project_only:fields(args.walks)
    else:write_manifest(args.walks)
    for tag,base,field,b in [('input','ours0','initial',False),('ours0','ours0','initial',True),
           ('random','random','random',False),('voxel','voxel','voxel',False),
           ('ours1','ours1','ours1',False),('ours3','ours3','ours3',False),('ours6','ours6','ours6',False)]:
        for zoom in [False,True]:project(tag,base,field,b,zoom)

if __name__=='__main__':main()
