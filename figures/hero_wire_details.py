"""Visibility-tested display triangulation on analytic cavity walls.

Edges are spherical arcs, projected by the detail camera and culled by the
actual rendered geometry. This is a visual depth cue, not a solver mesh.
"""
import argparse,json
import numpy as np
import pandas as pd
import trimesh
from render_mesh_hero import mi,OUT,BUNNY,PURPLE,scene_dict

def run(tag,method,count,centers_override=None,radius=.05):
    parts={k:OUT/f'{tag}-{k}.ply' for k in ['shell','section','cavity']
           if (OUT/f'{tag}-{k}.ply').exists()}
    scene=mi.load_dict(scene_dict(parts,PURPLE,32,1,zoom=True))
    c=pd.read_csv(BUNNY/'selected-centers.csv')
    c=c[c.method==method].sort_values('order')[['x','y','z']].values[:count]
    if centers_override is not None:c=np.asarray(centers_override,dtype=float)
    ico=trimesh.creation.icosphere(subdivisions=2,radius=1)
    e=ico.edges_unique; a=ico.vertices[e[:,0]]; b=ico.vertices[e[:,1]]
    t=np.linspace(0,1,9)
    q=(1-t[None,:,None])*a[:,None,:]+t[None,:,None]*b[:,None,:]
    q/=np.linalg.norm(q,axis=-1,keepdims=True)
    points=(c[:,None,None,:]+(.997*radius)*q[None,:,:,:]).reshape(-1,3)
    target=np.array([.065,-.16,.065]);d=-np.array([1.3,.52,3.5]);d/=np.linalg.norm(d)
    origins=np.ascontiguousarray(np.asarray((points-3*d).T),dtype=np.float32)
    hit=scene.ray_intersect(mi.Ray3f(mi.Point3f(origins),mi.Vector3f(d)))
    visible=(np.abs(np.asarray(hit.t)-3)<.001).reshape(len(c),len(e),len(t))
    left=np.cross([0,1,0],d);left/=np.linalg.norm(left);up=np.cross(d,left)
    relative=points-target
    uv=np.stack([.5-relative@left/.45,.5+relative@up/.45],axis=-1).reshape(len(c),len(e),len(t),2)
    segments=np.stack([uv[:,:,:-1],uv[:,:,1:]],axis=-2)[visible[:,:,:-1]&visible[:,:,1:]]
    (OUT/f'{tag}-wire.json').write_text(json.dumps(segments.tolist(),separators=(',',':')))
    print(tag,len(segments),'visible edge segments',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only');args=p.parse_args()
    for tag,method,n in [('voxel','Voxel adjoint TD',6),('random','Random feasible',6),
          ('ours1','TWoS + fresh validation',1),('ours3','TWoS + fresh validation',3),
          ('ours6','TWoS + fresh validation',6)]:
        if not args.only or args.only==tag:run(tag,method,n)
