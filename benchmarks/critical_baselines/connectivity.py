"""P1 FEM reproduction of Andrade et al.'s two-phase spectral operator.

This diagnostic reproduces equations (4)--(5), not their complete level-set
null-space optimizer. Geometric connectedness and exterior reachability are
computed separately. No connectivity guarantee is inferred from eta>0 alone.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from scipy.ndimage import label
from skfem import MeshTri,Basis,ElementTriP1,BilinearForm,asm
from skfem.helpers import dot,grad

OUT=Path(__file__).parent/'output'

def cavity(x,y,kind):
    if kind=='sealed_channel':return (np.abs(x)<.65)&(np.abs(y)<.12)
    if kind=='open_channel':return np.abs(y)<.12
    if kind=='two_open_components':return (np.abs(y-.45)<.12)|(np.abs(y+.45)<.12)
    if kind=='open_plus_island':return (np.abs(y+.45)<.12)|((x*x+(y-.4)**2)<.2**2)
    if kind=='thin_open_channel':return (np.abs(y)<.025)|((x*x+y*y)<.3**2)
    raise ValueError(kind)

@BilinearForm
def stiffness(u,v,w):return w.k*dot(grad(u),grad(v))
@BilinearForm
def mass(u,v,w):return w.k*u*v

def spectrum(n,eta,kind):
    mesh=MeshTri.init_tensor(np.linspace(-1,1,n),np.linspace(-1,1,n));basis=Basis(mesh,ElementTriP1())
    cent=mesh.p[:,mesh.t].mean(axis=1)
    phase=cavity(cent[0],cent[1],kind)
    k=np.where(phase,1.,eta)[:,None]*np.ones((1,basis.X.shape[1]))
    K=asm(stiffness,basis,k=k);M=asm(mass,basis,k=k)
    interior=basis.complement_dofs(basis.get_dofs())
    a=K[interior][:,interior];b=M[interior][:,interior]
    vals,vec=eigsh(a,k=3,M=b,sigma=0.,which='LM',tol=1e-10,
                  v0=np.random.default_rng(182).normal(size=len(interior)))
    residuals=[np.linalg.norm(a@v-l*b@v)/(np.linalg.norm(a@v)+abs(l)*np.linalg.norm(b@v))
               for l,v in zip(vals,vec.T)]
    return vals,float(max(residuals)),mesh.nelements

def topology(kind):
    # Independent fine Cartesian connectivity test. All cavities have resolved width.
    x=np.linspace(-1,1,801); xx,yy=np.meshgrid(x,x)
    mask=cavity(xx,yy,kind);labels,count=label(mask)
    exterior=set(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])-{0}
    return count,count-len(exterior)

def run():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    kinds=['sealed_channel','open_channel','two_open_components','open_plus_island','thin_open_channel']
    for n in [33,65,129]:
      for kind in kinds:
       components,islands=topology(kind)
       for eta in [1e-2,1e-4,1e-6]:
        start=time.perf_counter();vals,res,elements=spectrum(n,eta,kind)
        rows.append(dict(n=n,eta=eta,kind=kind,lambda1=vals[0],lambda2=vals[1],lambda3=vals[2],
                         residual=res,seconds=time.perf_counter()-start,elements=elements,
                         components=components,enclosed_components=islands))
       print('connectivity',n,kind,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'connectivity_spectrum.csv',index=False)
    # Homogeneous square independent eigenvalue sanity check: lambda1=pi²/2.
    val,res,_=spectrum(65,1.,'open_channel')
    assert abs(val[0]-np.pi**2/2)/(np.pi**2/2)<.005
    (OUT/'connectivity_manifest.json').write_text(json.dumps(dict(
      reference='https://doi.org/10.1007/s00158-026-04327-5',equations=[4,5],
      scope='spectral connectivity diagnostic; no level-set optimization or performance superiority claim',
      domain=[[-1,1],[-1,1]],boundary='Dirichlet on all exterior sides',
      cavity_coefficients=[1,1],solid_coefficients='k0=m0=eta',
      discretization='P1 triangles; centroid material labels; consistent mass',
      homogeneous_lambda1=float(val[0]),homogeneous_exact=float(np.pi**2/2),
      maximum_relative_eigen_residual=float(frame.residual.max())),indent=2))

if __name__=='__main__':run()
