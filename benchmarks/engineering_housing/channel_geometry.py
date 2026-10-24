"""Geometry-only feasibility checks, independent of rendering/PDE packages."""
import numpy as np

def strictly_overlapping_chain(centers, radius, relative_margin=1e-10):
    centers=np.asarray(centers,dtype=float)
    if centers.ndim!=2 or len(centers)==0 or not np.all(np.isfinite(centers)):
        return False
    if not np.isfinite(radius) or radius<=0:
        return False
    gaps=np.linalg.norm(np.diff(centers,axis=0),axis=1)
    return bool(np.all(gaps<2*radius*(1-relative_margin)))
