"""Shared 2-D geometry for the method diagrams: the peanut domain and genuine walk-on-spheres paths on it."""
import numpy as np


def peanut(cx, cy, sx_, sy_, n=240, tilt=0.08):
    """Closed two-lobed outline centred at (cx, cy), half extents about sx_ by sy_ (y down)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = (1 + 0.36 * np.cos(2 * t)) * (1 + tilt * np.cos(t))
    x = r * np.cos(t); y = r * np.sin(t)
    x = x / np.abs(x).max(); y = y / np.abs(y).max()
    return np.c_[cx + sx_ * x, cy + sy_ * y]


def seg_dist(p, P):
    """Distance from point p to the closed polygon P, and the nearest point on it."""
    A = P; B = np.roll(P, -1, 0)
    AB = B - A; AP = p - A
    t = np.clip((AP * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-12), 0, 1)
    Q = A + t[:, None] * AB
    d = np.hypot(*(Q - p).T)
    k = d.argmin()
    return d[k], Q[k]


def wos(z, P, rng, eps=0.6, holes=(), max_steps=200, rmax=None):
    """Walk on spheres from z inside polygon P (pt units). holes: [(centre, radius)] absorbing discs.
    Returns (points, radii, where) with where = 'outer' or the hole index."""
    pts = [np.array(z, float)]; rad = []
    p = np.array(z, float)
    for _ in range(max_steps):
        d, q = seg_dist(p, P)
        where = "outer"
        for k, (c, r) in enumerate(holes):
            dh = np.hypot(*(p - c)) - r
            if dh < d:
                d, where = dh, k
                q = c + (p - c) / np.hypot(*(p - c)) * r
        if d < eps:
            pts.append(q)
            return np.array(pts), np.array(rad), where
        if rmax is not None:  # any ball inside the domain keeps the mean-value property
            d = min(d, rmax)
        rad.append(d)
        a = rng.uniform(0, 2 * np.pi)
        p = p + d * np.array([np.cos(a), np.sin(a)])
        pts.append(p.copy())
    return np.array(pts), np.array(rad), None
