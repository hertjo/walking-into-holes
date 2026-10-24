"""Field panels shared by the supplement figures: masked heat maps, vertical colour bars, a simple 3-D projection."""
import math
import numpy as np
from splinx import draw as sx
from hstyle import *

TEMP_MAP = [(0.0, INK), (0.3, "#34299a"), (0.58, VIOLET), (0.84, LILAC), (1.0, "#f6f5ff")]
BEN_MAP = [(0.0, INK), (0.32, "#1f2fa8"), (0.64, CYAN), (0.87, MINT), (1.0, "#e6f8f0")]


def heat(P, arr, lo, hi, stops, name, upsample=4):
    """arr[i, j]: i along y (bottom row first); NaN cells transparent. Bilinear upsampling keeps the look smooth."""
    a = np.asarray(arr, float)
    if upsample > 1:
        ny, nx = a.shape
        yi = np.linspace(0, ny - 1, (ny - 1) * upsample + 1); xi = np.linspace(0, nx - 1, (nx - 1) * upsample + 1)
        y0 = np.floor(yi).astype(int).clip(0, ny - 2); x0 = np.floor(xi).astype(int).clip(0, nx - 2)
        ty = (yi - y0)[:, None]; tx = (xi - x0)[None, :]
        A = a[y0][:, x0]; B = a[y0][:, x0 + 1]; C = a[y0 + 1][:, x0]; D = a[y0 + 1][:, x0 + 1]
        a = (1 - ty) * ((1 - tx) * A + tx * B) + ty * ((1 - tx) * C + tx * D)
    mask = np.isfinite(a)
    rgb = sx.cmap(np.where(mask, a, lo), lo, hi, stops)
    rgba = np.dstack([rgb, np.uint8(mask * 255)])[::-1]
    sx.image(np.ascontiguousarray(rgba), P.x0, P.y0, P.w, P.h, name=name, smooth=True)


def vbar(x, y, w, h, lo, hi, stops, ticks, size=6.0, fmt=None, label=None):
    arr = sx.cmap(np.linspace(1, 0, 256)[:, None].repeat(2, 1), 0, 1, stops)
    sx.image(arr, x, y, w, h, name="vbar", smooth=True)
    sx.rect(x, y, w, h, fill=None, stroke=INK, width=0.35)
    for v in ticks:
        yy = y + h * (1 - (v - lo) / (hi - lo))
        sx.line(x + w, yy, x + w + 1.6, yy, stroke=INK, width=0.35)
        sx.tex(fmt(v) if fmt else f"${v:g}$", x + w + 2.6, yy, size, INK, "left", valign="middle")
    if label:
        ylabel(label, x + w + 22, y + h / 2, size=size)


def projector(elev, azim, scale, cx, cy):
    """Orthographic view like matplotlib's view_init(elev, azim): world (x, y, z) -> page (u, v) with v down."""
    e, a = math.radians(elev), math.radians(azim)
    right = np.array([-math.sin(a), math.cos(a), 0.0])
    up = np.array([-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e)])
    view = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])

    def f(p):
        p = np.atleast_2d(np.asarray(p, float))
        return np.c_[cx + scale * p @ right, cy - scale * p @ up], p @ view
    return f


def box(f, ext, col=GREY, w=0.6):
    ex, ey, ez = ext
    corners = np.array([[sx_ * ex, sy_ * ey, sz_ * ez] for sx_ in (-1, 1) for sy_ in (-1, 1) for sz_ in (-1, 1)])
    uv, _ = f(corners)
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(np.abs(corners[i] - corners[j]) > 1e-9) == 1:
                edges.append([uv[i].tolist(), uv[j].tolist()])
    sx.multi(edges, stroke=col, width=w, fill=None)


def sphere(u, v, r, col):
    """A shaded sphere: base disc, a soft darker rim and a small highlight."""
    sx.circle(u, v, r, fill=col, stroke=INK, width=0.3)
    sx.circle(u - 0.28 * r, v - 0.3 * r, 0.42 * r, fill="#ffffff", stroke=None, opacity=35)
    sx.circle(u - 0.33 * r, v - 0.36 * r, 0.16 * r, fill="#ffffff", stroke=None, opacity=70)


def raster_mesh(V, F, f, rect, color="#c9c6ff", k=8.0, light=(-0.45, -0.70, 0.65)):
    """Flat-shaded triangle mesh drawn back to front into an RGBA raster covering rect=(x, y, w, h) in page pt.
    f is a projector (see projector); returns the uint8 array for sx.image(..., *rect)."""
    from PIL import Image, ImageDraw
    x0, y0, w, h = rect
    W_, H_ = int(round(w * k)), int(round(h * k))
    T = V[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    L = np.asarray(light, float); L /= np.linalg.norm(L)
    inten = 0.6 + 0.4 * np.abs(n @ L)
    base = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], float)
    uv, depth = f(T.reshape(-1, 3))
    uv = uv.reshape(-1, 3, 2); depth = depth.reshape(-1, 3).mean(1)
    im = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    for i in np.argsort(depth):
        c = tuple(int(v) for v in np.clip(inten[i] * base, 0, 255)) + (255,)
        pts = [((p[0] - x0) * k, (p[1] - y0) * k) for p in uv[i]]
        dr.polygon(pts, fill=c, outline=c)
    return np.asarray(im)


def read_ascii_ply(path):
    """Vertices and triangle faces of an ASCII PLY (x y z vertices, '3 i j k' faces)."""
    with open(path) as fh:
        nv = nf = 0
        while True:
            line = fh.readline().strip()
            if line.startswith("element vertex"):
                nv = int(line.split()[-1])
            elif line.startswith("element face"):
                nf = int(line.split()[-1])
            elif line == "end_header":
                break
        V = np.loadtxt(fh, max_rows=nv, usecols=(0, 1, 2))
        F = np.loadtxt(fh, max_rows=nf, dtype=int)[:, 1:4]
    return V, F


def section(V, F, axis, cut):
    """Segments where the plane coordinate[axis] = cut crosses the mesh."""
    T = V[F]
    s = T[:, :, axis] - cut
    segs = []
    for i in np.flatnonzero((s.min(1) < 0) & (s.max(1) > 0)):
        hits = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if s[i, a] * s[i, b] < 0:
                w = -s[i, a] / (s[i, b] - s[i, a])
                hits.append(T[i, a] + w * (T[i, b] - T[i, a]))
        if len(hits) == 2:
            segs.append(np.vstack(hits))
    return segs
