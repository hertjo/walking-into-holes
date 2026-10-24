"""Accessible-channel plate inputs (display only). (1) Recolour the Mitsuba renders: render_professional.py writes
the section pixels as inferno(u / 0.07) after denoising, so they are inverted exactly and re-mapped through the paper
temperature map; the depth-tested ghost overlay (RGBA) is then composited as in the original. (2) The admissible-axis
mask of the priority panel, computed as in compose_professional.py (sensor/device clearances, the bounding box and the
exact point-mesh distance to the housing). Needs libigl. Writes assets/accessible/*.png and data/accessible_mask.npz."""
import os
import numpy as np
from PIL import Image
import matplotlib
import igl

REPO = os.environ.get("HOLES_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
A = f"{REPO}/benchmarks/accessible_channels/render_assets"
HERE = os.path.dirname(os.path.abspath(__file__))
INK, NAVY, VIOLET, LILAC = "#1e1b3a", "#0500b5", "#6b67ee", "#c9c6ff"
TEMP = [(0.0, INK), (0.3, "#34299a"), (0.58, VIOLET), (0.84, LILAC), (1.0, "#f6f5ff")]


def lut(stops, n=256):
    t = np.linspace(0, 1, n); ts = np.array([s for s, _ in stops])
    cs = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for _, c in stops])
    return np.stack([np.interp(t, ts, cs[:, i]) for i in range(3)], -1)


SRC = matplotlib.colormaps["inferno"](np.linspace(0, 1, 256))[:, :3]
DST = lut(TEMP)


def recolour(name, ghost=None):
    T = np.asarray(Image.open(f"{A}/{name}.png").convert("RGB"), float) / 255
    flat = T.reshape(-1, 3).copy()
    hsv = matplotlib.colors.rgb_to_hsv(flat)
    cand = np.flatnonzero((hsv[:, 1] > 0.2) | (hsv[:, 2] < 0.12))
    best = np.zeros(len(cand), int); dist = np.full(len(cand), 9.0)
    for s in range(0, len(cand), 20000):
        d = ((flat[cand[s:s + 20000], None, :] - SRC[None]) ** 2).sum(-1)
        best[s:s + 20000] = d.argmin(1); dist[s:s + 20000] = np.sqrt(d.min(1))
    sec = dist < 0.04
    flat[cand[sec]] = DST[best[sec]]
    out = flat.reshape(T.shape)
    if ghost:
        G = np.asarray(Image.open(f"{A}/{ghost}.png").convert("RGBA"), float) / 255
        out = out * (1 - G[..., 3:]) + G[..., :3] * G[..., 3:]
    Image.fromarray(np.uint8(np.clip(out * 255 + 0.5, 0, 255))).save(f"{HERE}/assets/accessible/{name}.png")
    return int(sec.sum())


for m in ["wos_hot", "twos_sparse", "mg57"]:
    print(m, recolour(m, f"{m}-ghost-overlay"), recolour(f"{m}-detail"), flush=True)

# admissible axes (compose_professional.py)
R, DEVICES = 0.075, np.array([[-.48, -.18, -.08], [.43, .19, -.07], [.02, -.31, -.02]])
sensor = np.array([.55, .18, -.03])
ini = np.load(f"{A}/initial-field.npz")
xx, yy = np.meshgrid(ini["x"], ini["y"])
forbidden = np.hypot(xx - sensor[0], yy - sensor[1]) < R + .13 + .02
for c in DEVICES:
    forbidden |= np.hypot(xx - c[0], yy - c[1]) < R + .07 + .02
forbidden |= (np.abs(xx) > .84) | (np.abs(yy) > .41)
v, f = igl.read_triangle_mesh(f"{REPO}/benchmarks/engineering_housing/assets/stylized-finned-housing.ply")
pts = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, -.03)))
d, _, _ = igl.point_mesh_squared_distance(pts, v, f)
forbidden |= np.sqrt(d).reshape(xx.shape) <= R + .023
np.savez_compressed(f"{HERE}/data/accessible_mask.npz", forbidden=forbidden)
print("mask", int(forbidden.sum()))
