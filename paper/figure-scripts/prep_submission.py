"""Recolour the matched-study renders (benchmarks/submission_study/render_assets) into the paper palette (display only).

render_study.py paints the planar section with viridis (relative cooling, unshaded), overlays a depth-tested ghost of
the removed top surface (grey .46/.49/.56 at alpha .09, .32 on silhouettes), then draws the sensor (#00ccda) and
heat-source (#ec9844) outlines without antialiasing. Here the ghost alpha and the colormap are inverted per pixel, the
cooling value is re-mapped through COOL and recomposited with the same alpha, the outlines get palette colours and the
method-tinted shells one neutral lavender tint."""
import os, sys
import numpy as np
from PIL import Image
import matplotlib
from hstyle import INK, NAVY, VIOLET, LILAC, CYAN, MINT

REPO = os.environ.get("HOLES_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
A = f"{REPO}/benchmarks/submission_study/render_assets"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/submission")
COOL = [(0.0, INK), (0.3, NAVY), (0.6, VIOLET), (0.84, LILAC), (1.0, "#f6f5ff")]
GHOST = np.array([.46, .49, .56])
SENSOR, SOURCE = CYAN, MINT


def lut(stops, n=256):
    t = np.linspace(0, 1, n); ts = np.array([s for s, _ in stops])
    cs = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for _, c in stops])
    return np.stack([np.interp(t, ts, cs[:, i]) for i in range(3)], -1)


def rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def nearest(px, table):
    best = np.zeros(len(px), int); dist = np.full(len(px), 9.0)
    for s in range(0, len(px), 20000):
        d = ((px[s:s + 20000, None, :] - table[None]) ** 2).sum(-1)
        best[s:s + 20000] = d.argmin(1); dist[s:s + 20000] = np.sqrt(d.min(1))
    return best, dist


SRC = matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))[:, :3]
DST = lut(COOL)


def recolour(name):
    T = np.asarray(Image.open(f"{A}/{name}.png").convert("RGB"), float) / 255
    H, W, _ = T.shape
    out = T.copy()
    flat = T.reshape(-1, 3)
    # outlines (exact colours)
    o_sen = np.abs(flat - rgb("#00ccda")).max(1) < 0.02
    o_src = np.abs(flat - rgb("#ec9844")).max(1) < 0.02
    best_d = np.full(len(flat), 9.0); best_i = np.zeros(len(flat), int); best_a = np.zeros(len(flat))
    hsv = matplotlib.colors.rgb_to_hsv(flat)
    cand = np.flatnonzero((hsv[:, 1] > 0.12) & ~o_sen & ~o_src)
    for a in (0.0, 0.09, 0.32):
        inv = (flat[cand] - a * GHOST) / (1 - a)
        i, d = nearest(inv, SRC)
        better = d < best_d[cand]
        best_d[cand[better]] = d[better]; best_i[cand[better]] = i[better]; best_a[cand[better]] = a
    sec = best_d < 0.05
    new = DST[best_i[sec]] * (1 - best_a[sec, None]) + best_a[sec, None] * GHOST
    o = out.reshape(-1, 3)
    o[sec] = new
    o[o_sen] = rgb(SENSOR); o[o_src] = rgb(SOURCE)
    # shells: remaining tinted pixels -> one lavender tint
    shell = (~sec) & (~o_sen) & (~o_src) & (hsv[:, 1] > 0.04) & (hsv[:, 1] < 0.6)
    h2 = hsv[shell].copy(); h2[:, 0] = 243 / 360; h2[:, 1] = np.clip(h2[:, 1] * 0.75, 0, 1)
    o[shell] = matplotlib.colors.hsv_to_rgb(h2)
    Image.fromarray(np.uint8(np.clip(o.reshape(H, W, 3) * 255 + 0.5, 0, 255))).save(f"{OUT}/{name}.png")
    return int(sec.sum()), int(shell.sum())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or sorted(f[:-4] for f in os.listdir(A) if f.endswith(".png"))
    for n in names:
        print(n, recolour(n), flush=True)
