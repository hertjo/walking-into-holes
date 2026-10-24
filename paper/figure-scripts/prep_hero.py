"""Recolour the path-traced Bunny renders of the hero into the paper palette (display only).

The thermal renders carry the field on the planar section through a colormap applied without shading (inferno for
temperature, viridis for benefit, white isotherms at alpha 55/255). Section pixels are found by differencing against
the field-free render of the same design, the colormap is inverted per pixel, and the value is re-mapped through the
paper's maps. The shells keep their path-traced shading and are tinted (TWoS lavender, baselines cyan-grey)."""
import os, colorsys
import numpy as np
from PIL import Image
import matplotlib
from hstyle import INK, NAVY, VIOLET, LILAC, CYAN, MINT

REPO = os.environ.get("HOLES_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
A = f"{REPO}/figures/hero-assets"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/hero")
TEMP = [(0.0, INK), (0.3, "#34299a"), (0.58, VIOLET), (0.84, LILAC), (1.0, "#f6f5ff")]
BEN = [(0.0, INK), (0.32, "#1f2fa8"), (0.64, CYAN), (0.87, MINT), (1.0, "#e6f8f0")]
ALPHA = 55 / 255
SPECS = [("input", "ours0", "temp", "twos"), ("ours0", "ours0", "ben", "twos"), ("random", "random", "temp", "base"),
         ("voxel", "voxel", "temp", "base"), ("ours1", "ours1", "temp", "twos"), ("budget6", "budget6", "temp", "twos"),
         ("budget12", "budget12", "temp", "twos")]


def lut(stops, n=256):
    t = np.linspace(0, 1, n); ts = np.array([s for s, _ in stops])
    cs = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for _, c in stops])
    return np.stack([np.interp(t, ts, cs[:, i]) for i in range(3)], -1)


def nearest(px, table):
    # px (n,3), table (256,3) -> index, distance
    best = np.zeros(len(px), int); dist = np.full(len(px), 9.0)
    for s in range(0, len(px), 200000):
        d = ((px[s:s + 200000, None, :] - table[None]) ** 2).sum(-1)
        best[s:s + 200000] = d.argmin(1); dist[s:s + 200000] = np.sqrt(d.min(1))
    return best, dist


def tint(rgb, hue, sat_scale):
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    hsv[..., 0] = hue
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 1)
    return matplotlib.colors.hsv_to_rgb(hsv)


def recolour(tag, base, kind, who, suffix):
    T = np.asarray(Image.open(f"{A}/thermal/{tag}{suffix}.png").convert("RGB"), float) / 255
    B = np.asarray(Image.open(f"{A}/{base}{suffix}.png").convert("RGB"), float) / 255
    src = matplotlib.colormaps["viridis" if kind == "ben" else "inferno"](np.linspace(0, 1, 256))[:, :3]
    dst = lut(BEN if kind == "ben" else TEMP)
    out = T.copy()
    sec = np.abs(T - B).max(-1) > 0.035
    px = T[sec]
    i1, d1 = nearest(px, src)
    i2, d2 = nearest((px - ALPHA) / (1 - ALPHA), src)
    iso = d2 < d1
    idx = np.where(iso, i2, i1); d = np.minimum(d1, d2)
    new = dst[idx]
    new[iso] = new[iso] * (1 - ALPHA) + ALPHA
    keep = d > 0.09  # cross marker, antialiased rim: keep as rendered
    new[keep] = px[keep]
    out[sec] = new
    # shells: saturated, non-section pixels of the field-free render
    hsv = matplotlib.colors.rgb_to_hsv(B)
    shell = (~sec) & (hsv[..., 1] > 0.08) & (hsv[..., 2] > 0.05)
    if who == "twos":
        out[shell] = tint(B[shell], 243 / 360, 0.85)
    else:
        out[shell] = tint(B[shell], 196 / 360, 0.55)
    # rim pixels kept from the thermal render that are still inferno-coloured: blend toward the shell tint
    Image.fromarray(np.uint8(np.clip(out * 255 + 0.5, 0, 255))).save(f"{OUT}/{tag}{suffix}.png")
    return int(sec.sum()), float(np.mean(keep)), float(np.mean(iso))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for tag, base, kind, who in SPECS:
        for suffix in ("", "-detail"):
            print(tag, suffix, recolour(tag, base, kind, who, suffix), flush=True)
