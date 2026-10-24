"""From a topological field to a connected, refined cooling channel. The run folder of this study is not tracked:
panels (a)-(e) use the values read back from the earlier vector figure (recover/housing_channels.py; the read-back
reproduces the stated r = 0.9982 and slope 1.341), panel (f) the values stated in the paper."""
import os, json
import numpy as np
from PIL import Image
import matplotlib
from splinx import draw as sx
from hstyle import *
import hstyle
from hfield import vbar, BEN_MAP
from s_common import panel, finish

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data/housing_channels.json")))
PL = D["placement"]
# compliance reduction (%) and standard error at matched nominal volume, as stated in the paper
F = [("disjoint", "TWoS", 68.964, 0.050, GREY), ("straight", "channel", 57.269, 0.080, CYAN),
     ("TD graph", "channel", 57.325, 0.048, VIOLET), ("TD +", "DiffWoS", 58.215, 0.037, NAVY)]
W, H = TEXTW, 300
K = 1.45  # enlargement of the old 3-D panels


def tint(key):
    im = np.asarray(Image.open(os.path.join(HERE, "assets/channels", key + ".png")).convert("RGBA")).astype(float) / 255
    rgb, a = im[..., :3], im[..., 3:]
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    s = hsv[..., 1] > 0.06
    hsv[..., 0] = np.where(s, 243 / 360, hsv[..., 0])
    if key == "tube":
        hsv[..., 1] = np.where(s, np.clip(hsv[..., 1] * 0.8, 0, 1), hsv[..., 1])
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.25, 0, 1)
    else:
        hsv[..., 1] = np.where(s, hsv[..., 1] * 0.9, hsv[..., 1])
        hsv[..., 2] = np.clip(hsv[..., 2] * 1.15 + 0.05, 0, 1)
    out = matplotlib.colors.hsv_to_rgb(hsv)
    return np.uint8(np.clip(np.dstack([out, a]) * 255 + 0.5, 0, 255))


def placer(key, x0, y0):
    p = PL[key]
    ox, oy = p["x"], p["y"]

    def f(px, py):
        return x0 + (px - ox) * K, y0 + (py - oy) * K
    return f, p["w"] * p["scale"] * K, p["h"] * p["scale"] * K


def shell(key, x0, y0):
    f, w, h = placer(key, x0, y0)
    sx.image(tint(key), x0, y0, w, h, name="ch-" + key, smooth=True)
    return f


def draw():
    old = hstyle.TICK, hstyle.LAB
    # (a) TD-weighted connection graph
    panel("a", "TD-weighted connection graph", 4, 10)
    f = shell("a", 8, 22)
    vmax = 0.003013
    dots = sorted(D["dots"], key=lambda d: d[2])
    cs = sx.cmap(np.array([d[2] for d in dots]), 0, vmax, BEN_MAP)
    for d, c in zip(dots, cs):
        u, v = f(d[0], d[1])
        sx.circle(u, v, 0.55, fill="#%02x%02x%02x" % tuple(c), stroke=None)
    vbar(146, 30, 4, 70, 0, vmax, BEN_MAP, [0, 1e-3, 2e-3, 3e-3], size=5.4, fmt=lambda t: f"${t * 1e3:.0f}$")
    sx.tex(r"$\times10^{-3}$", 144, 26, 5.2, INK, "left")
    # (b) connected ball-union channel
    panel("b", "connected ball-union channel", 162, 10)
    fb = shell("b", 164, 22)
    pt = PL["tube"]
    tx, ty = fb(pt["x"], pt["y"])
    sx.image(tint("tube"), tx, ty, pt["w"] * pt["scale"] * K, pt["h"] * pt["scale"] * K, name="ch-tube", smooth=True)
    sx.tex("22 overlapping balls; equal volume", 236, 130, 5.8, MUTED, "center")
    # (c) Differential WoS refinement, with the pre-refinement centre line dashed
    panel("c", "Differential WoS refinement", 318, 10)
    fc = shell("c", 318, 22)
    segs = [[list(fc(*q)) for q in s] for s in D["centre_line"]]
    sx.multi(segs, stroke=CYAN, width=0.7, fill=None)
    # (d) fresh-sample acceptance test
    panel("d", "fresh-sample acceptance test", 4, 152)
    P = sx.Plot(40, 162, 104, 104, (0.5, 6.5), (-1.02, 0.82))
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(P, [(i, f"${i}$") for i in range(1, 7)], [(v, f"${v:.2f}$") for v in (-1.0, -0.5, 0.0, 0.5)],
         xl="shape iteration", yl=r"paired change in $J$ ($\times10^{-4}$)", xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old
    P.line([0.5, 6.5], [0, 0], col=GREY, w=0.6, dash=[2.4, 1.6])
    for r in D["d"]:
        c = NAVY if r["accepted"] else VIOLET
        x = float(P.X(r["iteration"]))
        yl, yh = float(P.Y(r["lo"])), float(P.Y(r["hi"]))
        sx.multi([[[x, yl], [x, yh]], [[x - 2, yl], [x + 2, yl]], [[x - 2, yh], [x + 2, yh]]], stroke=c, width=0.6,
                 fill=None)
        marker("o" if r["accepted"] else "s", x, float(P.Y(r["change"])), c, r=2.0)
    legend_col([("accepted", NAVY, "o"), ("rejected", VIOLET, "s")], P.x0 + P.w - 42, P.y0 + P.h - 16, size=5.8,
               gap=8.4)
    # (e) directional-derivative check
    panel("e", "directional-derivative check", 162, 152)
    Q = sx.Plot(204, 162, 104, 104, (-0.0027, 0.0027), (-0.0027, 0.0027))
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    t = [(v, f"${v * 1e3:g}$") for v in (-0.002, -0.001, 0, 0.001, 0.002)]
    axes(Q, t, t, xl=r"Differential WoS ($\times10^{-3}$)", yl=r"central finite difference ($\times10^{-3}$)",
         xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old
    Q.line([-0.0027, 0.0027], [-0.0027, 0.0027], col=GREY, w=0.6, dash=[2.4, 1.6])
    for r in D["e"]:
        x = float(Q.X(r["diffwos"]))
        sx.line(x, float(Q.Y(r["lo"])), x, float(Q.Y(r["hi"])), stroke=NAVY, width=0.6)
        marker("o", x, float(Q.Y(r["central"])), NAVY, r=2.0)
    sx.tex(r"$r=0.998$, slope $=1.34$", Q.x0 + 4, Q.y0 + 8, 5.8, INK, "left", bg="#ffffff")
    # (f) equal-volume design quality
    panel("f", "equal-volume design quality", 330, 152)
    G = sx.Plot(362, 162, 100, 104, (-0.5, 3.5), (56.5, 70.8))
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(G, [(i, "") for i in range(4)], [(v, f"${v}$") for v in range(58, 71, 2)],
         yl=r"compliance reduction (\%)", grid="y")
    hstyle.TICK, hstyle.LAB = old
    for i, (l1, l2, val, se, c) in enumerate(F):
        x, y = float(G.X(i)), float(G.Y(val))
        sx.line(x, float(G.Y(val - se)), x, float(G.Y(val + se)), stroke=c, width=0.6)
        marker("o", x, y, c, r=2.4)
        sx.tex(f"{val:.1f}\\%", x, y - 5.5, 5.6, INK, "center", bg="#ffffff")
        sx.tex(l1, x, G.y0 + G.h + 7.5, 5.4, INK, "center")
        sx.tex(l2, x, G.y0 + G.h + 13.5, 5.4, INK, "center")
    gain = F[3][2] - F[2][2]
    sx.arrow((float(G.X(2.25)), float(G.Y(61.6))), (float(G.X(2.78)), float(G.Y(59.3))), col=MUTED, w=0.5, head=3.0)
    sx.tex(f"$+{gain:.2f}$ pp", float(G.X(2.2)), float(G.Y(62.2)), 5.6, MUTED, "center")
    finish()


sx.single("engineering-housing-channels", W, H, draw)
