"""Accessible through-channels: saved first-seed right-sensor designs (renders recoloured by prep_accessible.py), the
initial temperature and admissible priority, and the measured cooling and design time
(benchmarks/accessible_channels: render_assets/, output/results.csv)."""
import os, json
import numpy as np
from PIL import Image
from splinx import draw as sx
from hstyle import *
import hstyle
from hfield import heat, TEMP_MAP, BEN_MAP
from s_common import R, cload, jload, panel, finish

HERE = os.path.dirname(os.path.abspath(__file__))
A = R("benchmarks/accessible_channels/render_assets")
AS = os.path.join(HERE, "assets/accessible")
META = jload("benchmarks/accessible_channels/render_assets/manifest.json")
CFG = jload("benchmarks/accessible_channels/output/configuration.json")
DATA = cload("benchmarks/accessible_channels/output/results.csv")
SEED = META["source_seed"]
VMAX = META["temperature_scale_max"]
METH = [("wos_hot", "Hottest-point placement", "ranks temperature $u$", "Hottest", METHOD["hottest"]),
        ("twos_sparse", "TWoS placement", "ranks regional benefit $up$", "TWoS", METHOD["twos"]),
        ("mg57", "Multigrid placement", "ranks $up$ on a volumetric grid", "Multigrid", METHOD["multigrid"])]
W, H = TEXTW, 300


def crop(name):
    im = Image.open(os.path.join(AS, name + ".png")).convert("RGB")
    w, h = im.size
    im = im.crop((0, round(0.14 * h), w, round(0.93 * h))).resize((620, 490), Image.LANCZOS)
    return np.asarray(im)


def circ(name, n=300):
    a = np.asarray(Image.open(os.path.join(AS, name + ".png")).convert("RGB").resize((n, n), Image.LANCZOS))
    yy, xx = np.mgrid[:n, :n]
    alpha = np.clip(n / 2 - np.hypot(xx + 0.5 - n / 2, yy + 0.5 - n / 2) + 0.5, 0, 1)
    return np.dstack([a, np.uint8(alpha * 255)])


def draw():
    old = hstyle.TICK, hstyle.LAB
    IW, IH = 152.0, 152.0 * 0.79
    for i, (m, name, sub, _, c) in enumerate(METH):
        x = 2 + i * 156.0
        sx.tex(name, x + 3, 8, 7.6, c, "left")
        sx.tex(sub, x + 3, 16, 6.2, INK, "left")
        y = 20
        sx.image(crop(m), x, y, IW, IH, name=m, smooth=True)
        sp = np.load(os.path.join(A, f"{m}-projection.npz"))["sensor_pixel"]
        fx, fy = x + sp[0] * IW, y + (sp[1] - 0.14) / 0.79 * IH
        cx, cy, r = x + 0.82 * IW, y + (0.79 - 0.605) / 0.79 * IH, 0.17 / 0.79 * IH
        sx.line(fx, fy, cx - 0.62 * r, cy + 0.72 * r, stroke=c, width=0.6)
        sx.circle(fx, fy, 0.9, fill="#ffffff", stroke=c, width=0.5)
        sx.circle(cx, cy, r, fill="#ffffff", stroke=None)
        sx.image(circ(m + "-detail"), cx - r, cy - r, 2 * r, 2 * r, name=m + "-d", smooth=True)
        sx.circle(cx, cy, r, fill=None, stroke=c, width=0.9)
        row = [q for q in DATA if q["side"] == "1" and int(q["seed"]) == SEED and q["method"] == m][0]
        sx.tex(f"{100 * float(row['reduction']):.1f}\\% lower regional temperature", x + 3, y + IH + 9, 6.6, c, "left")
    # shared colour bar
    y = 20 + IH + 19
    arr = sx.cmap(np.linspace(0, 1, 256)[None, :].repeat(2, 0), 0, 1, TEMP_MAP)
    sx.image(arr, 318, y + 4, 100, 3.6, name="cb", smooth=True)
    sx.rect(318, y + 4, 100, 3.6, fill=None, stroke=INK, width=0.35)
    for t in (0, 0.5, 1):
        sx.tex(f"${t * VMAX:.3f}$", 318 + 100 * t, y + 10, 5.6, INK, "center", valign="top")
    sx.tex("temperature $u$ (shared scale)", 318, y + 1.5, 6.0, INK, "left")
    # bottom left: original temperature and admissible priority
    ini = np.load(os.path.join(A, "initial-field.npz"))
    mask = ini["mask"].astype(bool)
    forb = np.load(os.path.join(HERE, "data/accessible_mask.npz"))["forbidden"]
    xs, ys = ini["x"], ini["y"]
    geo = jload("benchmarks/accessible_channels/render_assets/twos_sparse-geometry.json")
    sensor = (0.55, 0.18)
    y0 = 196
    for j, (key, title, stops) in enumerate((("u", "original temperature $u$", TEMP_MAP),
                                              ("score", "admissible priority $up$", BEN_MAP))):
        x0 = 6 + j * 116
        panel("", title, x0, y0)
        w = 104.0; h = w * (ys[-1] - ys[0]) / (xs[-1] - xs[0])
        P = sx.Plot(x0, y0 + 6, w, h, (float(xs[0]), float(xs[-1])), (float(ys[0]), float(ys[-1])))
        val = ini[key].astype(float).copy(); val[~mask] = np.nan
        if key == "score":
            val[forb] = np.nan; val /= np.nanmax(val)
        heat(P, val, 0, VMAX if key == "u" else 1, stops, key, upsample=1)
        P.implicit(lambda X, Y: np.interp(0, [0, 1], [0, 1]) + 0 * X + _mask_interp(X, Y, xs, ys, mask), 0.5,
                   col=GREY, w=0.5)
        sx.circle(float(P.X(sensor[0])), float(P.Y(sensor[1])), 0.13 * w / (xs[-1] - xs[0]), fill=None, stroke=CYAN,
                  width=1.0)
        if key == "score":
            for c in geo["source_centers"]:
                sx.circle(float(P.X(c[0])), float(P.Y(c[1])), 1.8, fill=None, stroke=NAVY, width=0.8)
            sx.tex("blank: excluded axes; circles: final TWoS axes", x0 + w / 2, P.y0 + P.h + 8, 5.4, MUTED, "center")
    # priority colour bar
    arr = sx.cmap(np.linspace(1, 0, 256)[:, None].repeat(2, 1), 0, 1, BEN_MAP)
    sx.image(arr, 228, y0 + 8, 3.6, 56, name="pb", smooth=True)
    sx.rect(228, y0 + 8, 3.6, 56, fill=None, stroke=INK, width=0.35)
    sx.tex("$1$", 234, y0 + 8, 5.4, INK, "left", valign="middle"); sx.tex("$0$", 234, y0 + 64, 5.4, INK, "left", valign="middle")
    sx.tex("rel.", 226, y0 + 3, 5.2, INK, "left")
    # bottom right: cooling and design time over both sides and five seeds
    means, se, times = [], [], []
    for m, *_ in METH:
        g = [q for q in DATA if q["method"] == m]
        seeds = sorted({q["seed"] for q in g})
        paired = np.array([np.mean([100 * float(q["reduction"]) for q in g if q["seed"] == s]) for s in seeds])
        means.append(paired.mean()); se.append(paired.std(ddof=1) / np.sqrt(len(paired)))
        times.append(np.mean([float(q["complete_design_seconds"]) for q in g]))
    C = sx.Plot(302, 206, 92, 62, (0, 70), (-0.6, 2.6))
    T = sx.Plot(408, 206, 54, 62, (0, 1.65), (-0.6, 2.6))
    hstyle.TICK, hstyle.LAB = 6.0, 6.4
    axes(C, [(v, f"${v}$") for v in (0, 20, 40, 60)], [], grid="x")
    axes(T, [(v, f"${v:g}$") for v in (0, 0.5, 1, 1.5)], [], grid="x")
    hstyle.TICK, hstyle.LAB = old
    sx.tex(r"regional cooling (\%)", C.x0 + C.w / 2, 199, 6.6, INK, "center")
    sx.tex("design time (s)", T.x0 + T.w / 2, 199, 6.6, INK, "center")
    for j, ((m, _, _, short, c), mu, s_, t) in enumerate(zip(METH, means, se, times)):
        yc = float(C.Y(2 - j))
        sx.rect(C.x0, yc - 4.5, float(C.X(mu)) - C.x0, 9, fill=c, stroke=None)
        xl, xh = float(C.X(mu - s_)), float(C.X(mu + s_))
        sx.multi([[[xl, yc], [xh, yc]], [[xl, yc - 2], [xl, yc + 2]], [[xh, yc - 2], [xh, yc + 2]]], stroke=INK, width=0.5,
                 fill=None)
        sx.tex(f"{mu:.1f}\\%", xh + 2.5, yc, 6.0, INK, "left", valign="middle")
        sx.tex(short, C.x0 - 4, yc, 6.2, INK, "right", valign="middle")
        sx.rect(T.x0, yc - 4.5, float(T.X(t)) - T.x0, 9, fill=c, stroke=None)
        sx.tex(f"${t:.2f}$", float(T.X(t)) + 2.5, yc, 6.0, INK, "left", valign="middle")
    finish()


def _mask_interp(X, Y, xs, ys, mask):
    """Nearest-cell lookup of the material mask, for its outline."""
    i = np.clip(np.searchsorted(ys, Y) - 1, 0, len(ys) - 1)
    j = np.clip(np.searchsorted(xs, X) - 1, 0, len(xs) - 1)
    return mask[i, j].astype(float)


sx.single("accessible-channels", W, H, draw)
