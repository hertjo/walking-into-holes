"""Analytic-geometry internal cooling: slices (prep_internal_cooling.py), the saved TWoS cavities and the
independently evaluated trajectories (benchmarks/internal_cooling/output)."""
import os
import numpy as np
from splinx import draw as sx
from splinx.draw import nice_ticks
from hstyle import *
import hstyle
from hfield import heat, vbar, projector, box, sphere, TEMP_MAP, BEN_MAP
from s_common import cload, panel, finish

HERE = os.path.dirname(os.path.abspath(__file__))
S = np.load(os.path.join(HERE, "data/internal_cooling_slices.npz"))
TRAJ = cload("benchmarks/internal_cooling/output/design-trajectories.csv")
METH = [("TWoS + fresh validation", "TWoS + fresh validation", "TWoS + fresh", NAVY),
        ("TWoS + reused scout", "TWoS + reused scout", "TWoS + reuse", CYAN),
        ("Voxel adjoint TD", "Voxel adjoint TD", "voxel TD", VIOLET),
        ("Random feasible", "Random feasible", "random", GREY)]
W, H = TEXTW, 300
x, y = S["x"], S["y"]
EXT = S["half_extents"]
TMAX = float(np.nanmax(S["initial_temp"])); SMAX = float(np.nanmax(S["initial_score"]))
XL, YL = (float(x[0]), float(x[-1])), (float(y[0]), float(y[-1]))


def small_axes(P, xl="$x$", yl="$y$"):
    old = hstyle.TICK, hstyle.LAB
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(P, [(v, f"${v:g}$") for v in (-1, -0.5, 0, 0.5, 1)], [(v, f"${v:g}$") for v in (-0.5, 0, 0.5)], xl=xl, yl=yl,
         grid=False, xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old


def field_panel(x0, y0, arr, lo, hi, stops, name):
    w = 104.0; h = w * (YL[1] - YL[0]) / (XL[1] - XL[0])
    P = sx.Plot(x0, y0, w, h, XL, YL)
    heat(P, arr, lo, hi, stops, name)
    small_axes(P)
    return P


def draw():
    # (a) initial temperature
    panel("a", "initial temperature $u$", 4, 10)
    A = field_panel(26, 18, S["initial_temp"], 0, TMAX, TEMP_MAP, "temp0")
    vbar(A.x0 + A.w + 4, A.y0, 4, A.h, 0, TMAX, TEMP_MAP, [0, 0.05, 0.1], size=5.6)
    # (b) direct score with the selected sites in order
    panel("b", r"direct score $-D_T=u^2$", 162, 10)
    B = field_panel(182, 18, S["initial_score"], 0, SMAX, BEN_MAP, "score0")
    vbar(B.x0 + B.w + 4, B.y0, 4, B.h, 0, SMAX, BEN_MAP, [0, 0.004, 0.008, 0.012], size=5.6)
    for k, c in enumerate(S["centers"]):
        u, v = float(B.X(c[0])), float(B.Y(c[1]))
        sx.circle(u, v, 3.0, fill="#ffffff", stroke=INK, width=0.4)
        sx.tex(f"${k + 1}$", u, v, 4.6, INK, "center", valign="middle")
    # (c) the saved cavities in the block
    panel("c", "grid-free cooling cavities", 330, 10)
    f = projector(23, -55, 50.0, 398, 62)
    box(f, EXT)
    r = float(S["radius"])
    uv, depth = f(S["centers"])
    cols = [NAVY, "#2a3fc0", "#3a6fd8", CYAN, "#39c5c3", MINT, "#5fd3a6"]
    for i in np.argsort(depth):
        sphere(float(uv[i, 0]), float(uv[i, 1]), r * 50.0, cols[i % len(cols)])
    # (d) independently evaluated trajectories
    panel("d", "independently evaluated cooling", 4, 150)
    D = sx.Plot(36, 160, 108, 104, (-0.3, 7.3), (0.28, 1.04))
    old = hstyle.TICK, hstyle.LAB
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(D, [(i, f"${i}$") for i in range(8)], [(v, f"${v:.1f}$") for v in (0.4, 0.6, 0.8, 1.0)],
         xl="inserted cavities", yl="normalized compliance", xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old
    red = {}
    for key, _, short, c in METH:
        rs = [r_ for r_ in TRAJ if r_["method"] == key]
        it = np.array([float(r_["iteration"]) for r_ in rs]); ob = np.array([float(r_["objective"]) for r_ in rs])
        se = np.array([float(r_["objective_se"]) for r_ in rs])
        D.area(it, (ob - se) / ob[0], (ob + se) / ob[0], c, op=14)
        D.line(it, ob / ob[0], col=c, w=1.1)
        for a_, b_ in zip(it, ob / ob[0]):
            marker("o", float(D.X(a_)), float(D.Y(b_)), c, r=1.7)
        red[key] = 100 * (ob[0] - ob[-1]) / ob[0]
    legend_col([(m[1], m[3], "line") for m in METH], D.x0 + 4, float(D.Y(0.505)), size=5.6, gap=7.4)
    # (e) improvement after the last insertion
    panel("e", "improvement after 7 cavities", 162, 150)
    E = sx.Plot(214, 160, 92, 104, (0, 56), (0, 4))
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(E, [(v, f"${v}$") for v in range(0, 51, 10)], [], xl="compliance reduction (\\%)", grid="x", xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old
    for i, (key, _, short, c) in enumerate(METH):
        yc = E.y0 + E.h * (i + 0.5) / 4
        sx.rect(E.x0, yc - 8, float(E.X(red[key])) - E.x0, 16, fill=c, stroke=None)
        sx.tex(f"{red[key]:.1f}\\%", float(E.X(red[key])) + 2.5, yc, 6.0, INK, "left", valign="middle")
        sx.tex(short, E.x0 - 4, yc, 6.2, INK, "right", valign="middle")
    # (f) final temperature with the cavity sections
    panel("f", "final temperature $u$", 330, 150)
    F = field_panel(338, 158, S["final_temp"], 0, TMAX, TEMP_MAP, "temp1")
    vbar(F.x0 + F.w + 4, F.y0, 4, F.h, 0, TMAX, TEMP_MAP, [0, 0.05, 0.1], size=5.6)
    for c in S["centers"]:
        if abs(c[2]) < r:
            rr = float(np.sqrt(r * r - c[2] ** 2))
            sx.circle(float(F.X(c[0])), float(F.Y(c[1])), rr * F.w / (XL[1] - XL[0]), fill=None, stroke=MINT, width=1.0)
    finish()


sx.single("internal-cooling-benchmark", W, H, draw)
