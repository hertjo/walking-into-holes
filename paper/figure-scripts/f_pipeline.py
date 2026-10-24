"""Method diagram: the conditional insertion policy (score, freeze, check, insert) and the general-objective guard."""
import numpy as np
from splinx import draw as sx
from hstyle import *
from hgeom import wos

sx.OUT = "out"
W, H = 468, 150
PW = W / 4
TT, SUB, BODY = 7.6, 6.2, 6.6
DOM_FILL = "#f6f5fc"


def bean(cx, cy, a, b, n=220):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(t) + 0.05 * np.cos(2 * t)
    y = np.sin(t) * (1 + 0.14 * np.cos(t)) - 0.08 * np.sin(2 * t) ** 2
    return np.c_[cx + a * x, cy + b * y]


def head(i, title, sub):
    x = i * PW + PW / 2
    wn, wt = sx.texw(f"${i + 1}$", TT), sx.texw(title, TT)
    x0 = x - (wn + 4 + wt) / 2
    sx.tex(f"${i + 1}$", x0, 9.5, TT, VIOLET, "left")
    sx.tex(title, x0 + wn + 4, 9.5, TT, INK, "left")
    sx.tex(sub, x, 19.0, SUB, MUTED, "center")
    return x


def domain(cx, cy=52.0):
    D = bean(cx, cy, 44, 24)
    sx.poly(D, True, fill=DOM_FILL, stroke=INK, width=0.9)
    return D


def draw():
    for k in (1, 2, 3):
        sx.line(k * PW, 6, k * PW, 121, stroke=GRID, width=0.6)
    # 1 score
    x = head(0, "Score the current domain", "No trial cavity")
    domain(x)
    for (dx, dy, r, c) in [(-28, 47, 1.3, GREY), (-19, 39, 1.5, GREY), (-7, 54, 1.4, GREY), (-21, 58, 1.8, GREY),
                           (12, 41, 1.4, GREY), (-12, 66, 1.2, GREY), (22, 56, 2.0, VIOLET)]:
        sx.circle(x + dx, dy, r, fill=c, stroke=None, opacity=75 if c == GREY else None)
    sx.circle(x + 4, 55, 1.9, fill=VIOLET, stroke=None)
    sx.circle(x + 4, 55, 4.0, fill=None, stroke=VIOLET, width=0.7)
    sx.tex(r"$\widehat D_T(z)= -\operatorname{avg}_{i\ne j}(U_i-h_z)P_j$", x, 91, BODY, INK, "center")
    sx.tex("Rank feasible, separated candidates", x, 102, SUB, MUTED, "center")
    # 2 freeze
    x = head(1, "Freeze a finite proposal", "Fix center and radius")
    domain(x)
    c = (x + 10, 54.0)
    sx.circle(*c, 7.5, fill=None, stroke=VIOLET, width=0.8, dash=[2.2, 1.6])
    sx.circle(*c, 1.0, fill=VIOLET, stroke=None)
    sx.line(c[0], c[1], c[0] + 7.5, c[1], stroke=VIOLET, width=0.6)
    sx.tex(r"$\varepsilon$", c[0] + 3.6, c[1] - 2.0, 6.2, VIOLET, "center")
    sx.tex("Choose the proposal before validation", x, 91, BODY, INK, "center")
    sx.tex("The dashed circle is not yet a hole", x, 102, SUB, MUTED, "center")
    # 3 check
    x = head(2, "Check with fresh samples", "Account for all error terms")
    D = domain(x)
    z = np.array([x + 2.0, 57.0])

    def pick(base, cond):
        for seed in range(4000):
            pts, rad, _ = wos(z, D, np.random.default_rng(base + seed), eps=0.8, rmax=7.0)
            if cond(pts, rad):
                return pts, rad
    prim = pick(10, lambda p, r: 3 <= len(r) <= 6 and p[-1][1] < 34 and abs(p[-1][0] - x + 8) < 12)
    adj = pick(500, lambda p, r: 4 <= len(r) <= 7 and p[-1][0] > x + 38)
    for (pts, rad), col in ((prim, NAVY), (adj, CYAN)):
        for p, r in zip(pts[:-1], rad):
            sx.circle(p[0], p[1], r, fill=None, stroke=GREY, width=0.35, opacity=70)
    for (pts, rad), col in ((prim, NAVY), (adj, CYAN)):
        sx.poly(pts, stroke=col, width=0.8, fill=None)
    sx.circle(*z, 1.3, fill=VIOLET, stroke="#ffffff", width=0.4)
    sx.tex("$z$", z[0] - 3.5, z[1] + 6.0, 6.2, INK, "center")
    # covered upper bound on a number line
    y0 = 89.0
    sx.line(x - 34, y0, x + 34, y0, stroke=INK, width=0.5)
    zx = x + 20
    sx.line(zx, y0 - 5, zx, y0 + 3, stroke=INK, width=0.6)
    sx.tex("$0$", zx, y0 + 9.5, 6.0, INK, "center")
    lo, mid, hi = x - 28, x - 19, x + 6
    sx.line(lo, y0, hi, y0, stroke=VIOLET, width=1.8)
    sx.line(lo, y0 - 3, lo, y0 + 3, stroke=INK, width=0.5)
    sx.line(hi, y0 - 3, hi, y0 + 3, stroke=VIOLET, width=0.8)
    sx.circle(mid, y0, 1.4, fill=VIOLET, stroke=None)
    sx.tex("covered upper bound", x - 22, y0 - 6.0, SUB, VIOLET, "left")
    sx.tex(r"Pass only when the upper bound is $<0$", x, 108, SUB, INK, "center")
    sx.tex("Fail: resample under a valid schedule, or skip", x, 118, SUB - 0.4, MUTED, "center")
    # 4 insert
    x = head(3, "Insert, then refine", "Only after the test passes")
    domain(x)
    c = (x + 6, 54.0)
    sx.circle(*c, 7.5, fill="#ffffff", stroke=VIOLET, width=1.1)
    sx.circle(c[0] + 1.5, c[1] - 0.6, 7.1, fill=None, stroke=CYAN, width=0.6, dash=[1.8, 1.4])
    sx.arrow((c[0] + 6.0, c[1] - 5.0), (c[0] + 10.5, c[1] - 8.5), col=CYAN, w=0.6, head=2.6)
    sx.tex(r"$\Omega\leftarrow\Omega\setminus K_{\varepsilon,z}$", x, 91, BODY + 0.6, INK, "center")
    sx.tex("Optional Differential WoS shape step", x, 102, SUB, MUTED, "center")
    # guard
    sx.line(4, 126, W - 4, 126, stroke=GRID, width=0.6)
    sx.tex("General-objective guard", 6, 141, SUB + 0.2, INK, "left")
    sx.tex(r"$\widehat D_T+r_N(\delta)+b_{\rm term}+b_{\rm cache}+C_z\rho_z(\varepsilon)^\alpha<0$", W / 2 + 30, 141.5,
           BODY + 1.4, INK, "center")
    for msg in sx.check():
        print(msg)


sx.single("pipeline", W, H, draw)
