"""Finite-offset and seam-strip audit on the frozen pre-refinement channel. Its run folder is not tracked; the values
are read back from the earlier vector figure by recover/offset_sweep.py (data/offset_seam_sweep.json)."""
import os, json, math
import numpy as np
from splinx import draw as sx
from hstyle import *
import hstyle
from s_common import panel, finish

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data/offset_seam_sweep.json")))
W, H = TEXTW, 170
L2, L10 = math.log2, math.log10
SER = [("0.025", r"$\ell/r=0.025$", NAVY, "o"), ("0.051", r"$\ell/r=0.051$", CYAN, "s"),
       ("0.203", r"$\ell/r=0.203$", VIOLET, "^")]
OFF = [(0.025, "$0.025$"), (0.05, "$0.05$"), (0.1, "$0.10$"), (0.2, "$0.20$")]


def ax(P, xt, yt, xl, yl, grid=True):
    old = hstyle.TICK, hstyle.LAB
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(P, xt, yt, xl=xl, yl=yl, grid=grid, xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old


def draw():
    # (a) frozen directional audit
    panel("a", "frozen directional audit", 4, 10)
    P = sx.Plot(40, 20, 104, 112, (-2.7, 2.7), (-2.7, 2.7))
    ax(P, [(v, f"${v}$") for v in (-2, 0, 2)], [(v, f"${v}$") for v in (-2, -1, 0, 1, 2)],
       r"Differential WoS ($\times10^{-3}$)", r"central difference ($\times10^{-3}$)")
    P.line([-2.7, 2.7], [-2.7, 2.7], col=GREY, w=0.7, dash=[2.4, 1.6])
    for key, lab, c, mk in SER:
        for r in D["a"][key]:
            x, y = float(P.X(r["fd_diffwos"])), float(P.Y(r["central"]))
            sx.multi([[[float(P.X(r["fd_lo"])), y], [float(P.X(r["fd_hi"])), y]],
                      [[x, float(P.Y(r["c_lo"]))], [x, float(P.Y(r["c_hi"]))]]], stroke=c, width=0.6, fill=None)
            marker(mk, x, y, c, r=2.0)
    legend_col([(lab, c, mk) for _, lab, c, mk in SER], P.x0 + 6, P.y0 + 8, size=5.8, gap=8.4)
    # (b) offset convergence
    panel("b", "offset convergence", 162, 10)
    B = D["b"]
    Q = sx.Plot(196, 20, 104, 112, (L2(0.022), L2(0.23)), (0.9, 3.25))
    ax(Q, [(L2(v), l) for v, l in OFF], [(v, f"${v:.1f}$") for v in (1.0, 1.5, 2.0, 2.5, 3.0)],
       r"normalized offset $\ell/r$", "FD-on-estimate slope")
    Q.line([Q.xa, Q.xb], [1, 1], col=GREY, w=0.6, dash=[2.4, 1.6])
    X = [L2(v) for v in B["offset"]]
    Q.line(X, B["fd_slope"], col=NAVY, w=1.1)
    for a, b in zip(X, B["fd_slope"]):
        marker("o", float(Q.X(a)), float(Q.Y(b)), NAVY, r=1.9)
    # right axis: cosine similarity on 0.975 .. 1.000
    lo, hi = 0.975, 1.0025
    to = lambda c: Q.ya + (c - lo) / (hi - lo) * (Q.yb - Q.ya)
    Q.line(X, [to(c) for c in B["cosine"]], col=MINT, w=1.1)
    for a, b in zip(X, B["cosine"]):
        marker("s", float(Q.X(a)), float(Q.Y(to(b))), MINT, r=1.8)
    xr = Q.x0 + Q.w
    sx.line(xr, Q.y0, xr, Q.y0 + Q.h, stroke=INK, width=0.6)
    for v in (0.975, 0.98, 0.985, 0.99, 0.995, 1.0):
        yy = float(Q.Y(to(v)))
        sx.line(xr, yy, xr + 2.5, yy, stroke=INK, width=0.5)
        sx.tex(f"${v:.3f}$", xr + 4.2, yy, 6.0, MINT, "left", valign="middle")
    ylabel("cosine similarity", xr + 33, Q.y0 + Q.h / 2, size=6.6)
    legend_col([("FD-on-DiffWoS slope", NAVY, "o"), ("cosine similarity", MINT, "s")], Q.x0 + 6, float(Q.Y(2.45)), size=5.8,
               gap=8.4)
    # (c) safe-strip rate
    panel("c", r"safe-strip rate ($\widehat p=1.27$)", 352, 10)
    C = D["c"]
    G = sx.Plot(370, 20, 92, 112, (L2(0.022), L2(0.23)), (L10(0.0028), L10(0.7)))
    ax(G, [(L2(v), l) for v, l in OFF], [(k, rf"$10^{{{k}}}$") for k in (-2, -1)], r"normalized offset $\ell/r$", None)
    segs = []
    for k in (-3, -2, -1):
        for m in range(2, 10):
            v = k + L10(m)
            if G.ya <= v <= G.yb:
                segs.append([[G.x0, float(G.Y(v))], [G.x0 - 1.3, float(G.Y(v))]])
    sx.multi(segs, stroke=INK, width=0.3, fill=None)
    g = D["guide"]
    G.line([L2(v) for v in g["offset"]], [L10(v) for v in g["rate"]], col=GREY, w=0.7, dash=[2.4, 1.6])
    for key, lab, c, mk in (("all_exits", "all exits", NAVY, "o"), ("given_cavity_exit", "given cavity exit", VIOLET, "s")):
        rs = C[key]
        X = [L2(r["offset"]) for r in rs]
        G.line(X, [L10(r["rate"]) for r in rs], col=c, w=1.1)
        for r, a in zip(rs, X):
            x = float(G.X(a))
            sx.line(x, float(G.Y(L10(r["lo"]))), x, float(G.Y(L10(r["hi"]))), stroke=c, width=0.6)
            marker(mk, x, float(G.Y(L10(r["rate"]))), c, r=1.9)
    sx.tex("extra segment failures: 3 / 32.9M", G.x0 + 3, G.y0 + 7, 5.4, MUTED, "left", bg="#ffffff")
    legend_col([(r"$O(\ell)$ guide", GREY, "dash"), ("all exits", NAVY, "o"), ("given cavity exit", VIOLET, "s")],
               G.x0 + G.w - 58, G.y0 + G.h - 24, size=5.6, gap=8)
    finish()


sx.single("offset-seam-sweep", W, H, draw)
