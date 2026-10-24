"""Method diagram: the cross-fitted (leave-one-out) product estimator."""
import numpy as np
from splinx import draw as sx
from hstyle import *
from hgeom import peanut, wos

sx.OUT = "out"
W, H = 468, 172
TT, SUB, BODY = 7.6, 6.2, 7.0
C1, C2, C3 = 78.0, 234.0, 390.0
ADJ_FILL = "#e3f5fc"
HL = "#eeedfd"


def draw():
    sx.line(156, 6, 156, 138, stroke=GRID, width=0.6)
    sx.line(312, 6, 312, 138, stroke=GRID, width=0.6)
    # (a) reuse geometric work
    sx.tex("(a) Reuse geometric work", C1, 9.5, TT, INK, "center")
    sx.tex("Independent randomizations", C1, 19.0, SUB, MUTED, "center")
    D = peanut(C1, 66, 62, 25, tilt=0.12)
    D[:, 1] += 0.0
    sx.poly(D, True, fill="#f6f5fc", stroke=INK, width=0.9)
    z = np.array([C1 - 6.0, 69.0])
    walks = []
    conds = [lambda e: e[1] < 50 and e[0] < C1 - 18, lambda e: e[1] < 46 and C1 - 2 < e[0] < C1 + 26,
             lambda e: e[1] > 80 and e[0] > C1 + 10]
    for k, cond in enumerate(conds):
        for seed in range(6000):
            pts, rad, _ = wos(z, D, np.random.default_rng(1000 * k + seed), eps=0.8, rmax=6.0)
            if 4 <= len(rad) <= 8 and cond(pts[-1]):
                walks.append((pts, rad)); break
    pts, rad = walks[1]
    for p, r in zip(pts[:-1], rad):
        sx.circle(p[0], p[1], r, fill=None, stroke=GREY, width=0.35, opacity=70)
    for k, (pts, rad) in enumerate(walks):
        sx.poly(pts, stroke=VIOLET if k == 1 else MUTED, width=0.95 if k == 1 else 0.6, fill=None)
        e = pts[-1]
        off = {0: (-7, -5), 1: (8, -5), 2: (6, 6)}[k]
        sx.tex(f"$X_{k + 1}$", e[0] + off[0], e[1] + off[1], 6.8, VIOLET if k == 1 else MUTED, "center", valign="middle")
    sx.circle(*z, 1.4, fill=INK, stroke=None)
    sx.tex("$z$", z[0] - 4.0, z[1] + 6.5, 6.6, INK, "center")
    sx.tex(r"$X_i=(U_i,P_i)$", C1, 112, BODY + 0.4, INK, "center")
    sx.tex("Primal and adjoint may share the walk", C1, 123, SUB, MUTED, "center")
    # (b) hold out the matching adjoint
    sx.tex("(b) Hold out the matching adjoint", C2, 9.5, TT, INK, "center")
    sx.tex("Example: use the second primal", C2, 19.0, SUB, MUTED, "center")
    xp, xu, xa = C2 - 58, C2 - 12, C2 + 38
    sx.tex("pair", xp, 33, SUB - 0.2, MUTED, "center")
    sx.tex("primal", xu, 33, SUB - 0.2, NAVY, "center")
    sx.tex("adjoint", xa, 33, SUB - 0.2, MUTED, "center")
    ys = [48, 70, 92]
    sx.rect(C2 - 72, ys[1] - 9.5, 144, 19, radius=1.5, fill=HL, stroke=None)
    for i, y in enumerate(ys):
        sx.tex(f"$X_{i + 1}$", xp, y, BODY, VIOLET if i == 1 else MUTED, "center", valign="middle")
        if i == 1:
            sx.rect(xu - 19, y - 7.5, 38, 15, radius=1.5, fill="#ffffff", stroke=NAVY, width=0.7)
        sx.tex(f"$U_{i + 1}-h_z$", xu, y, BODY, NAVY if i == 1 else INK, "center", valign="middle")
        if i == 1:
            w = sx.tex("$P_2$", xa, y - 1.5, BODY, GREY, "center", valign="middle")
            sx.line(xa - w / 2 - 1.5, y - 0.8, xa + w / 2 + 1.5, y - 2.6, stroke=GREY, width=0.6)
            sx.tex("held out", xa, y + 7.0, SUB - 0.8, MUTED, "center", valign="middle")
        else:
            sx.rect(xa - 17, y - 7.5, 34, 15, radius=1.5, fill=ADJ_FILL, stroke=None)
            sx.tex(f"$P_{i + 1}$", xa, y, BODY, INK, "center", valign="middle")
    bx = xa + 24
    sx.poly([[bx - 3, ys[0]], [bx, ys[0]], [bx, ys[2]], [bx - 3, ys[2]]], stroke=CYAN, width=0.8, fill=None)
    sx.line(bx, (ys[0] + ys[2]) / 2, bx + 3, (ys[0] + ys[2]) / 2, stroke=CYAN, width=0.8)
    sx.tex(r"$\bar P_{-2}=\dfrac{P_1+P_3}{2}$", C2, 106, BODY, INK, "center", valign="top")
    # (c) multiply, then average
    sx.tex("(c) Multiply, then average", C3, 9.5, TT, INK, "center")
    sx.tex("Two independent factors per term", C3, 19.0, SUB, MUTED, "center")
    sx.tex(r"$C_2=-(U_2-h_z)\bar P_{-2}$", C3, 50, BODY + 1.4, INK, "center", valign="middle")
    sx.tex("Repeat this holdout for every pair", C3, 68, SUB, MUTED, "center")
    sx.tex(r"$\widehat D_T=\dfrac{1}{N}\sum\nolimits_{i=1}^N C_i$", C3, 88, BODY + 1.4, VIOLET, "center", valign="middle")
    sx.tex(r"$\bar P_{-i}=\dfrac{S_P-P_i}{N-1}$", C3, 114, BODY + 0.6, INK, "center", valign="middle")
    # strip
    sx.line(4, 143, W - 4, 143, stroke=GRID, width=0.6)
    w = sx.tex("One adjoint sum", 6, 160, SUB + 0.2, INK, "left")
    sx.tex(r"$\textstyle S_P=\sum_j P_j$", 6 + w + 10, 160, BODY + 0.4, INK, "left")
    sx.tex("$O(N)$ total work", C2, 160, SUB + 0.4, INK, "center")
    sx.tex("$N$ independent pairs; products share samples", C3, 160, SUB, MUTED, "center")
    for msg in sx.check():
        print(msg)


sx.single("crossfit", W, H, draw)
