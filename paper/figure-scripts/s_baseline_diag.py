"""Baseline components: fixed-topology tracking, gradient audit, connectivity spectrum, 3D channel overlap audit."""
import math
from collections import defaultdict
import numpy as np
from s_common import *

OUTD = "benchmarks/critical_baselines/output"
SH = cload(OUTD, "shape_optimization.csv")
AU = cload(OUTD, "shape_gradient_audit.csv")
EIG = cload(OUTD, "connectivity_spectrum.csv")
OV = jload(OUTD, "channel_overlap_audit.json")
KINDS = [("sealed_channel", "sealed channel", INK), ("open_channel", "open channel", MINT),
         ("two_open_components", "two open components", CYAN), ("open_plus_island", "open + island", VIOLET),
         ("thin_open_channel", "thin open channel", NAVY)]
W, H = TEXTW, 348
L = math.log10


def draw():
    pw, ph = 112, 104
    xs = [48, 202, 356]
    top1, top2 = 22, 204
    t = np.linspace(0, 2 * np.pi, 240)
    # (a) ellipse tracking
    A = sx.Plot(xs[0] - 4, top1, 120, ph, (-1.15, 1.15), (-1.0, 1.0))
    A.y0 += 0  # equal aspect: 2.3 x 2.0 data in 120 x 104 pt
    axes(A, [(v, f"${v:g}$") for v in (-1, -0.5, 0, 0.5, 1)], [(v, f"${v:g}$") for v in (-0.5, 0, 0.5)],
         xl="$x$", yl="$y$", grid=False)
    panel("a", "Fixed-topology tracking", 14, top1 - 10)
    A.line(0.85 * np.cos(t), 0.85 * np.sin(t), col=GREY, w=1.0, dash=[3, 2])
    for r in SH:
        if r["method"] == "probabilistic" and r["step"] == "30":
            A.line(float(r["a"]) * np.cos(t), float(r["b"]) * np.sin(t), col=METHOD["twos"], w=0.7)
    A.line(np.cos(t), 0.65 * np.sin(t), col=INK, w=1.6, dash=[1.2, 2.0])
    legend_grid([("target", INK, "dot"), ("initial", GREY, "dash"), ("MC final (5 seeds)", METHOD["twos"], "line")],
                A.x0 - 20, A.y0 + ph + 34, cols=2, colw=58)
    # (b) objective trajectories
    B = sx.Plot(xs[1] + 4, top1, pw - 4, ph, (-1, 31), (L(8e-9), L(2e-3)))
    axes(B, [(v, f"${v}$") for v in (0, 10, 20, 30)], logticks(-8, -3), xl="accepted or attempted iteration",
         yl="tracking objective")
    panel("b", "Same state oracle and line search", xs[1] - 24, top1 - 10)
    runs = defaultdict(list)
    for r in SH:
        runs[(r["method"], r["seed"])].append((int(r["step"]), float(r["objective"])))
    for (m, sd), v in sorted(runs.items(), key=lambda kv: kv[0][0] == "deterministic"):
        v.sort()
        B.line([a for a, _ in v], [L(b) for _, b in v], col=INK if m == "deterministic" else METHOD["twos"],
               w=1.3 if m == "deterministic" else 0.8)
    legend_col([("deterministic", INK, "line"), ("probabilistic", METHOD["twos"], "line")], B.x0 + 50, B.y0 + 7)
    # (c) gradient audit
    g = defaultdict(list)
    for r in AU:
        g[int(r["n"])].append(float(r["error"]))
    n = np.array(sorted(g))
    rm = np.array([math.sqrt(np.mean(np.square(g[k]))) for k in n])
    C = sx.Plot(xs[2] + 4, top1, pw - 4, ph, (L(n[0]) - 0.1, L(n[-1]) + 0.1), (L(2.8e-4), L(2.0e-3)))
    axes(C, [(L(k), f"${k:,}$".replace(",", "{,}")) for k in n],
         [(L(v), lab) for v, lab in [(3e-4, r"$3{\cdot}10^{-4}$"), (6e-4, r"$6{\cdot}10^{-4}$"), (1e-3, r"$10^{-3}$")]],
         xl="exit paths", yl="gradient RMSE")
    panel("c", "Exact-gradient audit", xs[2] - 24, top1 - 10)
    C.line(np.log10(n), np.log10(rm[0] * np.sqrt(n[0] / n)), col=GREY, w=0.9, dash=[3, 2])
    line_mk(C, np.log10(n), np.log10(rm), METHOD["twos"], "o", w=1.3, r=2.3)
    legend_col([("gradient RMSE", METHOD["twos"], "o"), (r"$n^{-1/2}$ reference", GREY, "dash")], C.x0 + 40, C.y0 + 7)
    # (d) connectivity geometries, each the square [-1,1]^2 squashed vertically
    panel("d", "Connectivity test geometries", 14, top2 - 10)
    bw, bh = 82, 13.0
    for j, (kind, lab, col) in enumerate(KINDS):
        x0, y0 = 20, top2 + 6 + j * 22
        X = lambda x: x0 + (x + 1) / 2 * bw
        Y = lambda y: y0 + (1 - (y + 1) / 2) * bh
        sx.rect(x0, y0, bw, bh, fill="#ffffff", stroke=GREY, width=0.5)
        band = lambda xa, xb, ya, yb: sx.rect(X(xa), Y(yb), X(xb) - X(xa), Y(ya) - Y(yb), fill=col, stroke=None)
        ell = lambda cx, cy, r: sx.poly(np.c_[X(cx + r * np.cos(t)), Y(np.clip(cy + r * np.sin(t), -1, 1))], True,
                                        fill=col, stroke=None)
        if kind == "sealed_channel":
            band(-0.65, 0.65, -0.12, 0.12)
        elif kind == "open_channel":
            band(-1, 1, -0.12, 0.12)
        elif kind == "two_open_components":
            band(-1, 1, 0.33, 0.57); band(-1, 1, -0.57, -0.33)
        elif kind == "open_plus_island":
            band(-1, 1, -0.57, -0.33); ell(0, 0.4, 0.2)
        else:
            band(-1, 1, -0.025, 0.025); ell(0, 0, 0.3)
        sx.tex(lab, x0 + bw + 6, y0 + bh / 2, ANN, INK, "left", valign="middle")
    # (e) first Dirichlet eigenvalue against material contrast
    E = sx.Plot(xs[1] + 4, top2, pw - 4, ph, (-6.2, -1.8), (L(1.5e-5), L(5)))
    axes(E, [(k, f"$10^{{{k}}}$") for k in (-6, -5, -4, -3, -2)], logticks(-4, 0),
         xl=r"material contrast $k_0=m_0$", yl=r"first eigenvalue $\lambda_1$")
    panel("e", "Exterior reachability spectrum", xs[1] - 24, top2 - 10)
    for kind, lab, col in KINDS:
        g = sorted([r for r in EIG if r["n"] == "129" and r["kind"] == kind], key=lambda r: float(r["eta"]))
        line_mk(E, [L(float(r["eta"])) for r in g], [L(float(r["lambda1"])) for r in g], col, "o", w=1.1, r=1.9)
    note("colors as in (d)", E.x0 + E.w - 3, float(E.Y(L(1e-4))), anchor="right")
    # (f) saved 3D channel overlap audit
    F = sx.Plot(xs[2] + 4, top2, pw - 4, ph, (-0.6, 2.6), (0, 1.15))
    axes(F, [(i, d["stage"]) for i, d in enumerate(OV)], [(v, f"${v:.1f}$") for v in np.arange(0, 1.01, 0.2)],
         yl="largest gap / diameter", grid="y")
    panel("f", "Saved 3D channel audit", xs[2] - 24, top2 - 10)
    bcol = {"refined": METHOD["twos"], "straight": GREY, "topology": VIOLET}
    for i, d in enumerate(OV):
        xa, xb = float(F.X(i - 0.32)), float(F.X(i + 0.32))
        ya, yb = float(F.Y(d["max_gap_over_diameter"])), float(F.Y(0))
        sx.rect(xa, ya, xb - xa, yb - ya, fill=bcol[d["stage"]], stroke=None)
        sx.tex(f"${d['max_gap_over_diameter']:.2f}$", (xa + xb) / 2, ya - 2.5, ANN - 0.4, INK, "center")
    F.line([-0.6, 2.6], [1, 1], col=MUTED, w=0.8, dash=[3, 2])
    note(r"strict overlap requires ratio $<1$", F.x0 + F.w / 2, float(F.Y(1)) - 3, anchor="center")
    finish()


sx.single("baseline-diagnostics", W, H, draw)
