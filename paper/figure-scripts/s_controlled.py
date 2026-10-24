"""Controlled experiments: vanishing-cavity variance, cross-fit bias, post-selection false insertions."""
import math
import numpy as np
from s_common import *

V = cload("experiments/output/variance_scaling.csv")
C = cload("experiments/output/crossfit_bias.csv")
S = cload("experiments/output/selection_coverage.csv")
SUM = jload("experiments/output/summary.json")
W, H = TEXTW, 186
L = math.log10


def series(rows, method, x, y):
    r = sorted([q for q in rows if q["method"] == method], key=lambda q: float(q[x]))
    return r, np.array([float(q[x]) for q in r]), np.array([float(q[y]) for q in r])


def draw():
    top, ph, pw = 22, 100, 110
    # (a) variance against relative capacity, log-log
    A = sx.Plot(48, top, pw, ph, (L(0.09), L(3.4)), (L(9e-5), L(9e-2)))
    axes(A, [(L(0.1), "$10^{-1}$"), (0, "$10^{0}$")], logticks(-4, -2),
         xl=r"relative capacity $\rho(\varepsilon)$", yl="empirical variance")
    panel("a", "vanishing-cavity variance", 18, top - 10)
    sty = [("Direct TWoS", METHOD["twos"], "o", "direct TWoS"),
           ("Coupled finite cavity", VIOLET, "s", "coupled finite cavity"),
           ("Independent finite cavity", GREY, "^", "independent finite cavity")]
    for m, col, mk, _ in sty:
        r, x, y = series(V, m, "capacity", "variance")
        A.area(np.log10(x), np.log10([float(q["variance_ci_low"]) for q in r]),
               np.log10([float(q["variance_ci_high"]) for q in r]), col, op=18)
        line_mk(A, np.log10(x), np.log10(y), col, mk, r=2.0, w=1.2)
    v = SUM["variance"]
    sl = [("direct", v["direct_loglog_slope"]), ("coupled", v["coupled_fd_loglog_slope"]),
          ("indep.", v["independent_fd_loglog_slope"])]
    yy = A.y0 + 8
    note("fitted slopes", A.x0 + pw - 2, yy, anchor="right")
    for k, (n, s) in enumerate(sl):
        note(f"{n} ${s + 0.0:+.2f}$", A.x0 + pw - 2, yy + 9 * (k + 1), anchor="right")
    legend_grid([(lab, col, mk) for _, col, mk, lab in sty], A.x0 - 30, A.y0 + ph + 34, cols=2, colw=62)
    # (b) bias against within-pair correlation
    B = sx.Plot(200, top, pw, ph, (-1.0, 1.0), (-0.3, 0.3))
    axes(B, [(v_, f"${v_:g}$") for v_ in (-0.5, 0, 0.5)], [(v_, f"${v_:g}$") for v_ in (-0.2, -0.1, 0, 0.1, 0.2)],
         xl="within-pair correlation", yl=r"bias in $\widehat D_T$")
    panel("b", "bias from shared randomness", 170, top - 10)
    B.line([-1, 1], [0, 0], col=INK, w=0.5)
    sty_b = [("Same-seed product", VIOLET, "o", None, "same-seed product"),
             ("Product of means", CYAN, "s", None, "product of means"),
             ("Independent batches", GREY, "^", [3, 2], "independent batches"),
             ("Cross-fit TWoS", METHOD["twos"], "o", None, "cross-fit TWoS")]
    for m, col, mk, dash, _ in sty_b:
        r, x, y = series(C, m, "correlation", "bias")
        if m in ("Same-seed product", "Product of means"):
            B.line(x, [float(q["theoretical_bias"]) for q in r], col=col, w=0.7, dash=[1, 1.6])
        line_mk(B, x, y, col, mk, r=1.9, w=1.1, dash=dash)
    legend_grid([(lab, col, mk) for _, col, mk, _, lab in sty_b], B.x0 - 22, B.y0 + ph + 34, cols=2, colw=74)
    note("dotted: theory", B.x0 + pw - 2, B.y0 + 8, anchor="right")
    # (c) false insertions against candidate count
    Cp = sx.Plot(352, top, pw, ph, (-0.6, 8.6), (-2, 104))
    axes(Cp, [(k, f"$2^{{{k}}}$") for k in (1, 3, 5, 7)], [(v_, f"${v_}$") for v_ in range(0, 101, 20)],
         xl="candidate sites $K$", yl=r"false insertions (\%)")
    panel("c", "adaptive search needs fresh data", 322, top - 10)
    Cp.line([-0.6, 8.6], [5, 5], col=INK, w=0.7, dash=[1, 1.6])
    sty_c = [("Reuse selected samples", VIOLET, "o", None, "reuse selected samples"),
             ("Bonferroni scout", GREY, "^", [3, 2], "Bonferroni scout"),
             ("Fresh validation", METHOD["twos"], "o", None, "fresh validation")]
    for m, col, mk, dash, _ in sty_c:
        r, x, y = series(S, m, "candidates", "false_accept_rate")
        lx = np.log2(x)
        Cp.area(lx, 100 * np.array([float(q["ci_low"]) for q in r]), 100 * np.array([float(q["ci_high"]) for q in r]),
                col, op=15)
        line_mk(Cp, lx, 100 * y, col, mk, r=1.9, w=1.1, dash=dash)
    legend_grid([(lab, col, mk) for _, col, mk, _, lab in sty_c] + [(r"$5\%$ target", INK, "dot")],
                Cp.x0 - 30, Cp.y0 + ph + 34, cols=2, colw=78)
    finish()


sx.single("controlled-experiments", W, H, draw)
