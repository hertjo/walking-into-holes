"""Application validation: certified finite insertion benefit, bound tightening, validation cost."""
import math
import numpy as np
from s_common import *

F = cload("benchmarks/minimum_benefit/output/validation-results.csv")
SUM = jload("benchmarks/minimum_benefit/output/summary.json")
SCALE = SUM["configuration"]["fixed_scales"]
GEO = {"bunny": VIOLET, "housing": CYAN}
W, H = TEXTW, 352
L = math.log10


def errbar(P, x, y, e, col, cap=2.2):
    X, y0, y1 = float(P.X(x)), float(P.Y(y - e)), float(P.Y(y + e))
    sx.multi([[[X, y0], [X, y1]], [[X - cap, y0], [X + cap, y0]], [[X - cap, y1], [X + cap, y1]]],
             stroke=col, width=0.7, fill=None)


def draw():
    pw, ph = 170, 104
    top1, top2 = 24, 212
    specs = [("bunny", "a", "Bunny: finite insertion benefit", 50, (0, 15.5), range(0, 15, 2)),
             ("housing", "b", "housing: finite insertion benefit", 286, (0, 31.5), range(0, 31, 5))]
    for kind, letter, title, x0, ylim, yt in specs:
        r = [q for q in F if q["geometry"] == kind]
        sc = SCALE[kind]
        x = np.array([int(q["attempt"]) for q in r])
        g = lambda k: np.array([float(q[k]) for q in r]) / sc * 100
        P = sx.Plot(x0, top1, pw, ph, (0.7, 6.3), (ylim[0] - 0.5 * (ylim[1] / 15.5), ylim[1]))
        axes(P, [(v, f"${v}$") for v in range(1, 7)], [(v, f"${v}$") for v in yt], xl="attempted insertion",
             yl=r"benefit / fixed scale (\%)" if letter == "a" else None, grid="y")
        panel(letter, title[0].upper() + title[1:], x0 - 34 if letter == "a" else x0 - 22, top1 - 11)
        P.line([0.7, 6.3], [0.1, 0.1], col=GREY, w=0.6, dash=[2, 1.6])
        P.line(x, g("source_only"), col=CYAN, w=1.3, dash=[1, 1.6])
        line_mk(P, x, g("floor"), VIOLET, "s", w=1.0, dash=[3, 2], r=1.9)
        line_mk(P, x, g("multigrid_reference"), GREY, "s", w=0.9, dash=[3, 2], r=2.0, open_=True)
        for a, b, e in zip(x, g("measured"), g("reference_se")):
            errbar(P, a, b, e, INK)
        line_mk(P, x, g("measured"), INK, "o", w=1.1, r=1.9)
        line_mk(P, x, g("bound"), METHOD["twos"], "o", w=1.5, r=2.2)
    legend_grid([("independent WoS reference", INK, "o"), ("multigrid reference", GREY, "s"),
                 ("average-trace certificate", METHOD["twos"], "o"), ("temperature-floor certificate", VIOLET, "s"),
                 ("source-only bound", CYAN, "dot"), (r"target ($0.1\%$)", GREY, "dash")],
                40, top1 + ph + 36, cols=3, colw=146)
    # (c) bound against walks
    C = sx.Plot(50, top2, pw, ph, (L(800), L(84000)), (6.5, 22))
    st = SUM["configuration"]["stages"]
    axes(C, [(L(n), f"${n:,}$".replace(",", "{,}")) for n in st], [(v, f"${v}$") for v in range(8, 22, 2)],
         xl="independent walks at the first candidate", yl=r"benefit bound / fixed scale (\%)", grid="y")
    panel("c", "Tightening the bound with more samples", 16, top2 - 11)
    for kind, col in GEO.items():
        s = SUM["results"][kind]["records"][0]["stages"]
        line_mk(C, [L(q["n"]) for q in s], [100 * q["benefit_lower"] / SCALE[kind] for q in s], col, "o", w=1.4,
                r=2.3)
    note("Housing", C.X(L(16384)), C.Y(19.8) - 6, col=GEO["housing"], anchor="right")
    note("Bunny", C.X(L(16384)), C.Y(10.2) - 6, col=GEO["bunny"], anchor="right")
    # (d) validation cost, log bars
    D = sx.Plot(286, top2, pw, ph, (-0.55, 2.55), (L(0.06), L(500)))
    axes(D, [(0, r"\shortstack{scout +\\rescore}"), (1, r"\shortstack{first passing\\prefixes}"), (2, r"\shortstack{full validation\\sweep}")],
         logticks(-1, 2), yl="seconds for six attempts", grid="y")
    panel("d", "Added cost of conservative validation", 264, top2 - 11)
    w = 0.34
    for off, kind in [(-w / 2, "bunny"), (w / 2, "housing")]:
        r = SUM["results"][kind]
        for i, v in enumerate([r["selection_seconds"], r["first_passing_prefix_seconds"], r["validation_seconds"]]):
            xa, xb = float(D.X(i + off - w / 2)), float(D.X(i + off + w / 2))
            ya, yb = float(D.Y(L(v))), float(D.Y(D.ya))
            sx.rect(xa, ya, xb - xa, yb - ya, fill=GEO[kind], stroke=None)
            sx.tex(f"${v:.2f}$", (xa + xb) / 2, ya - 2.5, ANN - 0.6, INK, "center")
    legend_col([("Bunny", GEO["bunny"], "s"), ("Housing", GEO["housing"], "s")], D.x0 + 6, D.y0 + 6)
    finish()


sx.single("application-minimum-benefit", W, H, draw)
