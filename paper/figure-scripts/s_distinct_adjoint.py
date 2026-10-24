"""Regional thermal protection with a distinct adjoint (fields, layout, reductions, retained-mesh cost)."""
import math
import numpy as np
from s_common import *

HERE = os.path.dirname(os.path.abspath(__file__))
FLD = np.load(os.path.join(HERE, "data/distinct_adjoint_fields.npz"))
LAY = json.load(open(os.path.join(HERE, "data/distinct_adjoint_layout.json")))
TAB = cload("benchmarks/distinct_adjoint/output/summary_table.csv")
PM = {r["method"]: r for r in jload("benchmarks/distinct_adjoint/output/summary.json")["persistent_means"]}
W, H = TEXTW, 314
L = math.log10


def circle_d(M, cx, cy, r, **look):
    """Circle given in data coordinates of the map window M (equal aspect)."""
    return sx.circle(float(M.X(cx)), float(M.Y(cy)), r * M.w / (M.xb - M.xa), **look)


def map_window(x0, y0, w):
    ext = FLD["u_extent"]
    h = w * (ext[3] - ext[2]) / (ext[1] - ext[0])
    return sx.Plot(x0, y0, w, h, (ext[0], ext[1]), (ext[2], ext[3]))


def outline(M, col=MUTED, w=0.6):
    for o in LAY["outline"]:
        o = np.array(o)
        sx.poly(M.P(o[:, 0], o[:, 1]), True, stroke=col, width=w, fill=None)


def draw():
    top, mw = 24, 136
    xs = [14, 170, 326]
    specs = [("u", "a", "Initial temperature $u$", -2, [2, 4, 6]),
             ("p", "b", "Sensor influence $p$", -3, [0.6, 1.2, 1.8]),
             ("score", "c", "Insertion score $up$", -5, [0.4, 0.8, 1.2])]
    sens = LAY["sensor"]
    for (name, letter, title, ex, tk), x0 in zip(specs, xs):
        M = map_window(x0, top, mw)
        lo, hi = FLD[name + "_range"]
        sx.image(rgba_field(FLD[name], lo, hi, FLD[name + "_alpha"]), M.x0, M.y0, M.w, M.h, smooth=True)
        outline(M)
        if name == "score":
            circle_d(M, sens[0], sens[1], LAY["exclusion_radius"], fill="#efeef6", stroke=GREY, width=0.5)
            sx.tex("no cavity centers", float(M.X(sens[0])), float(M.Y(sens[1] - LAY["exclusion_radius"])) + 3, 5.2,
                   MUTED, "center", valign="top")
        circle_d(M, sens[0], sens[1], sens[2], fill=None, stroke=CYAN, width=1.2)
        panel(letter, title, x0, top - 9)
        hbar(x0 + 14, M.y0 + M.h + 6, mw - 44, 5, lo / 10 ** ex, hi / 10 ** ex, [(v, f"${v:g}$") for v in tk],
             label=f"$\\times10^{{{ex}}}$")
    # (d) projected cavity locations
    top2 = 158
    M = map_window(xs[0], top2 + 14, mw)
    panel("d", "Projected cavity locations", xs[0], top2 - 9)
    outline(M, col=INK, w=0.7)
    circle_d(M, sens[0], sens[1], sens[2], fill=CYAN, stroke=CYAN, width=0.8, opacity=35)
    for c in LAY["mg_hot"]:
        circle_d(M, c[0], c[1], c[2], fill=None, stroke=VIOLET, width=1.0, dash=[2.4, 1.6])
    for c in LAY["twos_sparse"]:
        circle_d(M, c[0], c[1], c[2], fill=None, stroke=METHOD["twos"], width=1.1)
    legend_grid([("TWoS sparse", METHOD["twos"], "line"), ("MG hottest", VIOLET, "dash"),
                 ("protected sensor", CYAN, "o")], M.x0 + 2, M.y0 + M.h + 12, cols=2, colw=66)
    # (e) regional reductions per method and sensor side
    methods = ["random", "wos_hot", "mg_hot", "twos_sparse", "twos_dense", "mg57", "mg85", "mg_td", "mg169"]
    short = ["random", "WoS hot", "MG hot", "TWoS S", "TWoS L", "MG 57", "MG 85", "MG 113", "MG 169"]
    E = sx.Plot(200, top2 + 4, 118, 104, (-0.6, 8.6), (0, 65))
    axes(E, [(i, "") for i in range(9)], [(v, f"${v}$") for v in range(0, 61, 10)],
         yl=r"regional reduction (\%)", grid="y")
    for i, s in enumerate(short):
        f = lambda s=s: sx.tex(s, 0, 0, TICK - 0.6, INK, anchor="right", valign="middle")
        sx.move_to(sx.rotate(f, -60), (float(E.X(i)) - 3, E.y0 + E.h + 4 + 0.43 * sx.texw(s, TICK - 0.6)))()
    panel("e", "Objective-aware placement helps", 170, top2 - 9)
    for k, (side, col, mk) in enumerate([("-1", INK, "o"), ("1", GREY, "s")]):
        for i, m in enumerate(methods):
            r = next(q for q in TAB if q["side"] == side and q["method"] == m)
            x, y, e = i + (k - 0.5) * 0.22, float(r["reduction_percent"]), float(r["design_se_pp"])
            X = float(E.X(x))
            sx.multi([[[X, float(E.Y(y - e))], [X, float(E.Y(y + e))]]], stroke=col, width=0.7, fill=None)
            marker(mk, X, float(E.Y(y)), col, r=2.0)
    legend_box([("left sensor", INK, "o"), ("right sensor", GREY, "s")], float(E.X(4.3)), float(E.Y(24)))
    # (f) retained-mesh quality against complete design wall time
    F = sx.Plot(358, top2 + 4, 104, 104, (L(0.9), L(9)), (25, 65))
    axes(F, [(L(v), f"${v}$") for v in (1, 2, 4, 8)], [(v, f"${v}$") for v in range(25, 66, 10)],
         xl="complete design wall time (s)", yl=r"regional reduction (\%)")
    panel("f", "Retained-mesh design cost", 328, top2 - 9)
    pt = lambda m: (L(PM[m]["complete_design_seconds"]), 100 * PM[m]["reduction"])
    for names, col, labs in [(["mg57", "mg_td"], VIOLET, {"mg57": "57", "mg_td": "113"}),
                             (["twos_sparse", "twos_dense"], METHOD["twos"], {"twos_sparse": "sparse",
                                                                              "twos_dense": "larger"})]:
        P_ = [pt(m) for m in names]
        F.line([p[0] for p in P_], [p[1] for p in P_], col=col, w=1.2)
        for m, (x, y) in zip(names, P_):
            marker("o", float(F.X(x)), float(F.Y(y)), col, r=2.3)
            dy = 9 if m.startswith("twos") else -5
            note(labs[m], float(F.X(x)), float(F.Y(y)) + dy, ANN - 0.4, col, "center")
    for m, lab, mk in [("wos_hot", "WoS hottest", "o"), ("mg_hot", "MG hottest", "s")]:
        x, y = pt(m)
        marker(mk, float(F.X(x)), float(F.Y(y)), METHOD["hottest"], r=2.3)
        note(lab, float(F.X(x)), float(F.Y(y)) + 9, ANN - 0.4, MUTED, "center")
    note("TWoS budgets", float(F.X(L(1.9))), float(F.Y(43.5)), col=METHOD["twos"])
    note("multigrid nodes", float(F.X(L(1.9))), float(F.Y(43.5)) + 9, col=VIOLET)
    note(r"\shortstack[l]{one fresh process\\per design}", F.x0 + 4, F.y0 + 10)
    finish()


sx.single("distinct-adjoint-benchmark", W, H, draw)
