"""Same Bunny, two query workloads: accuracy--cost fronts, calibration scaling and the fresh-seed confirmation."""
import math
import numpy as np
from splinx import draw as sx
from hstyle import *
from hdata import rows, where
from ma_common import *

sx.OUT = "out"
W, H = 468, 246
HOLD = rows("benchmarks/matched_accuracy/output/conservative-holdout-summary.csv")
# Pointwise 95% bootstrap intervals (ms) of the mean query time over the 20 fresh seeds (10,000 resamples, seed
# 17092028, as in benchmarks/matched_accuracy/plot_bunny.py). The per-seed run folders (output/holdout/) are not
# tracked, so these endpoints are read back from the vector error bars of the figure that script drew.
CI_MS = {("original", "twos_rb"): (80.33, 105.01), ("original", "mg_update"): (101.22, 125.42),
         ("edited", "twos_rb"): (77.98, 91.24), ("edited", "mg_update"): (59.92, 90.56)}


def limits(rs, pad=0.12):
    xs = [L10(r["seconds_lo"]) for r in rs] + [L10(r["seconds_hi"]) for r in rs]
    ys = [L10(max(r["error_lo"], 1e-6)) for r in rs] + [L10(r["error_hi"]) for r in rs]
    return (min(xs) - pad, max(xs) + pad + 0.25), (min(ys) - pad, max(ys) + pad)


def front(x0, y0, geometry, title):
    curves = {}
    for m in ["twos", "twos_rb"]:
        curves[m] = sel(geometry, "original", 8, [m])
    curves["mg_update"] = fastest_per_budget(sel(geometry, "original", 8, ["mg", "mg_update"]))
    allr = [r for v in curves.values() for r in v]
    xl, yl = limits(allr)
    yl = (min(yl[0], L10(0.2) - 0.1), max(yl[1], L10(0.2) + 0.1))
    P = sx.Plot(x0, y0, 168, 76, xl, yl)
    logax(P, "cumulative query time (s)", "relative score error", title)
    target_line(P)
    for m in ["twos", "twos_rb", "mg_update"]:
        errorcurve(P, curves[m], m)
    return P


def calibration(x0, y0):
    g = sorted([r for r in SEL if r["geometry"] == "bunny" and r["state"] == "original" and r["target"] == 0.2],
               key=lambda r: r["queries"])
    L2 = math.log2
    P = sx.Plot(x0, y0, 168, 76, (L2(8) - 0.25, L2(512) + 0.25), (L10(0.1), L10(30)))
    logax_y = [(k, rf"$10^{{{k}}}$") for k in (-1, 0, 1)]
    axes(P, [(L2(q), f"${q}$") for q in (8, 32, 128, 512)], logax_y, xl="query sites",
         yl="best tested eligible time (s)", xl_dy=11.5)
    sx.tex(r"Localized objective, calibration, 20\% target", P.x0 + P.w / 2, P.y0 - 5.0, 6.8, INK, "center")
    for key, m in (("twos_rb", "twos_rb"), ("best_mg", "mg_update")):
        v = [r for r in g if r.get(key + "_seconds") is not None]
        X = np.array([L2(r["queries"]) for r in v])
        P.area(X, np.array([L10(r[key + "_seconds_lo"]) for r in v]), np.array([L10(r[key + "_seconds_hi"]) for r in v]),
               COL[m], op=16)
        P.line(X, [L10(r[key + "_seconds"]) for r in v], col=COL[m], w=0.9)
        for r in v:
            marker("o", float(P.X(L2(r["queries"]))), float(P.Y(L10(r[key + "_seconds"]))), COL[m], r=1.9)
    return P


def confirmation(x0, y0):
    P = sx.Plot(x0 + 58, y0, 110, 76, (0, 170), (0, 1))
    axes(P, [(v, f"${v}$") for v in range(0, 161, 40)], [], xl="query time (ms); lower is better", xl_dy=11.5,
         grid="x")
    sx.tex(r"Fresh seeds, 256 paths versus MG49", P.x0 + P.w / 2 - 29, P.y0 - 5.0, 6.8, INK, "center")
    for j, state in enumerate(["original", "edited"]):
        v = [r for r in HOLD if r["state"] == state][0]
        yc = P.y0 + P.h * (0.27 + 0.48 * j)
        for k, (val, m) in enumerate(((v["twos_seconds"] * 1000, "twos_rb"), (v["mg_seconds"] * 1000, "mg_update"))):
            yy = yc + (k - 0.5) * 11.0
            sx.rect(P.x0, yy - 4.6, float(P.X(val)) - P.x0, 9.2, fill=COL[m], stroke=None)
            lo, hi = CI_MS[(state, m)]
            xl, xh = float(P.X(lo)), float(P.X(hi))
            sx.multi([[[xl, yy], [xh, yy]], [[xl, yy - 2], [xl, yy + 2]], [[xh, yy - 2], [xh, yy + 2]]],
                     stroke=INK, width=0.5, fill=None)
            sx.tex(f"${val:.0f}$ ms", xh + 3, yy, 6.0, INK, "left", valign="middle", bg="#ffffff")
        sx.tex("Original" if state == "original" else "After cavity edit", P.x0 - 4, yc - 3.2, 6.4, INK, "right",
               valign="middle")
        ratio = rf"ratio ${v['speedup']:.2f}\times$, $[{v['speedup_lo']:.2f},\,{v['speedup_hi']:.2f}]$"
        sx.tex(ratio, P.x0 - 4, yc + 4.6, 5.4, MUTED, "right", valign="middle")
    return P


def draw():
    legend_centered(legend_items(["twos", "twos_rb", "mg_update"])[:2] + [("Reusable multigrid", VIOLET, "o")],
                    6.5, W)
    front(46, 30, "bunny", r"Localized sensor objective, 8 queries")
    front(46, 146, "bunny_compliance", r"Distributed-heating compliance, 8 queries")
    calibration(292, 30)
    confirmation(292, 146)
    for msg in sx.check():
        print(msg)


sx.single("matched-accuracy-bunny", W, H, draw)
