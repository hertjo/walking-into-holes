"""Matched-accuracy appendix figures: all budgets on the original and edited domains (8 and 512 sites), and the
fixed-budget query-count scaling. Data: benchmarks/matched_accuracy/output/summary.csv (as plot.py there)."""
import math
import numpy as np
from splinx import draw as sx
from hstyle import *
from ma_common import *

sx.OUT = "out"
W = TEXTW
L2 = math.log2


def limits(rs, pad=0.12):
    xs = [L10(max(r["seconds_lo"], 1e-9)) for r in rs] + [L10(r["seconds_hi"]) for r in rs]
    ys = [L10(max(r["error_lo"], 1e-6)) for r in rs] + [L10(r["error_hi"]) for r in rs]
    return (min(xs) - pad, max(xs) + pad + 0.3), (min(ys) - pad, max(ys) + pad + 0.1)


def budgets(state):
    H = 268

    def draw():
        legend_centered(legend_items(["twos", "twos_rb", "mg", "mg_update"]), 6.5, W, size=6.4)
        for i, q in enumerate([8, 512]):
            for j, (g, title) in enumerate(GEOM):
                x0, y0 = 30 + j * 113.0, 26 + i * 124.0
                rs = {m: sel(g, state, q, [m]) for m in ["twos", "twos_rb", "mg", "mg_update"]}
                allr = [r for v in rs.values() for r in v]
                xl, yl = limits(allr)
                yl = (min(yl[0], L10(0.2) - 0.1), max(yl[1], L10(0.2) + 0.1))
                P = sx.Plot(x0, y0, 82, 76, xl, yl)
                logax(P, "cumulative query time (s)" if i == 1 else None, "relative score error" if j == 0 else None,
                      f"{title}, {q} sites", tick=5.4, lab=6.0)
                target_line(P)
                for m in ["twos", "twos_rb", "mg", "mg_update"]:
                    errorcurve(P, rs[m], m, labels=(m != "mg"), lab_size=4.6, r=1.5, lab_off=(2.2, -1.8))
        for msg in sx.check():
            print(msg)

    sx.single(f"matched-accuracy-{state}", W, H, draw)


def scaling():
    H = 262
    LINES = [("twos", 256, None), ("twos", 4096, [3, 1.6]), ("mg", 49, None), ("mg", 129, [3, 1.6])]

    def draw():
        from hstyle import legend_row
        items = [(f"{'TWoS' if m == 'twos' else 'MG'} {b}", COL[m], "line" if d is None else "dash") for m, b, d in LINES]
        from splinx.draw import texw
        tot = sum(17 + texw(l, 6.4) + 14 for l, _, _ in items) - 14
        legend_row(items, W / 2 - tot / 2, 6.5, size=6.4, gap=14)
        for i, state in enumerate(["original", "edited"]):
            for j, (g, title) in enumerate(GEOM):
                x0, y0 = 30 + j * 113.0, 26 + i * 122.0
                v = [r for r in D if r["geometry"] == g and r["state"] == state and
                     any(r["method"] == m and r["budget"] == b for m, b, _ in LINES)]
                ys = [L10(r[k]) for r in v for k in ("seconds_lo", "seconds_hi") if r[k] > 0]
                P = sx.Plot(x0, y0, 82, 76, (L2(8) - 0.3, L2(512) + 0.3), (math.floor(min(ys)) - 0.05, max(ys) + 0.15))
                yt = [(k, rf"$10^{{{k}}}$") for k in range(math.ceil(P.ya), math.floor(P.yb) + 1)]
                import hstyle
                old = hstyle.TICK, hstyle.LAB
                hstyle.TICK, hstyle.LAB = 5.4, 6.0
                axes(P, [(L2(q), f"${q}$") for q in (8, 32, 128, 512)], yt, xl="query sites" if i == 1 else None,
                     yl="cumulative time (s)" if j == 0 else None, xl_dy=10.0)
                hstyle.TICK, hstyle.LAB = old
                sx.tex(f"{title}, {'original' if state == 'original' else 'after insertion'}", P.x0 + P.w / 2, P.y0 - 5.0,
                       6.2, INK, "center", maxw=P.w + 18)
                for m, b, dash in LINES:
                    g_ = sorted([r for r in v if r["method"] == m and r["budget"] == b], key=lambda r: r["queries"])
                    X = np.array([L2(r["queries"]) for r in g_])
                    P.area(X, np.array([L10(r["seconds_lo"]) for r in g_]), np.array([L10(r["seconds_hi"]) for r in g_]),
                           COL[m], op=12)
                    P.line(X, [L10(r["seconds"]) for r in g_], col=COL[m], w=0.9, dash=dash)
        for msg in sx.check():
            print(msg)

    sx.single("matched-accuracy-scaling", W, H, draw)


if __name__ == "__main__":
    budgets("original")
    budgets("edited")
    scaling()
