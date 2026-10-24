"""Shared drawing for the matched-accuracy figures (log-log accuracy--cost fronts with bootstrap bars)."""
import math
import numpy as np
from splinx import draw as sx
from hstyle import *
from hdata import rows
import hstyle
hstyle.TICK, hstyle.LAB = 6.0, 6.6

L10 = math.log10
D = rows("benchmarks/matched_accuracy/output/summary.csv")
SEL = rows("benchmarks/matched_accuracy/output/selected.csv")
COL = {"twos": CYAN, "twos_rb": NAVY, "mg": VIOLET, "mg_update": VIOLET}
DASH = {"twos": None, "twos_rb": None, "mg": [2.4, 1.6], "mg_update": None}
OPEN = {"mg": True}
NAME = {"twos": "Sampled-source TWoS", "twos_rb": "Analytic-source TWoS", "mg": "MG, field reuse",
        "mg_update": "MG, incremental"}
GEOM = [("housing", "Finned housing"), ("branch", "Branching manifold"), ("ring", "Annular spreader"),
        ("bunny", "Stanford bunny")]


def sel(geometry, state, queries, methods):
    return [r for r in D if r["geometry"] == geometry and r["state"] == state and r["queries"] == queries
            and r["method"] in methods]


def fastest_per_budget(rs):
    best = {}
    for r in rs:
        b = r["budget"]
        if b not in best or r["seconds"] < best[b]["seconds"]:
            best[b] = r
    return list(best.values())


def decades(lo, hi):
    """Integer decades covering [lo, hi] (log10 values)."""
    return list(range(math.floor(lo), math.ceil(hi) + 1))


def logax(P, xlab=None, ylab=None, title=None, tick=None, lab=None, ylab_dx=0.0, ylabels=True):
    xs = [k for k in decades(P.xa, P.xb) if P.xa - 1e-9 <= k <= P.xb + 1e-9]
    ys = [k for k in decades(P.ya, P.yb) if P.ya - 1e-9 <= k <= P.yb + 1e-9]
    import hstyle
    old = hstyle.TICK, hstyle.LAB
    hstyle.TICK, hstyle.LAB = tick or old[0], lab or old[1]
    lab = hstyle.LAB
    axes(P, [(k, rf"$10^{{{k}}}$") for k in xs], [(k, rf"$10^{{{k}}}$" if ylabels else "") for k in ys],
         xl=xlab, yl=ylab, title=None, xl_dy=11.5, yl_dx=ylab_dx)
    hstyle.TICK, hstyle.LAB = old
    # minor ticks
    segs = []
    for k in range(math.floor(P.xa), math.ceil(P.xb)):
        for m in range(2, 10):
            v = k + L10(m)
            if P.xa <= v <= P.xb:
                segs.append([[float(P.X(v)), P.y0 + P.h], [float(P.X(v)), P.y0 + P.h + 1.4]])
    for k in range(math.floor(P.ya), math.ceil(P.yb)):
        for m in range(2, 10):
            v = k + L10(m)
            if P.ya <= v <= P.yb:
                segs.append([[P.x0, float(P.Y(v))], [P.x0 - 1.4, float(P.Y(v))]])
    if segs:
        sx.multi(segs, stroke=INK, width=0.35, fill=None)
    if title:
        sx.tex(title, P.x0 + P.w / 2, P.y0 - 5.0, lab + 0.2, INK, "center", maxw=P.w + 16)


def errorcurve(P, rs, method, labels=True, lab_size=5.0, r=1.9, lab_off=(3.0, -2.6)):
    rs = sorted(rs, key=lambda q: q["budget"])
    if not rs:
        return
    c = COL[method]
    X = [L10(q["seconds"]) for q in rs]; Y = [L10(q["error"]) for q in rs]
    P.line(X, Y, col=c, w=0.9, dash=DASH[method])
    bars = []
    for q in rs:
        x, y = float(P.X(L10(q["seconds"]))), float(P.Y(L10(q["error"])))
        x0, x1 = float(P.X(L10(max(q["seconds_lo"], 1e-9)))), float(P.X(L10(q["seconds_hi"])))
        y0, y1 = float(P.Y(L10(max(q["error_lo"], 1e-9)))), float(P.Y(L10(q["error_hi"])))
        bars += [[[x0, y], [x1, y]], [[x, y0], [x, y1]], [[x0, y - 1.1], [x0, y + 1.1]], [[x1, y - 1.1], [x1, y + 1.1]],
                 [[x - 1.1, y0], [x + 1.1, y0]], [[x - 1.1, y1], [x + 1.1, y1]]]
    sx.multi(bars, stroke=c, width=0.45, fill=None)
    for q in rs:
        x, y = float(P.X(L10(q["seconds"]))), float(P.Y(L10(q["error"])))
        if OPEN.get(method):
            sx.circle(x, y, r * 0.85, fill="#ffffff", stroke=c, width=0.7)
        else:
            marker("o", x, y, c, r=r)
        if labels:
            sx.tex(f"${int(q['budget'])}$", x + lab_off[0], y + lab_off[1], lab_size, c, "left", valign="bottom", bg="#ffffff")


def target_line(P, v=0.2):
    if P.ya <= L10(v) <= P.yb:
        P.line([P.xa, P.xb], [L10(v), L10(v)], col=GREY, w=0.6, dash=[1.0, 1.6])


def legend_items(methods):
    return [(NAME[m], COL[m], "o" if not OPEN.get(m) else "open") for m in methods]


def legend_centered(items, y, W, size=6.6, gap=14):
    from splinx.draw import texw
    total = 0
    for lab, col, kind in items:
        total += (17 if kind in ("line", "dash", "dot", "band") else 10) + texw(lab, size) + gap
    total -= gap
    x = W / 2 - total / 2
    for lab, col, kind in items:
        if kind == "open":
            sx.circle(x + 4, y, 2.0, fill="#ffffff", stroke=col, width=0.7)
            x += 10
            x += sx.tex(lab, x, y, size, INK, "left", valign="middle") + gap
        else:
            x = legend_row([(lab, col, kind)], x, y, size=size, gap=gap) + gap
