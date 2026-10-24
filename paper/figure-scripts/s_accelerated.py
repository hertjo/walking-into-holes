"""Accelerated sampling: MSE against paths, wall time and distance queries (Poisson compliance, harmonic square)."""
import math
from collections import defaultdict
import numpy as np
from s_common import *

OUTD = "benchmarks/critical_baselines/output"
ST = cload(OUTD, "plot_statistics.csv")
RUNS = {"Poisson compliance": (cload(OUTD, "source_sampling_runs.csv"), "queries"),
        "Harmonic square diagnostic": ([r for r in cload(OUTD, "sampling_runs.csv") if r["domain"] == "square"],
                                       "distance_queries")}
COL = {"mc": GREY, "rqmc": VIOLET, "array": NAVY, "neighbors": MINT}
NAME = {"mc": "MC", "rqmc": "Sobol RQMC", "array": "Array-RQMC", "neighbors": "neighbor reuse"}
W, H = TEXTW, 322
L = math.log10


def series(problem):
    """method -> rows sorted by n with (paths, median seconds, mean queries, mean mse, lo, hi)."""
    runs, qk = RUNS[problem]
    out = {}
    for m in dict.fromkeys(r["method"] for r in runs):
        rows = []
        for s in sorted([q for q in ST if q["problem"] == problem and q["method"] == m], key=lambda q: int(q["n"])):
            g = [r for r in runs if r["method"] == m and r["n"] == s["n"]]
            rows.append((float(g[0]["paths"]), float(s["median_seconds"]), np.mean([float(r[qk]) for r in g]),
                         float(s["mean_mse"]), float(s["mse_lo"]), float(s["mse_hi"])))
        out[m] = np.array(rows)
    return out


def draw():
    pw, ph = 112, 96
    xs = [48, 202, 356]
    rows = [("Poisson compliance", 22, "topological-derivative MSE", (-7.75, -4.9)),
            ("Harmonic square diagnostic", 178, "harmonic-product MSE", (-6.45, -3.5))]
    xlab = ["total launched paths", "wall time (s)", "distance-query points"]
    for r, (prob, top, yl, ylim) in enumerate(rows):
        S = series(prob)
        for c in range(3):
            allx = np.concatenate([np.log10(a[:, c]) for a in S.values()])
            pad = 0.08 * (allx.max() - allx.min())
            P = sx.Plot(xs[c], top, pw, ph, (allx.min() - pad, allx.max() + pad), ylim)
            lo_d, hi_d = math.ceil(P.xa - 1e-9), math.floor(P.xb + 1e-9)
            axes(P, logticks(lo_d, hi_d), logticks(math.ceil(ylim[0]), math.floor(ylim[1])), xl=xlab[c],
                 yl=yl if c == 0 else None)
            panel("abcdef"[3 * r + c], prob, xs[c] - (34 if c == 0 else 26), top - 10)
            for m, a in S.items():
                x = np.log10(a[:, c])
                P.area(x, np.log10(a[:, 4]), np.log10(a[:, 5]), COL[m], op=16)
            for m, a in S.items():
                line_mk(P, np.log10(a[:, c]), np.log10(a[:, 3]), COL[m], "o", w=1.2, r=2.0)
    from splinx.draw import texw
    items = [(NAME[m] + (" (harmonic only)" if m == "neighbors" else ""), COL[m], "o")
             for m in ("mc", "rqmc", "array", "neighbors")]
    wsum = sum(10 + texw(l, ANN) for l, _, _ in items) + 16 * 3
    legend_row(items, W / 2 - wsum / 2, H - 6, size=ANN, gap=16)
    finish()


sx.single("accelerated-sampling", W, H, draw)
