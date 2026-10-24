"""Finned housing: predeclared multigrid sweep and the quality-time-memory tradeoff (included at 0.89 textwidth)."""
import numpy as np
from s_common import *

SW = sorted(cload("benchmarks/engineering_housing/output/multigrid-resolution-sweep.csv"), key=lambda r: int(r["grid_cells"]))
SUM = jload("benchmarks/engineering_housing/output/summary.json")
J0 = SUM["initial_objective"]
W, H = 0.89 * TEXTW, 168


def draw():
    top, ph = 22, 110
    A = sx.Plot(52, top, 150, ph, (230, 790), (0.066, 0.155))
    axes(A, [(v, f"${v}$") for v in (300, 400, 500, 600, 700)], [(v, f"${v:.2f}$") for v in (0.08, 0.10, 0.12, 0.14)],
         xl="fine-grid cells (thousands)", yl="relative RMSE")
    panel("a", "Predeclared multigrid tuning sweep", 14, top - 10)
    A.line([230, 790], [0.09, 0.09], col=MUTED, w=0.7, dash=[3, 2])
    note(r"$0.09$ accuracy target", A.x0 + A.w - 2, float(A.Y(0.09)) - 2.5, anchor="right")
    c = [int(r["grid_cells"]) / 1000 for r in SW]
    e = [float(r["relative_rmse"]) for r in SW]
    A.line(c, e, col=LILAC, w=1.2)
    off = {"93x61x47": (5, 1, "left"), "105x69x53": (5, -4, "left"), "113x73x57": (-2, 10, "right"),
           "121x77x61": (0, 18, "center"), "133x85x67": (0, 10, "right")}
    for r, x, y in zip(SW, c, e):
        X, Y = float(A.X(x)), float(A.Y(y))
        if r["selected"] == "True":
            sx.star(X, Y, 4.4, 5, 0.45, fill=VIOLET, stroke="#ffffff", width=0.4)
        else:
            marker("o", X, Y, GREY, r=2.3)
        dx, dy, an = off[r["grid"]]
        sx.tex(r["grid"].replace("x", r"$\times$"), X + dx, Y + dy, ANN - 0.6, MUTED, an, bg="#ffffff")
    B = sx.Plot(262, top, 140, ph, (0.25, 2.1), (67.9, 70.62))
    axes(B, [(v, f"${v:g}$") for v in (0.5, 1.0, 1.5, 2.0)], [(v, f"${v:.1f}$") for v in (68, 68.5, 69, 69.5, 70, 70.5)],
         xl="six-round selection time (s)", yl=r"compliance reduction (\%)")
    panel("b", "Quality, time and memory", 226, top - 10)
    sty = {"TWoS + validation": (METHOD["twos"], "TWoS sparse", (5, 9, "left")),
           "TWoS equal-time": (METHOD["twos"], "TWoS equal-time", (-6, 9, "right")),
           "geometric multigrid": (VIOLET, "multigrid", (-7, -2, "right")),
           "voxel FD/CG": (CYAN, "FD/CG", (6, 13, "left"))}
    for name, (col, lab, (dx, dy, an)) in sty.items():
        d = SUM["designs"][name]
        s = float(d["selection_seconds_mean"])
        red = 100 * (1 - float(d["aggregate_final_mean"]) / J0)
        se = 100 * float(d["aggregate_final_se"]) / J0
        mem = float(d["peak_rss_megabytes"])
        X, Y = float(B.X(s)), float(B.Y(red))
        sx.multi([[[X, float(B.Y(red - se))], [X, float(B.Y(red + se))]],
                  [[X - 2, float(B.Y(red - se))], [X + 2, float(B.Y(red - se))]],
                  [[X - 2, float(B.Y(red + se))], [X + 2, float(B.Y(red + se))]]], stroke=col, width=0.8, fill=None)
        r = np.sqrt(22.0 + 1.15 * mem) * 0.42
        if name == "TWoS equal-time":
            sx.circle(X, Y, r, fill="#ffffff", stroke=col, width=1.0)
        else:
            sx.circle(X, Y, r, fill=col, stroke="#ffffff", width=0.5)
        sx.tex(rf"\shortstack[{'l' if an == 'left' else 'r'}]{{{lab}\\{mem:.0f} MiB}}", X + dx, Y + dy, ANN - 0.4,
               INK, an, valign="top", bg="#ffffff")
    finish()


sx.single("engineering-housing-tradeoffs", W, H, draw)
