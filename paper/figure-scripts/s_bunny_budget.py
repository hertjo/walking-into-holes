"""Bunny budget study: cooling vs cavity count at two removed volumes, and the cost envelope."""
import numpy as np
from s_common import *

S = cload("benchmarks/bunny_budget/output/summary.csv")
ENV = jload("benchmarks/bunny_budget/output/cost-envelope.json")
ORDER = [("random", "random feasible", "o"), ("voxel", "voxel adjoint", "s"), ("twos", "TWoS", "o")]
COL = {"random": METHOD["random"], "voxel": METHOD["voxel"], "twos": METHOD["twos"]}
W, H = TEXTW, 176


def draw():
    pw, ph, top = 112, 96, 30
    xs0 = [44, 192, 340]
    yt = [(v, f"${v}$") for v in (20, 40, 60)]
    for k, (units, letter) in enumerate([("3", "a"), ("6", "b")]):
        g = [r for r in S if r["volume_units"] == units]
        frac = 100 * float(g[0]["removed_fraction"])
        P = sx.Plot(xs0[k], top, pw, ph, (1, 13), (15, 68))
        axes(P, [(v, f"${v}$") for v in (2, 4, 6, 12)], yt if k == 0 else [(v, f"${v}$") for v in (20, 40, 60)],
             xl="cavity count $n$", yl="heat-objective reduction (\\%)" if k == 0 else None, grid="y")
        panel(letter, f"${frac:.2f}\\%$ removed volume", P.x0 - 30, top - 12)
        for m, lab, mk in ORDER:
            rows = sorted([r for r in g if r["method"] == m], key=lambda r: int(r["count"]))
            n = np.array([int(r["count"]) for r in rows], float)
            P.area(n, np.array([float(r["cooling_min"]) for r in rows]),
                   np.array([float(r["cooling_max"]) for r in rows]), COL[m], op=16)
        for m, lab, mk in ORDER:
            rows = sorted([r for r in g if r["method"] == m], key=lambda r: int(r["count"]))
            line_mk(P, [int(r["count"]) for r in rows], [float(r["cooling_mean"]) for r in rows], COL[m], mk, r=2.2)
        radii = .05 * (int(units) / np.array([2, 4, 6, 12])) ** (1 / 3)
        sx.tex("radii " + ", ".join(f"${r:.3f}$" for r in radii), P.x0 + pw / 2, top + ph + 34, ANN, MUTED,
               "center")
    P = sx.Plot(xs0[2], top, pw, ph, (0, 4), (-0.5, 13))
    axes(P, [(v, f"${v}$") for v in range(5)], [(v, f"${v}$") for v in (0, 2, 4, 6, 12)],
         xl=r"assumed cost per cavity (\% of $J_0$)", yl="preferred tested count", grid="y")
    panel("c", "a cost-dependent choice", P.x0 - 30, top - 12)
    for a in [e for e in ENV if 100 * e["lower_cost"] < 4]:
        P.line([100 * a["lower_cost"], min(100 * a["upper_cost"], 4)], [a["count"]] * 2, col=COL["twos"], w=1.8)
    for a, b in zip(ENV[:-1], ENV[1:]):
        if a["upper_cost"] * 100 <= 4:
            P.line([a["upper_cost"] * 100] * 2, [a["count"], b["count"]], col=COL["twos"], w=0.7, dash=[1, 1.6])
    zero = next(s["lower_cost"] for s in ENV if s["count"] == 0)
    P.text(f"above ${100 * zero:.1f}\\%$: no cavities", 2.6, 9.6, ANN, MUTED, anchor="center")
    sx.tex(r"minimise $J/J_0+\kappa n$; includes no cavities", P.x0 + pw / 2, top + ph + 34, ANN, MUTED, "center")
    from splinx.draw import texw
    items = [(lab, COL[m], mk) for m, lab, mk in ORDER]
    wsum = sum(10 + texw(l, ANN) for l, _, _ in items) + 16 * 2
    legend_row(items, (xs0[0] + xs0[1] + pw) / 2 - wsum / 2 + 10, H - 6, size=ANN, gap=16)
    finish()


sx.single("bunny-budget", W, H, draw)
