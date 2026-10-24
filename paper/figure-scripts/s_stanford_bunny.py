"""Stanford Bunny curved-mesh stress test: repaired mesh, interior scores and final slice (prep_stanford_bunny.py),
the saved TWoS cavities and the fresh-path trajectories (benchmarks/stanford_bunny/output)."""
import os
import numpy as np
from splinx import draw as sx
from hstyle import *
import hstyle
from hfield import heat, vbar, projector, sphere, raster_mesh, TEMP_MAP, BEN_MAP
from s_common import cload, panel, finish

HERE = os.path.dirname(os.path.abspath(__file__))
S = np.load(os.path.join(HERE, "data/stanford_bunny_fields.npz"))
TRAJ = cload("benchmarks/stanford_bunny/output/design-trajectories.csv")
METH = [("TWoS + fresh validation", "TWoS + fresh", NAVY), ("TWoS + reused scout", "TWoS + reuse", CYAN),
        ("Voxel adjoint TD", "voxel TD", VIOLET), ("Random feasible", "random", GREY)]
W, H = TEXTW, 304
V = S["render_vertices"][:, [0, 2, 1]]   # display axes as in run_bunny_benchmark.py (x, z, y)
F = S["render_faces"]
lo, hi = V.min(0), V.max(0)
MID = 0.5 * (lo + hi)


def view(cx, cy, scale):
    f0 = projector(12, -68, scale, cx, cy)
    return lambda p: f0(np.atleast_2d(p) - MID)


def mesh_panel(x0, y0, w, h, col, op, name):
    f = view(x0 + w / 2, y0 + h / 2, 0.9 * w / (hi - lo).max())
    sx.image(raster_mesh(V, F, f, (x0, y0, w, h), color=col), x0, y0, w, h, name=name, smooth=True, opacity=op)
    return f


def draw():
    old = hstyle.TICK, hstyle.LAB
    # (a) repaired mesh
    panel("a", "canonical mesh, repaired", 4, 10)
    mesh_panel(14, 16, 126, 110, "#cfcde3", None, "bunny-a")
    sx.tex(f"{int(S['n_vertices']):,} vertices, {int(S['n_faces']):,} faces, 5 caps".replace(",", "{,}", 2),
           77, 132, 5.8, MUTED, "center")
    # (b) interior direct scores
    panel("b", r"interior direct score $-D_T=u^2$", 162, 10)
    f = mesh_panel(166, 16, 126, 110, "#b9b7cf", 22, "bunny-b")
    pts = S["candidates"][::2][:, [0, 2, 1]]
    sc = S["scores"][::2]
    smin, smax = float(S["scores"].min()), float(S["scores"].max())
    uv, depth = f(pts)
    cols = sx.cmap(sc, smin, smax, BEN_MAP)
    for i in np.argsort(depth):
        sx.circle(float(uv[i, 0]), float(uv[i, 1]), 1.15, fill="#%02x%02x%02x" % tuple(cols[i]), stroke=None)
    vbar(296, 30, 4, 80, smin, smax, BEN_MAP, [1e-4, 2e-4, 3e-4, 4e-4, 5e-4], size=5.4,
         fmt=lambda v: f"${v * 1e4:.0f}$")
    sx.tex(r"$\times10^{-4}$", 296, 26, 5.2, INK, "left")
    # (c) inserted cavities
    panel("c", "grid-free cavity insertion", 330, 10)
    f = mesh_panel(334, 16, 126, 110, "#b9b7cf", 26, "bunny-c")
    r = float(S["radius"])
    C = S["centers"][:, [0, 2, 1]]
    uv, depth = f(C)
    scale = 0.9 * 126 / (hi - lo).max()
    cols = [NAVY, "#2a3fc0", "#3a6fd8", CYAN, "#39c5c3", MINT]
    for i in np.argsort(depth):
        sphere(float(uv[i, 0]), float(uv[i, 1]), r * scale, cols[i % len(cols)])
    # (d) fresh-path compliance
    panel("d", "fresh-path compliance", 4, 154)
    D = sx.Plot(36, 164, 108, 104, (-0.3, 6.3), (0.28, 1.04))
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(D, [(i, f"${i}$") for i in range(7)], [(v, f"${v:.1f}$") for v in (0.4, 0.6, 0.8, 1.0)],
         xl="inserted cavities", yl="normalized compliance", xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old
    red = {}
    for key, short, c in METH:
        rs = [q for q in TRAJ if q["method"] == key]
        it = np.array([float(q["iteration"]) for q in rs]); ob = np.array([float(q["objective"]) for q in rs])
        se = np.array([float(q["objective_se"]) for q in rs])
        D.area(it, (ob - se) / ob[0], (ob + se) / ob[0], c, op=14)
        D.line(it, ob / ob[0], col=c, w=1.1)
        for a_, b_ in zip(it, ob / ob[0]):
            marker("o", float(D.X(a_)), float(D.Y(b_)), c, r=1.7)
        red[key] = 100 * (ob[0] - ob[-1]) / ob[0]
    legend_col([(m[0], m[2], "line") for m in METH], D.x0 + 4, float(D.Y(0.505)), size=5.6, gap=7.4)
    # (e) improvement after six cavities
    panel("e", "improvement after 6 cavities", 162, 154)
    E = sx.Plot(214, 164, 92, 104, (0, 58), (0, 4))
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(E, [(v, f"${v}$") for v in range(0, 51, 10)], [], xl="compliance reduction (\\%)", grid="x", xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old
    for i, (key, short, c) in enumerate(METH):
        yc = E.y0 + E.h * (i + 0.5) / 4
        sx.rect(E.x0, yc - 8, float(E.X(red[key])) - E.x0, 16, fill=c, stroke=None)
        sx.tex(f"{red[key]:.1f}\\%", float(E.X(red[key])) + 2.5, yc, 6.0, INK, "left", valign="middle")
        sx.tex(short, E.x0 - 4, yc, 6.2, INK, "right", valign="middle")
    # (f) final central slice
    panel("f", "final central slice $u$", 330, 154)
    x, y, t = S["x"], S["y"], S["final_temp"]
    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
    h = 100.0; w = h * (x[-1] - x[0]) / (y[-1] - y[0])
    P = sx.Plot(346, 164, w, h, (float(x[0]), float(x[-1])), (float(y[0]), float(y[-1])))
    heat(P, t, tmin, tmax, TEMP_MAP, "slice")
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(P, [(v, f"${v:g}$") for v in (-0.4, -0.2, 0, 0.2, 0.4) if x[0] <= v <= x[-1]],
         [(v, f"${v:g}$") for v in (-0.4, -0.2, 0, 0.2, 0.4) if y[0] <= v <= y[-1]], xl="$x$", yl="$y$", grid=False,
         xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old
    for c in S["centers"]:
        if abs(c[2]) < r:
            rr = float(np.sqrt(r * r - c[2] ** 2))
            sx.circle(float(P.X(c[0])), float(P.Y(c[1])), rr * P.w / (x[-1] - x[0]), fill=None, stroke=MINT, width=1.0)
    vbar(P.x0 + P.w + 4, P.y0, 4, P.h, tmin, tmax, TEMP_MAP, [0.002, 0.004, 0.006, 0.008, 0.01], size=5.4,
         fmt=lambda v: f"${v * 1e3:.0f}$")
    sx.tex(r"$\times10^{-3}$", P.x0 + P.w + 4, P.y0 - 3, 5.2, INK, "left")
    finish()


sx.single("stanford-bunny-benchmark", W, H, draw)
