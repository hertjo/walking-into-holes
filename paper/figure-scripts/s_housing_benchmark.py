"""Compiled engineering housing: procedural mesh, initial central-cut temperature, grid-free scores, the validated
design and the independently evaluated designs (benchmarks/engineering_housing: assets/ and output/)."""
import os, sys
import numpy as np
from splinx import draw as sx
from hstyle import *
import hstyle
from hfield import heat, vbar, projector, raster_mesh, read_ascii_ply, section, sphere, TEMP_MAP, BEN_MAP
from s_common import R, cload, jload, panel, finish

sys.path.insert(0, R("benchmarks/engineering_housing"))
import types  # noqa: E402
sys.modules.setdefault("igl", types.ModuleType("igl"))  # only the mesh-building helpers need libigl
from run_housing_benchmark import housing_sdf  # noqa: E402  (pure NumPy geometry of the housing)

O = "benchmarks/engineering_housing/output"
V, F = read_ascii_ply(R("benchmarks/engineering_housing/assets/stylized-finned-housing.ply"))
SUM = jload(O, "summary.json")
DES = SUM["designs"]; J0 = SUM["initial_objective"]; RAD = SUM["problem"]["cavity_radius"]
cand = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in cload(O, "field-visual-score.csv")])
vscore = np.array([float(r["score"]) for r in cload(O, "field-visual-score.csv")])
sl = cload(O, "field-slice-initial.csv")
NAMES = [("TWoS + validation", "TWoS +", "validation", NAVY), ("TWoS equal-time", "TWoS", "equal-time", CYAN),
         ("geometric multigrid", "geometric", "multigrid", VIOLET), ("random feasible", "random", "feasible", GREY)]
W, H = TEXTW, 300
lo, hi = V.min(0), V.max(0)
MID = 0.5 * (lo + hi)


def view(x0, y0, w, h, azim):
    s = 0.92 * w / (hi - lo)[:2].max()
    f0 = projector(20, azim, s, x0 + w / 2, y0 + h / 2)
    return (lambda p: f0(np.atleast_2d(p) - MID)), s


def cutaway(x0, y0, w, h, name):
    f, s = view(x0, y0, w, h, -50)
    rear = F[(V[F][:, :, 1] >= 0).all(1)]
    sx.image(raster_mesh(V, F, f, (x0, y0, w, h), color="#d9d7e8", light=(-0.35, -0.75, 0.72)), x0, y0, w, h,
             name=name + "-g", smooth=True, opacity=14)
    sx.image(raster_mesh(V, rear, f, (x0, y0, w, h), color="#a9a6c4", light=(-0.35, -0.75, 0.72)), x0, y0, w, h,
             name=name + "-r", smooth=True, opacity=70)
    rim = section(V, F, 1, 0.0)
    segs = [f(sg)[0].tolist() for sg in rim]
    sx.multi(segs, stroke=NAVY, width=0.45, fill=None, opacity=80)
    return f, s


def draw():
    old = hstyle.TICK, hstyle.LAB
    # (a) housing
    panel("a", "finned conduction housing", 4, 10)
    f, _ = view(10, 18, 132, 100, -58)
    sx.image(raster_mesh(V, F, f, (10, 18, 132, 100), color="#bdbad6", light=(-0.35, -0.75, 0.72)), 10, 18, 132, 100,
             name="housing-a", smooth=True)
    sx.tex(f"closed procedural mesh; {len(F):,} triangles".replace(",", "{,}", 1), 76, 128, 5.8, MUTED, "center")
    # (b) initial temperature on the central cut
    panel("b", r"initial temperature, central cut", 162, 10)
    xs = np.linspace(-1.14, 1.14, 72); zs = np.linspace(-0.34, 0.73, 56)
    gx, gz = np.meshgrid(xs, zs)
    pts = np.column_stack((gx.ravel(), np.zeros(gx.size), gz.ravel()))
    inside = housing_sdf(pts) < -0.003
    img = np.full(gx.size, np.nan)
    img[np.flatnonzero(inside)] = [float(r["u"]) for r in sl]
    img = img.reshape(gx.shape)
    vmax = float(np.nanquantile(img, 0.99))
    w = 120.0; h = w * (zs[-1] - zs[0]) / (xs[-1] - xs[0])
    P = sx.Plot(178, 26, w, h, (xs[0], xs[-1]), (zs[0], zs[-1]))
    heat(P, np.minimum(img, vmax), 0, vmax, TEMP_MAP, "slice")
    P.implicit(lambda x, z: housing_sdf(np.column_stack((np.ravel(x), np.zeros(np.size(x)), np.ravel(z)))).reshape(
        np.shape(x)), 0.0, col=INK, w=0.7)
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(P, [(v, f"${v:g}$") for v in (-1, -0.5, 0, 0.5, 1)], [(v, f"${v:g}$") for v in (-0.2, 0, 0.2, 0.4, 0.6)],
         xl="$x$", yl="$z$", grid=False, xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old
    for cx, cz in ((-0.48, -0.08), (0.43, -0.07)):
        u, v = float(P.X(cx)), float(P.Y(cz))
        sx.multi([[[u - 2, v - 2], [u + 2, v + 2]], [[u - 2, v + 2], [u + 2, v - 2]]], stroke=MINT, width=0.9, fill=None)
    vbar(P.x0 + P.w + 4, P.y0, 4, P.h, 0, vmax, TEMP_MAP, [0, 0.01, 0.02, 0.03], size=5.4)
    # (c) grid-free scores at the visible candidates
    panel("c", r"grid-free score $-D_T=u^2$", 330, 10)
    f, _ = cutaway(330, 18, 118, 100, "c")
    vis = cand[:, 1] >= 0
    smax = float(np.quantile(vscore, 0.985))
    uv, depth = f(cand[vis])
    cs = sx.cmap(vscore[vis], 0, smax, BEN_MAP)
    for i in np.argsort(depth):
        sx.circle(float(uv[i, 0]), float(uv[i, 1]), 1.2, fill="#%02x%02x%02x" % tuple(cs[i]), stroke=INK, width=0.1)
    vbar(452, 34, 4, 70, 0, smax, BEN_MAP, [0, 1e-3, 2e-3], size=5.4, fmt=lambda v: f"${v * 1e3:.0f}$")
    sx.tex(r"$\times10^{-3}$", 448, 30, 5.2, INK, "left")
    # (d) validated design
    panel("d", "validated grid-free design", 4, 152)
    f, s = cutaway(8, 160, 136, 104, "d")
    cav = np.array(DES["TWoS + validation"]["cavities"])
    uv, depth = f(cav)
    cols = [NAVY, "#2a3fc0", VIOLET, CYAN, "#39c5c3", MINT]
    for i in np.argsort(depth):
        u, v = float(uv[i, 0]), float(uv[i, 1])
        if cav[i, 1] >= 0:
            sphere(u, v, RAD * s, cols[i])
        else:
            sx.circle(u, v, RAD * s, fill=None, stroke=cols[i], width=0.6, dash=[1.2, 1.0])
        sx.tex(f"${i + 1}$", u, v - RAD * s - 1.5, 5.2, INK, "center")
    sx.tex("solid: rear cavities; dashed: cavities in the removed half", 76, 274, 5.6, MUTED, "center")
    # (e) independently evaluated design
    panel("e", "independently evaluated design", 162, 152)
    E = sx.Plot(196, 162, 108, 104, (-0.3, 6.3), (0, 104))
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(E, [(i, f"${i}$") for i in range(7)], [(v, f"${v}$") for v in range(0, 101, 20)], xl="inserted cavities",
         yl=r"compliance (\% of initial)", xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old
    for key, _, _, c in NAMES:
        ob = 100 * np.array(DES[key]["objectives"]) / J0
        E.line(np.arange(len(ob)), ob, col=c, w=1.1)
        for a, b in enumerate(ob):
            marker("o", float(E.X(a)), float(E.Y(b)), c, r=1.7)
    legend_col([(k, c, "line") for k, _, _, c in NAMES], E.x0 + 30, float(E.Y(31)), size=5.6, gap=7.0)
    # (f) final reduction over independent designs
    panel("f", "final reduction, independent designs", 318, 152)
    red = [100 * (1 - DES[k]["aggregate_final_mean"] / J0) for k, *_ in NAMES]
    err = [100 * DES[k]["aggregate_final_se"] / J0 for k, *_ in NAMES]
    G = sx.Plot(344, 162, 118, 104, (-0.6, 3.6), (0, 1.23 * max(r + e for r, e in zip(red, err))))
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(G, [(i, "") for i in range(4)], [(v, f"${v}$") for v in range(0, 81, 20)], yl=r"compliance reduction (\%)",
         grid="y")
    hstyle.TICK, hstyle.LAB = old
    for i, ((key, l1, l2, c), r_, e_) in enumerate(zip(NAMES, red, err)):
        x0, x1 = float(G.X(i - 0.34)), float(G.X(i + 0.34))
        sx.rect(x0, float(G.Y(r_)), x1 - x0, float(G.Y(0)) - float(G.Y(r_)), fill=c, stroke=None)
        xc = float(G.X(i))
        sx.multi([[[xc, float(G.Y(r_ - e_))], [xc, float(G.Y(r_ + e_))]],
                  [[xc - 2, float(G.Y(r_ - e_))], [xc + 2, float(G.Y(r_ - e_))]],
                  [[xc - 2, float(G.Y(r_ + e_))], [xc + 2, float(G.Y(r_ + e_))]]], stroke=INK, width=0.5, fill=None)
        top = float(G.Y(r_ + e_))
        sx.tex(f"{r_:.1f}\\%", xc, top - 8.5, 5.2, INK, "center")
        sx.tex(f"{DES[key]['selection_seconds_mean']:.2f} s select", xc, top - 2.5, 4.4, MUTED, "center")
        sx.tex(l1, xc, G.y0 + G.h + 7.5, 5.2, INK, "center")
        sx.tex(l2, xc, G.y0 + G.h + 13.5, 5.2, INK, "center")
    finish()


sx.single("engineering-housing-benchmark", W, H, draw)
