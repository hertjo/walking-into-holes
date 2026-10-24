"""Matched C++ Bunny study: actual Embree walk traces, the retained 32-walk TWoS and 54x64x48 voxel fields at the 384
common sites, accuracy/time, accuracy/memory and output-sensitive scaling (benchmarks/compiled_bunny/output).
The display mesh is the 7,000-face decimation of the repaired Bunny (prep_stanford_bunny.py)."""
import os, json, math
import numpy as np
from splinx import draw as sx
from hstyle import *
import hstyle
from hfield import vbar, projector, raster_mesh, BEN_MAP
from s_common import R, cload, panel, finish

HERE = os.path.dirname(os.path.abspath(__file__))
O = "benchmarks/compiled_bunny/output"
M = np.load(os.path.join(HERE, "data/stanford_bunny_fields.npz"))
VP = M["render_vertices"]; F = M["render_faces"]
pts = np.loadtxt(R(O, "evaluation-points.csv"), delimiter=",").reshape(-1, 3)
twos_field = np.array([float(r["score"]) for r in cload(O, "field-twos-0032.csv")])
voxel_field = np.array([float(r["score"]) for r in cload(O, "field-voxel-54x64x48.csv")])
TW = cload(O, "twos-scaling.csv"); VX = cload(O, "voxel-scaling.csv"); OUT = cload(O, "output-scaling.csv")
traces = {}
for r in cload(O, "walk-traces.csv"):
    traces.setdefault(int(r["path"]), []).append([float(r["x"]), float(r["y"]), float(r["z"])])
CUT = float(np.quantile(VP[:, 2], 0.46))
W, H = TEXTW, 300
D3 = lambda p: np.asarray(p)[..., [0, 2, 1]]
lo, hi = D3(VP).min(0), D3(VP).max(0)
MID = 0.5 * (lo + hi)
L10 = math.log10


def view(x0, y0, w, h):
    s = 0.9 * w / (hi - lo).max()
    f0 = projector(12, -68, s, x0 + w / 2, y0 + h / 2)
    return (lambda p: f0(np.atleast_2d(D3(p)) - MID)), s


def shell(x0, y0, w, h, name):
    f, s = view(x0, y0, w, h)
    rear = F[(VP[F][:, :, 2] >= CUT).all(1)]
    sx.image(raster_mesh(VP, F, f, (x0, y0, w, h), color="#d9d7e8"), x0, y0, w, h, name=name + "-g", smooth=True,
             opacity=16)
    sx.image(raster_mesh(VP, rear, f, (x0, y0, w, h), color="#a9a6c4"), x0, y0, w, h, name=name + "-r", smooth=True,
             opacity=62)
    return f


def logaxes(P, xt, yt, xl, yl):
    old = hstyle.TICK, hstyle.LAB
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(P, xt, yt, xl=xl, yl=yl, xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old


def dec(a, b):
    return [(k, rf"$10^{{{k}}}$") for k in range(math.ceil(a - 1e-9), math.floor(b + 1e-9) + 1)]


def draw():
    T, V = NAVY, VIOLET
    # (a) walk traces
    panel("a", "Embree-accelerated sphere walks", 4, 10)
    f = shell(14, 16, 126, 112, "a")
    cols = [NAVY, "#2a3fc0", VIOLET, CYAN, "#39c5c3", MINT]
    longest = sorted(traces, key=lambda k: len(traces[k]), reverse=True)[:6]
    for c, k in zip(cols, longest):
        uv, _ = f(np.array(traces[k]))
        sx.poly(uv.tolist(), stroke=c, width=0.7, fill=None, opacity=90)
        for p in uv[:: max(1, len(uv) // 8)]:
            sx.circle(float(p[0]), float(p[1]), 0.9, fill=c, stroke=None)
    sx.tex("cutaway shell; actual C++ paths; no volume grid", 77, 134, 5.6, MUTED, "center")
    # (b), (c) fields at the visible sites
    vis = pts[:, 2] >= CUT
    vmax = float(np.quantile(np.r_[twos_field, voxel_field], 0.985))
    for letter, title, field, x0 in (("b", r"direct TWoS score $-D_T$", twos_field, 158),
                                     ("c", r"voxel-adjoint score $-D_T$", voxel_field, 306)):
        panel(letter, title, x0 - 4, 10)
        f = shell(x0, 16, 126, 112, letter)
        uv, depth = f(pts[vis])
        cs = sx.cmap(field[vis], 0, vmax, BEN_MAP)
        for i in np.argsort(depth):
            sx.circle(float(uv[i, 0]), float(uv[i, 1]), 1.35, fill="#%02x%02x%02x" % tuple(cs[i]), stroke=INK, width=0.12)
    vbar(442, 30, 4, 84, 0, vmax, BEN_MAP, [0, 1e-4, 2e-4, 3e-4, 4e-4], size=5.4, fmt=lambda v: f"${v * 1e4:.0f}$")
    sx.tex(r"$\times10^{-4}$", 440, 26, 5.2, INK, "left")
    # (d) accuracy versus wall time
    panel("d", "accuracy versus wall time", 4, 152)
    tt = [L10(float(r["wall_seconds"])) for r in TW]; te = [L10(float(r["relative_rmse"])) for r in TW]
    vt = [L10(float(r["wall_seconds"])) for r in VX]; ve = [L10(float(r["relative_rmse"])) for r in VX]
    P = sx.Plot(36, 162, 106, 104, (min(tt + vt) - 0.15, max(tt + vt) + 0.3), (min(te + ve) - 0.08, max(te + ve) + 0.08))
    logaxes(P, dec(P.xa, P.xb), [(L10(v), f"${v:g}$") for v in (0.1, 0.2, 0.3, 0.5) if P.ya <= L10(v) <= P.yb],
            "end-to-end field time (s)", "relative field RMSE")
    P.line(tt, te, col=T, w=1.1); P.line(vt, ve, col=V, w=1.1)
    for a, b in zip(tt, te):
        marker("o", float(P.X(a)), float(P.Y(b)), T, r=1.8)
    for a, b in zip(vt, ve):
        marker("s", float(P.X(a)), float(P.Y(b)), V, r=1.8)
    for r, a, b in list(zip(TW, tt, te))[::2]:
        sx.tex(f"{int(float(r['walks']))}w", float(P.X(a)) + 2.5, float(P.Y(b)) + 6.0, 4.8, T, "left", bg="#ffffff")
    for r, a, b in list(zip(VX, vt, ve))[::2]:
        sh = json.loads(r["shape"])
        sx.tex(f"${sh[0]}\\times{sh[1]}\\times{sh[2]}$", float(P.X(a)) + 2.5, float(P.Y(b)) - 2.5, 4.6, V, "left",
               bg="#ffffff")
    legend_col([("C++ TWoS", T, "o"), ("C++ voxel FD/CG", V, "s")], P.x0 + P.w - 56, P.y0 + 6, size=5.6, gap=8)
    # (e) accuracy versus memory (x reversed: lower error to the right)
    panel("e", "accuracy versus memory", 162, 152)
    tm = [float(r["peak_rss_megabytes"]) for r in TW]; vm = [float(r["peak_rss_megabytes"]) for r in VX]
    xa, xb = max(te + ve) + 0.08, min(te + ve) - 0.08
    Q = sx.Plot(196, 162, 106, 104, (xa, xb), (0, 125))
    logaxes(Q, [(L10(v), f"${v:g}$") for v in (0.5, 0.3, 0.2, 0.1) if xb <= L10(v) <= xa],
            [(v, f"${v}$") for v in range(0, 121, 20)], "relative field RMSE (lower is better)", "peak RSS (MiB)")
    Q.line(te, tm, col=T, w=1.1); Q.line(ve, vm, col=V, w=1.1)
    for a, b in zip(te, tm):
        marker("o", float(Q.X(a)), float(Q.Y(b)), T, r=1.8)
    for a, b in zip(ve, vm):
        marker("s", float(Q.X(a)), float(Q.Y(b)), V, r=1.8)
    sx.tex("TWoS stores no volumetric operator", Q.x0 + 4, Q.y0 + 8, 5.6, MUTED, "left", bg="#ffffff")
    legend_col([("C++ TWoS", T, "o"), ("C++ voxel FD/CG", V, "s")], Q.x0 + 4, Q.y0 + 46, size=5.6, gap=8)
    # (f) output-sensitive scaling
    panel("f", "output-sensitive scaling", 318, 152)
    s_ = [L10(float(r["sites"])) for r in OUT]
    ot = [L10(float(r["twos_seconds"])) for r in OUT]; ov = [L10(float(r["voxel_seconds"])) for r in OUT]
    G = sx.Plot(352, 162, 106, 104, (min(s_) - 0.1, max(s_) + 0.1), (min(ot + ov) - 0.1, max(ot + ov) + 0.1))
    logaxes(G, dec(G.xa, G.xb), [(L10(v), f"${v:g}$") for v in (0.2, 0.5, 1, 2) if G.ya <= L10(v) <= G.yb],
            "requested derivative sites", "end-to-end field time (s)")
    faster = [a for a, b, c in zip(s_, ot, ov) if b < c]
    k = int(np.argmin(np.abs(np.array(ot) - np.array(ov))))
    G.area(np.array([min(s_), max(faster)]), np.array([G.ya, G.ya]), np.array([G.yb, G.yb]), CYAN, op=10)
    sx.tex(f"timing crossover near {int(float(OUT[k]['sites']))} outputs", G.x0 + 3, G.y0 + 7, 5.4, NAVY, "left")
    G.line(s_, ot, col=T, w=1.1); G.line(s_, ov, col=V, w=1.1)
    for a, b, c in zip(s_, ot, ov):
        marker("o", float(G.X(a)), float(G.Y(b)), T, r=1.8)
        marker("s", float(G.X(a)), float(G.Y(c)), V, r=1.8)
    legend_col([("C++ TWoS, 512 walks/site", T, "o"), (r"C++ voxel, $96\times112\times84$", V, "s")],
               G.x0 + 4, G.y0 + 17, size=5.4, gap=8)
    finish()


sx.single("compiled-bunny-scaling", W, H, draw)
