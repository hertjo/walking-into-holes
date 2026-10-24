"""Independent tetrahedral FEM audit: finest-mesh facets, central slice and candidate scores from the deterministic
rerun in prep_external_fem.py; the 512-walk TWoS field from benchmarks/engineering_housing/output."""
import os, math
import numpy as np
from splinx import draw as sx
from hstyle import *
import hstyle
from hfield import heat, vbar, projector, TEMP_MAP, BEN_MAP
from s_common import panel, finish
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
S = np.load(os.path.join(HERE, "data/external_fem.npz"))
cand, fem, twos = S["candidates"], S["fem"], S["twos"]
# Panel (d): end-to-end times of the three resolutions (s) and their relative RMSE against the finest field. The
# errors and the finest time are stated in the paper; the two coarse times are read back from the earlier vector
# figure (prep_external_fem.py, data/external_fem_d_raw.json), whose run folder is not tracked. Marker widths there
# (pt) encode measured peak RSS and are kept as relative sizes.
RES = ["41\\times27\\times23", "61\\times39\\times33", "81\\times51\\times43"]
TIMES = [1.50, 5.02, 20.37]
ERRS = [0.1285, 0.0802, None]
SIZES = [3.52, 4.97, 5.95]
W, H = TEXTW, 236
L10 = math.log10


def draw():
    old = hstyle.TICK, hstyle.LAB
    # (a) cutaway boundary facets and the 36 highest-ranking sites
    panel("a", "independent tetrahedral FEM audit", 4, 10)
    T = S["facets"]
    f = projector(22, -58, 76.0, 104, 112)
    uv, depth = f(T.reshape(-1, 3))
    uv = uv.reshape(-1, 3, 2); d = depth.reshape(-1, 3).mean(1)
    order = np.argsort(d)
    sx.multi([np.vstack([uv[i], uv[i][:1]]).tolist() for i in order], stroke="#8b87b5", width=0.08, fill=None,
             opacity=55)
    hot = np.argsort(fem)[-36:]
    puv, pd = f(cand[hot])
    lo, hi = float(fem[hot].min()), float(fem[hot].max())
    cs = sx.cmap(fem[hot], lo, hi, BEN_MAP)
    for i in np.argsort(pd):
        sx.circle(float(puv[i, 0]), float(puv[i, 1]), 1.9, fill="#%02x%02x%02x" % tuple(cs[i]), stroke="#ffffff",
                  width=0.3)
    vbar(196, 70, 4, 80, lo, hi, BEN_MAP, [0.002, 0.0025, 0.003], size=5.4, fmt=lambda v: f"${v * 1e3:.1f}$")
    sx.tex(r"$u_h^2\ (\times10^{-3})$", 194, 64, 5.4, INK, "left")
    sx.tex(f"{int(S['vertices']):,} vertices, {int(S['tets']):,} tetrahedra".replace(",", "{,}"), 104, 196, 5.8,
           MUTED, "center")
    # (b) central-slice field
    panel("b", "central-slice field", 236, 10)
    x, z, v = S["x"], S["z"], S["slice"]
    vmax = float(np.nanmax(v))
    P = sx.Plot(254, 20, 76, 76 * (z[-1] - z[0]) / (x[-1] - x[0]) * 2.0, (float(x[0]), float(x[-1])),
                (float(z[0]), float(z[-1])))
    heat(P, v, 0, vmax, TEMP_MAP, "slice", upsample=2)
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(P, [(t, f"${t:g}$") for t in (-1, 0, 1)], [(t, f"${t:g}$") for t in (-0.2, 0, 0.2)], xl="$x$", yl="$z$",
         grid=False, xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old
    vbar(P.x0 + P.w + 4, P.y0, 4, P.h, 0, vmax, TEMP_MAP, [0, 0.0005, 0.001, 0.0015], size=5.4,
         fmt=lambda t: f"${t * 1e3:.1f}$")
    sx.tex(r"$\times10^{-3}$", P.x0 + P.w + 2, P.y0 - 3, 5.2, INK, "left")
    # (c) identical 768 sites after one fitted positive scale
    panel("c", "identical 768 sites", 364, 10)
    slope = float(np.dot(fem, twos) / np.dot(fem, fem))
    sc = slope * fem
    lim = 1.04 * max(float(sc.max()), float(twos.max()))
    Q = sx.Plot(386, 20, 76, 76, (0, lim), (0, lim))
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    ticks = [(t, f"${t * 1e3:g}$") for t in (0, 0.001, 0.002, 0.003, 0.004) if t <= lim]
    axes(Q, ticks, ticks, xl=r"scaled FEM $u_h^2$ ($\times10^{-3}$)", yl=r"TWoS $u^2$ ($\times10^{-3}$)", xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old
    Q.line([0, lim], [0, lim], col=INK, w=0.6, dash=[2.4, 1.6])
    xc = sx.cmap(cand[:, 0], -1.0, 1.0, [(0, NAVY), (0.5, LILAC), (1, MINT)])
    for i in range(len(sc)):
        sx.circle(float(Q.X(sc[i])), float(Q.Y(twos[i])), 0.75, fill="#%02x%02x%02x" % tuple(xc[i]), stroke=None,
                  opacity=75)
    rho = float(pearsonr(fem, twos).statistic)
    sx.tex(f"Pearson $r={rho:.3f}$", Q.x0 + 3, Q.y0 + 7, 5.8, INK, "left", bg="#ffffff")
    # (d) convergence and cost
    panel("d", "mesh convergence and resource cost", 236, 130)
    floor = ERRS[1] / 2.7
    errs = [e if e is not None else floor for e in ERRS]
    G = sx.Plot(276, 140, 186, 66, (L10(1.1), L10(28)), (L10(0.025), L10(0.16)))
    hstyle.TICK, hstyle.LAB = 6.0, 6.6
    axes(G, [(L10(t), f"${t:g}$") for t in (2, 5, 10, 20)], [(L10(t), f"${t:g}$") for t in (0.03, 0.05, 0.1)],
         xl="end-to-end FEM time (s)", yl="relative RMSE", xl_dy=10.5)
    hstyle.TICK, hstyle.LAB = old
    G.line([L10(t) for t in TIMES[:2]], [L10(e) for e in errs[:2]], col=NAVY, w=1.0)
    for k, (t, e, s, lab) in enumerate(zip(TIMES, errs, SIZES, RES)):
        x0, y0 = float(G.X(L10(t))), float(G.Y(L10(e)))
        r = 1.0 * s
        if k < 2:
            marker("o", x0, y0, NAVY, r=r / 2 + 0.6)
        else:
            sx.poly([[x0 - r * 0.55, y0 - r * 0.45], [x0 + r * 0.55, y0 - r * 0.45], [x0, y0 + r * 0.5]], True,
                    fill=VIOLET, stroke="#ffffff", width=0.4)
        if k < 2:
            sx.tex(f"${lab}$", x0 + 4, y0 - 4, 5.4, INK, "left", bg="#ffffff")
        else:
            sx.tex(f"${lab}$ (reference)", x0 - 5, y0 - 5, 5.4, INK, "right", bg="#ffffff")
    sx.tex("marker area encodes peak RSS; triangle is the reference", G.x0 + G.w - 2, G.y0 + 7, 5.2, MUTED,
           "right", bg="#ffffff")
    finish()


sx.single("external-fem-audit", W, H, draw)
