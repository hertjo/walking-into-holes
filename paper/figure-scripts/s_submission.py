"""Matched geometry and pipeline study: refined designs (pipeline), accessible channels (channels), actual-geometry
estimators (estimators) and the three geometries (geometries).

Renders: benchmarks/submission_study/render_assets, recoloured by prep_submission.py. Numbers: the study's
output/*.csv and assets/geometry-audit.json. Usage: python s_submission.py [pipeline channels estimators geometries]"""
import os, sys, math, json
import numpy as np
from PIL import Image
from splinx import draw as sx
from splinx.draw import nice_ticks, texw
from hstyle import *
import hstyle
from s_common import R, cload, jload, finish

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET = os.path.join(HERE, "assets/submission")
RAW = R("benchmarks/submission_study/render_assets")
W = TEXTW
GEOM = [("housing", "Finned housing"), ("branch", "Bifurcating manifold"), ("ring", "Ribbed annular spreader")]
COL = {"twos": METHOD["twos"], "mg": METHOD["multigrid"], "mg49": METHOD["multigrid"], "nearest": METHOD["nearest"],
       "hot": METHOD["hottest"], "random": METHOD["random"], "coupled-fd": METHOD["crn"], "hit-correction": METHOD["harmonic"]}
NAME = {"nearest": "Nearest sensor", "twos": "TWoS", "mg": "Multigrid TD (65)", "mg49": "Multigrid TD (49)",
        "hot": "Hottest point", "random": "Random"}
MK = {"twos": "o", "mg": "s", "mg49": "s", "nearest": "D", "hot": "^", "random": "x", "coupled-fd": "^", "hit-correction": "D"}
DASH = {"twos": None, "mg": None, "mg49": [2.4, 1.6], "nearest": [3.2, 1.4, 0.8, 1.4], "hot": None, "random": [0.8, 1.6]}
OPEN = {"mg49"}
COOL = [(0.0, INK), (0.3, NAVY), (0.6, VIOLET), (0.84, LILAC), (1.0, "#f6f5ff")]


def num(rs):
    out = []
    for r in rs:
        q = {}
        for k, v in r.items():
            try:
                q[k] = float(v)
            except (TypeError, ValueError):
                q[k] = float("nan") if v in ("", None) else v
        out.append(q)
    return out


DES = num(cload("benchmarks/submission_study/output/design-results.csv"))
GRP = num(cload("benchmarks/submission_study/output/group-results.csv"))
VOL = num(cload("benchmarks/submission_study/output/volume-audit.csv"))
EST = num(cload("benchmarks/submission_study/output/estimator-results.csv"))


def grp(**kw):
    r = [g for g in GRP if all(g[k] == v for k, v in kw.items())]
    assert len(r) == 1, kw
    return r[0]


def mk(kind, x, y, col, r=2.1, open_=False):
    fill = "#ffffff" if open_ else col
    edge = col if open_ else "#ffffff"
    if kind == "D":
        q = r * 1.25
        sx.poly([[x, y - q], [x + q, y], [x, y + q], [x - q, y]], True, fill=fill, stroke=edge, width=0.5)
    elif kind == "x":
        q = r * 0.85
        sx.multi([[[x - q, y - q], [x + q, y + q]], [[x - q, y + q], [x + q, y - q]]], stroke=col, width=0.8, fill=None)
    else:
        marker(kind, x, y, fill, r=r, edge=edge)


def legend_items(x, y, items, size=ANN, gap=11):
    xx = x
    for lab, m in items:
        c = COL[m]
        sx.poly([[xx, y], [xx + 14, y]], stroke=c, width=1.0, dash=DASH.get(m), fill=None)
        mk(MK[m], xx + 7, y, c, r=1.9, open_=m in OPEN)
        xx += 17
        xx += sx.tex(lab, xx, y, size, INK, "left", valign="middle") + gap
    return xx - gap


def legend_width(items, size=ANN, gap=11):
    return sum(17 + texw(l, size) + gap for l, _ in items) - gap


def img(path, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im)


def circ(path, n):
    a = img(path, n)
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(xx + 0.5 - n / 2, yy + 0.5 - n / 2)
    alpha = np.clip((n / 2 - rr) + 0.5, 0, 1)
    return np.dstack([a, np.uint8(alpha * 255)])


def cbar(x, y, w, h, label):
    arr = sx.cmap(np.linspace(0, 1, 256)[None, :].repeat(2, 0), 0, 1, COOL)
    sx.image(arr, x, y, w, h, name="cbar", smooth=True)
    sx.rect(x, y, w, h, fill=None, stroke=INK, width=0.35)
    for t in range(0, 101, 20):
        xx = x + t / 100 * w
        sx.line(xx, y + h, xx, y + h + 1.6, stroke=INK, width=0.35)
        sx.tex(f"${t}$", xx, y + h + 2.4, 5.6, INK, "center", valign="top")
    sx.tex(label, x + w + 5, y + h / 2, ANN, INK, "left", valign="middle")


def axes_small(P, xt, yt, xl, yl):
    old = hstyle.TICK, hstyle.LAB
    hstyle.TICK, hstyle.LAB = 6.0, 6.4
    axes(P, xt, yt, xl=xl, yl=yl, grid="y", xl_dy=11.0)
    hstyle.TICK, hstyle.LAB = old


# ---------------------------------------------------------------- refined designs / channels
def plate(study):
    S = 104.0         # render side (pt)
    RH = 143.0        # row height
    TOP = 18.0
    H = TOP + 3 * RH + 16
    meths = ["random", "hot", "nearest", "twos", "mg", "mg49"]

    def draw():
        items = [(NAME[m], m) for m in meths]
        legend_items(W / 2 - legend_width(items) / 2, 6.5, items)
        for row, (g, label) in enumerate(GEOM):
            y0 = TOP + row * RH
            sx.tex(label, 2, y0 + 8, 7.6, INK, "left")
            for col, m in enumerate(["nearest", "twos", "mg"]):
                x = 2 + col * 113.0
                state = "refine8" if study == "pipeline" else "initial"
                gg = grp(study=study, geometry=g, method=m, state=state)
                lab = NAME[m] + (" + DWoS" if study == "pipeline" else "")
                sx.tex(lab, x + 2, y0 + 18.5, 6.4, COL[m], "left")
                iy = y0 + 21
                name = f"{g}-{study}-{m}"
                sx.image(img(os.path.join(ASSET, name + ".png"), 560), x - 2, iy, S, S, name=name, smooth=True)
                meta = json.load(open(os.path.join(RAW, name + ".json")))
                px, py = meta["sensor_pixel"]
                sxp, syp = x - 2 + px * S, iy + py * S
                cx, cy, r = x - 2 + S - 17.5, iy + 21.0, 17.0
                ang = math.atan2(syp - cy, sxp - cx)
                sx.line(cx + r * math.cos(ang), cy + r * math.sin(ang), sxp, syp, stroke=COL[m], width=0.55, opacity=75)
                sx.circle(cx, cy, r, fill="#ffffff", stroke=None)
                sx.image(circ(os.path.join(ASSET, name + "-detail.png"), 260), cx - r, cy - r, 2 * r, 2 * r,
                         name=name + "-d", smooth=True)
                sx.circle(cx, cy, r, fill=None, stroke=COL[m], width=0.8)
                sx.tex(f"{gg['reduction_pct']:.1f}\\% mean sensor cooling", x + 2, iy + S + 7.0, 6.2, INK, "left")
                if study == "pipeline":
                    sub = f"{gg['seconds']:.2f} s $\\,|\\,$ {gg['backend_peak_rss_mb']:.1f} MiB backend"
                else:
                    sub = f"{gg['seconds']:.2f} s $\\,|\\,$ mean radius {gg['radius']:.3f}"
                sx.tex(sub, x + 2, iy + S + 14.0, 5.6, MUTED, "left")
            quality(study, g, y0)
        y = TOP + 3 * RH + 3
        cbar(8, y, 110, 4.0, "local cooling (\\%)")
        xx = 238
        sx.circle(xx, y + 2, 2.6, fill=None, stroke=CYAN, width=0.9)
        xx += 5 + sx.tex("protected sensor", xx + 5, y + 2, ANN, INK, "left", valign="middle") + 12
        sx.circle(xx, y + 2, 2.6, fill=None, stroke=MINT, width=0.9)
        sx.tex("complete heat-source support", xx + 5, y + 2, ANN, INK, "left", valign="middle")
        finish()

    sx.single(f"submission-{study}", W, H, draw)


def quality(study, g, y0):
    a = [d for d in DES if d["study"] == study and d["geometry"] == g]
    rs = [x for x in GRP if x["study"] == study and x["geometry"] == g]
    xmax = 1.08 * max(d["seconds"] for d in a)
    ylo = min(0.0, 1.1 * min(x["cooling_ci95_low"] for x in rs))
    yhi = max(0.1, 1.1 * max(x["cooling_ci95_high"] for x in rs))
    ph = 70.0 if study == "channels" else 80.0
    P = sx.Plot(372, y0 + 24, 90, ph, (0, xmax), (ylo, yhi))
    xt = [(v, l) for v, l in nice_ticks(0, xmax, 3)]
    yt = [(v, l) for v, l in nice_ticks(max(0, ylo), yhi, 4)]
    axes_small(P, xt, yt, "complete design time (s)", "sensor cooling (\\%)")
    sx.tex("quality and total cost", P.x0, y0 + 18.5, 6.4, INK, "left")
    for m in ["random", "hot", "nearest", "twos", "mg", "mg49"]:
        c = COL[m]
        aa = [d for d in a if d["method"] == m]
        if study == "pipeline":
            states = ["initial", "refine4", "refine8"] + (["refine16"] if m in ("hot", "random") else [])
        else:
            states = ["initial"]
        xs, ys, lo, hi = [], [], [], []
        for st in states:
            b = [d for d in aa if d["state"] == st]
            gg = grp(study=study, geometry=g, method=m, state=st)
            for d in b:
                sx.circle(float(P.X(d["seconds"])), float(P.Y(d["reduction_pct"])), 0.9, fill=c, stroke=None, opacity=30)
            xs.append(np.mean([d["seconds"] for d in b])); ys.append(np.mean([d["reduction_pct"] for d in b]))
            lo.append(gg["cooling_ci95_low"]); hi.append(gg["cooling_ci95_high"])
        if len(xs) > 1:
            P.area(np.array(xs), np.array(lo), np.array(hi), c, op=14)
        else:
            sx.line(float(P.X(xs[0])), float(P.Y(lo[0])), float(P.X(xs[0])), float(P.Y(hi[0])), stroke=c, width=1.2,
                    opacity=80)
        for st, x, y in zip(states, xs, ys):
            gg = grp(study=study, geometry=g, method=m, state=st)
            sx.line(float(P.X(gg["seconds_ci95_low"])), float(P.Y(y)), float(P.X(min(gg["seconds_ci95_high"], xmax))),
                    float(P.Y(y)), stroke=c, width=0.6, opacity=60)
        if len(xs) > 1:
            P.line(xs, ys, col=c, w=0.9, dash=DASH[m])
        for x, y in zip(xs, ys):
            mk(MK[m], float(P.X(x)), float(P.Y(y)), c, r=1.8, open_=m in OPEN)
    if study == "channels":
        v = [q for q in VOL if q["geometry"] == g]
        sx.tex(f"volume target $0.0168$; audit ${min(q['volume'] for q in v):.5f}$--${max(q['volume'] for q in v):.5f}$",
               P.x0 - 22, P.y0 + P.h + 26, 5.4, MUTED, "left")


# ---------------------------------------------------------------- estimators
def logticks_between(lo, hi):
    return [(k, rf"$10^{{{k}}}$") for k in range(math.ceil(lo - 1e-9), math.floor(hi + 1e-9) + 1)]


def minor(P):
    segs = []
    for k in range(math.floor(P.xa), math.ceil(P.xb)):
        for m in range(2, 10):
            v = k + math.log10(m)
            if P.xa <= v <= P.xb:
                segs.append([[float(P.X(v)), P.y0 + P.h], [float(P.X(v)), P.y0 + P.h + 1.3]])
    for k in range(math.floor(P.ya), math.ceil(P.yb)):
        for m in range(2, 10):
            v = k + math.log10(m)
            if P.ya <= v <= P.yb:
                segs.append([[P.x0, float(P.Y(v))], [P.x0 - 1.3, float(P.Y(v))]])
    if segs:
        sx.multi(segs, stroke=INK, width=0.3, fill=None)


def estimators():
    RH, TOP = 132.0, 18.0
    H = TOP + 3 * RH + 2
    L = math.log10

    def draw():
        items = [("CRN subtraction", "coupled-fd"), ("Harmonic correction", "hit-correction"), ("Direct TWoS", "twos"),
                 ("Multigrid", "mg")]
        x = W / 2 - legend_width(items) / 2
        legend_items(x, 6.5, items)
        for row, (g, label) in enumerate(GEOM):
            y0 = TOP + row * RH
            a = [e for e in EST if e["geometry"] == g]
            sx.tex(label, 2, y0 + 8, 7.6, INK, "left")
            sx.image(img(os.path.join(ASSET, f"{g}-opaque.png"), 560), 0, y0 + 12, 108, 108, name=g, smooth=True)
            # variance against trial radius
            v = [e for e in a if e["budget"] == 2048]
            ys = [L(e[k]) for e in v for k in ("variance_ci95_low", "variance_ci95_high", "normalized_variance")
                  if e[k] and e[k] > 0]
            P = sx.Plot(150, y0 + 14, 128, 88, (L(0.0085), L(0.094)), (math.floor(min(ys) * 2) / 2 - 0.15,
                                                                         math.ceil(max(ys) * 2) / 2 + 0.15))
            old = hstyle.TICK, hstyle.LAB
            hstyle.TICK, hstyle.LAB = 6.0, 6.4
            axes(P, [(L(r), f"${r:g}$") for r in (0.01, 0.02, 0.04, 0.08)], logticks_between(P.ya, P.yb),
                 xl="trial-cavity radius", yl="relative variance", xl_dy=11.0)
            hstyle.TICK, hstyle.LAB = old
            minor(P)
            t = [e for e in v if e["method"] == "twos"][0]
            P.area(np.array([P.xa, P.xb]), np.array([L(t["variance_ci95_low"])] * 2),
                   np.array([L(t["variance_ci95_high"])] * 2), COL["twos"], op=16)
            P.line([P.xa, P.xb], [L(t["normalized_variance"])] * 2, col=COL["twos"], w=1.4)
            for m, dash in (("coupled-fd", [3, 1.6]), ("hit-correction", None)):
                b = sorted([e for e in v if e["method"] == m and e["variance_status"] != "unresolved_all_zero"],
                           key=lambda e: e["radius"])
                if not b:
                    continue
                X = np.array([L(e["radius"]) for e in b])
                P.area(X, np.array([L(max(e["variance_ci95_low"], 10 ** P.ya)) for e in b]),
                       np.array([L(e["variance_ci95_high"]) for e in b]),
                       COL[m], op=16)
                P.line(X, [L(e["normalized_variance"]) for e in b], col=COL[m], w=1.0, dash=dash)
                for e in b:
                    mk(MK[m], float(P.X(L(e["radius"]))), float(P.Y(L(e["normalized_variance"]))), COL[m], r=2.1)
            if g == "ring":
                sx.tex(r"$r=0.01,\,0.02$: all corrections zero", P.x0 + 4, P.y0 + P.h * 0.47, 5.4,
                       COL["hit-correction"], "left", bg="#ffffff")
                sx.tex("variance unresolved", P.x0 + 4, P.y0 + P.h * 0.47 + 6.5, 5.4, COL["hit-correction"], "left",
                       bg="#ffffff")
            # error against time
            xs = [L(e[k]) for e in a for k in ("seconds", "seconds_ci95_low", "seconds_ci95_high") if e[k] == e[k] and e[k] > 0]
            ys = [L(e[k]) for e in a for k in ("relative_rmse", "rmse_ci95_low", "rmse_ci95_high") if e[k] == e[k] and e[k] > 0]
            Q = sx.Plot(326, y0 + 14, 136, 88, (min(xs) - 0.2, max(xs) + 0.2), (min(ys) - 0.25, max(ys) + 0.25))
            hstyle.TICK, hstyle.LAB = 6.0, 6.4
            axes(Q, logticks_between(Q.xa, Q.xb), logticks_between(Q.ya, Q.yb), xl="time for eight query sites (s)",
                 yl="relative RMS error", xl_dy=11.0)
            hstyle.TICK, hstyle.LAB = old
            minor(Q)
            for m in ["coupled-fd", "hit-correction", "twos", "mg"]:
                c = COL[m]
                for e in sorted([e for e in a if e["method"] == m], key=lambda e: e["seconds"]):
                    x, y = float(Q.X(L(e["seconds"]))), float(Q.Y(L(e["relative_rmse"])))
                    segs = []
                    if e["rmse_ci95_high"] == e["rmse_ci95_high"]:
                        segs.append([[x, float(Q.Y(L(max(e["rmse_ci95_low"], 10 ** Q.ya))))], [x, float(Q.Y(L(e["rmse_ci95_high"])))]])
                    if e["seconds_ci95_low"] == e["seconds_ci95_low"] and e["seconds_ci95_low"] > 0:
                        segs.append([[float(Q.X(L(e["seconds_ci95_low"]))), y], [float(Q.X(L(e["seconds_ci95_high"]))), y]])
                    if segs:
                        sx.multi(segs, stroke=c, width=0.5, fill=None, opacity=60)
                    mk(MK[m], x, y, c, r=2.0, open_=e["variance_status"] == "unresolved_all_zero")
                    if m == "mg":
                        sx.tex(f"${int(e['budget'])}$", x + 2.5, y + 7.5, 5.4, c, "left", bg="#ffffff")
        finish()

    sx.single("submission-estimators", W, H, draw)


# ---------------------------------------------------------------- geometries
def geometries():
    audit = jload("benchmarks/submission_study/assets/geometry-audit.json")
    details = {"housing": ("Parallel fins, narrow gaps,", "and mounting bosses"),
               "branch": ("Branching transport, thin ribs,", "and an elevated return loop"),
               "ring": ("Curved transport, a central bridge,", "and closely spaced ribs")}
    H = 200.0

    def draw():
        for col, (g, label) in enumerate(GEOM):
            x = col * 156.0
            sx.tex(label, x + 8, 8, 7.6, INK, "left")
            sx.image(img(os.path.join(ASSET, f"{g}-opaque.png"), 720), x + 2, 12, 150, 150, name=g, smooth=True)
            sx.tex(details[g][0], x + 8, 170, 6.4, INK, "left")
            sx.tex(details[g][1], x + 8, 178, 6.4, INK, "left")
            sx.tex(f"{audit[g]['triangles']:,} triangles; watertight input surface".replace(",", "{,}"), x + 8, 188, 5.8,
                   MUTED, "left")
        finish()

    sx.single("submission-geometries", W, H, draw)


if __name__ == "__main__":
    which = sys.argv[1:] or ["pipeline", "channels", "estimators", "geometries"]
    for w in which:
        {"pipeline": lambda: plate("pipeline"), "channels": lambda: plate("channels"), "estimators": estimators,
         "geometries": geometries}[w]()
