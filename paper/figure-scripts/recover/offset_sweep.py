"""Read back the finite-offset / seam-strip audit (old supplement Fig. S22, vector, page 60 of the anonymous
supplement PDF): its run folder benchmarks/engineering_housing/output/channels/offset_sweep/ is not tracked.
Marker centres and error-bar ends are mapped through the axis tick marks. Writes data/offset_seam_sweep.json."""
import json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from svgprims import prims, inside

REPO = os.environ.get("HOLES_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
svg = "/tmp/offset_sweep_p60.svg"
subprocess.run(["pdftocairo", "-svg", "-f", "60", "-l", "60",
                f"{REPO}/paper/Walking_into_Holes_SIGGRAPH_Supplement.pdf", svg], check=True)
P = [p for p in prims(svg) if inside(p, (40, 430, 580, 600))]


def lin(p0, v0, p1, v1):
    return lambda p: v0 + (p - p0) * (v1 - v0) / (p1 - p0)


def logm(p0, v0, p1, v1):
    a, b = np.log10(v0), np.log10(v1)
    return lambda p: 10 ** (a + (p - p0) * (b - a) / (p1 - p0))


def markers(col, box):
    out = []
    for p in P:
        if p["fill"] == col and p["stroke"] == col and len(p["pts"]) in (5, 10) and inside(p, box):
            out.append(p["pts"][:-1].mean(0))
    return np.array(sorted(out, key=lambda q: q[0]))


def segs(col, box, vertical):
    out = []
    for p in P:
        q = p["pts"]
        if p["stroke"] == col and len(q) == 2 and inside(p, box):
            d = np.abs(q[1] - q[0])
            L = d.max()
            if 3.45 < L < 3.6:  # error-bar caps (3.52 pt)
                continue
            if (vertical and d[0] < 0.01 and d[1] > 0.05) or (not vertical and d[1] < 0.01 and d[0] > 0.05):
                out.append(q)
    return out


# (a) frozen directional audit, x and y in units of 1e-3
ax_ = lin(101.43, -2, 187.94, 2); ay = lin(558.18, -2, 478.06, 2)
A = {}
for col, lr in (("#0b7285", 0.025), ("#e07a1f", 0.051), ("#6a51a3", 0.203)):
    box = (104, 460, 200, 575)
    pts = [m for m in markers(col, (100, 460, 200, 575))]
    xb = segs(col, box, False); yb = segs(col, box, True)
    rows = []
    for m in pts:
        hx = [s for s in xb if abs(s[0, 1] - m[1]) < 0.3 and min(s[:, 0]) - 0.5 <= m[0] <= max(s[:, 0]) + 0.5]
        hy = [s for s in yb if abs(s[0, 0] - m[0]) < 0.3 and min(s[:, 1]) - 0.5 <= m[1] <= max(s[:, 1]) + 0.5]
        hx = max(hx, key=lambda s: abs(s[1, 0] - s[0, 0])) if hx else np.array([[m[0], m[1]]] * 2)
        hy = max(hy, key=lambda s: abs(s[1, 1] - s[0, 1])) if hy else np.array([[m[0], m[1]]] * 2)
        rows.append({"fd_diffwos": float(ax_(m[0])), "central": float(ay(m[1])),
                     "fd_lo": float(ax_(hx[:, 0].min())), "fd_hi": float(ax_(hx[:, 0].max())),
                     "c_lo": float(ay(hy[:, 1].max())), "c_hi": float(ay(hy[:, 1].min()))})
    A[str(lr)] = rows
# (b) offset convergence
bx = logm(266.39, 0.025, 370.97, 0.2); by = lin(566.8, 1.0, 475.09, 3.0); br = lin(571.38, 0.975, 472.74, 1.0)
slope = markers("#0b7285", (262, 460, 380, 575)); cos = markers("#2a9d55", (262, 460, 380, 575))
slope = slope[slope[:, 0] > 263]; cos = cos[cos[:, 0] > 263]
# drop the legend entries (they sit at x ~ 273.5 near the top)
slope = np.array([q for q in slope if not (abs(q[0] - 273.53) < 0.1)])
cos = np.array([q for q in cos if not (abs(q[0] - 273.53) < 0.1)])
B = {"offset": [float(bx(q[0])) for q in slope], "fd_slope": [float(by(q[1])) for q in slope],
     "cosine": [float(br(q[1])) for q in cos]}
# (c) safe-strip rate
cx = logm(441.24, 0.025, 545.82, 0.2); cy = logm(547.39, 1e-2, 501.51, 1e-1)
C = {}
for col, key in (("#c23b22", "all_exits"), ("#6a51a3", "given_cavity_exit")):
    ms = [q for q in markers(col, (437, 460, 560, 570)) if not abs(q[0] - 485.0) < 0.1]
    vb = segs(col, (437, 460, 560, 570), True)
    rows = []
    for m in ms:
        hy = [s for s in vb if abs(s[0, 0] - m[0]) < 0.3]
        hy = max(hy, key=lambda s: abs(s[1, 1] - s[0, 1])) if hy else np.array([[m[0], m[1]]] * 2)
        rows.append({"offset": float(cx(m[0])), "rate": float(cy(m[1])), "lo": float(cy(hy[:, 1].max())),
                     "hi": float(cy(hy[:, 1].min()))})
    C[key] = rows
guide = [p for p in P if p["stroke"] == "#687176" and inside(p, (437, 460, 560, 570)) and len(p["pts"]) >= 2
         and abs(p["pts"][-1, 0] - p["pts"][0, 0]) > 30]
G = None
if guide:
    q = guide[0]["pts"]
    G = {"offset": [float(cx(q[0, 0])), float(cx(q[-1, 0]))], "rate": [float(cy(q[0, 1])), float(cy(q[-1, 1]))]}
out = os.path.join(os.path.dirname(__file__), "..", "data/offset_seam_sweep.json")
json.dump({"source": "old supplement Fig. S22 (vector read-back)", "a": A, "b": B, "c": C, "guide": G},
          open(out, "w"), indent=1)
print(json.dumps({"a": {k: len(v) for k, v in A.items()}, "b": B, "c": {k: [round(r["rate"], 4) for r in v]
      for k, v in C.items()}, "guide": G}, indent=1))
