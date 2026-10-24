"""Recover the regional benchmark's plane fields (u, p, up) and first-seed cavity layout from the old figure.

The figure was drawn from run folders (side*/seed*/, figure/) that are not tracked; its rasters and paths are the
only retained copy. Writes data/distinct_adjoint_fields.npz and data/distinct_adjoint_layout.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from pdfrecover import *

REPO = os.environ.get("HOLES_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
pg = page(os.path.join(REPO, "benchmarks/distinct_adjoint/output/distinct-adjoint-benchmark.pdf"))
# column offsets of the three top panels (pt); axis maps from the tick marks (x: -1 .. 1, y: -0.5 .. 0.5)
cols = [0.0, 244.8, 489.6]
fields = {}
imgs = [im for im in images(pg) if im[2].shape[:2] == (157, 249)]
# colour-bar spans and ticks (pt) -> vmax with vmin = 0 (all three bars start at their data minimum ~ 0)
bars = [((74.22, 214.84), (118.57, 2e-2), (207.42, 6e-2)),
        ((319.02, 459.64), (354.38, 0.6e-3), (425.11, 1.8e-3)),
        ((563.82, 704.44), (608.64, 0.4e-5), (698.29, 1.2e-5))]
for k, (name, im) in enumerate(zip(["u", "p", "score"], imgs)):
    off = cols[k]
    ax = Axis((65.25 + off, -1.0), (223.81 + off, 1.0), (123.26, -0.5), (43.97, 0.5))
    (b0, b1), (t0, v0), (t1, v1) = bars[k]
    slope = (v1 - v0) / (t1 - t0)
    vmin, vmax = v0 - (t0 - b0) * slope, v0 + (b1 - t0) * slope
    t, err = invert_cmap(im[2], "magma")
    a = im[3] if im[3] is not None else np.full(t.shape, 255, np.uint8)
    val = vmin + t * (vmax - vmin)
    x0, y0, x1, y1 = im[1]
    fields[name] = val.astype(np.float32)
    fields[name + "_alpha"] = a
    fields[name + "_extent"] = np.array([ax.x(x0), ax.x(x1), ax.y(y1), ax.y(y0)])
    fields[name + "_range"] = np.array([vmin, vmax])
    print(name, "range", vmin, vmax, "median cmap error", float(np.median(err[a > 128])), "extent",
          fields[name + "_extent"])
# outline (grey contour) and cavity circles, in data coordinates of panel d (x ticks as panel a; y shifted)
ax_d = Axis((65.25, -1.0), (223.81, 1.0), (368.9, -0.5), (289.3, 0.5))
yt = ticks(pg, (40, 250, 50, 400), vertical=False)
xt = ticks(pg, (50, 390, 240, 400), vertical=True)
print("panel d ticks", xt, yt)
ax_d = Axis((xt[0], -1.0), (xt[-1], 1.0), (yt[-1], -0.5), (yt[0], 0.5))
ax_a = Axis((65.25, -1.0), (223.81, 1.0), (123.26, -0.5), (43.97, 0.5))
layout = {"outline": [], "mg_hot": [], "twos_sparse": [], "sensor": None, "exclusion_radius": None}
for d in drawings(pg, (50, 20, 240, 145)):
    if d.get("color") and abs(d["color"][0] - 0.333) < 0.01 and len(d["items"]) > 50:
        P = path_points(d)
        layout["outline"].append(np.c_[ax_a.x(P[:, 0]), ax_a.y(P[:, 1])].round(5).tolist())
for d in drawings(pg, (50, 270, 240, 400)):
    col = d.get("color")
    if not col or len(d["items"]) < 4:
        continue
    r = d["rect"]
    cx, cy, rad = ax_d.x((r.x0 + r.x1) / 2), ax_d.y((r.y0 + r.y1) / 2), (r.x1 - r.x0) / 2 / (xt[-1] - xt[0]) * 2
    rgb = tuple(round(c, 3) for c in col)
    if rgb[0] > 0.7 and rgb[2] < 0.4:
        layout["mg_hot"].append([round(float(cx), 4), round(float(cy), 4), round(float(rad), 4)])
    elif rgb[0] > 0.5 and rgb[2] > 0.6:
        layout["twos_sparse"].append([round(float(cx), 4), round(float(cy), 4), round(float(rad), 4)])
    elif rgb[2] > 0.7 and rgb[0] < 0.3 and d.get("fill"):
        layout["sensor"] = [round(float(cx), 4), round(float(cy), 4), round(float(rad), 4)]
for d in drawings(pg, (540, 20, 730, 145)):
    if d.get("fill") and d["fill"][0] > 0.9 and abs(d["fill"][0] - d["fill"][2]) < 0.02 and len(d["items"]) >= 4:
        r = d["rect"]
        layout["exclusion_radius"] = round(float((r.x1 - r.x0) / 2 / 79.28), 4)
print(json.dumps({k: (v if k != "outline" else [len(o) for o in v]) for k, v in layout.items()}))
np.savez_compressed(os.path.join(OUT, "distinct_adjoint_fields.npz"), **fields)
json.dump(layout, open(os.path.join(OUT, "distinct_adjoint_layout.json"), "w"))
