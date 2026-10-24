"""Read back the connected-channel figure (old supplement Fig. S21, page 60 of the anonymous supplement PDF); its run
folder benchmarks/engineering_housing/output/channels/ is not tracked. Vector markers are mapped through the axis
ticks; the scatter colours of panel (a) are inverted through inferno (colour bar 0 .. 0.003013); the three rasterised
3-D shells and the channel tube are extracted with their placements. Writes data/housing_channels.json and
assets/channels/*.png (RGBA)."""
import json, os, subprocess, sys, io, base64, re
import numpy as np
from PIL import Image
import matplotlib
sys.path.insert(0, os.path.dirname(__file__))
from svgprims import prims, inside

REPO = os.environ.get("HOLES_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
svg = "/tmp/housing_channels_p60.svg"
subprocess.run(["pdftocairo", "-svg", "-f", "60", "-l", "60",
                f"{REPO}/paper/Walking_into_Holes_SIGGRAPH_Supplement.pdf", svg], check=True)
P = prims(svg)
reg = [p for p in P if inside(p, (60, 190, 560, 345))]
lin = lambda p0, v0, p1, v1: (lambda p: v0 + (p - p0) * (v1 - v0) / (p1 - p0))


def mk(col, box=(60, 190, 560, 345)):
    return [p["pts"][:-1].mean(0) for p in P if p["fill"] == col and p["stroke"] == col and len(p["pts"]) >= 5
            and inside(p, box)]


def bars(col, x):
    s = [p["pts"] for p in reg if p["stroke"] == col and len(p["pts"]) == 2 and abs(p["pts"][0, 0] - x) < 0.05
         and abs(p["pts"][1, 0] - x) < 0.05]
    s = max(s, key=lambda q: abs(q[1, 1] - q[0, 1]))
    return s[:, 1].max(), s[:, 1].min()


# (d) paired change in J (x 1e-4) per shape iteration
dx = lin(88.65, 1, 207.57, 6); dy = lin(335.86, -1.0, 236.49, 0.75)
D = []
for col, acc, bc in (("#277da1", True, "#70a9bf"), ("#d55e57", False, "#df9d98")):
    for m in mk(col, (80, 230, 215, 320)):
        lo, hi = bars(bc, m[0])
        D.append({"iteration": int(round(dx(m[0]))), "change": float(dy(m[1])), "lo": float(dy(lo)), "hi": float(dy(hi)),
                  "accepted": acc})
D.sort(key=lambda r: r["iteration"])
# (e) central difference against Differential WoS
ex = lin(267.97, -0.002, 363.02, 0.002); ey = lin(320.75, -0.002, 243.68, 0.002)
E = []
for m in mk("#663f8c"):
    lo, hi = bars("#9a86b0", m[0])
    E.append({"diffwos": float(ex(m[0])), "central": float(ey(m[1])), "lo": float(ey(lo)), "hi": float(ey(hi))})
E.sort(key=lambda r: r["diffwos"])
# (a) scatter: page positions and values
inf = matplotlib.colormaps["inferno"](np.linspace(0, 1, 256))[:, :3]
dots = []
for p in P:
    if p["stroke"] is None and p["fill"] and p["fill"].startswith("#") and len(p["pts"]) >= 5 and inside(p, (100, 100, 210, 180)):
        c = np.array([int(p["fill"][i:i + 2], 16) / 255 for i in (1, 3, 5)])
        k = int(((inf - c) ** 2).sum(1).argmin())
        if ((inf[k] - c) ** 2).sum() < 0.003:
            dots.append([float(v) for v in p["pts"][:-1].mean(0)] + [k / 255 * 0.003013, float(np.ptp(p["pts"][:, 0]))])
# (b), (c) control markers and the pre-refinement centre line
ctrl = {k: [list(map(float, m)) for m in mk("#183f49", box)] for k, box in (("b", (260, 100, 425, 180)),
                                                                             ("c", (425, 100, 560, 180)))}
centre = [p["pts"].tolist() for p in P if p["stroke"] == "#f3b65d" and inside(p, (425, 100, 560, 180))]
# rasters and placements
s = open(svg).read()
os.makedirs(f"{HERE}/assets/channels", exist_ok=True)
place = {}
for key, rgb_id, a_id in (("a", "source-6", "source-7"), ("b", "source-18", "source-19"), ("c", "source-24", "source-25"),
                          ("tube", "source-12", "source-13")):
    def img(i):
        m = re.search(r'<image id="' + i + r'"[^>]*xlink:href="data:image/png;base64,([^"]+)"', s)
        return Image.open(io.BytesIO(base64.b64decode(m.group(1))))
    rgb = img(rgb_id).convert("RGB"); al = img(a_id).convert("L")
    t = re.search(r'<use xlink:href="#' + rgb_id + r'" transform="matrix\(([^)]*)\)"', s).group(1)
    a = [float(v) for v in re.split(r"[ ,]+", t.strip())]
    place[key] = {"x": a[4], "y": a[5], "scale": a[0], "w": rgb.size[0], "h": rgb.size[1]}
    im = rgb.copy(); im.putalpha(al)
    im.save(f"{HERE}/assets/channels/{key}.png")
json.dump({"source": "old supplement Fig. S21 (vector read-back)", "d": D, "e": E, "dots": dots, "controls": ctrl,
           "centre_line": centre, "placement": place}, open(f"{HERE}/data/housing_channels.json", "w"), indent=1)
print(len(D), len(E), len(dots), {k: len(v) for k, v in ctrl.items()}, len(centre), place)
print(D); print(E)
