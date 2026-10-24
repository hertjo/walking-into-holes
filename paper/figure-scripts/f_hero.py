"""Hero (teaser): where new cooled cavities should appear in the Stanford Bunny.

Surfaces are the path-traced renders of the saved designs (recoloured into the paper palette by prep_hero.py); every
label, number, outline, colour bar, the cooling plot and the mechanism panel are vector Splinx drawing."""
import json, os
import numpy as np
from PIL import Image
from splinx import draw as sx
from hstyle import *
from hdata import REPO, rows, js, where
from hgeom import peanut, wos

sx.OUT = "out"
HERE = os.path.dirname(os.path.abspath(__file__))
ASSET = os.path.join(HERE, "assets/hero")
W, H = 468, 252
T_SIZE, S_SIZE, N_SIZE = 6.8, 5.1, 5.2
BASE_COL = "#7fb3c6"  # baseline shells are tinted cyan-grey in the renders; their outlines match

# ---------------------------------------------------------------- data
traj = rows("benchmarks/stanford_bunny/output/design-trajectories.csv")


def reduction(method, n):
    g = {r["iteration"]: r["objective"] for r in where(traj, method=method)}
    return 100 * (1 - g[n] / g[0])


alloc = {r["tag"]: r for r in js("figures/hero-assets/budget-render-manifest.json")["designs"]}
meta = js("figures/hero-assets/thermal/manifest.json")
acc = js("figures/hero-assets/thermal/accuracy-check.json")
sc64 = where(rows("benchmarks/compiled_bunny/output/output-scaling.csv"), sites=64)[0]
budget = rows("benchmarks/bunny_budget/output/summary.csv")
full = [r for r in budget if r["volume_units"] == 6]
half = [r for r in budget if r["volume_units"] == 3 and r["method"] == "twos"]
TMAX = meta["temperature_max"]

PANELS = [  # tag, col, row, title, subtitle, colour
    ("input", 0, 0, "Initial temperature", "Internal heating; cooled boundaries", VIOLET),
    ("random", 1, 0, "Random placement", f"6 cavities; {reduction('Random feasible', 6):.1f}\\% lower heat objective", BASE_COL),
    ("voxel", 2, 0, "Voxel-adjoint placement", f"6 cavities; {reduction('Voxel adjoint TD', 6):.1f}\\% lower heat objective", BASE_COL),
    ("ours0", 0, 1, "Where to insert", r"Predicted benefit $-D_T=u^2$", VIOLET),
    ("ours1", 1, 1, r"TWoS (1 cavity)", f"{reduction('TWoS + fresh validation', 1):.1f}\\% lower heat objective", VIOLET),
    ("budget6", 2, 1, r"TWoS (6 large cavities)", f"{alloc['budget6']['cooling_percent']:.2f}\\% lower heat objective", VIOLET),
    ("budget12", 3, 1, r"TWoS (12 small cavities)", f"{alloc['budget12']['cooling_percent']:.2f}\\% lower heat objective", VIOLET),
]

X0, COLW = 90.0, (468.0 - 90.0) / 4
ROWY = [0.0, 126.0]
IMG = 96.0  # image box side (pt)
PX = 520    # embedded pixels per side (≈ 480 dpi at IMG pt)


def load(tag, detail=False):
    im = Image.open(os.path.join(ASSET, f"{tag}{'-detail' if detail else ''}.png")).convert("RGB")
    n = 300 if detail else PX
    im = im.resize((n, n), Image.LANCZOS)
    a = np.asarray(im)
    if detail:  # antialiased circular mask
        yy, xx = np.mgrid[:n, :n]
        r = np.hypot(xx + 0.5 - n / 2, yy + 0.5 - n / 2)
        alpha = np.clip((n / 2 - r) * 1.0 + 0.5, 0, 1)
        a = np.dstack([a, np.uint8(alpha * 255)])
    return a


def cbar(x, y, w, h, stops, ticks, label, size=N_SIZE, label_side="top"):
    n = 256
    arr = sx.cmap(np.linspace(0, 1, n)[None, :].repeat(2, 0), 0, 1, stops)
    sx.image(arr, x, y, w, h, name="cbar", smooth=True)
    sx.rect(x, y, w, h, fill=None, stroke=INK, width=0.35)
    for t, lab in ticks:
        xx = x + t * w
        sx.line(xx, y + h, xx, y + h + 1.6, stroke=INK, width=0.35)
        sx.tex(lab, xx, y + h + 2.4, size, INK, "center", valign="top")
    if label:
        if label_side == "top":
            sx.tex(label, x, y - 2.0, size + 0.2, INK, "left")
        else:
            sx.tex(label, x + w + 3, y + h / 2, size, INK, "left", valign="middle")


def panel(tag, col, row, title, sub, colour):
    x = X0 + col * COLW + 1.5
    y = ROWY[row]
    sx.tex(title, x, y + 7.5, T_SIZE, INK, "left", maxw=COLW - 3)
    sx.tex(sub, x, y + 15.0, S_SIZE, MUTED, "left", maxw=COLW - 3)
    iy = y + 19.0
    s = IMG
    sx.image(load(tag), x - 4, iy, s, s, name=tag, smooth=True)
    # zoom: same world point as the detail camera; the leader marks the zoom centre
    cx, cy, r = x - 4 + 0.81 * s, iy + (1 - 0.79) * s, 0.192 * s
    ax_, ay_ = x - 4 + 0.54 * s, iy + (1 - 0.37) * s
    bx_, by_ = x - 4 + 0.69 * s, iy + (1 - 0.63) * s
    sx.line(ax_, ay_, bx_, by_, stroke=colour, width=0.6)
    sx.circle(ax_, ay_, 0.85, fill="#ffffff", stroke=colour, width=0.5)
    sx.circle(cx, cy, r, fill="#ffffff", stroke=None)
    sx.image(load(tag, True), cx - r, cy - r, 2 * r, 2 * r, name=tag + "-d", smooth=True)
    wire = os.path.join(REPO, "figures/hero-assets", f"{tag}-wire.json")
    if os.path.exists(wire):
        segs = np.array(json.load(open(wire)))
        P = segs * (2 * r) + np.array([cx - r, cy - r])
        P[..., 1] = 2 * (cy) - P[..., 1]  # data y up -> page y down, about the circle centre
        keep = (np.hypot(P[..., 0] - cx, P[..., 1] - cy) < r - 0.3).all(1)
        sx.multi(P[keep].tolist(), stroke="#2a2440", width=0.18, fill=None, opacity=70)
    sx.circle(cx, cy, r, fill=None, stroke=colour, width=0.8)
    return x, iy


def budget_plot():
    x = X0 + 3 * COLW + 1.5
    sx.tex("Similar cooling, half the volume", x, 8.0, T_SIZE, INK, "left", maxw=COLW - 2)
    fr = full[0]["removed_fraction"] * 100; hr = half[0]["removed_fraction"] * 100
    sx.tex(f"Full / half removal: {fr:.2f}\\% / {hr:.2f}\\%", x, 16.0, S_SIZE, MUTED, "left")
    P = sx.Plot(x + 16, 34, COLW - 21, 30, (1.2, 12.8), (18, 60))
    import hstyle
    hstyle.TICK = 5.2
    axes(P, [(2, "$2$"), (4, "$4$"), (6, "$6$"), (12, "$12$")], [(20, "$20$"), (40, "$40$"), (60, "$60$")],
         grid="y", xl_dy=9.5)
    hstyle.TICK = 7.0
    sx.tex("cavities (radius set by volume)", P.x0 + P.w / 2, P.y0 + P.h + 9.0, N_SIZE, INK, "center", valign="top")
    ylabel("cooling (\\%)", P.x0 - 12.0, P.y0 + P.h / 2, size=N_SIZE)
    series = [("random", METHOD["random"], None, full), ("voxel", METHOD["voxel"], None, full),
              ("twos", METHOD["twos"], [2.4, 1.6], half), ("twos", METHOD["twos"], None, full)]
    for m, c, dash, src in series:
        g = sorted([r for r in src if r["method"] == m], key=lambda r: r["count"])
        xs = [r["count"] for r in g]
        P.area(np.array(xs), np.array([r["cooling_min"] for r in g]), np.array([r["cooling_max"] for r in g]), c, op=16)
        P.line(xs, [r["cooling_mean"] for r in g], col=c, w=1.0, dash=dash)
        for r in g:
            if dash:
                sx.circle(float(P.X(r["count"])), float(P.Y(r["cooling_mean"])), 1.35, fill="#ffffff", stroke=c, width=0.6)
            else:
                marker("o", float(P.X(r["count"])), float(P.Y(r["cooling_mean"])), c, r=1.6, edge="#ffffff")
    # legend: two short rows above the plot
    legend_row([("TWoS (full)", METHOD["twos"], "line"), ("Voxel", METHOD["voxel"], "line")], x + 1, 22.0, size=5.2, gap=5)
    legend_row([("TWoS (half)", METHOD["twos"], "dash"), ("Random", METHOD["random"], "line")], x + 1, 28.0, size=5.2, gap=5)
    return P


def callouts():
    x = X0 + 3 * COLW + 1.5
    sp = sc64["voxel_seconds"] / sc64["twos_seconds"]
    ram = 100 * (1 - sc64["twos_peak_rss_megabytes"] / sc64["voxel_peak_rss_megabytes"])
    sx.tex(f"64 sites: ${sp:.2f}\\times$ faster; {ram:.0f}\\% less RAM", x, 85.5, S_SIZE, NAVY, "left", maxw=COLW - 2)
    sx.tex(f"TD error: {100 * acc['twos_rrmse_16384']:.1f}\\% (ours) vs {100 * acc['voxel_rrmse_16384']:.1f}\\% (voxel)",
           x, 92.5, S_SIZE, INK, "left", maxw=COLW - 2)
    sx.tex("Separate query test; 16,384-walk reference", x, 99.0, N_SIZE - 0.2, MUTED, "left", maxw=COLW - 2)
    cbar(x + 1, 109.0, COLW - 12, 3.2, TEMP_MAP,
         [(0, f"${0:.4f}$"), (0.5, f"${TMAX / 2:.4f}$"), (1, f"${TMAX:.4f}$")], None)
    return x


# ---------------------------------------------------------------- mechanism panel (vector)
TEMP_MAP = [(0.0, INK), (0.3, "#34299a"), (0.58, VIOLET), (0.84, LILAC), (1.0, "#f6f5ff")]
BEN_MAP = [(0.0, INK), (0.32, "#1f2fa8"), (0.64, CYAN), (0.87, MINT), (1.0, "#e6f8f0")]
PRIMAL, ADJ, ZC = NAVY, CYAN, VIOLET


def mechanism():
    x0, w = 1.0, 82.0
    # --- finite-cavity differencing
    sx.tex("Finite-cavity differencing", x0, 8.0, T_SIZE - 0.4, INK, "left", maxw=w)
    sx.tex("Perturb the domain, then subtract", x0, 15.5, N_SIZE, MUTED, "left", maxw=w)
    D = peanut(x0 + 40, 41, 37, 15.5)
    sx.poly(D, True, fill="#f4f3f9", stroke=INK, width=0.9)
    src = np.array([x0 + 17.0, 43.0]); hole = (np.array([x0 + 58.5, 39.0]), 2.3)
    miss, hit, ends = [], None, []
    for seed in range(3000):
        pts, rad, where_ = wos(src, D, np.random.default_rng(seed), eps=0.7, holes=[hole], rmax=6.5)
        n = len(pts)
        if where_ == 0 and hit is None and 8 <= n <= 22:
            hit = pts
        elif where_ == "outer" and len(miss) < 4 and 7 <= n <= 22:
            e = pts[-1]
            if pts[:, 0].max() > src[0] + 18 and e[0] > src[0] + 6 and all(np.hypot(*(e - f)) > 16 for f in ends):
                miss.append(pts); ends.append(e)
        if hit is not None and len(miss) >= 4:
            break
    for pts in miss:
        sx.poly(pts, stroke=GREY, width=0.4, fill=None, opacity=80)
    sx.poly(hit, stroke=ADJ, width=0.75, fill=None)
    sx.circle(*hole[0], hole[1], fill="#ffffff", stroke=ADJ, width=0.9)
    sx.circle(*src, 1.2, fill=INK, stroke=None)
    # callouts
    sx.poly([[hole[0][0] + 1.4, hole[0][1] - 2.0], [x0 + 66, 23.0], [x0 + 80, 23.0]], stroke=ADJ, width=0.4, fill=None)
    sx.tex("Tiny trial hole", x0 + 80, 21.6, N_SIZE, INK, "right")
    sx.poly([[hole[0][0] + 1.2, hole[0][1] + 2.0], [x0 + 68, 59.5], [x0 + 80, 59.5]], stroke=ADJ, width=0.4, fill=None)
    sx.tex("Rare hit", x0 + 80, 64.6, N_SIZE, INK, "right")
    sx.poly([[x0 + 22, 52.0], [x0 + 18, 60.0], [x0 + 3, 60.0]], stroke=MUTED, width=0.4, fill=None)
    sx.tex("Most walks miss", x0 + 2, 64.6, N_SIZE, MUTED, "left")
    sx.tex(r"$\dfrac{J(\Omega\setminus B_\varepsilon)-J(\Omega)}{\operatorname{cap}(B_\varepsilon)}$", x0 + 41, 69.0,
           6.6, INK, "center", valign="top")
    sx.tex("Tiny target; amplified noise", x0 + 41, 92.5, N_SIZE, MUTED, "center")
    sx.line(x0, 97.5, x0 + w - 2, 97.5, stroke=GRID, width=0.6)
    # --- TWoS
    sx.tex(r"TWoS: score, check, insert", x0, 107.5, T_SIZE - 0.4, VIOLET, "left", maxw=w)

    def step(n, y, text):
        sx.circle(x0 + 4.0, y - 2.0, 3.2, fill="#ffffff", stroke=VIOLET, width=0.5)
        sx.tex(f"${n}$", x0 + 4.0, y - 2.0, 5.2, VIOLET, "center", valign="middle")
        sx.tex(text, x0 + 10.0, y, N_SIZE, INK, "left")
    step(1, 117.0, "Query the original domain")
    D2 = peanut(x0 + 40, 143, 37, 16.5)
    sx.poly(D2, True, fill="#efeefb", stroke=INK, width=0.9)
    z = np.array([x0 + 47.0, 141.5])
    def pick(base, cond):
        for seed in range(4000):
            pts, rad, _ = wos(z, D2, np.random.default_rng(base + seed), eps=0.8, rmax=5.5)
            if cond(pts, rad):
                return pts, rad
    best = pick(100, lambda p, r: 5 <= len(r) <= 9 and p[-1][1] > 150 and p[-1][0] < x0 + 30 and p[:, 0].min() > x0 + 12)
    adj = pick(900, lambda p, r: 4 <= len(r) <= 8 and p[-1][1] < 134 and p[-1][0] > x0 + 58)
    for (pts, rad), c in ((best, PRIMAL), (adj, ADJ)):
        for p, r in zip(pts[:-1], rad):
            sx.circle(p[0], p[1], r, fill=None, stroke=c, width=0.3, opacity=60)
        sx.poly(pts, stroke=c, width=0.8, fill=None)
        sx.circle(*pts[-1], 0.9, fill="#ffffff", stroke=c, width=0.5)
    sx.circle(*z, 1.5, fill=ZC, stroke="#ffffff", width=0.4)
    sx.tex("$z$", z[0] + 1.0, z[1] + 9.0, N_SIZE, ZC, "center")
    sx.tex("No trial hole", x0 + 12.0, 137.0, 4.8, MUTED, "left")
    sx.tex("Primal $u$", x0 + 12.0, 144.0, N_SIZE, PRIMAL, "left")
    sx.tex("Adjoint $p$", x0 + 81, 124.5, N_SIZE, INK, "right")
    step(2, 170.0, "Combine independent batches")
    sx.tex(r"$\widehat D_T=-\operatorname{avg}_{i\neq j}U_iP_j$", x0 + 42, 181.5, 6.8, VIOLET, "center")
    step(3, 195.0, "Fresh check")
    sx.tex("then insert", x0 + 10.0, 202.0, N_SIZE, VIOLET, "left")
    sx.arrow((x0 + 42, 197.5), (x0 + 56, 197.5), col=VIOLET, w=0.6, head=3.0)
    D3 = peanut(x0 + 69, 197.5, 11.5, 6.0)
    sx.poly(D3, True, fill="#efeefb", stroke=INK, width=0.7)
    sx.circle(x0 + 71.5, 197.0, 1.8, fill="#ffffff", stroke=VIOLET, width=0.6)


def draw():
    mechanism()
    sx.line(86.0, 2.0, 86.0, H - 8, stroke=GRID, width=0.6)
    for p in PANELS:
        x, iy = panel(*p)
        if p[0] == "budget6":
            sx.tex(f"Full budget: 1.529\\% removed; $r=0.0500$", x, iy + IMG + 6.0, 5.0, VIOLET, "left", maxw=COLW - 4)
        if p[0] == "budget12":
            sx.tex(f"Half budget: 0.765\\% removed; $r=0.0315$", x, iy + IMG + 6.0, 5.0, NAVY, "left", maxw=COLW - 4)
        if p[0] == "ours0":
            cbar(x + 2, iy + IMG + 0.5, 34, 2.6, BEN_MAP, [(0, "$0$"), (1, "$1$")], None, size=4.8)
            sx.tex("relative benefit; cross: first selected site", x + 39, iy + IMG + 1.8, 4.8, MUTED, "left", valign="middle")
    budget_plot()
    callouts()
    x = X0 + 3 * COLW + 1.5
    sx.tex("Interior temperature $u$ (shared scale)", x + 1, 106.5, N_SIZE, INK, "left", maxw=COLW - 4)
    for msg in sx.check():
        print(msg)


if __name__ == "__main__":
    sx.single("mesh-hero", W, H, draw)
