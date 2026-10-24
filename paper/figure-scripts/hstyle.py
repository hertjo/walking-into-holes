"""House style for the Walking-into-Holes figures: matches the Beckmann paper (Latin Modern labels, lavender grid,
left/bottom spines, frameless legends) in the paper palette."""
import math
import numpy as np
from splinx import draw as sx

INK = "#1e1b3a"; NAVY = "#0500b5"; VIOLET = "#6b67ee"; LILAC = "#c9c6ff"; CYAN = "#3fc4f0"; MINT = "#2fbf8f"
GRID = "#E4E2F2"; MUTED = "#5d5a7a"; GREY = "#a19fb8"
# one colour per method, used in every figure of the paper
METHOD = {
    "twos": NAVY,          # TWoS (ours), direct insertion score
    "multigrid": VIOLET,   # multigrid / voxel-adjoint topological derivative (finest grid)
    "multigrid_coarse": LILAC,  # coarser multigrid (open marker, violet edge)
    "voxel": VIOLET,       # voxel FD/CG adjoint
    "nearest": MINT,       # nearest-sensor placement
    "hottest": CYAN,       # hottest-point placement
    "random": GREY,        # random feasible placement
    "crn": GREY,           # finite-cavity CRN subtraction
    "harmonic": MINT,      # finite-cavity harmonic correction
    "fem": CYAN,           # external FEM reference
}
TICK, LAB, ANN = 7.0, 8.0, 6.8



def ylabel(s, x, y, size=LAB):
    f = lambda: sx.tex(s, 0, 0, size, INK, anchor="center", valign="middle")
    sx.move_to(sx.rotate(f, -90), (x, y))()


def axes(P, xt, yt, xl=None, yl=None, title=None, grid=True, yl_dx=None, xl_dy=15.5, spines="lb"):
    """Ticks as [(value, label)] in the plot's data coordinates."""
    x0, y0, w, h = P.x0, P.y0, P.w, P.h
    segs = []
    if grid in (True, "x"):
        segs += [[[float(P.X(a)), y0], [float(P.X(a)), y0 + h]] for a, _ in xt]
    if grid in (True, "y"):
        segs += [[[x0, float(P.Y(b))], [x0 + w, float(P.Y(b))]] for b, _ in yt]
    if segs:
        sx.multi(segs, stroke=GRID, width=0.5, fill=None)
    sx.poly([[x0, y0], [x0, y0 + h], [x0 + w, y0 + h]], stroke=INK, width=0.6, fill=None)
    ticks = [[[float(P.X(a)), y0 + h], [float(P.X(a)), y0 + h + 2.5]] for a, _ in xt]
    ticks += [[[x0, float(P.Y(b))], [x0 - 2.5, float(P.Y(b))]] for b, _ in yt]
    sx.multi(ticks, stroke=INK, width=0.5, fill=None)
    for a, lab in xt:
        if lab:
            sx.tex(lab, float(P.X(a)), y0 + h + 4.2, TICK, INK, "center", valign="top")
    ywid = 0.0
    for b, lab in yt:
        if lab:
            ywid = max(ywid, sx.tex(lab, x0 - 4.2, float(P.Y(b)), TICK, INK, "right", valign="middle"))
    if xl:
        sx.tex(xl, x0 + w / 2, y0 + h + xl_dy, LAB, INK, "center", valign="top")
    if yl:
        ylabel(yl, x0 - 4.2 - ywid - 7 - (yl_dx or 0), y0 + h / 2)
    if title:
        sx.tex(title, x0 + w / 2, y0 - 8, LAB, INK, "center")


def logticks(a, b, labels=None):
    """Decades between 10^a and 10^b (a, b integers) as (log10 value, label)."""
    out = []
    for k in range(a, b + 1):
        out.append((k, labels.get(k, rf"$10^{{{k}}}$") if labels else rf"$10^{{{k}}}$"))
    return out


def marker(kind, x, y, col, r=2.6, edge="#ffffff"):
    """Filled marker with a thin white edge: o circle, s square, ^ triangle."""
    if kind == "o":
        sx.circle(x, y, r, fill=col, stroke=edge, width=0.5)
    elif kind == "s":
        q = r * 0.9
        sx.rect(x - q, y - q, 2 * q, 2 * q, fill=col, stroke=edge, width=0.5)
    elif kind == "^":
        q = r * 1.25
        sx.poly([[x, y - q], [x + q * 0.95, y + q * 0.62], [x - q * 0.95, y + q * 0.62]], True,
                fill=col, stroke=edge, width=0.5)


def legend_row(items, x, y, size=ANN, gap=12):
    """items: (label, col, kind) with kind line|dash|dot|o|s|^ ; left to right from x at baseline-middle y."""
    xx = x
    for lab, col, kind in items:
        if kind in ("line", "dash", "dot"):
            dash = {"line": None, "dash": [3, 2], "dot": [1, 1.6]}[kind]
            sx.poly([[xx, y], [xx + 14, y]], stroke=col, width=1.4 if kind == "line" else 1.0, dash=dash, fill=None)
            xx += 17
        elif kind == "band":
            sx.rect(xx, y - 3, 14, 6, fill=col, stroke=None, opacity=55)
            xx += 17
        else:
            marker(kind, xx + 4, y, col)
            xx += 10
        xx += sx.tex(lab, xx, y, size, INK, "left", valign="middle") + gap
    return xx - gap


def legend_col(items, x, y, size=ANN, gap=10.5):
    yy = y
    for it in items:
        legend_row([it], x, yy, size)
        yy += gap
    return yy


# sequential map for fields drawn in Splinx (heat maps, colour bars): indigo -> navy -> violet -> cyan -> mint
SEQ_H = [(0.0, INK), (0.28, NAVY), (0.55, VIOLET), (0.8, CYAN), (1.0, MINT)]
TEXTW = 468.0  # c2f text width in pt (6.5 in); draw every figure at this width (or a fraction) so fonts print at size
