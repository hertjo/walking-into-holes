"""Shared helpers for the supplement figures (on top of hstyle)."""
import os, json, csv
from hstyle import *
from splinx import draw as sx

REPO = os.environ.get("HOLES_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sx.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def R(*p):
    return os.path.join(REPO, *p)


def jload(*p):
    return json.load(open(R(*p)))


def cload(*p):
    with open(R(*p)) as f:
        return list(csv.DictReader(f))


def panel(letter, text, x, y, size=LAB):
    """Beckmann-style panel label '(a) text', left aligned at baseline y."""
    text = text[:1].upper() + text[1:] if text and text[0].isalpha() else text
    s = f"({letter}) {text}" if letter else text
    return sx.tex(s, x, y, size, INK, "left")


def line_mk(P, xs, ys, col, kind="o", w=1.3, dash=None, r=2.4, open_=False):
    P.line(xs, ys, col=col, w=w, dash=dash)
    for a, b in zip(xs, ys):
        if open_:
            marker(kind, float(P.X(a)), float(P.Y(b)), "#ffffff", r=r, edge=col)
        else:
            marker(kind, float(P.X(a)), float(P.Y(b)), col, r=r)


def finish():
    for msg in sx.check():
        print(msg)


def legend_box(items, x, y, size=ANN, gap=9.0, pad=2.0):
    """A column legend on a white backing (so grid lines don't run through it); y is the first row's middle."""
    from splinx.draw import texw
    w = max((17 if k in ("line", "dash", "dot", "band") else 10) + texw(l, size) for l, _, k in items)
    sx.rect(x - pad, y - gap / 2 - pad + 0.5, w + 2 * pad, gap * len(items) + 2 * pad - 1, fill="#ffffff", stroke=None)
    return legend_col(items, x, y, size, gap)


def note(s, x, y, size=ANN, col=None, anchor="left"):
    return sx.tex(s, x, y, size, col or MUTED, anchor, bg="#ffffff")


def legend_grid(items, x, y, cols=2, colw=None, size=ANN, gap=9.0):
    """Legend items in a grid of ``cols`` columns starting at (x, y) (first row's middle)."""
    from splinx.draw import texw
    if colw is None:
        colw = max((17 if k in ("line", "dash", "dot", "band") else 10) + texw(l, size) for l, _, k in items) + 8
    for i, it in enumerate(items):
        legend_row([it], x + (i % cols) * colw, y + (i // cols) * gap, size)


# light-to-dark field map for scalar fields on a white page (high values dark, so cyan/mint marks stay visible)
FIELD = [(0.0, "#f7f6ff"), (0.25, LILAC), (0.55, VIOLET), (0.8, NAVY), (1.0, INK)]


def rgba_field(vals, lo, hi, alpha=None, stops=FIELD):
    rgb = sx.cmap(vals, lo, hi, stops)
    a = np.full(rgb.shape[:2], 255, np.uint8) if alpha is None else np.asarray(alpha, np.uint8)
    return np.dstack([rgb, a])


def hbar(x, y, w, h, lo, hi, ticks, stops=FIELD, label=None, size=TICK - 0.6):
    """Horizontal colour bar with ticks [(value, label)] below and an optional label to the right."""
    t = np.linspace(lo, hi, 256)[None, :].repeat(2, 0)
    sx.image(sx.cmap(t, lo, hi, stops), x, y, w, h, smooth=True)
    sx.rect(x, y, w, h, fill=None, stroke=INK, width=0.4)
    for v, lab in ticks:
        xx = x + (v - lo) / (hi - lo) * w
        sx.line(xx, y + h, xx, y + h + 2, stroke=INK, width=0.4)
        sx.tex(lab, xx, y + h + 3.2, size, INK, "center", valign="top")
    if label:
        sx.tex(label, x + w + 3, y + h / 2, size, INK, "left", valign="middle")


# diverging map: negative navy, zero near-white, positive mint
DIVH = [(0.0, NAVY), (0.25, VIOLET), (0.5, "#f7f6ff"), (0.75, CYAN), (1.0, MINT)]


def vbar(x, y, w, h, lo, hi, ticks, stops=FIELD, size=TICK - 0.8):
    """Vertical colour bar (low at the bottom) with tick labels on the right."""
    t = np.linspace(hi, lo, 256)[:, None].repeat(2, 1)
    sx.image(sx.cmap(t, lo, hi, stops), x, y, w, h, smooth=True)
    sx.rect(x, y, w, h, fill=None, stroke=INK, width=0.4)
    for v, lab in ticks:
        yy = y + (hi - v) / (hi - lo) * h
        sx.line(x + w, yy, x + w + 2, yy, stroke=INK, width=0.4)
        sx.tex(lab, x + w + 3.2, yy, size, INK, "left", valign="middle")
