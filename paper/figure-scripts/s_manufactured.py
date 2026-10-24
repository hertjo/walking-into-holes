"""Manufactured harmonic fields on the unit disk: exact u, p, -up and the three product-estimator error maps."""
import numpy as np
from s_common import *

M = cload("experiments/output/manufactured_field.csv")
SUM = jload("experiments/output/summary.json")["spatial"]
W, H = TEXTW, 300


def grid(key):
    xs = np.array(sorted({round(float(r["x"]), 9) for r in M}))
    ys = np.array(sorted({round(float(r["y"]), 9) for r in M}))
    A = np.full((len(ys), len(xs)), np.nan)
    ix = {v: i for i, v in enumerate(xs)}
    iy = {v: i for i, v in enumerate(ys)}
    for r in M:
        A[iy[round(float(r["y"]), 9)], ix[round(float(r["x"]), 9)]] = float(r[key])
    return A


def nice(lo, hi, n=5):
    from splinx.plot.charts import nice_ticks
    return [float(v) for v, _ in nice_ticks(lo, hi)]


def draw():
    errs = np.concatenate([np.abs([float(r[k]) for r in M]) for k in
                           ("same_seed_error", "product_means_error", "crossfit_error")])
    elim = float(np.quantile(errs, 0.985))
    maps = [("true_u", "a", "Harmonic field $u$", None), ("true_p", "b", "Harmonic field $p$", None),
            ("true_td", "c", "Exact product $-up$", None),
            ("same_seed_error", "d", "Same-seed product error", SUM["same_seed_rmse"]),
            ("product_means_error", "e", "Product-of-means error", SUM["product_means_rmse"]),
            ("crossfit_error", "f", "Cross-fit TWoS error", SUM["crossfit_rmse"])]
    D = 112.0  # drawn diameter of the unit disk (pt)
    for k, (key, letter, title, rmse) in enumerate(maps):
        col, row = k % 3, k // 3
        x0, y0 = 10 + col * 156, 22 + row * 146
        A = grid(key)
        fin = A[np.isfinite(A)]
        if k >= 3:
            lo, hi, stops = -elim, elim, DIVH
        elif k == 2:
            lim = float(np.abs(fin).max()); lo, hi, stops = -lim, lim, DIVH
        else:
            lo, hi, stops = float(fin.min()), float(fin.max()), FIELD
        cx, cy = x0 + D / 2, y0 + D / 2
        s = 0.88 * D / 2
        img = rgba_field(np.nan_to_num(A[::-1], nan=lo), lo, hi, np.where(np.isfinite(A[::-1]), 255, 0), stops)
        sx.image(img, cx - s, cy - s, 2 * s, 2 * s, smooth=True)
        sx.circle(cx, cy, D / 2, fill=None, stroke=INK, width=0.9)
        sx.circle(cx, cy, s, fill=None, stroke=MUTED, width=0.45, dash=[2, 2])
        panel(letter, title, x0, y0 - 9)
        tk = [v for v in nice(lo, hi) if lo <= v <= hi]
        if k >= 3:
            tk = [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]
        vbar(x0 + D + 8, y0 + 4, 5, D - 8, lo, hi, [(v, f"${v:g}$") for v in tk], stops)
        if rmse is not None:
            note(f"RMSE ${rmse:.3f}$", x0 + 1, y0 + D - 1, size=ANN, col=INK)
    finish()


sx.single("manufactured-field", W, H, draw)
