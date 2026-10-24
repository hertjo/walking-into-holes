"""Recover numbers that only survive inside the old matplotlib PDFs (their run folders are not in the repository).

Needs PyMuPDF, NumPy, Pillow and Matplotlib (for the colour maps used by the old figures). Values are read back
exactly from what the old figures drew: rasterised fields are inverted through their colour map, vector paths are
mapped through the axis tick marks.
"""
import io
import numpy as np
import pymupdf
from PIL import Image
from matplotlib import colormaps


def page(path, k=0):
    return pymupdf.open(path)[k]


def images(pg):
    """[(xref, bbox, rgb uint8 HxWx3, alpha HxW or None)] in drawing order."""
    out = []
    doc = pg.parent
    for info in pg.get_image_info(xrefs=True):
        x = info["xref"]
        if not x:
            continue
        d = doc.extract_image(x)
        im = Image.open(io.BytesIO(d["image"])).convert("RGB")
        rgb = np.asarray(im)
        alpha = None
        if d.get("smask"):
            sm = doc.extract_image(d["smask"])
            alpha = np.asarray(Image.open(io.BytesIO(sm["image"])).convert("L"))
        out.append((x, tuple(info["bbox"]), rgb, alpha))
    return out


def invert_cmap(rgb, name):
    """RGB pixels -> position t in [0, 1] along the named colour map (nearest of 1024 samples), and the RGB error."""
    lut = (colormaps[name](np.linspace(0, 1, 1024))[:, :3] * 255).astype(np.float32)
    px = rgb.reshape(-1, 3).astype(np.float32)
    t = np.empty(len(px), np.float32)
    err = np.empty(len(px), np.float32)
    for i in range(0, len(px), 4096):
        d = ((px[i:i + 4096, None, :] - lut[None]) ** 2).sum(-1)
        j = d.argmin(1)
        t[i:i + 4096] = j / 1023.0
        err[i:i + 4096] = np.sqrt(d[np.arange(len(j)), j])
    return t.reshape(rgb.shape[:2]), err.reshape(rgb.shape[:2])


class Axis:
    """Affine map page pt -> data from two (pt, value) pairs per axis."""

    def __init__(self, xa, xb, ya, yb):
        """xa, xb: (pt, value) on x; ya, yb: (pt, value) on y."""
        (self.px0, self.x0), (self.px1, self.x1) = xa, xb
        (self.py0, self.y0), (self.py1, self.y1) = ya, yb

    def x(self, p):
        return self.x0 + (np.asarray(p, float) - self.px0) / (self.px1 - self.px0) * (self.x1 - self.x0)

    def y(self, p):
        return self.y0 + (np.asarray(p, float) - self.py0) / (self.py1 - self.py0) * (self.y1 - self.y0)


def drawings(pg, rect=None):
    out = []
    for d in pg.get_drawings():
        r = d["rect"]
        if rect is None or (r.x0 >= rect[0] - 0.5 and r.y0 >= rect[1] - 0.5 and r.x1 <= rect[2] + 0.5
                            and r.y1 <= rect[3] + 0.5):
            out.append(d)
    return out


def path_points(d):
    """Polyline vertices of a drawing (line ends and curve ends)."""
    pts = []
    for it in d["items"]:
        if it[0] == "l":
            for p in it[1:3]:
                pts.append((p.x, p.y))
        elif it[0] == "c":
            for p in (it[1], it[4]):
                pts.append((p.x, p.y))
        elif it[0] == "re":
            r = it[1]
            pts += [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
    # drop consecutive duplicates
    out = [pts[0]] if pts else []
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) > 1e-6 or abs(p[1] - out[-1][1]) > 1e-6:
            out.append(p)
    return np.array(out)


def ticks(pg, rect, vertical):
    """Tick-mark positions (short black lines) inside rect: x of vertical ticks or y of horizontal ones."""
    out = []
    for d in drawings(pg, rect):
        r = d["rect"]
        if len(d["items"]) == 1 and d["items"][0][0] == "l":
            if vertical and abs(r.x1 - r.x0) < 0.05 and 2 < r.y1 - r.y0 < 5:
                out.append(r.x0)
            if not vertical and abs(r.y1 - r.y0) < 0.05 and 2 < r.x1 - r.x0 < 5:
                out.append(r.y0)
    return sorted(set(round(v, 3) for v in out))
