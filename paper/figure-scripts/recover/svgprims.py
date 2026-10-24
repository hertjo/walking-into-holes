"""Primitive extraction from a cairo SVG export (pdftocairo -svg) of an old vector figure: every <path> (also those
reached through <use>) as absolute page-coordinate polylines with its stroke and fill colours. Used only to read back
plotted values of figures whose run folders are not tracked in the repository."""
import re
import xml.etree.ElementTree as ET
import numpy as np

NS = {"s": "http://www.w3.org/2000/svg", "x": "http://www.w3.org/1999/xlink"}
XL = "{http://www.w3.org/1999/xlink}href"


def mat(t):
    M = np.eye(3)
    for kind, args in re.findall(r"(\w+)\(([^)]*)\)", t or ""):
        a = [float(v) for v in re.split(r"[ ,]+", args.strip()) if v]
        if kind == "matrix":
            N = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif kind == "translate":
            N = np.array([[1, 0, a[0]], [0, 1, a[1] if len(a) > 1 else 0], [0, 0, 1]])
        elif kind == "scale":
            N = np.diag([a[0], a[1] if len(a) > 1 else a[0], 1])
        else:
            continue
        M = M @ N
    return M


def parse_d(d):
    toks = re.findall(r"[MLCZHVmlczhv]|-?[\d.]+(?:e-?\d+)?", d)
    subs, cur, i, cmd, pos = [], [], 0, None, np.zeros(2)
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t; i += 1
            if cmd in "Zz":
                if cur:
                    cur.append(cur[0]); subs.append(cur); cur = []
            continue
        if cmd == "M":
            if cur:
                subs.append(cur)
            pos = np.array([float(toks[i]), float(toks[i + 1])]); cur = [pos]; i += 2; cmd = "L"
        elif cmd == "L":
            pos = np.array([float(toks[i]), float(toks[i + 1])]); cur.append(pos); i += 2
        elif cmd == "C":
            pos = np.array([float(toks[i + 4]), float(toks[i + 5])]); cur.append(pos); i += 6
        elif cmd == "H":
            pos = np.array([float(toks[i]), pos[1]]); cur.append(pos); i += 1
        elif cmd == "V":
            pos = np.array([pos[0], float(toks[i])]); cur.append(pos); i += 1
        else:
            i += 1
    if cur:
        subs.append(cur)
    return [np.array(s) for s in subs]


def rgb(s):
    if not s or s == "none":
        return None
    m = re.match(r"rgb\(([\d.]+)%,\s*([\d.]+)%,\s*([\d.]+)%\)", s)
    if not m:
        return s
    return "#%02x%02x%02x" % tuple(int(round(float(v) * 2.55)) for v in m.groups())


def prims(path):
    tree = ET.parse(path)
    root = tree.getroot()
    defs = {}
    for el in root.iter():
        if el.get("id"):
            defs[el.get("id")] = el
    out = []

    def walk(el, M, inherited):
        tag = el.tag.split("}")[-1]
        M2 = M @ mat(el.get("transform"))
        st = dict(inherited)
        for k in ("stroke", "fill"):
            if el.get(k) is not None:
                st[k] = el.get(k)
        if el.get("style"):
            for kv in el.get("style").split(";"):
                if ":" in kv:
                    k, v = kv.split(":", 1)
                    if k.strip() in ("stroke", "fill"):
                        st[k.strip()] = v.strip()
        if tag in ("defs", "clipPath", "mask", "symbol") and el is not root:
            return
        if tag == "path" and el.get("d"):
            for sp in parse_d(el.get("d")):
                P = (M2 @ np.c_[sp, np.ones(len(sp))].T).T[:, :2]
                out.append({"stroke": rgb(st.get("stroke")), "fill": rgb(st.get("fill")), "pts": P})
        elif tag == "use":
            ref = defs.get((el.get(XL) or el.get("href") or "").lstrip("#"))
            if ref is not None:
                T = M2 @ mat(f"translate({el.get('x', 0)},{el.get('y', 0)})")
                for ch in ([ref] if ref.tag.split('}')[-1] == "path" else list(ref)):
                    walk(ch, T, st)
        for ch in list(el):
            walk(ch, M2, st)

    walk(root, np.eye(3), {})
    return out


def inside(p, box):
    x0, y0, x1, y1 = box
    c = p["pts"].mean(0)
    return x0 <= c[0] <= x1 and y0 <= c[1] <= y1
