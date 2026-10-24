"""External FEM audit inputs. The finest scikit-fem solve (81x51x43 background) is deterministic, so it is rerun
exactly as run_external_fem_baseline.py does (its output folder is not tracked): candidate scores, the central-slice
u_h^2 field and the cutaway boundary facets. The Pearson correlation with the 512-walk TWoS field is checked against
the value in the paper (0.9943). Panel (d) needs the measured times of all three resolutions, which only survive in
the earlier vector figure, so its three markers are read back from that figure (page 59 of the anonymous supplement).
Needs scikit-fem 11.0.0. Writes data/external_fem.npz and data/external_fem_d.json."""
import json, os, subprocess, sys
import numpy as np

REPO = os.environ.get("HOLES_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{REPO}/benchmarks/engineering_housing")
import types
sys.modules.setdefault("igl", types.ModuleType("igl"))
import run_external_fem_baseline as fem  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

housing = fem.load_housing_module()
src = fem.HERE / "output"
cand = fem.read_points(src / "points-reference-score.csv")
twos = fem.read_scores(src / "field-reference-score.csv")
mesh, basis, state, scores, rec = fem.solve_fem(housing, (81, 51, 43), cand)
r = float(pearsonr(scores, twos).statistic)
print("vertices", rec["vertices"], "tets", rec["tetrahedra"], "pearson", r, flush=True)
x = np.linspace(-1.08, 1.08, 220); z = np.linspace(-0.30, 0.34, 110)
xx, zz = np.meshgrid(x, z)
pts = np.column_stack((xx.ravel(), np.zeros(xx.size), zz.ravel()))
inside = housing.housing_sdf(pts) < -0.025
vals = np.full(len(pts), np.nan)
vals[inside] = np.square(fem.interpolate_in_chunks(basis, state, pts[inside], chunk_size=96))
tri = fem.cutaway_facets(mesh)
np.savez_compressed(f"{HERE}/data/external_fem.npz", candidates=cand, fem=scores, twos=twos, x=x, z=z,
                    slice=vals.reshape(xx.shape), facets=tri, vertices=rec["vertices"], tets=rec["tetrahedra"])

# panel (d) read-back from an earlier build of the paper (not distributed); its result is data/external_fem_d_raw.json
OLD = f"{REPO}/paper/Walking_into_Holes_SIGGRAPH_Supplement.pdf"
if not os.path.exists(OLD):
    sys.exit(0)
sys.path.insert(0, f"{HERE}/recover")
from svgprims import prims, inside as within  # noqa: E402
svg = "/tmp/external_fem_p59.svg"
subprocess.run(["pdftocairo", "-svg", "-f", "59", "-l", "59", OLD, svg],
               check=True)
P = prims(svg)
print(sorted({(p["stroke"], p["fill"]) for p in P if p["fill"] in ("#2563a6", "#e56b2f") or p["stroke"] in ("#2563a6",)}))
marks = [p for p in P if p["fill"] in ("#2563a6", "#e56b2f") and len(p["pts"]) >= 4]
out = [{"fill": p["fill"], "centre": p["pts"][:-1].mean(0).tolist(), "size": float(np.ptp(p["pts"][:, 0]))} for p in marks]
ticks = [p["pts"].tolist() for p in P if p["stroke"] == "#000000" and len(p["pts"]) == 2
         and np.hypot(*(p["pts"][1] - p["pts"][0])) < 5 and within(p, (300, 250, 600, 400))]
json.dump({"markers": out, "ticks": ticks}, open(f"{HERE}/data/external_fem_d_raw.json", "w"), indent=1)
print(json.dumps(out, indent=1))
