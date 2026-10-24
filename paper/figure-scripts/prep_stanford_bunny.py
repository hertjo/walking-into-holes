"""Recompute the field data of the Stanford Bunny stress-test figure exactly as run_bunny_benchmark.py does after its
design loop: the repaired mesh (written to a scratch path, not into the repository), the 1,024 interior candidates,
their initial direct scores (field_agreement, seed+300) and the final central slice (seed+400) for the saved TWoS
centres. Checks the recomputed score/voxel agreement against summary.json. Needs libigl 2.6.3.
Writes data/stanford_bunny_fields.npz."""
import csv, json, os, sys, tempfile
from pathlib import Path
import numpy as np

REPO = os.environ.get("HOLES_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
B = Path(REPO) / "benchmarks/stanford_bunny"
sys.path.insert(0, str(B))
import run_bunny_benchmark as rb  # noqa: E402

summary = json.load(open(B / "output/summary.json"))
cfg = summary["configuration"]
config = rb.Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()
                      if k in rb.Config.__dataclass_fields__})
with tempfile.TemporaryDirectory() as tmp:
    mesh = rb.load_and_repair_mesh(B / "assets/bun_zipper.ply", Path(tmp) / "bunny.ply", config)
candidates = rb.sobol_interior(mesh, config.candidate_count, config.cavity_radius + config.outer_clearance, config.seed + 1)
corr, rrmse, scores = rb.field_agreement(candidates, mesh, config)
print("agreement", corr, rrmse, summary["initial_field_agreement"], flush=True)
rows = list(csv.DictReader(open(B / "output/selected-centers.csv")))
ours = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows if r["method"] == "TWoS + fresh validation"])
x, y, t = rb.evaluate_slice(mesh, ours, config)
out = Path(__file__).resolve().parent / "data/stanford_bunny_fields.npz"
np.savez_compressed(out, candidates=candidates, scores=scores, x=x, y=y, final_temp=t, centers=ours,
                    radius=config.cavity_radius, render_vertices=mesh.render_vertices, render_faces=mesh.render_faces,
                    n_vertices=len(mesh.vertices), n_faces=len(mesh.faces))
print(out, float(np.nanmax(t)), float(scores.max()))
