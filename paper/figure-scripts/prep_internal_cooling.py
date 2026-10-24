"""Recompute the two plane slices of the analytic-geometry cooling benchmark (initial and final temperature, initial
score) exactly as run_reference_benchmark.py does after its design loop: same seeds (seed+200, seed+201), the saved
TWoS centres (output/selected-centers.csv) and the recorded configuration. Writes data/internal_cooling_slices.npz."""
import csv, json, os, sys
import numpy as np

REPO = os.environ.get("HOLES_REPO", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
B = os.path.join(REPO, "benchmarks/internal_cooling")
sys.path.insert(0, B)
import run_reference_benchmark as rb  # noqa: E402

cfg = json.load(open(os.path.join(B, "output/summary.json")))["configuration"]
config = rb.Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items() if k in rb.Config.__dataclass_fields__})
assert config == rb.Config(**{k: getattr(config, k) for k in rb.Config.__dataclass_fields__})
rows = list(csv.DictReader(open(os.path.join(B, "output/selected-centers.csv"))))
ours = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows if r["method"] == "TWoS + fresh validation"])
x, y, t0, s0 = rb.evaluate_slice(np.empty((0, 3)), config, config.seed + 200)
_, _, t1, _ = rb.evaluate_slice(ours, config, config.seed + 201)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/internal_cooling_slices.npz")
np.savez_compressed(out, x=x, y=y, initial_temp=t0, initial_score=s0, final_temp=t1, centers=ours,
                    radius=config.cavity_radius, half_extents=np.array(config.half_extents))
print(out, float(np.nanmax(t0)), float(np.nanmax(s0)), float(np.nanmax(t1)))
