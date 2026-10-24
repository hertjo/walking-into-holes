# 3D internal-cooling benchmark

This is the correctness-first application benchmark for **Walking into Holes**.
It solves

\[
-\Delta u=f,\qquad u=0\text{ on the outer box and cooling-cavity boundaries},
\]

with an affine positive heat source. The compliance objective is
\(J=\int_\Omega f u\,dx\), so the adjoint equals the state and the
capacity-normalized Dirichlet-cavity derivative is \(D_T=-u^2\).

The domain is represented only by analytic distance queries to a box and a
union of spheres. No grid is used by the TWoS method. A structured finite-
difference solve is included as a discretized adjoint/topological-derivative
reference; it is a pilot baseline, not a substitute for the final FEM/level-set
comparison.

## Run

From the paper root:

```bash
make cooling-reference
```

or directly:

```bash
python benchmarks/internal_cooling/run_reference_benchmark.py
```

The run is fixed to seed `20260912` and writes:

- `output/design-trajectories.csv`: independently evaluated compliance after
  every insertion;
- `output/selected-centers.csv`: ordered cavity centers for every method;
- `output/summary.json`: configuration, metrics, field agreement, timing, and
  dependency versions;
- `output/internal-cooling-benchmark.pdf`: paper-ready vector figure;
- `output/internal-cooling-benchmark.png`: inspection preview.

## Compared methods

1. **TWoS + fresh validation:** noisy global scout, cross-fitted \(u^2\), fresh
   high-sample validation of a shortlist, and selection by a lower confidence
   bound.
2. **TWoS + reused scout:** deliberately anti-conservative ablation that
   inserts the largest scouting estimate without fresh data.
3. **Voxel adjoint TD:** a `49 x 33 x 19` finite-difference reference that
   selects the largest interpolated \(u^2\).
4. **Random feasible:** equal-radius insertions subject to identical clearance
   constraints.

All final designs are reevaluated by fresh WoS paths on a common scrambled
Sobol probe set. The figure and summary explicitly label this as an analytic-
geometry reference benchmark. Timing claims are made only by the separate compiled C++ studies.

## Production implementation boundary

The intended final stack is:

- **Python 3.12:** configuration, seeds, statistics, baseline orchestration,
  CSV/JSON output, and plotting;
- **C++20 + Eigen:** batched walks, cross-fitting, cache ownership, and
  deterministic parallel reductions;
- **libigl:** mesh loading, repair checks, winding numbers, and reference
  closest-point operations;
- **Embree 4:** production BVH traversal and point/ray queries;
- **nanobind:** zero-copy NumPy-facing bindings;
- **Polyscope:** interactive debugging only; publication rendering remains a
  deterministic script.

Candidate point arrays and cavity geometry enter the future compiled kernel;
batched state, adjoint, walk-length, and diagnostic arrays return to Python.
