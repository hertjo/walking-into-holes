# Compiled Stanford Bunny scaling benchmark

This experiment replaces the Python geometry/walk kernel with a deterministic
C++17 implementation backed by Embree 4 point queries and OpenMP. It measures
the initial topological-derivative field at common interior sites and compares
accuracy, wall time, process memory, and surface-complexity scaling against a
matched C++ Cartesian finite-difference/conjugate-gradient reference. Both
methods use the same Embree mesh scene and CPU-thread cap.

The study is deliberately described as a **CPU reference benchmark**. The
voxel code is useful for correctness and scaling diagnostics and the shared
executable also contains the matrix-free geometric-multigrid baseline used by
the housing study. Neither is a substitute for a production FEM, BEM, or GPU
comparison or by itself supports a state-of-the-art speed claim.

## Dependencies

```bash
python -m pip install embreex==4.4.0 cmake==4.1.0 libigl==2.6.3 \
  numpy scipy matplotlib
```

The runner discovers the Embree CMake package installed by `embreex`. A system
Embree installation can instead be selected with `EMBREE_DIR`.

## Reproduce

From the paper root:

```bash
make compiled-bunny
```

The seeded run builds the executable, validates Embree closest-triangle
distances against libigl, computes a 2,048-walk independent reference field,
sweeps nine TWoS path budgets, eight voxel resolutions, nine output counts,
and four surface resolutions. It emits raw fields and summaries as well as
the main `output/compiled-bunny-scaling.pdf`, a supplemental surface-scaling
PDF, and high-resolution PNG versions of both.

The C++ executable can also be used directly:

```bash
/tmp/walking-into-holes-embree-build/wih_twos \
  --mesh benchmarks/stanford_bunny/output/stanford-bunny-watertight.ply \
  --points benchmarks/compiled_bunny/output/evaluation-points.csv \
  --output /tmp/field.csv --summary /tmp/summary.json \
  --walks 32 --repeats 3 --threads 8
```

The executable accepts ASCII triangular PLY meshes. Evaluation points must be
known interior points with one `x,y,z` triple per CSV row.

For the embedded-domain geometric-multigrid solver, pass `--mode mg` and an odd
grid shape such as `--grid 113,73,57`. The solver uses full weighting,
trilinear prolongation, weighted-Jacobi smoothing, and a residual-minimizing
coarse correction; tolerances and smoothing counts are exposed as `--mg-*`
flags and recorded in each summary JSON.

For fixed-radius cavity-center and radius derivatives, pass `--mode diffwos`,
`--cavities`, `--cavity-radius`, and `--diff-offset`. Each primary harmonic
walk terminates on the current boundary. On an internal Dirichlet sphere it
launches a nested Poisson WoS estimate at the inward normal offset and writes
the Differential WoS estimators `dc<i>_{x,y,z}` and `dR<i>`, together with
per-coordinate standard errors. Overlapping spheres are interpreted as one
ball-union cavity; only exposed boundary patches receive exit probability.
The engineering channel runner maps these dense center gradients to its seven
control points.

The stream key hashes the base seed, repeat, site, and sample index separately.
This prevents adjacent evaluation seeds from merely permuting the same finite
sample-key set, which is essential for valid fresh-seed uncertainty estimates.
