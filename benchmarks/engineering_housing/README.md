# Stylized engineering-housing benchmark

This benchmark generates a closed, manifold, license-free finned conduction
housing from an analytic signed-distance construction, extracts its triangle
surface with marching cubes, and measures a six-cavity design task. The state
solves

```text
-Laplacian(u) = f
```

with zero temperature on the exterior and inserted spherical-cavity
boundaries. The localized volumetric source models two chips and a regulator;
the compliance objective is the integral of `f*u` over the remaining material.

Five sequential-design variants use identical candidate and cavity budgets:

- sparse C++/Embree TWoS scouting followed by fresh shortlist validation;
- a near-matched-time TWoS budget with more scout and validation walks;
- tuned, matrix-free geometric multigrid on an embedded Cartesian domain;
- a sparse Cartesian finite-difference/CG adjoint field;
- uniform random feasible insertion.

All objective values are reevaluated with independent TWoS paths on fixed
Sobol probes. The two TWoS selectors and random placement use five complete
independent design runs; deterministic final designs use ten fresh objective
streams. The multigrid resolution is chosen by a recorded, predeclared sweep:
the smallest tested grid with relative RMSE at most 0.09 against a separate
512-walk field. Run with:

```bash
python benchmarks/engineering_housing/run_housing_benchmark.py
```

The generated mesh, every queried field, solver summary, trajectory and
design-replicate CSVs, multigrid tuning sweep, aggregate JSON, and vector/raster
publication figures are retained. This is a stylized engineering geometry,
not proprietary CAD; the multigrid reference is tuned and measured in house,
but it is not a production FEM, BEM, or GPU topology-optimization system.

## Connected cooling-channel benchmark

The companion experiment turns the topological field into one connected
channel rather than six disconnected cavities:

```bash
make engineering-channels
```

It scores 3,072 interior Sobol candidates, selects two separated high-value
terminals, and runs Dijkstra on a topology-weighted k-nearest-neighbor graph.
Seven controls parameterize a fixed 22-ball union. Differential WoS estimates
all 66 center-coordinate derivatives in one nested-walk pass (the executable
also reports 22 radius derivatives), and the dense gradients are mapped to
the controls by the exact interpolation transpose. Endpoints are fixed and
the proposed direction is projected to preserve centerline length to first
order.

Every tested shape is recalibrated to the target removed volume with a fixed
$2^{20}$-sample Sobol measure. Backtracking compares paired common-seed
objective estimates; Student-t upper bounds use a predeclared Bonferroni
allocation across all three scales and six rounds. Final method values use
separate streams. A high-sample audit compares the Differential WoS gradient
to central finite differences along six pre-seeded, feasible, high-signal
tangent directions. The output directory `output/channels/` retains the
controls, dense spheres, refinement statistics, derivative audit, every raw
field and solver summary, aggregate JSON, and vector/raster figure.
