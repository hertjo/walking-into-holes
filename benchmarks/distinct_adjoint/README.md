# Distinct-adjoint regional cooling benchmark

The state solves `-Delta u=f` using the existing housing source. The adjoint
solves `-Delta p=q`, with a compact observation bump on a protected sensor ball.
The objective is `integral(q*u)/integral(q)`; its insertion ranking is `u*p`.
Holes cannot intersect the sensor ball or its 0.02 protective gap. All holes
have radius 0.095 and all designs contain six holes.

Build the independent executable (existing compliance binaries are preserved):

```bash
cmake -S benchmarks/distinct_adjoint -B /tmp/wih-adjoint-build \
  -DCMAKE_BUILD_TYPE=Release -Dembree_DIR=/path/to/embree/cmake
cmake --build /tmp/wih-adjoint-build --parallel 2
OPENBLAS_NUM_THREADS=1 python3 benchmarks/distinct_adjoint/run_benchmark.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/distinct_adjoint/audit_benchmark.py
cmake -S benchmarks/distinct_adjoint -B /tmp/wih-adjoint-server-build \
  -DCMAKE_BUILD_TYPE=Release -Dembree_DIR=/path/to/embree/cmake
cmake --build /tmp/wih-adjoint-server-build --parallel 2
OPENBLAS_NUM_THREADS=1 python3 benchmarks/distinct_adjoint/run_persistent.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/distinct_adjoint/plot_results.py
```

Use `WIH_ADJOINT_OUTPUT=/absolute/fresh/path` for an independent rerun and
`WIH_ADJOINT_EXE` for a different build location. Run performance stages
sequentially on an otherwise idle machine; do not parallelize competing methods.
The primary configuration and source/executable hashes are retained in output.
The persistent experiment uses `WIH_PERSISTENT_OUTPUT` for its separate
output directory and a separately built executable. It executes actual
adaptive designs, holding only the fixed outer-mesh structure across requests.
Each method/design gets a fresh server; startup and shutdown are timed.
All six selected centers must match the corresponding original run exactly
before an independently measured objective is reused. The before/after
comparison changes this backend overhead for both numerical methods.
Original executable-source snapshots are retained in `output/implementation`.
Python dependencies include NumPy, SciPy, pandas, Matplotlib, libigl, and SymPy;
the latter independently differentiates the manufactured counterexample.
The first complete seed was inspected as a pilot before the remaining four
predeclared seeds ran. No primary objective/budget was changed after that pilot.
The extra multigrid resolution sweep was specified after the pilot and is
reported as supplementary. The `setup-failure` directory records an early
empty-cavity-CSV interface error before any successful numerical measurements.

Primary methods: sparse TWoS (12 scout / 192 rescore paths), larger-budget TWoS
(48 / 768), stochastic hottest-point placement (12 / 192), deterministic
multigrid topological derivatives, deterministic multigrid hottest-point
placement, and random feasible placement. Every rescore uses 24 sites and a
fresh stream. Multigrid uses the same operator and geometry hierarchy for its
two sources; it is not charged for rebuilding between state and adjoint solves.

The core timing includes boundary-acceleration construction, source queries,
solves, and field extraction. Solver wall time additionally includes subprocess
startup, CSV I/O, and result loading. Driver time adds candidate filtering and
ranking. Common candidate-generation time is retained separately and added
when reporting the complete design-driver wall time. Independent numerical
reference evaluation is offline verification, separately timed and not silently
included in an algorithm speedup. This fixed-count experiment does not enable
finite-benefit certification. Never transfer the compliance-only square-energy
certificate to the q-weighted objective.

All comparisons use fresh WoS evaluation on four independently scrambled
128-point sensor-ball blocks with 256 walks per point, and a separate finer
multigrid solve. References are numerical approximations, not exact ground
truth. First-seed designs additionally receive a fourfold shell refinement and
a 225x145x113 grid check. The same evaluation probes and streams are shared
between methods to reduce comparison noise and are separate from selection.
Five candidate/design seeds are paired across methods; uncertainty across
complete designs is kept distinct from conditional probe-block uncertainty.

The benchmark proves neither global optimality nor manufacturing feasibility.
Both TWoS and deterministic TD exploit the same objective-aware coefficient.
An advantage over hottest-point ranking supports the value of the adjoint;
only matched resource comparisons distinguish Monte Carlo from volumetric TD.
