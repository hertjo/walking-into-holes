# Critical baseline experiments, 14 September 2026

This package measures the accelerated topological field and validates the
components needed for fair shape/connectivity comparisons. It does not claim
to reproduce a complete competing optimizer where only a component is run.

## Run

Install numpy, scipy, pandas, matplotlib, hilbertcurve, scikit-fem. For upstream
parity also install qmcpy and tqdm (CC=gcc CXX=g++ may be needed for qmctoolscl).
From the project root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/critical_baselines/sampling.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/critical_baselines/source_sampling.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/critical_baselines/shape_tracking.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/critical_baselines/connectivity.py
python3 benchmarks/critical_baselines/plot_results.py
python3 benchmarks/critical_baselines/summarize.py
```

The source and generated manifests fix parameters. Two-replication N=64 runs
were used only to validate execution, and are excluded from final statistics.
All 32 retained replications and all methods are retained without seed selection.

## What is measured

* Genuine Poisson compliance on the disk: -Delta u=1, zero Dirichlet data,
  u=p=(1-|x|^2)/4, and topological derivative -u^2. MC, Sobol RQMC, and
  Array-RQMC integrate each constant-source Green contribution exactly as
  r^2/4. Five sites, four independent batches, three path budgets and 32
  replications. Terminal-only neighbor reuse is deliberately excluded because
  it would not correctly account for the source term. The largest-budget
  ratio is summarized with a paired bootstrap over full replications.
* MC, scrambled-Sobol RQMC, shifted-Korobov Array-RQMC, and variance-weighted
  neighbor reuse, using matched harmonic boundary functions, 25 candidate sites,
  four independent batches, 64/256/1024 paths per batch, 32 replications, and
  shell tolerance 1e-7. Each method gets the same number of launched paths.
  Costs include RNG initialization, distance calls, sorting, and reweighting.
  Wall time is Python implementation time on this environment, not a tuned
  production solver comparison. Harmonic timings were collected in a shared
  runtime with other benchmark activity and are exploratory; path/query
  comparisons are unaffected. The final Poisson campaign was repeated with
  other benchmark jobs stopped, using exactly the same complete seed set;
  the earlier timing data are retained under output/exploratory-timing.
  The postprocessing kernel uses dimensionally
  normalized coordinates, which avoids the radius typo in the HTML rendering
  of the general-dimensional Poisson kernel. We use the main-text weights (18),
  not the stronger supplemental variance weights.
* Elliptic-domain Poisson tracking: exact state available to both optimizers;
  a two-parameter specialization of the probabilistic shape representation,
  uniform weighted starts instead of signed rejection sampling, compared with
  an independently derived deterministic pullback gradient. Same objective,
  bounds, step schedule and Armijo test. This isolates derivative sampling and
  descent, and intentionally does not measure PINN training or a boundary FEM
  Riesz map. Five stochastic designs and one deterministic trajectory.
* Andrade spectral operator (4): P1 triangular FEM with consistent mass,
  Dirichlet exterior boundary, k0=m0 in {1e-2,1e-4,1e-6}, 33/65/129 nodes per
  side. Five geometries distinguish connected cavities from absence of sealed
  void islands. This is an eigenproblem reproduction, not their full null-space
  level-set optimizer. Positive eigenvalues at finite contrast are not exact
  connectivity certificates; connectivity is separately checked geometrically.

## Provenance and fidelity

Array-RQMC reference: https://github.com/hoval58/RQMC-WoS at
`5b4909207386438ecadd891b5fd4f3122e9dd75d`, unit_disk_array_rqmc.py.
Our implementation uses the authors' tabulated 2D Korobov generators and
13-bit Hilbert sorting. `verify_upstream.py --upstream /path/to/RQMC-WoS`
executes their update routine with common random shifts and our boundary
function, comparing four cases to tolerance 1e-12. Upstream source is not
redistributed. Hilbert ordering is separately checked against hilbertcurve on
1,000 random points. Ordinary RQMC uses SciPy's scrambled Sobol implementation,
which is a documented implementation choice rather than bitwise QMCPy parity.

Neighbor reference: https://arxiv.org/abs/2404.17692v3, equations (17)--(18).
All paths are included; deterministic geometric weights preserve unbiasedness.
Within a full independent batch, paths are reused across all candidates.

Shape reference: https://arxiv.org/abs/2603.01141v1, equation (3.1) and
Remark 3.1. Author code: https://github.com/max-wuer/ProbabilisticShape at
`52a894588ba0235150b401f9cae97b0d2747347a`. The native application imports dolfin
and torch and uses a learned state, signed-start estimates, a boundary FEM
metric and mesh updates. None of those extra components is represented by an
oracle-state timing. A full matched application remains to be ported and run.

Connectivity reference: https://doi.org/10.1007/s00158-026-04327-5. Their
Remark 1 explicitly distinguishes void-island elimination from connectedness.
No original optimization code link was found on the article page. A complete
comparison requires matching external ports, physical PDE, volume constraint,
filters, null-space updates, and eigenvalue thresholds before timing.

## Statistical unit and limits

Off-diagonal cross-fitting is performed over four independent complete batches.
Applying the iid theorem to paths inside an RQMC array is invalid. The included
proof gives an exact biased antithetic example and a common-cache counterexample.
Bootstrap intervals across the 32 replications describe measured MSE, not
finite-insertion certification. Terminal projection introduces bounded bias.
No experimental error bar proves the asymptotic remainder used in certification.

The saved 3D channel designs pass strict consecutive-ball overlap checks. The
new geometry guard rejects gaps/tangencies after radius recalibration. These
designs remain interior cavities; they have no exterior fluid inlet or outlet.
