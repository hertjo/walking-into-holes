# Submission-strengthening campaign

This is a new, isolated campaign. Historical experiments are not overwritten.
The protocol, executable hash, requests, selected designs, raw fields, independent
validation blocks, and inclusive design times are saved under `output/`.
`pilot/` timings are development diagnostics and must not enter reported tables.

## Four questions

1. On actual three-dimensional meshes, how do the variance, numerical bias,
   ranking, and cost of direct capacity-normalized topological scores compare
   with common-random-number finite-cavity differencing and multigrid?
2. Does nucleation improve an entire shape-design pipeline when all
   initializations can subsequently use Differential WoS? Compare random,
   deterministic hottest-point, deterministic TD, and TWoS initialization.
   Give hot/random 16 refinement rounds versus 8 for TD initializations, and
   evaluate checkpoints at 0, 4, 8, and (where available) 16 rounds.
3. Do channel-placement findings survive matched removed volume and genuinely
   fixed heat input? Protect the complete supports of compact heat sources,
   calibrate a common channel radius against a shared volume sample, and audit
   volume independently with four independent scrambled Sobol samples.
4. Do the findings transfer to distinct geometry? Use a finned housing,
   bifurcating manifold with thin ribs and a return loop, and an annular spreader
   with a narrow bridge and closely spaced vertical ribs. All are actual closed
   triangle meshes, not illustrated proxy shapes.

## PDE and exact normalization

Solve `-Delta u = f` with zero Dirichlet values on the exterior and cooling voids.
Every heat component is `A (1-|x-c|^2/a^2)_+^2`, with total power
`32 pi A a^3 / 105`. The entire support remains in the material. The objective
is `J = integral q u`, where the sensor bump `q` has unit integral. Thus the
positive small-ball benefit per capacity is `u(z) p(z)`, with `-Delta p=q`.
A ball's capacity is `4 pi r`. Cooling walls remain ideal isothermal boundaries;
there is no fluid simulation or claim about pumping/manufacturability.

## Attribution and comparison scope

The Differential WoS component is our specialized implementation of Miller,
Sawhney, Crane and Gkioulekas, SIGGRAPH 2024, Algorithm 2 / Equation 13:
https://arxiv.org/html/2405.12964v4

Cross-checked against the authors' code at commit
`f2b10ebd44904b9c81c063d80e244f0d26868f33`, notably
`zombie/include/zombie/point_estimation/walk_on_stars.h`, terminal contribution
and recursive boundary data. This campaign does not run the authors' full
application optimizer. Fixed-radius spherical translations specialize their
boundary velocity; the regional objective, initialization, backtracking,
clearance constraints, and candidate pool are local benchmark choices.
Differential WoS is not generally restricted to topology-preserving changes.
Here its fixed six-cavity parameterization lets us isolate initialization.
The off-diagonal product estimator is established prior methodology, not itself
our novelty. The finite-cavity baseline is a local common-random-number coupling,
not a claim to the best possible coupling. Multigrid is a local embedded Cartesian
finite-difference solver, not a paper-specific FEM implementation.

## Fairness and limitations

- Identical meshes, heat, sensors, candidate seeds, cavity constraints and final
  evaluation blocks. Six disjoint equal-radius spheres have exactly equal volume.
- Complete design times include outer Embree construction, per-call I/O, adaptive
  decisions, and refinement. Input mesh generation/loading into the shared Python
  geometry workflow and independent evaluation are separate. Channel calibration
  point generation is shared preprocessing; radius calibration is timed.
- Solver memory is the backend process high-water RSS, not total driver memory.
  Distance queries and parity rays are instrumented separately; walk steps are
  not relabeled as geometry queries. Python candidate/volume geometry queries are
  excluded from backend counters but their design-time costs are included.
- All measured runs execute sequentially with four OpenMP threads. Numerical
  residual convergence is not continuum accuracy; grid-resolution checks and
  independent Monte Carlo evaluations remain separate.
- The channel heuristic queries a point on each candidate axis. It is not a
  derivative or certificate for the whole finite channel. Common-radius
  recalibration may modify previously inserted channels.
- A finite-radius FD score and an infinitesimal TD have different expectations.
  Variance about each method's own mean is reported separately from error
  against the high-budget infinitesimal reference. Small-hole grid differences
  below grid resolution are explicitly marked unresolved.
- Fresh line-search samples reduce reuse, but acceptance is heuristic. Neither
  the optimization runs nor plots imply application-level certification.

## Run

Build `study.cpp` with CMake and Embree 4; set `WIH_STUDY_BINARY` if needed.
Run `geometry.py`, `check_gradient.py` (after a pilot initialization), then
`run_study.py --stage all`. Resume skips completed designs/studies and preserves
original measured times; unfinished calls are repeated. Run final auditing and
rendering only after measured runs finish, avoiding CPU contention.

## Uncertainty visualization

`uncertainty.py` uses 4,000 paired percentile bootstrap replicates. Optimization
ribbons describe pointwise 95% intervals for mean cooling, resampling five design
seeds and four shared independent evaluation blocks. Individual runs remain
visible. Isolated channel results use vertical bars; horizontal bars describe
uncertainty in mean complete-design runtime. Curves interpolate checkpoints,
not continuous-time optimization measurements.

Estimator variance ribbons bootstrap sixteen eight-site repeat vectors and
sixteen independent reference batches. They quantify uncertainty in the measured
variance, not the spread of a single estimator output. RMS-error bars include
reference sampling uncertainty. Deterministic MG has no Monte Carlo field variance; its RMS-error bar reflects
the sampled reference only. Five additional timing repeats per grid, with
rotated resolution order and a retained outer mesh, support runtime bars.
Each repetition rebuilds the hierarchy and must reproduce the original field. These small-sample intervals are pointwise, approximate,
and do not cover numerical bias or provide finite-insertion certification.

From the project root, `make submission-study PYTHON=python3` builds all three
backends, resumes the measurements, audits the outputs, and composes all four
figures. Set `EMBREE_DIR` to the installed Embree 4 CMake package directory when
it is not automatically found. The build requires CMake, a C++17 compiler,
OpenMP, and Embree 4. Python dependencies include NumPy, SciPy, pandas, trimesh,
libigl, scikit-image, Embreex, Matplotlib, Pillow, Mitsuba, Dr.Jit, manifold3d,
and the Open Image Denoise binding used by the existing hero renderer.
`make submission-study-figures PYTHON=python3` recreates the figure composition
from completed data; its renderer also retains reusable physical field caches.

An additional nearest-sensor control chooses the closest feasible candidate to
the sensor center without a PDE solve. It shares the exact candidate pools,
clearances, volume calibration, and eight-round refinement schedule of TD
initializations. This was added during the campaign to test whether simple
proximity explains performance on the localized observation task. It is run by
`run_coarse.py` alongside the cheaper MG49 baseline, with identical independent
evaluation and uncertainty calculations.

Independent references receive an additional check when a first-seed case has
more than 1 percentage point change from grid 97 to 129, or more than 2 points
of MC/grid129 disagreement for any method. All six methods in a triggered case
are then evaluated on grids 193 and 257. This is an accuracy diagnostic outside
all reported design times; neither reference is declared exact ground truth.
The cutaway field renderer uses longest-axis grid count 193.

`python3 benchmarks/submission_study/resume_complete.py` is the resumable serial
entry point for interrupted long runs. It records the active stage and logs
under `output/`; completed design JSON files preserve their original measured
costs. An incomplete design is rerun in full. Successful numerical completion
still requires inspecting the resulting figures and integrating the findings
into the manuscript.

### Sparse-query scheduling audit

The eight-site direct-score timing is replayed with OpenMP chunk size one rather than four to expose all four configured workers. `run_sparse_timings.py` checks all replayed u, p, score, and same-index fields bitwise against the original samples. Query plots use these timing repeats; original raw times and all complete-design campaign times are retained unchanged. This is a scheduling correction made after inspecting the initial results, not a change to the estimator or sampling budgets.
