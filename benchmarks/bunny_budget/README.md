# Equal-volume cavity allocation on the Stanford Bunny

This exploratory study answers whether more cavities help when **removed
volume is held fixed**. It does not claim a physically optimal cavity count.

## Protocol

- Same repaired 69,674-triangle Bunny, height 1, as the original experiment.
- `-Delta u = 1 + 0.65*(y+0.5)`, with **ideal zero-temperature exterior and
  cavity walls**. No fluid flow, cooling-access cost, or mechanical model.
- Two removed volumes: `3*(4*pi/3)*0.05^3` and `6*(4*pi/3)*0.05^3`.
- Counts 2, 4, 6, 12. All balls within a design have the common radius
  `0.05*(volume_units/count)^(1/3)`. Radii are set by the volume constraint,
  not optimized after insertion. The balls are disjoint, so volumes are exact.
- Three independent design seeds for each allocation; 1,024 common candidate
  points for the three methods within each seed/allocation.
- TWoS: 12-path scouting, top-20 shortlist, 128 independent paths for rescoring,
  maximizing `score - 1.96*score_se`. This ranking heuristic is **not** a
  post-selection finite-insertion confidence certificate.
- Voxel-adjoint TD: compiled FD/CG on a 96x112x84 grid, rescored after every
  insertion. This is finer than the original Python Bunny design baseline.
- Random: uniform choice among remaining feasible candidates at each step.
- Minimum outer wall 0.012; minimum cavity-to-cavity gap 0.014. Each design
  starts from the unperforated Bunny and relocates its centers from scratch.

## Independent objective evaluation

Four independently scrambled Sobol interior blocks of 512 points each, with
128 independent WoS paths per active point. All designs share those blocks
and base walk seeds; evaluation randomness is separate from selection.
Points in removed material contribute exactly zero. The objective is the
original-volume-normalized integral of `f*u`, not the mean over surviving
material. Cooling percentages average the four paired block ratios to the
unperforated evaluation. Initial and perforated solves both use epsilon
0.0005 and a 512-step cap; cap hits cause failure.

The shaded plot ranges show minimum/maximum across three complete design
runs, **not confidence intervals**. The separately retained block standard
error describes evaluation uncertainty conditional on those selected designs
and omits termination bias. Blocks are shared between designs; do not treat
all design/block combinations as independent observations.

The count-cost plot minimizes the mean measured `J/J0 + kappa*n` among the
tested full-budget allocations and an unperforated option. Kappa is an
**assumed**, dimensionless per-cavity cost, not an estimated manufacturing or
operating cost. Exact intersections of these affine costs define the finite
envelope. Three seeds and four evaluation blocks support an exploratory
tradeoff, not a global optimum or precise switching thresholds.

## Interpretation

At equal volume, total cooling area grows as `n^(1/3)`, and the sum of
isolated ball capacities grows as `n^(2/3)`. That sum is not the capacity of
their union; it is not used as an exact prediction. The paper appendix proves
these geometry identities and monotonicity for *nested* ideal cooled
removals. Different equal-volume allocations are not nested. A volume-only
penalty is constant within each sweep and cannot express a preferred count.

## Reproduce

From the project root, with the existing C++/Embree dependencies installed:

```bash
OPENBLAS_NUM_THREADS=1 python3 benchmarks/bunny_budget/run_study.py --workers 2
OPENBLAS_NUM_THREADS=1 python3 benchmarks/bunny_budget/check_termination.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/bunny_budget/audit.py
python3 benchmarks/bunny_budget/analyze.py
```

Two independent design jobs can run concurrently; each oracle gets four
threads. **These elapsed times are not performance results.** Every selected
center, candidate, oracle field, solver summary, and evaluation block is
retained. A configuration/mesh/solver fingerprint rejects mismatched reuse.
The script resumes completed designs and oracle calls after interruption.
The initial cache and all design result records are written atomically.

The termination check fixes the first design seed, full volume budget, counts
2 and 12, and both TD selectors in advance. It reevaluates those four designs
and the initial state with a fourfold thinner termination shell (0.000125),
using the same saved point blocks and base seeds. This is a sensitivity check,
not exact ground truth or an upper bound on the remaining numerical bias.
An additional first-seed check of the half-volume 12-cavity versus full-volume
6-cavity comparison was added after observing that allocation result; the
summary labels it separately from the pre-specified endpoint checks.
