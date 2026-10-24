# Matched-accuracy sparse insertion queries

Frozen on 2026-09-17 before the measured sweep. `output/protocol.json` is authoritative.

Four existing geometries, five random seeds, two domain states, four nested
query counts, three walk budgets, and four multigrid resolutions are retained.
The same fixed sites are used by every method. The inserted cavity is fixed by
geometry before scores are computed. The objective uses distinct compact primal
and adjoint sources; the Bunny therefore differs from the earlier compliance
crossover. This is a new harder test, not a replication of that crossover.

The multigrid implementation uses the existing geometric V-cycle and masked
finite-difference operator. It caches both solved fields for additional queries
on the same domain, and invalidates the cache on changes to the domain, source,
resolution, or solver controls. Each budget and repetition has a fresh process;
mesh setup is measured separately. Timing runs execute sequentially, with a
rotated method order. Raw requests, numerical outputs, and timing metadata are
retained. Solver residuals are algebraic convergence diagnostics, not PDE error.

Accuracy is relative to MG257, with MG193 refinement and independent Monte Carlo
spot checks. These are numerical checks, not exact ground truth. A 20% target
is primary; 10% and 40% targets are sensitivity analyses. For a query prefix,
the refinement discrepancy must be at most one quarter of the target before a
winner is selected. The independent spot check must also be resolved and
consistent under the declared analysis gate. Unresolved cases remain visible.
Bootstrap intervals describe sampling/timing variability, not discretization
or truncation bias. A selected fastest configuration is the best among tested
budgets, not proof of global state of the art.

Build with the study CMake project and target `wih_reuse_query`; execute
`python3 benchmarks/matched_accuracy/run.py` from the project root. The binary
path is currently `/tmp/wih-reuse-build/wih_reuse_query`. No existing experiment
outputs are changed. Numerical-reference requests are excluded from workload
times. The reference and baseline use related discretizations, so independent
Monte Carlo checks and unresolved-case reporting are essential.

## Disclosed follow-up experiments

`incremental_protocol.json` adds outer-grid-mask reuse and warm starts after
cavity insertion. `analytic_protocol.json` adds exact source integration for
the compact radial loads. `compliance_protocol.json` checks the earlier
Bunny distributed-heating PDE type, using one MG solve for the self-adjoint
objective and analytic affine-source integration for TWoS. These were added
after early original results and are not represented as prospectively frozen
parts of the initial four-workload sweep. Their configurations were fixed
before their own measurements.

Build: `python3 benchmarks/matched_accuracy/build.py`. Run the primary sweep
with `run.py`, then the extensions with `build_and_extend.sh`. The recorded
protocol guards against silently mixing binaries or machines. To run an
independent reproduction, preserve the supplied output folder under a different
name before starting a fresh output folder. Do not discard the recorded data.
`check_integral.py` compares the compiled analytic formula to independent direct
Green-volume quadrature. The appendix proves its ideal-arithmetic expectation
and variance properties; numerical tests do not certify floating-point bias.

## Stream-identity correctness amendment

The legacy XOR stream layout aliases a small set of seed/walk identities at
4096 paths. Budgets up to 1024, the high-seed reference batches, and the fresh
holdout seeds do not have this structural overlap. `stream_repair_protocol.json`
describes the correction. `finalize_corrected.sh` enumerates the keys, replaces
all twelve affected path identities at every affected site, reconstructs the
primal/adjoint/cross-product sums by deterministic replay, and checks the result
against full clean runs. The index selection does not depend on path values.
Superseded outputs remain under `output/legacy-streams`; analysis uses corrected
fields only. All additional replay work is charged to corrected query time.
The supplied `wih_clean_*` targets reproduce the corrected benchmark sampling
schedule directly. The legacy targets are retained to reproduce the historical
raw outputs and their correction, not as a recommended general-purpose stream
layout. Formal estimator statements use the explicitly stated ideal independent
sampling model, not a proof of PRNG independence.

## Final interpretation and independent confirmation

The four calibration-selected apparent wins do not pass both fresh-seed rules.
A separately declared fixed-256-path confirmation on 20 further seeds passes
both rules only for eight original-domain distributed-heating Bunny queries:
89.9 ms versus 113.9 ms for MG49 (1.27x; pointwise interval 1.05–1.50).
After an edit, MG warm starts are faster in the mean. These timings exclude
common geometry loading; they are not full design times or a universal SOTA
claim. The numerical target is reference-relative, not certified continuum error.
See both holdout CSVs, `conservative_confirmation_protocol.json`, and the report.

All 360 repeat-query cache checks agree within eight machine epsilons after CSV
serialization (maximum absolute difference 1.36e-20). The initial raw-bitwise
assertion was too strict for round-tripped decimal files; no solver fields were
changed to make that check pass.
