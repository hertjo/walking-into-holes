# Controlled experiment suite

Run the complete seeded suite from the project root:

```bash
python experiments/run_controlled_experiments.py
```

The script emits raw CSV files and a machine-readable `summary.json` under
`experiments/output/`, plus the paper figures
`figures/controlled-experiments.pdf` and `figures/manufactured-field.pdf`.

The suite tests four narrow claims:

1. **Rare-event scaling.** In a concentric annulus, the exact probability that
   Brownian motion started at radius `r` hits an inner disk of radius
   `epsilon` is `log(1/r)/log(1/epsilon)`. This yields an exact controlled
   finite-cavity quotient with the variance scaling analyzed in the paper.
2. **Correlation bias.** Bivariate Gaussian oracle pairs sweep arbitrary
   within-pair correlation and compare same-seed multiplication, multiplication
   of sample means, leave-diagonal-out cross-fitting, and independent batches.
3. **Adaptive selection.** A Gaussian null field compares reusing the selected
   scouting estimate, a simultaneous Bonferroni bound, and fresh validation.
4. **Spatial WoS validation.** A standard vectorized Walk on Spheres solves two
   manufactured harmonic unit-disk problems. Their exact product is known at
   every candidate, so all displayed field errors are measured directly.

These are controlled diagnostics, not the final application benchmarks. In
particular, they do not establish an end-to-end wall-clock or design-quality
advantage over FEM/level-set, Differential WoS, spline/IGA topological
derivatives, or neural topology optimization. Those comparisons require the
application implementations and matched-budget protocol described in the
paper.

