# Matched experiment results

All entries below are generated from the saved measurements. Point estimates do not establish a winner, equivalence, or a guarantee on another geometry.

Design intervals jointly resample five paired design seeds and four shared evaluation blocks. SD describes the observed spread across five evaluated designs, including evaluation noise. Intervals are approximate, pointwise, and do not cover numerical bias.

## Pipeline

| Geometry | Method | Mean cooling (%) | Seed SD (pp) | 95% interval | Complete time (s) |
|---|---|---:|---:|---:|---:|
| housing | TWoS | 64.990 | 1.412 | [63.723, 66.203] | 11.767 |
| housing | Multigrid TD (65) | 65.095 | 0.588 | [64.251, 65.937] | 8.661 |
| housing | Multigrid TD (49) | 64.507 | 0.666 | [63.675, 65.258] | 7.988 |
| housing | Nearest sensor | 63.557 | 2.603 | [61.309, 65.599] | 7.076 |
| housing | Hottest point | 42.736 | 16.500 | [27.884, 51.064] | 13.138 |
| housing | Random | 21.878 | 15.367 | [10.867, 34.924] | 10.691 |
| branch | TWoS | 16.402 | 0.939 | [15.503, 17.381] | 51.238 |
| branch | Multigrid TD (65) | 16.266 | 1.918 | [14.569, 17.689] | 43.761 |
| branch | Multigrid TD (49) | 16.444 | 1.989 | [14.682, 17.906] | 43.207 |
| branch | Nearest sensor | 17.335 | 0.804 | [16.435, 18.269] | 41.486 |
| branch | Hottest point | 0.139 | 0.040 | [0.012, 0.353] | 13.429 |
| branch | Random | 5.437 | 2.006 | [3.899, 7.371] | 46.650 |
| ring | TWoS | 11.356 | 0.878 | [10.421, 12.260] | 35.779 |
| ring | Multigrid TD (65) | 11.681 | 1.359 | [10.503, 12.885] | 29.673 |
| ring | Multigrid TD (49) | 11.831 | 1.527 | [10.505, 13.170] | 30.303 |
| ring | Nearest sensor | 12.194 | 2.120 | [10.255, 13.628] | 29.955 |
| ring | Hottest point | 0.007 | 0.001 | [0.000, 0.022] | 10.726 |
| ring | Random | 1.594 | 3.541 | [-0.000, 4.835] | 18.630 |

## Channels

| Geometry | Method | Mean cooling (%) | Seed SD (pp) | 95% interval | Complete time (s) |
|---|---|---:|---:|---:|---:|
| housing | TWoS | 37.786 | 5.476 | [32.918, 42.347] | 4.477 |
| housing | Multigrid TD (65) | 42.109 | 3.004 | [39.133, 44.679] | 2.844 |
| housing | Multigrid TD (49) | 42.109 | 3.004 | [39.133, 44.679] | 2.370 |
| housing | Nearest sensor | 39.367 | 4.773 | [34.933, 43.052] | 1.688 |
| housing | Hottest point | 29.484 | 5.641 | [24.560, 33.796] | 2.728 |
| housing | Random | 7.309 | 1.740 | [5.575, 9.288] | 2.098 |
| branch | TWoS | 2.291 | 0.143 | [1.744, 2.889] | 6.546 |
| branch | Multigrid TD (65) | 2.588 | 0.159 | [2.045, 3.194] | 1.629 |
| branch | Multigrid TD (49) | 2.566 | 0.227 | [1.996, 3.199] | 1.160 |
| branch | Nearest sensor | 2.622 | 0.176 | [2.052, 3.233] | 0.826 |
| branch | Hottest point | 1.251 | 1.139 | [0.397, 2.184] | 1.596 |
| branch | Random | 0.931 | 0.290 | [0.572, 1.214] | 0.929 |
| ring | TWoS | 1.750 | 0.270 | [1.479, 1.992] | 8.137 |
| ring | Multigrid TD (65) | 1.933 | 0.164 | [1.742, 2.116] | 2.723 |
| ring | Multigrid TD (49) | 1.890 | 0.126 | [1.727, 2.007] | 2.911 |
| ring | Nearest sensor | 2.019 | 0.111 | [1.880, 2.146] | 1.246 |
| ring | Hottest point | 0.005 | 0.003 | [-0.001, 0.014] | 2.874 |
| ring | Random | 0.203 | 0.254 | [0.015, 0.439] | 1.399 |

## Actual-mesh estimator variance

All entries use 2,048 paths per query site, eight fixed sites, and sixteen repetitions. Ratios compare measured sums of site variances. The finite-cavity estimators target finite differences and retain radius bias relative to the infinitesimal coefficient. These ratios are descriptive point estimates, not confidence bounds.

| Geometry | Trial radius | CRN variance / TWoS variance | Harmonic-correction variance / TWoS variance | Observed correction hit fraction |
|---|---:|---:|---:|---:|
| housing | 0.08 | 19.6 | 8.02 | 0.060608 |
| housing | 0.04 | 59.4 | 12 | 0.026283 |
| housing | 0.02 | 188 | 7.47 | 0.011959 |
| housing | 0.01 | 682 | 35.7 | 0.005867 |
| branch | 0.08 | 386 | 12.9 | 0.043484 |
| branch | 0.04 | 743 | 26.4 | 0.01741 |
| branch | 0.02 | 1.56e+03 | 59.5 | 0.0076485 |
| branch | 0.01 | 4.72e+03 | 13.8 | 0.0036354 |
| ring | 0.08 | 293 | 0.901 | 0.0017586 |
| ring | 0.04 | 1.87e+03 | 2.1 | 0.00059509 |
| ring | 0.02 | 1.13e+03 | unresolved (all zero) | 0.00028992 |
| ring | 0.01 | 3.35e+03 | unresolved (all zero) | 0.00015259 |

## Interpretation limits

- The shape-refinement component is a specialized Differential WoS implementation shared by all initializations; it is not a reproduction of the authors' full application.
- Sensor proximity and tuned multigrid are informed controls. Improvements over hottest-point placement alone do not establish superiority over these controls.
- Channel heat input is fixed by protected compact source supports; removed volume is calibrated and independently audited numerically. No fluid flow is solved.
- Numerical references and rendering fields are approximate. See `reference-audit.csv` and `finite-grid-audit.csv`; sampling intervals do not enclose their discretization or termination errors.
- Query timings replay direct TWoS with chunk size one to use all four workers at eight sites. Component fields match the original outputs bitwise. Complete-design timings retain their original scheduling.
- Runs were serial on a managed shared machine. Timing intervals include algorithmic and machine variation.
- Renderings show the first predetermined seed; statistics use all five. Older hero bands remain observed ranges over three designs, not 95% confidence intervals.
