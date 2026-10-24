# Walking into Holes — research hero video

82 seconds, 2560 × 1440, 24 fps. H.264 MP4 with a 1920 × 1080 companion export.
The film is designed for silent autoplay, with short labels and equations on screen. An optional SRT caption file supplies fuller explanation. There is no audio track or third-party music.

## Story

| Time | Scene | Evidence |
|---|---|---|
| 00–07 | Design question; the outer shell fades to a curved scoop | Original intact geometry and auxiliary steady field |
| 07–19 | Matched heating of intact and 12-cavity designs | Backward-Euler solutions of the heat equation, same source and model time |
| 19–31 | Actual WoS paths synchronized with Penrose construction | Reproducible paths on the actual mesh and accumulated source contributions |
| 31–43 | Small-cavity sensitivity, then the leave-one-out estimator | Penrose equations and worked three-pair construction |
| 43–55 | Sequential insertion checkpoints and subsequent cooling | Saved 0/1/3/6-cavity designs; auxiliary thermal relaxation on each fixed domain |
| 55–65 | Equal-volume placement comparison | Retained random / TWoS / voxel-adjoint benchmark realization |
| 65–76 | Six large vs twelve small cavities | First prespecified budget-study seed, with ranges over three seeds |
| 76–82 | Method recap | Twelve-cavity design |

## Numerical interpretation

The thermal model is -Δu = 1 + 0.65(y + 0.5), with u = 0 on the exterior and cavity walls. The objective is source-weighted temperature over fixed probes in the original domain. For this compliance objective the adjoint equals the primal: p = u. The general method supports distinct adjoints; this example does not demonstrate their advantage over temperature-only placement.

The ellipsoidal scoop is purely a visualization window. It never changes the PDE. Only cavities exposed by the visible scoop can be seen; all cavities remain in the physical design. The faint shell shows the removed part of the exterior.

All temperature panels share the interval [0, 0.025]. Curved-surface temperatures are trilinearly interpolated from an auxiliary three-dimensional Cartesian-grid solve on the **physical domain**, with the exact saved cavity centers and radii. Surface lighting conveys curvature, and the fine contours are isotherms. These visualization fields are separate from the retained WoS benchmark objective estimates printed on screen. No volumetric grid is introduced into the TWoS design algorithm.

The transient visualization solves ∂u/∂τ − Δu = f, with dimensionless unit diffusivity, zero physical boundary temperature, and source f = 1 + 0.65(y + 0.5). Heat-up starts from u = 0 on each fixed design. After an insertion checkpoint, cooling starts from the preceding design's steady field restricted to the new domain. The solver uses a seven-point Cartesian Dirichlet-mask operator and backward Euler, with 49 stored times through τ = 0.065. Smooth playback interpolates the computed snapshots; the on-screen clock specifies model time. A topology change itself is a discrete edit, not simulated material growth.

`assets/scooped/thermal-audit.json` records residuals and monotonicity. `validation.json` records a manufactured Dirichlet-eigenmode check, a 112-to-144 grid sensitivity check, and a time-step-halving check. Grid refinement changed the intact and twelve-cavity mean steady temperatures by approximately 1.4% and 2.7%; halving the time steps changed the intact heat-up mean curve by at most 0.75% of its steady mean. Residuals are not continuum-error bounds. This auxiliary visualization is not a new benchmark, a continuum ground truth, or a claimed transient TWoS method.

The animated walk traces are new illustrative paths with recorded seeds. Directions are uniform on the 3D sphere. A sphere radius is 0.9999 times the distance to the actual triangle boundary; termination uses an epsilon of 0.0005. They are not advertised as the optimization paths or timing trials. The left Penrose diagram is an orthographic projection of a separate, genuine 3D walk inside an analytic capsule. Its 56 steps use independent normalized-Gaussian directions from three fixed, unselected seeds and exact boundary distances. The capsule and Bunny examples are synchronized by relative progress, not step-by-step spatial correspondence. A 3D sphere exit may project inside its 2D silhouette. The moving straight segments illustrate jumps, not Brownian trajectories. Finite-epsilon stopping is approximate; the displayed samples are not claimed to be exactly unbiased.

Subtle SVG material gradients shade Penrose geometry without moving shapes or labels. The cavity is shaded as a concavity. Three true great circles replace the Bunny sphere’s dense triangle wireframe.

`assets/walk-independent-audit.json` independently checks all 48 Bunny steps using NumPy point-to-triangle distances and solid-angle winding numbers, separately from the VTK oracle that generated them. It also replays seeds and verifies source contributions. `assets/capsule-audit.json` checks all 56 capsule steps, analytic containment, projection, and all 288 walk-animation frames. The direction-law justification is rotational invariance of normalized independent standard Gaussians; a retained moment diagnostic is not presented as proof of a finite sample’s uniformity.

The optimization checkpoints remain saved steady-state designs. The explicitly labeled auxiliary heat sequence and post-insertion relaxation illustrate their thermal behavior. No fluid flow or physical cavity-growth model is asserted. Independent rescoring is not described as the separate certified insertion experiment.

The scoring diagram uses the same leave-one-out identity as Figure 2 of the paper. Retain a temperature estimate, hold out its paired adjoint, average the other adjoints, then multiply. Averaging these terms gives the exact all-cross-pairs estimator. The displayed example has three pairs and zero cavity-wall temperature. Each product has independent factors; the products themselves can share samples.

The comparison shows 39.46% reduction for random placement, 47.77% for TWoS, and 48.06% for voxel-adjoint TD in one retained realization. This supports useful placement versus random selection; it does not establish superiority over grid-based adjoint quality or a universal speed advantage.

The budget pair reports 47.35% for six radius-0.05 cavities and 47.07% for twelve radius-0.031498 cavities. The latter removes half as much volume. Ranges over the three design seeds are 46.78–48.22% and 47.03–47.67%, respectively. These are seed ranges, not confidence intervals.

`assets/evidence.json` records exact values, sources, rendering conventions, and interpretation.

## Reproduce

Use Python 3.12, Node 22, Inkscape, ffmpeg, and the standard fonts-lmodern / fonts-urw-base35 packages. Install `video/requirements.txt` in an isolated environment and `npm ci` at the project root for Penrose 3.3.1.

```bash
python3 video/prepare_video.py
OPENBLAS_NUM_THREADS=1 python3 video/prepare_scooped.py --grid 144
OPENBLAS_NUM_THREADS=1 python3 video/validate_thermal.py
python3 video/prepare_diagrams.py
python3 video/prepare_depth_diagrams.py
node scripts/render-penrose.mjs video-estimator video-sensitivity video-increment video-heat-equation video-capsule video-sphere video-topology video-model
python3 video/shade_penrose.py
inkscape figures/video-estimator.svg -w 1600 -o video/assets/estimator.png
inkscape figures/video-capsule.svg -w 1600 -o video/assets/capsule.png
inkscape figures/video-sphere.svg -w 720 -o video/assets/sphere.png
inkscape figures/video-topology.svg -w 1600 -o video/assets/topology.png
inkscape figures/video-model.svg -w 1600 -o video/assets/model.png
inkscape figures/video-sensitivity.svg -w 1600 -o video/assets/sensitivity.png
inkscape figures/video-increment.svg -w 1600 -o video/assets/increment.png
inkscape figures/video-heat-equation.svg -w 2080 -o video/assets/heat-equation.png
python3 video/audit_walks.py
python3 video/audit_capsule.py
python3 video/update_evidence.py
python3 video/make_video.py --preview
python3 video/audit_video.py
python3 video/make_video.py --out /absolute/path/Walking_into_Holes_Hero_1440p.mp4
```

The supplied Penrose PNGs and auxiliary thermal arrays allow the final command to run without reinstalling Node or Inkscape or recomputing the heat solves. `TWOS_VIDEO_OUTPUT=/absolute/output/path python3 video/render_film.py` renders four independent chapter ranges concurrently, assembles the 1440p master, creates the 1080p export, and verifies frame counts and duration. An EGL-capable OpenGL implementation is needed for VTK's offscreen renderer. The factory `vtkRenderWindow` selects EGL in the production environment. Mesa software rendering is sufficient; no GPU is required.

`python3 video/audit_video.py` checks the recorded geometry, displayed metrics, and inspected text layouts. The rendered sample frames also require visual inspection.

The source bundle includes the precise mesh parts and saved scientific fields needed to recreate the film, with metadata/results from the underlying studies. It is not the complete benchmark repository; the full manuscript/benchmark archive is distributed separately.

## Suggested LinkedIn text

Where should a new cooling cavity go?

Our research project, *Walking into Holes*, uses Monte Carlo topological derivatives to score where a cavity would help—by querying the current geometry with random walks, before making the hole.

This video shows the method on an idealized Stanford Bunny conduction benchmark, with a curved cutaway, actual heat-diffusion visualizations, equal-volume baselines, and a cavity-budget study. In the illustrated budget comparison, twelve smaller cavities give nearly the same cooling benefit as six larger ones while removing half as much material.

The grid-based adjoint baseline remains competitive. The paper examines where pointwise Monte Carlo queries help, along with the assumptions and limits of the theory.
