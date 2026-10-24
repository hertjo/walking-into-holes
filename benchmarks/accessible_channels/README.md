# Accessible through-channel comparison

This is a separate experiment built on the saved distinct-adjoint backend.
It does not rerun or replace the prior 90-design / 70-persistent-rerun study.

Each method inserts six channels. Every channel is a vertical union of eighteen
equal balls, radius 0.075 and center spacing 0.075. Its endpoints are beyond the
outer mesh bounds. The union contains a straight open core of radius
`0.075*sqrt(3)/2`; the appendix proves this by a nearest-center argument.
Axes avoid the sensor and three radius-0.07 component balls, with 0.02 gap.
The mutual channel gap is at least 0.025. Ports move with the chosen axes;
there is no fixed inlet/outlet manifold or fluid-flow calculation.

The point topological coefficient `u*p` at z=-0.03 is a **placement heuristic**
for the entire channel. It is not a channel derivative. The single-ball
finite-benefit certificate is not used. All channel walls remain ideal
zero-temperature Dirichlet sinks, and Gaussian source tails are restricted to
surviving material. Protected balls do not convert this to a fixed-power device
or conjugate heat-transfer model.

## Reproduce

Build the retained-mesh executable as described in `../distinct_adjoint/README.md`
at `/tmp/wih-adjoint-server-build/wih_adjoint`. Then run sequentially:

```bash
OPENBLAS_NUM_THREADS=1 python3 benchmarks/accessible_channels/run_accessible.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/accessible_channels/audit_accessible.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/accessible_channels/plot_accessible.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/accessible_channels/render_professional.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/accessible_channels/compose_professional.py
```

Set `WIH_ACCESSIBLE_OUTPUT` to a fresh absolute path to retain a separate rerun.
Existing completed results are reused; incomplete measured designs cause an
error so partial timing cannot silently become a complete-design measurement.
Python dependencies are the same as the distinct-adjoint benchmark; no new PDE
backend was introduced. The exact configuration and source/binary hashes were
written before the first measurements. Run solver timings sequentially.

There are five methods, two sensor objectives on one housing, and five seeds:
50 unique designs. Sparse TWoS and WoS hottest share the 12/192 scout/rescore
budgets; MG evaluates the same point TD on grids 57x37x29 and 113x73x57.
Every method shares 512 candidate axes and the same geometric predicates.
Both numerical solver families retain the outer mesh for a complete design.
End-to-end timing includes common candidate setup, all adaptive requests,
geometry setup, process startup/shutdown, ranking, and file I/O. Offline
reference evaluation is separate.

Four independent scrambled 128-point probe blocks, 256 fresh walks per probe,
and a 113x73x57 MG solve evaluate every completed design. First-seed designs
receive shell and grid refinements. All references remain approximate. Four
scrambled 32,768-point blocks estimate material/source removal using numerical
boundary-mesh membership. Count/radius/extent are matched, but removed material
is **not constrained equal** because the housing thickness is nonuniform.

The source archive retains raw per-round fields, chosen axes, final channels,
reference fields, runtime records, volume probes, and audit results. Geometric
access is established; physical cooling feasibility, a channel-specific
derivative/certificate, and an independent expert proof review remain open.

## Professional comparison figure

`render_professional.py` uses the same Mitsuba/Manifold/OIDN pipeline as the
Bunny hero, with actual saved first-seed right-sensor geometries. It renders
the remaining material after subtracting tessellated ball-union channels and
an identical circular display cut. The display cut is never passed to a PDE
solver. Exterior and channel walls are shaded; section pixels receive a
separate 169x109x85 multigrid temperature field on the uncut physical domain.
All methods share the camera, lighting, zoom center, and absolute color scale.
Bilinear interpolation is for display only; scientific field values are not
denoised or individually rescaled. OIDN filters illumination before section
colors are applied. Rendering requires Mitsuba, trimesh, manifold3d, and OIDN
in addition to the numerical benchmark dependencies.

`compose_professional.py` writes the paper figure and an 8000-pixel PNG. Its
single-design labels come from the retained independent WoS evaluations;
the aggregate bars come from all ten runs of each displayed method. Cooling
error bars use five seed averages, with both sensor cases paired within each
seed. The full 50-design study includes random and finer-grid comparisons in
the accompanying table. The original diagnostic plot can be regenerated as
`accessible-channels-diagnostic.pdf/png`; all physical experimental results
remain unchanged by the figure redesign. Render and composition manifests
record scale, geometry, data source, and uncertainty conventions.

Run `python3 benchmarks/accessible_channels/ghost_shell.py` after rendering
and before composition to restore a faint shell of the removed bars. It uses
the original physical geometry with saved channels, excludes artificial
window faces, and depth-tests the shell against surviving opaque material.
Surface opacity is 4.7%; edge opacity is 24.3%. Magnified sensor views remain
unobstructed. The underlying temperature arrays and original renders are
preserved. The composed PNG is 8000 by 4300 pixels.

Both figure composers check rendered text bounds before exporting.
`python3 scripts/check-hero-labels.py` additionally checks the Penrose hero
schematic and the shared left edge of its three numbered steps.
