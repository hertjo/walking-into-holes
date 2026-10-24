"""Record the revised film's sources and interpretation without changing metrics."""
from pathlib import Path
import json
V=Path(__file__).parent;A=V/'assets';p=A/'evidence.json';d=json.loads(p.read_text())
d.update(
 copy_revision={'description':'Removed repeated subtitles, slogans, footer branding, and long explanatory blocks. Kept equations, numerical scales, benchmark scope, and brief scene labels.','audit':'copy-audit.json'},
 video='82 seconds, 2560 x 1440, 24 fps; no audio required',
 temperature='Auxiliary 3D finite-difference temperature fields on the saved physical geometries, evaluated on a curved display cut. All panels share [0,0.025]. Isotherms are actual scalar contours. The source arrays and numerical audits are in assets/scooped.',
 animation='Heat-up from u=0 on the intact and 12-cavity geometries, plus thermal relaxation after saved sequential insertion checkpoints. Time is dimensionless; transitions between saved design checkpoints are edits, not a continuous topology-changing physical process.',
 cutaway='An ellipsoidal scoop, clipped against the actual outer surface and saved spherical cavities. The removed outer patch is faintly visible. This display operation never changes a PDE domain.',
 cavity_growth='Saved cavities appear as discrete geometry changes. Subsequent heat diffusion is solved on each fixed new domain, with the preceding design steady state restricted to that domain as initial data.',
 diagram='Penrose 3.3.1 equations and projected 3D capsule geometry, with SVG material gradients applied without moving any geometry. The 3D zero-Dirichlet small-cavity law is followed by the exact leave-one-out form of the all-cross-pairs estimator. Each product has independent factors; products can share samples. The walk contribution shown applies to the affine source of this example.',
 walk_visualization='Two separate examples: exact analytic 3D capsule and actual Bunny mesh. Both use independent normalized-Gaussian directions with recorded, unselected seeds. Left diagram is an orthographic projection, so exits need not lie on the drawn 2D rim. Straight interpolated segments show jumps, not Brownian trajectories. Finite epsilon stopping is approximate; no exact-unbiasedness claim is made for the stopped paths.',
 walk_audits={name:json.loads((A/name).read_text()) for name in ['walk-independent-audit.json','capsule-audit.json']},
 query_animation='Illustrative query markers are a deterministic subset of points on the curved display surface. The final marker is near the projection of the first retained insertion location. These markers are not optimization samples.',
 surface_renderer='VTK 9.5.2, actual outer mesh and clipped ellipsoidal/spherical surfaces, consistently oriented interior normals, smooth Phong shading, orthographic camera, faint removed-window shell, thin rims and display-only contact shadow.',
 scalar_rendering='Trilinear interpolation of auxiliary 3D solution values to the curved display surface; linear interpolation between computed temporal snapshots. Fixed global colormap; surface lighting conveys curvature. No fitted or invented thermal history.',
 transient_scope='The transient solve is an explanatory visualization, not a claimed transient TWoS method or a new paper benchmark. Published comparison percentages remain the saved WoS objective estimates. No coolant-flow simulation is asserted.',
 thermal_validation=json.loads((A/'scooped/validation.json').read_text()),
 thermal_solver=json.loads((A/'scooped/thermal-audit.json').read_text())
)
p.write_text(json.dumps(d,indent=2)+'\n')
