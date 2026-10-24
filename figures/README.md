# Bunny renders for the hero figure

`hero-assets/` holds the path-traced Stanford Bunny views used by Figure 1 (intact domain, saved random,
voxel-adjoint and TWoS designs, close-ups, wire details and the thermal/benefit sections). The paper figure itself
is drawn from these files by `paper/figure-scripts/f_hero.py`.

Regenerate the renders with `make hero-renders` (Mitsuba on the CPU, fixed seed; see `requirements-render.txt`):

- `render_mesh_hero.py` builds display geometry for the saved designs and path-traces the views;
- `hero_wire_details.py` adds sparse display triangles on the cavity walls of the close-ups;
- `hero_thermal.py` projects fresh temperature and benefit fields onto the section;
- `hero_budget.py` renders the six- and twelve-cavity designs of the count/volume study.

The large opening is a visualization window, not a physical hole; display tessellations never enter the solver.
`layout_checks.py` is a small text-overlap check used by the benchmark plotting scripts.
